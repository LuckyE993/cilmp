from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
import os.path as osp
import math

from llm_representation_utils import resolve_llm_representation_path

"""CLIP 主体网络与 CILMP/IVLP 改造实现。

文件结构：
1) 原始 CLIP 基础模块（ResNet/ViT/Transformer）；
2) CILMP 相关低秩干预模块（LoReFT + 条件化版本 + LoRA 投影）；
3) 带 Prompt 注入的残差块 ResidualAttentionBlock_IVLP；
4) 从预训练 state_dict 自动反推结构并构建模型。
"""

class Bottleneck(nn.Module):
    """ModifiedResNet 使用的瓶颈残差块。"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """将二维特征图展平后做一次自注意力池化，取 cls-like token 输出。"""
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # NCHW -> (HW)NC，并在最前面拼接全局均值 token 作为聚合查询。
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual 主干
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """兼容半精度输入的 LayerNorm（内部用 fp32 计算再转回）。"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """CLIP 中常用的近似 GELU 激活。"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """标准 Transformer 残差块（原始 CLIP 版本）。"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LoraProjection(nn.Module):
    """低秩投影层：in_dim -> out_dim，通过 A/B 两个低秩矩阵实现。"""
    def __init__(self, in_dim, low_rank_dim, out_dim):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(low_rank_dim, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, low_rank_dim))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        # 先映射到低秩空间，再映射回目标维度。
        x_dtype = x.dtype
        x = x.to(torch.float32)
        after_A = F.linear(x, self.lora_A)
        after_B = F.linear(after_A, self.lora_B)
        return after_B.to(x_dtype)



class LowRankRotateLayer(nn.Module):
    """LoReFT 使用的旋转层 R，参数用正交初始化。"""

    def __init__(self, n, m):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, embed_dim, low_rank_dimension):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = rotate_layer
        self.learned_source = torch.nn.Linear(embed_dim, low_rank_dimension)
        self.dropout = torch.nn.Dropout(0.1)
        self.act_fn = ACT2FN["linear"]
        
    def forward(self, base):
        # LoReFT: h + R^T(Wh+b-Rh)
        base_dtype = base.dtype
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base_dtype))



class ConditionalLoreftIntervention(nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, embed_dim, low_rank_dimension, ctx_dim, n_cls):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, low_rank_dimension)
        self.rotate_layer = rotate_layer
        self.learned_source = torch.nn.Linear(embed_dim * 2, low_rank_dimension)
        self.dropout = torch.nn.Dropout(0.1)
        self.act_fn = ACT2FN["linear"]
        self.image_feature_learned_weight = nn.Sequential(
            torch.nn.Linear(ctx_dim, low_rank_dimension),
            torch.nn.Linear(low_rank_dimension, embed_dim)
        )
        self.n_cls = n_cls
        self.low_rank_dimension = low_rank_dimension





    def forward(self, base, img_feature):
        """图像条件化 LoReFT。

        参数：
        - base: [n_cls, d]（类别 token 表示）
        - img_feature: [bs, ctx_dim]（每个样本的图像特征）
        返回：
        - [bs, n_cls, d]，即每个样本对应一套类别表示。
        """
        base_org = base
        img_feature_shape = img_feature.shape  
        processed_img_feature = self.image_feature_learned_weight(img_feature) 
        # 与 base 做逐维交互，形成条件输入。
        rd = processed_img_feature.unsqueeze(1) * base.unsqueeze(0)
        bs = rd.shape[0]
        base2 = torch.cat([base.unsqueeze(0).repeat(bs, 1, 1), rd], dim=-1)
        processed_base = self.learned_source(base2)
        base_shifted = processed_base
        base_dtype = base.dtype
        rotated_base = self.rotate_layer(base_org)
        output = []
        # 逐样本执行 LoReFT 变换。
        for base_shifted_i in base_shifted:
            tmp = base_org + torch.matmul(
                (self.act_fn(base_shifted_i) - rotated_base), self.rotate_layer.weight.T
            )
            output.append(self.dropout(tmp.to(base_dtype)))
        
        return torch.stack(output)



class ResidualAttentionBlock_IVLP(nn.Module):
    """支持 Prompt 注入与 LLM 干预的残差注意力块（IVLP/CILMP 扩展版）。"""
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # 约定：第 0 层不注入深层 prompt；从第 1 层开始可注入。
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    # 文本分支：初始化文本上下文 token，并加载每类 LLM 表示。
                    self.n_ctx_text = design_details["language_ctx"]
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                    classnames = design_details['classnames']
                    llm_hidden_representations = []
                    llm_rep_length = []
                    prefix_length = design_details['prefix_length']
                    suffix_length = design_details['suffix_length']
                    llm_embed_dim = 4096
                    low_rank_dimension = design_details['low_rank_dimension']
                    max_len = 0
                    min_len = 1000
                    n_cls = len(classnames)
                    dataset_name = design_details.get("dataset_name", "")
                    for name in classnames:
                        path = resolve_llm_representation_path(
                            classname=name,
                            dataset_name=dataset_name,
                            classnames=classnames,
                        )

                        llm_hidden_rep = torch.load(path)
                        max_len = max(max_len, len(llm_hidden_rep))
                        min_len = min(min_len, len(llm_hidden_rep))
                        llm_hidden_representations.append(llm_hidden_rep)
                        llm_rep_length.append(len(llm_hidden_rep))
                    self.llm_rep_length = llm_rep_length
                    self.total_length = max_len
                    self.min_rep_length = min_len
                    self.prefix_length = prefix_length
                    self.suffix_length = suffix_length
                    # 变长表示 padding 到统一长度，便于批量操作。
                    self.llm_prompt = torch.empty(len(llm_hidden_representations), max_len, llm_embed_dim).to(ctx_vectors.dtype)
                    for index, hr in enumerate(llm_hidden_representations):
                        self.llm_prompt[index, :len(hr), :] = hr.to(ctx_vectors.dtype)
                    self.llm_prompt = self.llm_prompt.cuda().half()
    
                    self.prefix_length = prefix_length
                    self.suffix_length = suffix_length
                    self.low_rank_dimension = low_rank_dimension
                    # 只对前后缀 token 干预，中间主体 token 保持。
                    self.prefix_intervention = ConditionalLoreftIntervention(llm_embed_dim, low_rank_dimension, d_model, n_cls)
                    self.suffix_intervention = ConditionalLoreftIntervention(llm_embed_dim, low_rank_dimension, d_model, n_cls)
                    self.lora_proj = LoraProjection(llm_embed_dim, low_rank_dimension, d_model)

                else:
                    # 视觉分支：仅维护可学习视觉 prompt token。
                    self.n_ctx_visual = design_details["vision_ctx"]
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False


    def forward_llm(self, img_fea):
        """根据图像特征生成条件化后的 LLM token（映射到 d_model）。"""
        prefix = self.llm_prompt[:, :self.prefix_length] 
        after_prefix_int = []
        for i in range(self.prefix_length):
            rep = prefix[:, i]
            after_prefix_int.append(self.prefix_intervention(rep, img_fea).unsqueeze(-2))
        after_prefix_int = torch.cat(after_prefix_int, dim=-2)  

        bs = after_prefix_int.shape[0]
        # 快路径：类别表示长度一致，可完全向量化处理。
        if self.llm_rep_length[0] ==  self.llm_rep_length[-1]:
            total_length = self.llm_rep_length[0]
            suffix = self.llm_prompt[:, total_length - self.suffix_length:] 
            after_suffix_int = []
            for i in range(self.suffix_length):
                rep = suffix[:, i]  
                after_suffix_int.append(self.suffix_intervention(rep, img_fea).unsqueeze(-2))  
            after_suffix_int = torch.cat(after_suffix_int, dim=-2) 
            unchanged = self.llm_prompt[:, self.prefix_length: total_length - self.suffix_length] 
            unchanged = unchanged.unsqueeze(0).repeat(bs, 1, 1, 1)
            after_int = torch.cat([
                after_prefix_int,
                unchanged,
                after_suffix_int
            ], dim=-2)
            bs2, n_cls2, len_total2, dim2 = after_int.shape
            assert bs == bs2
            after_int = after_int.view(-1, len_total2, dim2) 
            n_cls, max_len, dim = after_int.shape
            after_int = after_int.view(-1, dim)  
            final_output = self.lora_proj(after_int)
            final_output = final_output.reshape(n_cls, max_len, -1)
            final_output_dim = final_output.shape[-1]
            final_output = final_output.view(bs2, n_cls2, len_total2, final_output_dim)
            return final_output
        else: 
            # 慢路径：类别长度不一致，逐类构建后回填。
            after_suffix_int = []
            unchanged = []
            for i in range(len(self.llm_prompt)):  
                after_suffix_int_i = []
                suffix = self.llm_prompt[i, self.llm_rep_length[i] - self.suffix_length:, :].unsqueeze(0)  
                unchanged_i = self.llm_prompt[i, self.prefix_length:self.llm_rep_length[i] - self.suffix_length, :].unsqueeze(0) 
                unchanged_i = unchanged_i.unsqueeze(0).repeat(bs, 1, 1, 1)  
                unchanged.append(unchanged_i)
                for j in range(self.suffix_length):
                    rep = suffix[:, j] 
                    after_suffix_int_i.append(self.suffix_intervention(rep, img_fea).unsqueeze(-2))  
                after_suffix_int_i = torch.cat(after_suffix_int_i, dim=-2) 
                after_suffix_int.append(after_suffix_int_i)
            output_shape = (bs,) + self.llm_prompt.shape
            after_int = torch.zeros(output_shape).cuda()
            for i in range(len(self.llm_prompt)): 
                after_int_i = torch.cat([
                    after_prefix_int[:, i].unsqueeze(1), 
                    unchanged[i],  
                    after_suffix_int[i]
                ], dim=-2) 
                after_int[:, i, :self.llm_rep_length[i], :] = after_int_i.squeeze(1)
            bs2, n_cls2, len_total2, dim2 = after_int.shape
            assert bs == bs2
            after_int = after_int.view(-1, len_total2, dim2) 
            n_cls, max_len, dim = after_int.shape
            after_int = after_int.view(-1, dim) 
            final_output = self.lora_proj(after_int)
            final_output = final_output.reshape(n_cls, max_len, -1)
            final_output_dim = final_output.shape[-1]
            final_output = final_output.view(bs2, n_cls2, len_total2, final_output_dim)
            return final_output


    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        """支持两种输入：
        1) 纯张量 x（普通 Transformer）；
        2) 字典 {"x", "img_feature"}（文本分支需要图像条件干预）。
        """
        if type(x) == dict:
            dict_flag = True
        else:
            dict_flag = False
        if type(x) == dict:
            img_feature = x['img_feature']
            x = x['x']
            img_feature = img_feature.to(torch.float16)
            x = x.to(torch.float16)

        if self.add_prompt:
            if not self.text_layer:
                # 视觉分支：替换末尾 token 为可学习视觉 prompt。
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # 文本分支：注入上下文 token，并将类别 LLM token 替换为条件化结果。
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                x = torch.cat([prefix, textual_context, suffix], dim=0)
                x = x.permute(1, 0, 2) 
                new_llm_prompt = self.forward_llm(img_feature) 
                bs, n_cls, len_total, dim2 = new_llm_prompt.shape
                x = x.view(bs, n_cls, -1, dim2) 
                len_total3, dim3 = x.shape[2:]
                for i in range(x.shape[1]): 
                    x[:, i, 1 + self.n_ctx_text: 1 + self.n_ctx_text + self.llm_rep_length[i], :] = new_llm_prompt[:, i, :self.llm_rep_length[i], :]
                x = x.view(-1, len_total3, dim3)
                x = x.permute(1, 0, 2)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))


        if dict_flag:
            return {"x": x, "img_feature": img_feature}

        return x


class Transformer(nn.Module):
    """Transformer 容器。

    当 trainer 为 IVLP/VPT 时使用改造版残差块；
    否则回退为原始 CLIP 残差块。
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        else:
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, img_feature=None):
        # 文本分支在 CILMP 下可额外携带 img_feature 用于条件干预。
        if img_feature is not None:
            param = {'x': x, 'img_feature': img_feature}
            return self.resblocks(param)
        return self.resblocks(x)



class VisionTransformer(nn.Module):
    """CLIP 的 ViT 视觉编码器，支持视觉 prompt（VPT）注入。"""
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            n_ctx = design_details["vision_ctx"]
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, img_feature=None):
        # 图像切 patch 并线性映射到 token 序列。
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        if self.VPT_shallow:
            # 在 token 序列末尾追加视觉 prompt token。
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, img_feature)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x




class CLIP(nn.Module):
    """CLIP 主模型（视觉编码器 + 文本编码器 + 对比式 logits）。"""
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        # ResNet 与 ViT 两种视觉主干均兼容。
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                design_details=design_details
            )
        # 文本 Transformer 中启用 prompt 的深度。
        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        """按 CLIP 习惯初始化各模块参数。"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """构建因果 mask（上三角 -inf）。"""
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        # 取 EOT 位置作为句子特征。
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        # 标准 CLIP：归一化特征后做双向相似度。
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """将可转换参数转为 fp16，减少显存占用。"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    """从 checkpoint 自动反推结构参数并构建 CLIP 模型。"""
    vit = "visual.proj" in state_dict

    if vit:
        # ViT: 由 patch/pos embedding 反推分辨率与层数。
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        # ResNet: 由 layer{1..4} 的 block 数反推结构。
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    # 一些元数据键不属于可学习参数，加载前删除。
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        # 为兼容改造结构，允许非严格加载并打印缺失键。
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
