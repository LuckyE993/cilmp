# CILMP Training Logic

本文档只解释当前仓库里的“训练实际是怎么跑起来的”，不是论文方法综述。重点是把从 shell 脚本到一个 batch 的前向、反向、保存 checkpoint 的完整调用链串起来。

说明：文中提到的 `Dassl.pytorch/...` 文件不在当前仓库内，而是来自 `cilmp/requirements.txt` 固定的 `Dassl.pytorch` 依赖版本。

## 0. 先看结论

这套训练逻辑的核心路径是：

`scripts/cilmp/train.sh` -> `train.py` -> `Dassl` 的 `build_trainer(cfg)` -> `trainers/cilmp.py::CILMP` -> `Dassl` 的 `TrainerX.run_epoch()` -> `CILMP.forward_backward()` -> `CustomCLIP.forward()` -> `VLPromptLearner.forward()` + `TextEncoder.forward()` -> `F.cross_entropy(...)`

训练结束后，代码不会选最佳验证集模型，而是直接用最后一轮的当前模型做测试。这里的“最后一轮当前模型”又不是纯最后一轮权重，而是混入了一个高斯加权平均逻辑。

## 1. 实际入口文件

### 1.1 训练脚本

入口脚本是 `cilmp/scripts/cilmp/train.sh`。

它做了几件事：

- 固定 `TRAINER=CILMP`
- 固定配置文件 `configs/trainers/CILMP/vit_b16.yaml`
- 对 `SEED=1,2,3` 依次训练
- 输出目录命名为 `output/${DATASET}/CILMP/vit_b16_16shots/seed${SEED}`
- 传入 `DATASET.NUM_SHOTS 16` 和 `TEST.PER_CLASS_RESULT True`

对应代码见：

- `cilmp/scripts/cilmp/train.sh:1-27`

### 1.2 工作目录要求

这个脚本内部执行的是：

```bash
python train.py \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/${TRAINER}/${CFG}.yaml
```

因此它默认假设当前工作目录是仓库里的子目录 `cilmp/`，也就是：

```text
/home/qixuan/cilmp/cilmp
```

如果你站在仓库根目录 `/home/qixuan/cilmp` 直接执行脚本，这些相对路径会对不上。

### 1.3 README 和实际脚本路径不完全一致

`README.md` 写的是 `bash scripts/train.sh ${DATASET}`，但仓库里真实存在的是 `scripts/cilmp/train.sh`。以代码目录结构为准。

## 2. 配置是如何合并的

`train.py` 的配置顺序非常重要，最终优先级如下：

1. `Dassl` 默认配置
2. `train.py::extend_cfg()` 里额外加入的 CILMP/PromptSRC/IVLP 相关字段
3. 数据集 yaml，例如 `configs/datasets/isic.yaml`
4. 训练器 yaml，例如 `configs/trainers/CILMP/vit_b16.yaml`
5. 命令行参数，例如 `--root --seed --trainer`
6. 命令最后的 `opts`，例如 `DATASET.NUM_SHOTS 16`

对应函数：

- `cilmp/train.py::extend_cfg`
- `cilmp/train.py::reset_cfg`
- `cilmp/train.py::setup_cfg`

关键点：

- 默认测试相关配置来自 `Dassl`，其中 `TEST.FINAL_MODEL = "last_step"`，不是 `best_val`
- `vit_b16.yaml` 把训练改成了：
  - batch size 64
  - test batch size 100
  - 100 epoch
  - SGD
  - cosine scheduler
  - fp16
  - text/vision prompt depth 都是 12
  - prefix/suffix intervention 长度都为 4

对应文件：

- `cilmp/train.py:73-155`
- `cilmp/configs/trainers/CILMP/vit_b16.yaml:1-45`
- `Dassl.pytorch/dassl/config/defaults.py:11-216`

## 3. 训练器是怎么被注册和构造的

`train.py` 顶部先显式 import 了：

- `import trainers.cilmp`
- 一组医学数据集模块，例如 `datasets.isic`、`datasets.adam` 等

这一步不是“多余 import”，而是为了触发注册：

- `CILMP` 通过 `@TRAINER_REGISTRY.register()` 注册到 `Dassl`
- 各数据集类通过 `@DATASET_REGISTRY.register()` 注册到 `Dassl`

之后 `build_trainer(cfg)` 才能按字符串 `cfg.TRAINER.NAME == "CILMP"` 找到对应类。

对应文件：

- `cilmp/train.py:4-22`
- `cilmp/trainers/cilmp.py:487-488`
- `Dassl.pytorch/dassl/engine/build.py:1-8`

## 4. DataLoader 和数据集 split 的真实来源

### 4.1 Dassl 先构造 DataManager

`CILMP(cfg)` 继承自 `Dassl` 的 `TrainerX`。在基类 `SimpleTrainer.__init__()` 里，会先调用：

1. `build_data_loader()`
2. `build_model()`
3. `build_evaluator()`

也就是说，模型构建时已经拿得到 `self.dm.dataset.classnames`。

对应文件：

- `Dassl.pytorch/dassl/engine/trainer.py:341-358`

### 4.2 DataManager 会同时构造 train / val / test loader

`Dassl` 的 `DataManager` 默认会：

- 从注册表里按 `cfg.DATASET.NAME` 构造 dataset
- 按训练增强构造 `train_loader_x`
- 按测试增强构造 `val_loader` 和 `test_loader`

当前仓库的默认 train/test 变换分别来自：

- 训练：`random_resized_crop`, `random_flip`, `normalize`
- 测试：`build_transform(..., is_train=False)` 生成的测试变换

对应文件：

- `Dassl.pytorch/dassl/data/data_manager.py:51-149`
- `cilmp/configs/trainers/CILMP/vit_b16.yaml:8-13`

### 4.3 医学数据集通常直接读取固定 split JSON

以 `ISIC` 为例：

- 数据集目录是 `${DATASET.ROOT}/isic`
- split 文件是 `split_isic.json`
- 通过 `OxfordPets.read_split()` 直接读出 `train, val, test`

也就是说，这些医学数据集不是训练时在线切分，而是依赖预先准备好的 split 文件。

对应文件：

- `cilmp/datasets/isic.py:19-48`
- `cilmp/datasets/oxford_pets.py:122-138`

### 4.4 `NUM_SHOTS=16` 对当前医学数据集基本不起作用

虽然训练脚本传了 `DATASET.NUM_SHOTS 16`，但像 `ISIC`、`ADAM` 这类医学数据集类并没有调用 few-shot 采样逻辑，因此这个参数多数情况下只是保留了 Prompt Learning 代码框架的接口。

这也和脚本里的注释一致：

```bash
SHOTS=16 # doesn't use
```

## 5. 模型构建：真正的 CILMP 在哪几个文件里

训练时最关键的不是一个文件，而是三个层次叠在一起：

1. `trainers/cilmp.py`
2. `clip/model.py`
3. `Dassl` 基类训练循环

### 5.1 `load_clip_to_cpu()`：先下载 OpenAI CLIP，再换成自定义 CLIP

`trainers/cilmp.py::load_clip_to_cpu()` 会：

1. 按 backbone 名称从 `clip._MODELS` 里取下载链接
2. 下载原始 CLIP 权重
3. 调用本仓库改写过的 `clip.build_model(...)`

这里最容易误解的一点是：`design_details["trainer"]` 被硬编码成了 `"IVLP"`，不是 `"CILMP"`。这说明 CILMP 的实现是复用 IVLP/PromptSRC 风格的“支持深层 prompt 的 CLIP 骨架”，再在其上叠加医学知识干预逻辑。

对应文件：

- `cilmp/trainers/cilmp.py:21-52`
- `cilmp/clip/clip.py:23-66`
- `cilmp/clip/model.py:743-780`

### 5.2 `CILMP.build_model()`：用数据集 classnames 初始化模型

`CILMP.build_model()` 做了这些事：

1. 从 `self.dm.dataset.classnames` 拿类别名
2. 加载 CLIP 主干
3. 构造 `CustomCLIP`
4. 冻结大部分参数
5. 建 optimizer 和 lr scheduler
6. 注册模型名为 `"VLPromptLearner"`
7. 初始化高斯参数平均相关状态

对应文件：

- `cilmp/trainers/cilmp.py:492-543`

### 5.3 哪些参数真的会训练

代码的冻结策略不是“只训练 prompt_learner”这么简单。

`CILMP.build_model()` 中的逻辑是：

- 名字里包含 `prompt_learner` 的参数默认保留可训练
- 但 `prompt_learner.ZS_image_encoder` 强制冻结
- 名字不含 `prompt_learner` 的参数里，只要包含：
  - `VPT`
  - `intervention`
  - `lora_proj`
  也会被设成可训练
- 其余全部冻结

因此实际可训练部分包括：

- 浅层文本 prompt 参数 `ctx`
- 条件干预模块 `prefix_intervention` / `suffix_intervention`
- 将 LLM 表征投到 CLIP 文本维度的 `lora_proj`
- `clip/model.py` 里深层 visual/text prompt 相关的 `VPT_*`
- `clip/model.py` 里深层的 `intervention` / `lora_proj`

## 6. Prompt 和医学知识是怎么接进来的

### 6.1 `VLPromptLearner` 负责浅层 prompt 组装

`VLPromptLearner` 是训练链路里最关键的模块之一。

它做了几件事：

1. 初始化可学习文本上下文 `ctx`
2. 根据类别集合，去相对路径 `llm_representations/.../*.pth` 读取每一类的 LLM 表征
3. 统计每个类别的表征长度 `llm_rep_length`
4. 构造带有 `H H H ...` 占位符的 tokenized prompt
5. 构造 prefix 和 suffix 的条件 LoReFT 干预模块
6. 用 `LoraProjection` 把 4096 维 LLM 表征投到 CLIP 文本维度

对应文件：

- `cilmp/trainers/cilmp.py:202-338`

### 6.2 这里依赖一个仓库外部资源：`llm_representations`

当前仓库里没有 `llm_representations/` 目录，但代码训练时会直接读取：

- `llm_representations/isic/*.pth`
- `llm_representations/adam/*.pth`
- `llm_representations/kvasir/*.pth`
- ...

所以“代码逻辑能读懂”不等于“当前仓库可直接启动训练”。要真正训练，还必须补齐这批 `.pth` 文件。

从代码看，`torch.load(path)` 返回的对象至少应当是形如 `[L_h, 4096]` 的张量，因为后面会按 `len(...)` 当作序列长度，并送入低秩投影层。

### 6.3 论文公式和代码对应

这部分和论文《Medical Knowledge Intervention Prompt Tuning for Medical Image Classification》第 III 节可以直接对照：

- 论文式 (13) 条件干预
  - 代码：`ConditionalLoreftIntervention.forward()`
  - 文件：`cilmp/trainers/cilmp.py:144-183`
- 论文式 (9) 低秩投影
  - 代码：`LoraProjection.forward()`
  - 文件：`cilmp/trainers/cilmp.py:187-200`
- 论文式 (10) 拼接自适应疾病提示
  - 代码：`VLPromptLearner.construct_prompts()`
  - 文件：`cilmp/trainers/cilmp.py:340-355`

### 6.4 代码里不止“浅层 prompt”，还有“深层 prompt”

如果你只看 `trainers/cilmp.py`，会以为所有 prompt 都在 `VLPromptLearner` 里构造完了。但实际上 `clip/model.py` 还会在 CLIP 的 Transformer 层内部再做一遍 prompt 注入。

关键逻辑在：

- `ResidualAttentionBlock_IVLP`
- `Transformer`
- `VisionTransformer`

这部分实现了两类事情：

- 视觉分支的浅层和深层 VPT
- 文本分支的深层 prompt 注入，以及在每一层内部再次插入经过干预的 LLM prompt

对应文件：

- `cilmp/clip/model.py:291-537`
- `cilmp/clip/model.py:541-589`

所以训练时真正生效的是：

- 浅层文本 prompt：`VLPromptLearner`
- 深层文本 prompt：`clip/model.py` 内部 block
- 视觉 prompt：`clip/model.py` 内部 block

## 7. 一个 batch 的前向过程

### 7.1 外层训练循环来自 Dassl

`trainer.train()` 最终会走到 `TrainerX.run_epoch()`：

1. `self.set_model_mode("train")`
2. 遍历 `train_loader_x`
3. 每个 batch 调一次 `self.forward_backward(batch)`

对应文件：

- `Dassl.pytorch/dassl/engine/trainer.py:583-637`

### 7.2 CILMP 自己只重写了 `forward_backward()`

`CILMP.forward_backward()` 的流程是：

1. `parse_batch_train()` 取出 `img` 和 `label`
2. `self.model(image, label)` 做前向
3. 非 `amp` 模式下取返回 tuple 的第 0 个元素作为 loss
4. `loss.backward()`
5. `optim.step()`
6. 每个 epoch 最后一个 batch 才更新 scheduler，并做高斯参数平均

对应文件：

- `cilmp/trainers/cilmp.py:546-581`

### 7.3 `CustomCLIP.forward()`：图像特征和文本特征如何产生

`CustomCLIP.forward()` 是训练时真正算 logits 的地方：

1. `self.image_encoder(image)` 提取图像特征
2. 图像特征做 L2 normalize
3. `self.prompt_learner(image_features)` 生成当前 batch 条件化的 prompt
4. `self.text_encoder(...)` 生成每个类别的文本特征
5. 文本特征做 L2 normalize
6. 计算 `logit_scale * image_features @ text_features^T`
7. 训练态下返回 `F.cross_entropy(logits, label)` 及一堆附加对象

对应文件：

- `cilmp/trainers/cilmp.py:444-484`

### 7.4 当前训练真正用到的损失只有交叉熵

虽然代码里还保留了 `fixed_embeddings`、`zero_shot_features`、`zero_shot_logits` 这些变量名，看起来像 PromptSRC/蒸馏式损失残留，但当前 CILMP 训练分支里它们最终都是 `None`，真正反向传播的只有：

```python
F.cross_entropy(logits, label)
```

对应文件：

- `cilmp/trainers/cilmp.py:475-482`

这和论文式 (14) 的监督对比形式在实现上是一致的：本质上就是对图像特征和各类别文本特征的相似度做 softmax 分类。

## 8. Scheduler 和高斯参数平均

### 8.1 学习率更新不是每个 batch，都不是每个 optimizer step

`forward_backward()` 只在一个 epoch 的最后一个 batch 上执行：

- `self.update_lr()`

所以这里的 cosine scheduler 是按 epoch 更新，而不是按 iteration 更新。

### 8.2 这份代码有一个额外的高斯加权平均逻辑

这是当前仓库训练逻辑里最特别、也最容易漏掉的一段：

- 每个 epoch 结束时复制当前 `model.state_dict()`
- 用一个高斯权重 `self.gauss[epoch_idx]` 去加权
- 累加到 `self.previous_model_gpa`
- 训练结束时把累计后的权重重新 load 回模型

对应文件：

- `cilmp/trainers/cilmp.py:531-580`

### 8.3 但它不是“所有参数都平均”

`state_dict_weighting()` 和 `state_dict_add()` 对参数做了分流：

- 名字里包含 `intervention` 或 `lora_proj` 的参数不做高斯平均
- 这些参数在累计时始终保留“当前 epoch 的最新值”
- 其余参数才参与高斯加权求和

也就是说，最终测试用的模型是：

- 大部分 prompt / backbone 相关参数：高斯加权平均结果
- intervention / lora projection：最后一个 epoch 的最新权重

对应文件：

- `cilmp/trainers/cilmp.py:584-606`

### 8.4 为什么最后保存的是 `model.pth.tar-100`

`Dassl.save_model()` 存 checkpoint 时使用的是 `epoch + 1`。

所以：

- 第 0 个 epoch 结束会保存成 `model.pth.tar-1`
- 第 99 个 epoch 结束会保存成 `model.pth.tar-100`

由于默认 `CHECKPOINT_FREQ=0`，因此只会在最后一个 epoch 保存一次。

对应文件：

- `Dassl.pytorch/dassl/engine/trainer.py:422-443`
- `Dassl.pytorch/dassl/utils/torchtools.py:27-63`

## 9. 训练结束后会发生什么

训练结束会进入 `Dassl` 的 `after_train()`：

1. 打印 `Finish training`
2. 因为 `TEST.NO_TEST=False`，会继续做测试
3. 因为 `TEST.FINAL_MODEL="last_step"`，不会加载 best model
4. 直接对当前内存中的模型执行 `self.test()`

对应文件：

- `Dassl.pytorch/dassl/engine/trainer.py:402-420`

由于 CILMP 在最后一个 batch 后已经把高斯平均后的权重 load 回当前模型，所以这里测试的就是那份“高斯平均后的最终模型”。

## 10. 训练链路中的关键文件和关键函数

### 10.1 必看文件

- `cilmp/scripts/cilmp/train.sh`
- `cilmp/train.py`
- `cilmp/trainers/cilmp.py`
- `cilmp/clip/model.py`
- `cilmp/configs/trainers/CILMP/vit_b16.yaml`
- `cilmp/configs/datasets/*.yaml`
- `cilmp/datasets/*.py`
- `Dassl.pytorch/dassl/engine/trainer.py`
- `Dassl.pytorch/dassl/data/data_manager.py`

### 10.2 必看函数

- `train.py::setup_cfg`
- `train.py::main`
- `CILMP.build_model`
- `CILMP.forward_backward`
- `VLPromptLearner.__init__`
- `VLPromptLearner.forward_llm`
- `VLPromptLearner.construct_prompts`
- `CustomCLIP.forward`
- `TextEncoder.forward`
- `ConditionalLoreftIntervention.forward`
- `TrainerX.run_epoch`
- `SimpleTrainer.after_train`

## 11. 论文到代码的最短映射

如果你现在只想快速建立“论文公式 -> 代码位置”的脑图，可以先记这几个点：

- 论文 III-B / III-C 的“医学知识干预”
  - `cilmp/trainers/cilmp.py::ConditionalLoreftIntervention`
- 论文 III-B 的“低秩线性投影”
  - `cilmp/trainers/cilmp.py::LoraProjection`
- 论文 III-B 的“拼接自适应疾病提示”
  - `cilmp/trainers/cilmp.py::construct_prompts`
- 论文 III-D 的训练目标
  - `cilmp/trainers/cilmp.py::CustomCLIP.forward`
- 论文实验设置里的 deep prompt
  - `cilmp/clip/model.py::ResidualAttentionBlock_IVLP`

## 12. 训练逻辑里的几个易错点

### 12.1 当前仓库缺少 `llm_representations`

这是实际运行训练前必须补齐的外部资源。

### 12.2 默认是 `fp16`，不是 `amp`

`vit_b16.yaml` 里 `PREC: "fp16"`，所以默认分支走的是：

- 半精度模型
- 非 autocast
- 常规 `loss.backward()`

不是自动混合精度训练。

### 12.3 `amp` 分支从代码上看是有风险的

`forward_backward()` 的 `amp` 分支里：

```python
loss = model(image, label)
```

但训练态 `model(image, label)` 返回的是 tuple，不是纯标量 loss。默认配置不会走到这条分支，所以日常训练不受影响；但如果你后面把 `PREC` 改成 `amp`，要优先检查这里。

### 12.4 输出目录名里的 `16shots` 只是命名习惯

对当前医学数据集训练逻辑来说，这个名字并不代表真的在做 16-shot 采样。
