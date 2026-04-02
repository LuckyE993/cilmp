# CILMP Validation Logic

本文档只解释当前仓库里的“验证/评估实际是怎么跑的”。这里刻意把“代码里的 validation/test 流程”和“论文里汇报实验指标的逻辑”拆开，因为两者并不完全一致。

说明：文中提到的 `Dassl.pytorch/...` 文件不在当前仓库内，而是来自 `cilmp/requirements.txt` 固定的 `Dassl.pytorch` 依赖版本。

## 0. 先看结论

这套仓库默认没有“训练中每个 epoch 跑验证集挑最优模型”的流程。

默认行为其实是：

- 训练结束后直接测试最后模型
- `eval.sh` 也是直接加载训练输出并在 test set 上评估
- 默认评估指标只有 `accuracy`、`error_rate`、`macro_f1`
- `AUC` 和 `Kappa` 并没有在当前代码里实现

## 1. 代码里“验证”和“测试”的区别

### 1.1 DataManager 的确会构造 `val_loader`

只要数据集 split 里有 `val`，`Dassl` 的 `DataManager` 就会建出：

- `train_loader_x`
- `val_loader`
- `test_loader`

对应文件：

- `Dassl.pytorch/dassl/data/data_manager.py:115-149`

### 1.2 但默认训练流程不会用 `val_loader`

原因有两个：

1. 默认配置里 `TEST.FINAL_MODEL = "last_step"`
2. `Dassl.after_epoch()` 只有在 `TEST.FINAL_MODEL == "best_val"` 时才会每个 epoch 跑一次 `self.test(split="val")`

当前默认配置来自 `Dassl`：

- `TEST.SPLIT = "test"`
- `TEST.FINAL_MODEL = "last_step"`

对应文件：

- `Dassl.pytorch/dassl/config/defaults.py:203-216`
- `Dassl.pytorch/dassl/engine/trainer.py:422-443`

所以按默认脚本训练时：

- 不会用验证集选模型
- 只会在训练结束后跑一次最终测试

这也和论文实现细节里的描述一致：训练过程中不使用验证集。

## 2. 评估入口文件

### 2.1 单独评估脚本

入口脚本是 `cilmp/scripts/cilmp/eval.sh`。

它会：

- 对 `SEED=1,2,3` 循环
- 进入每个训练输出目录
- 调用 `python train.py --eval-only --model-dir ${DIR}`
- 同时打开 `TEST.PER_CLASS_RESULT True`

对应文件：

- `cilmp/scripts/cilmp/eval.sh:1-26`

### 2.2 评估脚本也默认要求工作目录在 `cilmp/`

和训练脚本一样，`eval.sh` 里使用的也是：

```bash
python train.py
```

以及相对配置路径 `configs/...`。所以它也默认假设你当前所在目录是：

```text
/home/qixuan/cilmp/cilmp
```

### 2.3 训练结束后的自动测试

如果你走的是训练脚本而不是 `eval.sh`，那么 `trainer.train()` 结束后，`Dassl.after_train()` 也会自动调用一次 `self.test()`。

所以这个仓库里有两种“评估发生的时机”：

1. 训练完自动测一次
2. 之后再用 `eval.sh` 单独测一次

## 3. `eval-only` 模式的完整调用链

默认评估调用链如下：

`scripts/cilmp/eval.sh` -> `train.py --eval-only` -> `build_trainer(cfg)` -> `trainer.load_model(model_dir, epoch)` -> `trainer.test()` -> `evaluator.process()` -> `evaluator.evaluate()`

对应关键位置：

- `cilmp/train.py:158-180`
- `cilmp/trainers/cilmp.py:619-643`
- `Dassl.pytorch/dassl/engine/trainer.py:445-473`
- `Dassl.pytorch/dassl/evaluation/evaluator.py:50-125`

这里要特别注意：`build_trainer(cfg)` 内部会先完整构造 `CILMP(cfg)`，而 `CILMP(cfg)` 的基类初始化又会先 `build_data_loader()`、再 `build_model()`。所以即便是 `--eval-only`，代码也会先把整套 `CustomCLIP`、prompt learner、LLM 表征读取逻辑全部跑一遍，然后才会加载 checkpoint。

## 4. `load_model()` 到底加载哪个 checkpoint

### 4.1 CILMP 重写了默认 `load_model()`

这是验证逻辑里最关键的一个特殊点。

`Dassl` 默认会加载 `model-best.pth.tar`。但 CILMP 自己重写了 `load_model()`，默认文件名被改成了：

```python
model_file = "model.pth.tar-100"
```

只有显式传了 `--load-epoch` 时，才会改成 `model.pth.tar-{epoch}`。

对应文件：

- `cilmp/trainers/cilmp.py:619-643`

### 4.2 为什么默认正好是 `model.pth.tar-100`

因为训练配置的 `MAX_EPOCH=100`，而 `Dassl` 保存 checkpoint 时用的是 `epoch + 1` 作为编号，所以最后一个 epoch 会保存为：

```text
model.pth.tar-100
```

模型实际路径是：

```text
${DIR}/VLPromptLearner/model.pth.tar-100
```

这里的 `VLPromptLearner` 来自训练时的：

```python
self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
```

对应文件：

- `cilmp/trainers/cilmp.py:529`
- `Dassl.pytorch/dassl/utils/torchtools.py:51-63`

### 4.3 加载时会删掉两个 buffer

加载 checkpoint 时，CILMP 还会手动删掉：

- `prompt_learner.token_prefix`
- `prompt_learner.token_suffix`

然后用 `strict=False` 加载。

这说明这两个 buffer 视为“可按当前类别名重新生成的静态内容”，不要求和 checkpoint 完全匹配。

对应文件：

- `cilmp/trainers/cilmp.py:633-643`

## 5. `trainer.test()` 真正做了什么

`Dassl` 的通用测试流程如下：

1. `self.set_model_mode("eval")`
2. `self.evaluator.reset()`
3. 决定评估 split
4. 遍历 `data_loader`
5. `parse_batch_test(batch)` 拿到 `img` 和 `label`
6. `model_inference(input)` 拿到 logits
7. `evaluator.process(output, label)` 累加预测结果
8. `evaluator.evaluate()` 输出最终指标

对应文件：

- `Dassl.pytorch/dassl/engine/trainer.py:445-485`

### 5.1 split 的选择逻辑

如果显式传入 `split="val"` 且 `val_loader` 存在，就评估 val。

否则：

- 用 `cfg.TEST.SPLIT`
- 但如果 `val_loader` 不存在，仍会自动回退到 `test`

默认配置里 `cfg.TEST.SPLIT = "test"`，所以 `eval.sh` 实际评估的是 test set，不是 val set。

## 6. 验证时模型前向是怎么走的

### 6.1 `model_inference()` 默认就是 `self.model(input)`

`CILMP` 没有重写 `model_inference()`，所以验证时直接调用 `CustomCLIP.forward(image)`。

对应文件：

- `Dassl.pytorch/dassl/engine/trainer.py:475-476`

### 6.2 验证时仍然是“图像条件化 prompt”

这点很重要。

`CustomCLIP.forward()` 在没有传 `label` 时会走推理分支，但它依然会：

1. 先算 `image_features`
2. 再把 `image_features` 送进 `prompt_learner(image_features)`
3. 让 prompt 生成过程依赖当前图像

也就是说，验证/测试时不是用固定 prompt，而是继续使用图像自适应的 prompt。

对应文件：

- `cilmp/trainers/cilmp.py:457-484`

### 6.3 预测分数是什么

最终输出仍然是：

```python
logit_scale * image_feature @ normalized(text_feature).T
```

这和论文式 (15) 的推理逻辑一致。本代码没有显式对 logits 做 softmax 再存概率，因为分类评估只需要比较大小，argmax 结果与 softmax 后 argmax 相同。

## 7. 验证指标是怎么计算的

### 7.1 当前代码只实现了三类主指标

`Dassl` 默认分类评估器会输出：

- `accuracy`
- `error_rate`
- `macro_f1`

如果 `TEST.PER_CLASS_RESULT=True`，还会再输出：

- 每个类别的 accuracy
- `perclass_accuracy`

如果 `TEST.COMPUTE_CMAT=True`，还会额外保存混淆矩阵。

对应文件：

- `Dassl.pytorch/dassl/evaluation/evaluator.py:26-125`

### 7.2 `eval.sh` 和 `train.sh` 都开启了 per-class result

脚本最后都传了：

```bash
TEST.PER_CLASS_RESULT True
```

因此无论你是：

- 训练结束后的自动测试
- 还是手动执行 `eval.sh`

都会打印每个类别的分类正确率。

对应文件：

- `cilmp/scripts/cilmp/train.sh:18-26`
- `cilmp/scripts/cilmp/eval.sh:15-25`

### 7.3 论文里的 AUC / Kappa 不在当前代码里

论文第 IV-B 写了四项指标：

- Accuracy
- F1-score
- AUROC
- Kappa score

但当前仓库的默认 evaluator 并没有实现：

- AUC
- Kappa

所以现在这份代码的“验证逻辑”和论文里的“结果汇报逻辑”并不是完全一致的。

如果你后面要严格复现实验表格，这一点必须额外补代码。

## 8. 训练完成后的自动测试到底测的是哪份模型

默认训练流程结束时，`after_train()` 不会重新从磁盘加载 checkpoint，而是直接测试当前内存中的模型。

对 CILMP 来说，当前模型在最后一个 epoch 的最后一个 batch 后已经被替换为：

- 大部分参数做过高斯加权平均后的状态
- `intervention` / `lora_proj` 保留最新 epoch 的状态

所以训练后自动测试和 `eval.sh` 的默认加载目标，理论上都指向同一份“100 epoch 结束后的最终模型”。

## 9. 当前仓库里“validation”最容易混淆的几个点

### 9.1 有 `val_loader`，但默认不用它挑模型

这是代码结构和默认运行策略之间最容易让人误判的地方。

### 9.2 `eval.sh` 名叫 eval，但默认测的是 test，不是 val

因为默认 `TEST.SPLIT="test"`，脚本也没有改这个配置。

### 9.3 默认加载不是 `model-best.pth.tar`

因为 `CILMP.load_model()` 覆盖掉了 `Dassl` 默认行为，直接指定加载 `model.pth.tar-100`。

### 9.4 输出指标和论文不完全一致

当前代码只能直接得到：

- accuracy
- macro_f1
- error_rate
- per-class accuracy

不能直接得到：

- AUROC
- Kappa

### 9.5 `eval-only` 也依赖 `llm_representations`

评估时不是“直接把 checkpoint 反序列化完就结束”，而是会先构建 `VLPromptLearner`。因此如果仓库外部依赖目录 `llm_representations/.../*.pth` 不存在，验证流程会在 `build_model()` 阶段就失败，而不是在 `load_model()` 阶段失败。

## 10. 验证链路中的关键文件和关键函数

### 10.1 必看文件

- `cilmp/scripts/cilmp/eval.sh`
- `cilmp/train.py`
- `cilmp/trainers/cilmp.py`
- `Dassl.pytorch/dassl/engine/trainer.py`
- `Dassl.pytorch/dassl/evaluation/evaluator.py`

### 10.2 必看函数

- `train.py::main`
- `CILMP.load_model`
- `CustomCLIP.forward`
- `SimpleTrainer.test`
- `Classification.process`
- `Classification.evaluate`

## 11. 如果你把“代码验证逻辑”压缩成一句话

这份仓库默认的验证/评估逻辑其实就是：

“加载 `output/.../VLPromptLearner/model.pth.tar-100`，在 test set 上用图像条件化 prompt 做前向，取 argmax，汇报 accuracy、macro-f1 和 per-class accuracy。”

## 12. 后续你读代码时建议先抓这条主线

建议按下面顺序读最省力：

1. `scripts/cilmp/eval.sh`
2. `train.py` 里的 `eval_only` 分支
3. `trainers/cilmp.py::load_model`
4. `Dassl::test`
5. `CustomCLIP.forward`
6. `Dassl::Classification.evaluate`

这样你能最快分清楚：

- “哪里在选模型”
- “哪里在跑前向”
- “哪里在算最终指标”
- “论文里哪些指标并没有落到当前代码里”
