# CE-CSL Gloss Recognition V1 — Codex 工作说明

本文档用于指导 Codex 后续接手 `experiments/ce_csl_gloss_recognition_v1` 实验线时的修改、调参和新增实验。

当前实验目标：基于 CE-CSL 数据集，使用已经提取好的 MediaPipe 特征训练连续手语 gloss 序列识别模型。当前主 baseline 为 **BiLSTM + CTC Loss**。

---

## 0. 最重要的协作规则

### 0.1 每次调整必须新建版本，不允许覆盖旧实验

后续每次做实验或调参，都必须新建脚本、新建 checkpoint 目录、新建 log 目录。

禁止直接覆盖当前已有的有效实验目录，尤其是：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta
D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta
```

如果要做新实验，应该使用新版本命名，例如：

```text
versions/v004_raw_delta_ter_scheduler/train.py
versions/v005_raw_delta_hidden384/train.py
versions/v006_raw_delta_delta/train.py
```

对应输出目录也必须独立，例如：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_ter_scheduler
D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_ter_scheduler
```

每个实验只改一个主要变量。不要一次同时改 hidden size、dropout、feature mode、scheduler、sampler、batch size 等多个变量，否则无法判断到底是哪项改动有效。

---

## 1. 当前项目位置

项目根目录：

```text
MySssb/
```

当前实验目录：

```text
experiments/ce_csl_gloss_recognition_v1/
```

主要代码结构：

```text
experiments/ce_csl_gloss_recognition_v1/
  docs/
    FEATURE_SPEC.md

  scripts/
    01_manifest/
    02_inspect_raw/
    03_features/
    04_training_checks/
    05_train/

  src/
    ce_csl/
      __init__.py
      dataset.py
      model.py
      ctc_decode.py
      ctc_beam_decode.py
      metrics.py  # 若不存在，可按需新建
```

数据集根目录固定为：

```text
D:\CE-CSL\CE-CSL
```

当前特征文件目录：

```text
D:\CE-CSL\CE-CSL\processed\features
```

当前 CTC ready 清单目录：

```text
D:\CE-CSL\CE-CSL\processed\ctc_ready
```

不要把数据集、`.npy`、checkpoint、日志文件提交进 Git。

---

## 2. 当前已经完成的链路

已经完成并验证：

```text
1. manifest 构建
2. 全量视频特征提取
3. 特征完整性检查
4. CTC ready 检查
5. PyTorch Dataset / DataLoader 检查
6. BiLSTM-CTC forward + CTCLoss 检查
7. overfit 20 小样本验证
8. subset 500 拟合验证
9. full train v1 raw baseline
10. full train v2 raw_delta baseline
11. full train v3 raw_delta_reg 对照实验
12. greedy vs beam search 解码对照
13. token 频率与错误类型诊断
```

---

## 3. 当前核心模型

文件：

```text
experiments/ce_csl_gloss_recognition_v1/src/ce_csl/model.py
```

当前模型：

```text
BiLstmCtcModel
```

结构：

```text
B × T × input_dim
↓
Linear(input_dim → 256)
↓
LayerNorm
↓
ReLU
↓
Dropout
↓
2-layer BiLSTM(hidden_size=256, bidirectional=True)
↓
Dropout
↓
Linear(hidden_size * 2 → vocab_size)
↓
log_softmax
```

CTC blank id：

```text
BLANK_ID = 0
```

当前词表大小：

```text
vocab_size = 3840
```

---

## 4. Dataset 当前能力

文件：

```text
experiments/ce_csl_gloss_recognition_v1/src/ce_csl/dataset.py
```

`CeCslGlossDataset` 已支持：

```python
feature_mode="raw"
feature_mode="raw_delta"
```

### 4.1 raw

输入维度：

```text
T × 166
```

### 4.2 raw_delta

读取原始 `T × 166` 后，在 Dataset 中临时构造一阶差分：

```text
delta[t] = feature[t] - feature[t - 1]
```

最终输入：

```text
T × 332
```

注意：`raw_delta` 不会重新生成 `.npy` 文件，只是在训练时临时拼接特征。

---

## 5. 当前实验结果

### 5.1 v1：raw baseline

脚本：

```text
experiments/ce_csl_gloss_recognition_v1/versions/v001_raw/train.py
```

输出目录：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc
D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc
```

结果：

```text
best_dev_TER  = 0.7228
best_dev_loss = 5.4695
```

---

### 5.2 v2：raw_delta baseline

脚本：

```text
experiments/ce_csl_gloss_recognition_v1/versions/v002_raw_delta/train.py
```

输出目录：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta
D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta
```

结果：

```text
best_dev_TER  = 0.7018
best_dev_loss = 5.3171
```

当前最佳模型：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta\best_dev_ter.pt
```

结论：

```text
raw_delta 明确优于 raw。
```

主要提升来源：

```text
1. 正确 token 数增加
2. delete / 漏词减少
3. 中长句 TER 有改善
```

---

### 5.3 v3：raw_delta + 稍强正则 + dev_loss scheduler

脚本：

```text
experiments/ce_csl_gloss_recognition_v1/versions/v003_raw_delta_reg/train.py
```

输出目录：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_reg
D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_reg
```

配置变化：

```text
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 3e-4
input_dropout = 0.3
lstm_dropout = 0.4
output_dropout = 0.4
scheduler = ReduceLROnPlateau(dev_loss)
```

结果：

```text
best_dev_TER  = 0.7069
best_dev_loss = 5.2157
```

结论：

```text
v3 dev_loss 更稳，但 dev_TER 不如 v2。
正则和学习率调度压得略重。
当前最佳仍然是 v2 raw_delta。
```

---

## 6. beam search 对照结果

脚本：

```text
experiments/ce_csl_gloss_recognition_v1/versions/v001_raw/evaluate_beam_search.py
```

基于 v1 raw 的 best model 进行测试。

结果：

```text
greedy                TER=0.7228
beam3_top30_bonus0    TER=0.7228
beam5_top30_bonus0    TER=0.7237
beam5_top50_bonus0    TER=0.7237
beam5_top50_bonus02   TER=0.7232
beam10_top50_bonus02  TER=0.7232
```

结论：

```text
beam search 基本没有收益。
当前瓶颈不主要在解码，而在模型概率分布、特征表达和训练策略。
```

---

## 7. 诊断结论

### 7.1 v1 raw 诊断

```text
totalReferenceTokens = 2327
totalCorrectTokens   = 669
totalErrorTokens     = 1658
totalSubstitute      = 885
totalDelete          = 773
```

### 7.2 v2 raw_delta 诊断

```text
totalReferenceTokens = 2327
totalCorrectTokens   = 724
totalErrorTokens     = 1603
totalSubstitute      = 903
totalDelete          = 700
```

对比：

```text
correct +55
delete  -73
substitute +18
```

结论：

```text
raw_delta 主要减少漏词，并提升整体正确 token 数。
但替换错误仍然很多。
```

---

## 8. token 频率问题

v2 raw_delta 按 train 频次分桶：

```text
00_oov          acc=0.0000
01_once         acc=0.0156
02_count_2_3    acc=0.0309
03_count_4_10   acc=0.0442
04_count_11_50  acc=0.1861
05_count_51+    acc=0.5761
```

结论：

```text
1. OOV 不是主因，dev OOV token 只有 14 个。
2. 真正困难是长尾低频词。
3. train 出现 <= 10 次的 token 基本学不动。
4. 高频 token 能学，但仍不够稳定。
```

---

## 9. 当前主要问题画像

当前最佳 v2 raw_delta 的问题：

```text
1. 预测仍偏短，但比 v1 改善。
2. 低频 token 几乎识别不动。
3. 高频 token 仍有较高替换错误。
4. 长句明显更难。
5. dev_loss 与 dev_TER 不完全同步。
6. 单纯 beam search 无明显收益。
```

当前最佳结果：

```text
best_dev_TER = 0.7018
```

后续实验应以此为基准，任何新实验都要和这个值比较。

---

## 10. 后续推荐实验顺序

### 实验 A：v4 raw_delta + dev_TER scheduler

目的：保留 v2 的训练强度，但让学习率调度看最终识别指标 `dev_TER`，而不是 `dev_loss`。

建议复制：

```text
experiments/ce_csl_gloss_recognition_v1/versions/v002_raw_delta/train.py
```

生成：

```text
experiments/ce_csl_gloss_recognition_v1/versions/v004_raw_delta_ter_scheduler/train.py
```

输出目录：

```text
D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_ter_scheduler
D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_ter_scheduler
```

推荐配置：

```python
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

input_dropout = 0.2
lstm_dropout = 0.3
output_dropout = 0.3
```

调度器：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
    threshold=1e-4,
    min_lr=5e-5,
)
```

每轮验证后：

```python
scheduler.step(dev_ter)
```

目标：

```text
尝试将 best_dev_TER 从 0.7018 压到 0.69x。
```

---

### 实验 B：hidden_size=384

如果 v4 没有明显提升，尝试增加模型容量。

建议复制 v2：

```text
versions/v002_raw_delta/train.py
```

生成：

```text
versions/v005_raw_delta_hidden384/train.py
```

改模型配置：

```python
model_config = {
    "input_dim": MODEL_INPUT_DIM,
    "projection_dim": 256,
    "hidden_size": 384,
    "num_layers": 2,
    "vocab_size": vocab_size,
    "input_dropout": 0.2,
    "lstm_dropout": 0.3,
    "output_dropout": 0.3,
}
```

注意：

```text
BiLSTM 输出维度会从 512 变成 768。
model.py 中 output_layer 已经使用 hidden_size * 2，一般不用额外改。
```

目标：

```text
观察更大 hidden_size 是否能改善高频 token 稳定性和长句识别。
```

---

### 实验 C：raw_delta + delta_delta

如果 hidden_size 没有明显提升，可继续增强特征。

输入：

```text
raw:         x_t
delta:       x_t - x_{t-1}
delta_delta: delta_t - delta_{t-1}
```

最终维度：

```text
T × 498
```

需要在 `dataset.py` 中增加：

```python
feature_mode="raw_delta_delta"
```

注意：不要覆盖 raw 和 raw_delta。新模式必须向后兼容。

---

### 实验 D：低频样本重采样

如果诊断持续显示低频 token 完全学不动，可以考虑 WeightedRandomSampler。

思路：

```text
包含低频 token 的样本获得更高采样权重。
```

注意：这会改变训练分布，可能提升低频，但也可能损伤高频。需要单独实验，不要和模型结构改动混在一起。

---

## 11. Codex 修改原则

每次实验只改一个主要变量。

不要一次同时改：

```text
hidden_size
dropout
feature_mode
scheduler
batch_size
sampler
```

否则无法判断哪项有效。

每个新实验都要：

```text
1. 新建独立 train_xxx.py。
2. 新建独立 checkpoint 输出目录。
3. 新建独立 log 输出目录。
4. 保留 v2 raw_delta baseline。
5. 跑完后记录 best_dev_TER / best_dev_loss。
6. 对 best_dev_ter.pt 做预测诊断。
7. 必要时做 token 频率诊断。
```

禁止直接覆盖当前最佳实验目录：

```text
full_bilstm_ctc_raw_delta
```

---

## 12. 当前验收基准

当前最佳 baseline：

```text
v2 raw_delta
best_dev_TER = 0.7018
```

新实验有效的最低标准：

```text
best_dev_TER < 0.7018
```

较明显提升：

```text
best_dev_TER <= 0.69
```

值得重点保留：

```text
best_dev_TER <= 0.68
```

如果新实验只是降低 `dev_loss`，但 `dev_TER` 不如 v2，则不能算最终识别效果提升。

---

## 13. 常用运行命令

全量 v2 raw_delta 训练：

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v002_raw_delta\train.py
```

v3 训练：

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v003_raw_delta_reg\train.py
```

raw_delta 预测诊断：

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v002_raw_delta\inspect_predictions.py
```

raw_delta token 频率诊断：

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v002_raw_delta\analyze_token_frequency.py
```

beam search 评估：

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v001_raw\evaluate_beam_search.py
```

---

## 14. 重要注意事项

1. `feature_mode="raw_delta"` 时，模型输入维度必须是：

```text
MODEL_INPUT_DIM = 332
```

2. `feature_mode="raw"` 时，模型输入维度必须是：

```text
MODEL_INPUT_DIM = 166
```

3. 如果 checkpoint 里的模型配置是 `input_dim=332`，但 Dataset 没有传 `feature_mode="raw_delta"`，会报输入维度错误。

4. Windows 下 DataLoader 暂时使用：

```python
NUM_WORKERS = 0
```

如需提速，可以单独实验 `NUM_WORKERS = 2`，但不要和训练策略改动混在一起。

5. 当前 GPU 可用，训练日志中应显示：

```text
device: cuda
```

6. 当前显卡为 RTX 5060，CUDA 版 PyTorch 已可用。

7. 不要把 `D:\CE-CSL\CE-CSL` 下的数据文件加入 Git。

---

## 15. 版本目录约定

后续正式实验优先放到：

```text
experiments/ce_csl_gloss_recognition_v1/versions/
```

当前已整理：

```text
versions/
  README.md
  v001_raw/
    train.py
    README.md
  v002_raw_delta/
    train.py
    README.md
  v003_raw_delta_reg/
    train.py
    README.md
  v004_raw_delta_ter_scheduler/
    train.py
    README.md
  v005_raw_delta_hidden384/
    train.py
    README.md
  v006_raw_delta_delta/
    train.py
    README.md
```

旧的 `scripts/05_train/train_full*.py` 已移动到 `versions/`；版本专属评估/诊断脚本也放在对应版本目录。`scripts/05_train` 只保留小样本训练和训练链路检查工具。

新版本建议使用：

```text
versions/v007_raw_delta_lowfreq_sampler/train.py
```

每个版本文件夹至少包含：

```text
train.py
README.md
```

版本 README 需要记录：

```text
1. 主要改动
2. 运行命令
3. checkpoint/log 输出目录
4. best_dev_TER / best_dev_loss
5. 实验结论
```

---

## 16. 已完成新增实验结论

v4 / v5 / v6 / v7 已跑完：

```text
v2 raw_delta baseline:
  best_dev_TER  = 0.7018
  best_dev_loss = 5.3171

v4 raw_delta + dev_TER scheduler:
  best_dev_TER  = 0.7039
  best_dev_loss = 5.3171
  结论：接近 v2，但未超过。

v5 raw_delta + hidden_size=384:
  best_dev_TER  = 0.8101
  best_dev_loss = 6.3493
  结论：明显变差。

v6 raw_delta_delta:
  best_dev_TER  = 0.7379
  best_dev_loss = 5.6325
  结论：能学，但不如 raw_delta，后段过拟合更明显。

v7 raw_delta + lowfreq sampler:
  best_dev_TER  = 0.7563
  best_dev_loss = 6.0332
  结论：单纯低频样本重采样没有救起低频桶，反而拉低整体。
```

当前最佳仍然是：

```text
v2 raw_delta
best_dev_TER = 0.7018
```

---

## 17. 当前推荐的下一步

v7 的负结果说明不要继续加大 sampler。下一步应该回到 v2 raw_delta，有意识地改输入表达或时序建模。

优先考虑：

```text
versions/v008_raw_delta_tcn_frontend/train.py
```

核心思想：

```text
保留 v2 raw_delta 的有效配置；
在 BiLSTM 前面加一个轻量 Temporal Conv frontend；
让模型先在局部时间窗口里抽取短动作模式，再交给 LSTM/CTC。
```

注意：

```text
1. v8 不改 feature_mode，不改 sampler，不改训练集分布。
2. 只在模型前端加局部时序卷积，优先验证是否能降低 delete/substitute。
3. 如果 v8 不行，再考虑 attention/Conformer 或显式关键点归一化特征。
```

v7 频率诊断对比：

```text
v2 tokenAccuracy = 0.3111
v7 tokenAccuracy = 0.2557

bucket acc:
00_oov           v2 0.0000 -> v7 0.0000
01_once          v2 0.0156 -> v7 0.0234
02_count_2_3     v2 0.0309 -> v7 0.0062
03_count_4_10    v2 0.0442 -> v7 0.0155
04_count_11_50   v2 0.1861 -> v7 0.1165
05_count_51_plus v2 0.5761 -> v7 0.5029
```
