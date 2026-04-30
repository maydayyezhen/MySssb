# CE-CSL Gloss Recognition Versions

本目录按实验版本保存正式全量训练入口和版本专属评估/诊断工具。旧的 `scripts/05_train/train_full*.py` 已移动到这里；`scripts/05_train` 保留小样本训练和训练链路检查工具。

数据、checkpoint、log 仍然写到 `D:\CE-CSL\CE-CSL\processed` 下，不放进 Git。

## Current Best

当前 full vocab 最佳训练路线：

```text
v018_raw_delta_full_vocab_from_topk1000_pretrain
best_dev_TER(greedy) = 0.6816
best_dev_loss = 4.6301
```

当前 full vocab 最低验证 TER 解码结果：

```text
v018_raw_delta_full_vocab_from_topk1000_pretrain/best_dev_ter.pt
beam_size = 3
top_k_per_frame = 20
blank_penalty = 0.0
token_insert_bonus = 0.0
dev_TER = 0.6798
```

当前稳定 baseline：

```text
v002_raw_delta
best_dev_TER = 0.7018
```

相比 v002，v018 + beam3 解码将 dev TER 从 `0.7018` 降到 `0.6798`，绝对提升 `0.0220`。

## Version Summary

| Version                                          |                                          Main change | Best dev TER | Best dev loss | Conclusion                                             |
| ------------------------------------------------ | ---------------------------------------------------: | -----------: | ------------: | ------------------------------------------------------ |
| v001_raw                                         |                                 raw feature baseline |       0.7228 |        5.4695 | 被 raw_delta 超过                                         |
| v002_raw_delta                                   |                                 raw + delta baseline |       0.7018 |        5.3171 | 原始稳定 baseline                                          |
| v003_raw_delta_reg                               |                            更强正则 + dev_loss scheduler |       0.7069 |        5.2157 | loss 更稳但 TER 不如 v2                                     |
| v004_raw_delta_ter_scheduler                     |                                    dev_TER scheduler |       0.7039 |        5.3171 | 接近 v2 但未超过                                             |
| v005_raw_delta_hidden384                         |                                      hidden_size=384 |       0.8101 |        6.3493 | 明显变差                                                   |
| v006_raw_delta_delta                             |                            raw + delta + delta_delta |       0.7379 |        5.6325 | 能学但不如 raw_delta                                        |
| v007_raw_delta_lowfreq_sampler                   |                          raw_delta + lowfreq sampler |       0.7563 |        6.0332 | 未改善低频，且拉低整体                                            |
| v008_raw_delta_tcn_frontend                      |                             raw_delta + TCN frontend |       0.8775 |        6.4803 | 早停，明显落后 v2                                             |
| v009_raw_delta_presence_mask                     |                       raw_delta + hand presence mask |       0.8711 |        6.5103 | 早停，明显落后 v2                                             |
| v010_raw_delta_packed_lstm                       |                            raw_delta + packed BiLSTM |       0.8466 |        6.3419 | 早停，明显落后 v2                                             |
| v011_raw_delta_transformer_ctc                   |                          raw_delta + Transformer CTC |       0.7520 |        5.5395 | 能学但输出更短，不如 v2                                          |
| v012_raw_delta_finetune_v2_lr2e4                 |             raw_delta + fine-tune v2 best checkpoint |       0.6940 |        7.2620 | 早期最佳训练路线；配合 beam/blank 解码最低 0.6889                     |
| v013_raw_delta_blankpenalty_finetune             |                  raw_delta + blank-penalty fine-tune |       0.6910 |        7.3869 | epoch0 解码校准有效，继续训练无提升                                  |
| v014_raw_delta_finetune_swa                      |             raw_delta + fine-tune with SWA averaging |       0.6932 |        7.3367 | SWA 未胜出，beam/blank 最低 0.6914                           |
| v015_raw_delta_finetune_seed2026                 |                      raw_delta + fine-tune seed 2026 |      Not run |       Not run | 已创建但按用户要求暂停，未训练                                        |
| v016_raw_delta_controlled_vocab                  |    raw_delta + top_k_1000 controlled vocab + `<unk>` |       0.6519 |        3.9080 | 诊断实验；证明长尾词表明显拖累模型，但 `<unk>` 会简化任务，不能直接视作 full vocab 提升 |
| v017_raw_delta_topk1000_filtered                 | raw_delta + top_k_1000 closed vocab filtered samples |       0.7881 |        5.5513 | 负结果；过滤样本导致训练集从 4973 降到 2138，损失训练信号，不适合作为主线             |
| v018_raw_delta_full_vocab_from_topk1000_pretrain |        v016 encoder pretrain -> full vocab fine-tune |       0.6816 |        4.6301 | 当前 full vocab 最强训练路线；受控词表预训练迁移有效                       |

## Key Findings

### 1. `raw_delta` 是当前最稳定的输入特征

`v002_raw_delta` 将 `v001_raw` 的 dev TER 从 `0.7228` 降到 `0.7018`，说明动作变化量对 CE-CSL gloss recognition 有明确帮助。后续实验应继续以 `raw_delta` 为主要输入基线。

### 2. 小幅调参收益有限

`v003` 到 `v015` 主要围绕正则、scheduler、hidden size、额外 delta、采样、TCN、presence mask、packed LSTM、Transformer CTC、fine-tune 与 SWA 做调整。多数版本没有超过 `v002` 或只获得很小收益，说明当前瓶颈不只是学习率、dropout、hidden size 这类局部超参数。

### 3. 全词表长尾是重要瓶颈

`v016_raw_delta_controlled_vocab` 使用 `top_k_1000 + <unk>` 后，controlled vocab dev TER 达到 `0.6519`。这说明原始 3840 级 gloss 词表中的长尾 token 明显拖累模型。

但该结果不能直接当作 full vocab 成绩，因为 dev 中低频 token 被映射为 `<unk>`，任务被简化。

### 4. 不能粗暴过滤含低频 token 的样本

`v017_raw_delta_topk1000_filtered` 只保留完全由 top_k_1000 token 构成的样本：

```text
train original sample size = 4973
train filtered sample size = 2138
train removed sample size = 2835

dev original sample size = 515
dev filtered sample size = 267
dev removed sample size = 248
```

该版本 best dev TER 为 `0.7881`，明显差于 v002 与 v016。结论是：过滤样本会严重损失训练信号，不适合作为解决长尾问题的主线。

### 5. 受控词表预训练可以迁移到 full vocab

`v018_raw_delta_full_vocab_from_topk1000_pretrain` 从 `v016` 的 `best_dev_ter.pt` 加载 encoder 权重：

```text
loaded encoder keys = 20
skipped keys = 2
```

跳过的参数为 full vocab 维度不匹配的输出层：

```text
output_layer.weight
output_layer.bias
```

该版本回到完整 3840 gloss 词表后，greedy dev TER 达到 `0.6816`，优于 v002 的 `0.7018`。说明 top_k1000 controlled vocab 任务可以作为有效预训练任务，为 full vocab 任务提供更好的 encoder 初始化。

### 6. V18 的提升主要来自减少替换错误

对比 v002 与 v018 的 token 频率诊断：

```text
V2:
totalCorrectTokens = 724
totalSubstitute    = 903
totalDelete        = 700
tokenAccuracy      = 0.3111

V18:
totalCorrectTokens = 766
totalSubstitute    = 811
totalDelete        = 750
tokenAccuracy      = 0.3292
```

V18 的主要变化：

```text
正确 token +42
替换错误 -92
删除错误 +50
```

结论：预训练迁移让模型更少把一个 token 错认成另一个 token，但输出仍然偏短，删除错误仍然较多。

### 7. V18 的 beam search 有小幅收益，但 blank penalty / insert bonus 无收益

V18 decode calibration 结果：

```text
greedy_blank0:
TER = 0.6816
avgPredLen = 3.1107

beam3_top20_blank0_bonus0:
TER = 0.6798
avgPredLen = 3.1184
```

加入 blank penalty 或 token insert bonus 后 TER 反而变差，说明当前模型虽然输出偏短，但强行鼓励更长输出会引入更多错词。当前最优解码配置是：

```text
beam_size = 3
top_k_per_frame = 20
blank_penalty = 0.0
token_insert_bonus = 0.0
```

## Current Recommended Next Step

下一步建议创建：

```text
v019_raw_delta_pretrain_diff_lr
```

思路：继续使用 v018 的预训练迁移路线，但采用差分学习率：

```text
encoder / backbone: 2e-4
output_layer:       1e-3
```

理由：v018 证明预训练 encoder 有效，但统一 `1e-3` 可能会过度扰动已经学好的 encoder。差分学习率可以让前半截稳定微调，同时让重新初始化的 full vocab 输出层更快学习。

## Version-specific Tools

| Version                                          | Tools                                                                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| v001_raw                                         | `inspect_predictions.py`, `analyze_token_frequency.py`, `evaluate_beam_search.py`                                |
| v002_raw_delta                                   | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v007_raw_delta_lowfreq_sampler                   | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v008_raw_delta_tcn_frontend                      | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v009_raw_delta_presence_mask                     | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v010_raw_delta_packed_lstm                       | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v011_raw_delta_transformer_ctc                   | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v012_raw_delta_finetune_v2_lr2e4                 | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v013_raw_delta_blankpenalty_finetune             | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v014_raw_delta_finetune_swa                      | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v015_raw_delta_finetune_seed2026                 | `inspect_predictions.py`, `analyze_token_frequency.py`                                                           |
| v016_raw_delta_controlled_vocab                  | `build_controlled_vocab.py`, `controlled_dataset.py`, `train_top_k_1000.py`, `inspect_top_k_1000_predictions.py` |
| v017_raw_delta_topk1000_filtered                 | `closed_vocab_dataset.py`, `train.py`                                                                            |
| v018_raw_delta_full_vocab_from_topk1000_pretrain | `train.py`, `inspect_predictions.py`, `analyze_token_frequency.py`, `evaluate_decode_calibration.py`             |

## Notes for Future Experiments

1. 不要覆盖已有版本目录。每个新方向都创建新的 `vXXX_*` 目录。
2. 所有 checkpoint / log / prediction diagnosis 写入 `D:\CE-CSL\CE-CSL\processed`，不要提交到 Git。
3. 每个版本应至少记录：训练配置、best dev TER、best dev loss、checkpoint 路径、诊断结论。
4. 对 full vocab 模型，最终优先比较 `best_dev_TER`，不要只看 `best_dev_loss`。当前多次实验显示，`best_dev_loss` 往往输出偏短，TER 不一定最优。
5. 解码校准需要单独记录，因为训练 checkpoint 的 greedy TER 和 beam/penalty 后的最低 TER 不是同一个指标。
6. 低频 token 是长期瓶颈。`<unk>` 合并可以作为诊断或预训练任务，但不能直接代表 full vocab 实际识别能力。
