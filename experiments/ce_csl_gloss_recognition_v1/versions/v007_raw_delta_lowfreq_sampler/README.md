# v007 raw_delta lowfreq sampler

在 v002 raw_delta baseline 基础上，只修改训练采样策略：

- 保持 `feature_mode = "raw_delta"`
- 保持 `MODEL_INPUT_DIM = 332`
- 保持 BiLSTM-CTC 模型结构、dropout、学习率、weight decay 不变
- 新增 `WeightedRandomSampler`，提高包含低频 gloss token 的样本被抽到的概率

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v007_raw_delta_lowfreq_sampler\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v007_raw_delta_lowfreq_sampler\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v007_raw_delta_lowfreq_sampler\analyze_token_frequency.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `hidden_size = 256`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`
- `LOWFREQ_TARGET_COUNT = 10`
- `LOWFREQ_MIN_SAMPLE_WEIGHT = 1.0`
- `LOWFREQ_MAX_SAMPLE_WEIGHT = 3.0`
- `LOWFREQ_SAMPLER_REPLACEMENT = True`

## Sampler

对训练集统计每个 gloss token 的出现次数。单个 token 的采样系数为：

```text
sqrt(LOWFREQ_TARGET_COUNT / token_count)
```

然后裁剪到 `[1.0, 3.0]`。一个样本的最终权重取其 target token 中最大的权重。

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_lowfreq_sampler`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_lowfreq_sampler`

## Result

- `best_dev_TER = 0.7563`
- `best_dev_loss = 6.0332`
- best TER checkpoint epoch: 59
- v7 overall token accuracy: 0.2557
- v2 overall token accuracy: 0.3111

## Conclusion

未超过 v002，且低频桶没有有效改善。

| Bucket | v002 acc | v007 acc |
|---|---:|---:|
| `00_oov` | 0.0000 | 0.0000 |
| `01_once` | 0.0156 | 0.0234 |
| `02_count_2_3` | 0.0309 | 0.0062 |
| `03_count_4_10` | 0.0442 | 0.0155 |
| `04_count_11_50` | 0.1861 | 0.1165 |
| `05_count_51_plus` | 0.5761 | 0.5029 |

结论：单纯低频样本重采样不划算。它只让 `once` 桶从 0.0156 到 0.0234 有很小提升，但损伤了其他低频桶和高频桶，整体 TER 从 v002 的 0.7018 退到 0.7563。
