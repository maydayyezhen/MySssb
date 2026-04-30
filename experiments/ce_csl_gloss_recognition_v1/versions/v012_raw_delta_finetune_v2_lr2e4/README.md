# v012 raw_delta_finetune_v2_lr2e4

从 `v002_raw_delta` 的 `best_dev_ter.pt` 继续训练，模型和输入完全保持 v2 配置，只把学习率降到 `2e-4`，用于验证当前最佳 basin 里是否还能继续降低 TER。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v012_raw_delta_finetune_v2_lr2e4\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v012_raw_delta_finetune_v2_lr2e4\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v012_raw_delta_finetune_v2_lr2e4\analyze_token_frequency.py
```

## Config

- Source checkpoint: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta\best_dev_ter.pt`
- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `hidden_size = 256`
- `learning_rate = 2e-4`
- `epochs = 80`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_finetune_v2_lr2e4`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_finetune_v2_lr2e4`

## Result

- Running.

## Conclusion

- Pending.
