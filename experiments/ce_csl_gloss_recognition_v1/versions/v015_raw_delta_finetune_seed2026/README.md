# v015 raw_delta_finetune_seed2026

从 `v002_raw_delta` 的最佳权重继续训练，保持 v12 的低学习率 fine-tune 设置，但改用 `seed=2026` 并加入 early stop。目标是验证 fine-tune 对 batch order / seed 的敏感性，寻找更好的局部点。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v015_raw_delta_finetune_seed2026\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v015_raw_delta_finetune_seed2026\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v015_raw_delta_finetune_seed2026\analyze_token_frequency.py
```

## Config

- Source checkpoint: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta\best_dev_ter.pt`
- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `learning_rate = 2e-4`
- `seed = 2026`
- `early_stop_patience = 12`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_finetune_seed2026`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_finetune_seed2026`

## Result

- Running.

## Conclusion

- Pending.
