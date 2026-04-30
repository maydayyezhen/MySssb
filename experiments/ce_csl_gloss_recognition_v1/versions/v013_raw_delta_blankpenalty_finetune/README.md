# v013 raw_delta_blankpenalty_finetune

从 `v012_raw_delta_finetune_v2_lr2e4` 的最佳权重继续训练，在 CTC loss 和 greedy decode 前对 blank log-prob 加惩罚并重新归一化，目标是缓解当前最主要的输出偏短和 deletion 问题。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v013_raw_delta_blankpenalty_finetune\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v013_raw_delta_blankpenalty_finetune\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v013_raw_delta_blankpenalty_finetune\analyze_token_frequency.py
```

## Config

- Source checkpoint: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_finetune_v2_lr2e4\best_dev_ter.pt`
- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `learning_rate = 1e-4`
- `blank_logit_penalty = 0.45`
- `early_stop_patience = 12`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_blankpenalty_finetune`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_blankpenalty_finetune`

## Result

- Running.

## Conclusion

- Pending.
