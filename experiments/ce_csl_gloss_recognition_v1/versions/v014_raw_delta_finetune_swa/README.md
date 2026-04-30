# v014 raw_delta_finetune_swa

从 `v002_raw_delta` 的最佳权重重新做低学习率续训，同时从第 3 轮开始维护 SWA 风格的等权平均模型。目标是稳定 v012 在早期出现的收益，降低单个 checkpoint 抖动。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v014_raw_delta_finetune_swa\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v014_raw_delta_finetune_swa\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v014_raw_delta_finetune_swa\analyze_token_frequency.py
```

## Config

- Source checkpoint: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta\best_dev_ter.pt`
- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `learning_rate = 2e-4`
- `swa_start_epoch = 3`
- `early_stop_patience = 12`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_finetune_swa`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_finetune_swa`

## Result

- Best checkpoint:
  - `best_dev_TER = 0.6932`
  - `best_dev_TER_epoch = 14`
  - `best_dev_TER_source = model`
  - `best_dev_loss = 7.3367`
- SWA did not beat the normal model checkpoint.
- Best SWA values stayed around `0.6966-0.6979` after the running average stabilized.
- Decode probes on v14 best checkpoint:
  - Greedy: `TER = 0.6932`
  - Greedy + blank penalty `0.45`: `TER = 0.6927`
  - Best observed beam decode: `beam_size=3`, `top_k=20`, blank penalty `0.30`, token insert bonus `0.10`, `TER = 0.6914`

## Conclusion

- SWA/weight averaging smoothed the checkpoint behavior but also washed out the better early local point.
- v14 greedy slightly beats v12 greedy (`0.6932` vs `0.6940`), but v14 decode calibration does not beat v12 decode calibration (`0.6914` vs `0.6889`).
- This route is useful as evidence that averaging is not the missing lever.
