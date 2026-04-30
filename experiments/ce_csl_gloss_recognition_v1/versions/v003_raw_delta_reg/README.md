# v003 raw_delta_reg

raw_delta + 稍强正则 + dev_loss scheduler。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v003_raw_delta_reg\train.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `LEARNING_RATE = 8e-4`
- `WEIGHT_DECAY = 3e-4`
- `input_dropout = 0.3`
- `lstm_dropout = 0.4`
- `output_dropout = 0.4`
- scheduler: `ReduceLROnPlateau(dev_loss)`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_reg`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_reg`

## Result

- `best_dev_TER = 0.7069`
- `best_dev_loss = 5.2157`

## Conclusion

dev_loss 更稳，但最终 TER 不如 v002。不要继续在这个强正则方向微调太久。

