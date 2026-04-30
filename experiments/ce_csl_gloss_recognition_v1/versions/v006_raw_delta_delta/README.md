# v006 raw_delta_delta

raw + delta + delta_delta 特征实验。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v006_raw_delta_delta\train.py
```

## Config

- `feature_mode = "raw_delta_delta"`
- `MODEL_INPUT_DIM = 498`
- `hidden_size = 256`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_delta`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_delta`

## Result

- `best_dev_TER = 0.7379`
- `best_dev_loss = 5.6325`

## Conclusion

二阶差分能学，但没有超过 raw_delta，后段过拟合更明显。

