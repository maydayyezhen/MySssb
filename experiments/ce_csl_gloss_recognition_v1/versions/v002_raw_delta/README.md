# v002 raw_delta

当前最佳 baseline。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v002_raw_delta\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v002_raw_delta\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v002_raw_delta\analyze_token_frequency.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `hidden_size = 256`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta`

## Result

- `best_dev_TER = 0.7018`
- `best_dev_loss = 5.3171`

## Conclusion

当前最佳版本。后续实验都应优先和这个结果对比。
