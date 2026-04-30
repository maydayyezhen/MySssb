# v001 raw

原始 raw feature baseline。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v001_raw\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v001_raw\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v001_raw\analyze_token_frequency.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v001_raw\evaluate_beam_search.py
```

## Config

- `feature_mode = "raw"`
- `MODEL_INPUT_DIM = 166`
- `hidden_size = 256`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc`

## Result

- `best_dev_TER = 0.7228`
- `best_dev_loss = 5.4695`

## Conclusion

raw baseline 有效，但明显不如 v002 raw_delta。
