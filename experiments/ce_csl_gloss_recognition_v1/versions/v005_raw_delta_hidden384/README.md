# v005 raw_delta_hidden384

raw_delta + `hidden_size=384`。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v005_raw_delta_hidden384\train.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `hidden_size = 384`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_hidden384`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_hidden384`

## Result

- `best_dev_TER = 0.8101`
- `best_dev_loss = 6.3493`

## Conclusion

单纯增加 hidden size 明显变差。在当前学习率和正则配置下不建议继续。

