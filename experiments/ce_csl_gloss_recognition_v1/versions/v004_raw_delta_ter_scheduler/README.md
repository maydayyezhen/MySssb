# v004 raw_delta_ter_scheduler

raw_delta + dev_TER scheduler。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v004_raw_delta_ter_scheduler\train.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `hidden_size = 256`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`
- scheduler: `ReduceLROnPlateau(dev_TER)`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_ter_scheduler`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_ter_scheduler`

## Result

- `best_dev_TER = 0.7039`
- `best_dev_loss = 5.3171`

## Conclusion

第 42 轮学习率降到 `0.0005` 后有后段收益，但仍未超过 v002。

