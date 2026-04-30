# v009 raw_delta presence mask

在 v002 raw_delta baseline 基础上，只修改输入特征：

- 保持 BiLSTM-CTC 模型结构和训练策略
- 使用 `feature_mode = "raw_delta_presence"`
- 输入从 `raw_delta` 的 332 维变成 335 维
- 额外加入每帧 `left_hand_present / right_hand_present / arm_present` 三个 mask

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v009_raw_delta_presence_mask\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v009_raw_delta_presence_mask\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v009_raw_delta_presence_mask\analyze_token_frequency.py
```

## Config

- `feature_mode = "raw_delta_presence"`
- `MODEL_INPUT_DIM = 335`
- `hidden_size = 256`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_presence_mask`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_presence_mask`

## Result

- early stopped at epoch 12
- `best_dev_TER = 0.8711`
- `best_dev_loss = 6.5103`

## Conclusion

负结果。显式 presence mask 没有改善收敛，epoch 12 仍明显落后 v002。
