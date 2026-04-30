# v010 raw_delta packed LSTM

在 v002 raw_delta baseline 基础上，只修改 BiLSTM 的 padding 处理：

- 保持 `feature_mode = "raw_delta"`
- 保持 `MODEL_INPUT_DIM = 332`
- 保持 BiLSTM-CTC 主体结构、dropout、学习率、训练策略
- LSTM 前使用 `pack_padded_sequence`
- LSTM 后使用 `pad_packed_sequence(total_length=max_T)` 还原形状

## Why

旧模型直接把 padded batch 输入双向 LSTM。CTC 会忽略 `input_lengths` 之后的输出，但双向 LSTM 的 backward 方向已经看过 padding 零帧，短样本末端表示会被 padding 干扰。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v010_raw_delta_packed_lstm\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v010_raw_delta_packed_lstm\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v010_raw_delta_packed_lstm\analyze_token_frequency.py
```

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_packed_lstm`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_packed_lstm`

## Result

- early stopped at epoch 13
- `best_dev_TER = 0.8466`
- `best_dev_loss = 6.3419`

## Conclusion

负结果。虽然修复了 padding 处理，但收敛明显慢于 v002，说明 padding 干扰不是当前主瓶颈。
