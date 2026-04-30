# v008 raw_delta TCN frontend

在 v002 raw_delta baseline 基础上，只修改模型前端：

- 保持 `feature_mode = "raw_delta"`
- 保持 `MODEL_INPUT_DIM = 332`
- 保持 shuffle 训练，不使用低频重采样
- 在输入投影后、BiLSTM 前加入 3 层轻量时序卷积残差块

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v008_raw_delta_tcn_frontend\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v008_raw_delta_tcn_frontend\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v008_raw_delta_tcn_frontend\analyze_token_frequency.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `projection_dim = 256`
- `hidden_size = 256`
- `temporal_kernel_size = 5`
- `temporal_dilations = [1, 2, 4]`
- `temporal_dropout = 0.1`
- `input_dropout = 0.2`
- `lstm_dropout = 0.3`
- `output_dropout = 0.3`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_tcn_frontend`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_tcn_frontend`

## Result

- early stopped at epoch 12
- `best_dev_TER = 0.8775`
- `best_dev_loss = 6.4803`

## Conclusion

负结果。v2 在 epoch 11 已经到 `dev_TER = 0.7714`，v8 到 epoch 12 仍只有 `0.8775`，说明这个 TCN frontend 没有带来可用收益，提前停止节省时间。
