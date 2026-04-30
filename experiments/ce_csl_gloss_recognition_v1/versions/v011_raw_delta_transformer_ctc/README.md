# v011 raw_delta Transformer CTC

换时序建模范式：

- 保持 `feature_mode = "raw_delta"`
- 使用 Transformer Encoder + CTC
- 使用 sinusoidal positional encoding
- 使用 `src_key_padding_mask` 显式屏蔽 padding 帧
- 学习率调低到 `5e-4`

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v011_raw_delta_transformer_ctc\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v011_raw_delta_transformer_ctc\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v011_raw_delta_transformer_ctc\analyze_token_frequency.py
```

## Config

- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `projection_dim = 256`
- `num_layers = 4`
- `num_heads = 4`
- `feedforward_dim = 1024`
- `input_dropout = 0.2`
- `transformer_dropout = 0.2`
- `output_dropout = 0.2`
- `learning_rate = 5e-4`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_transformer_ctc_raw_delta`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_transformer_ctc_raw_delta`

## Result

- `best_dev_TER = 0.7520`
- `best_dev_loss = 5.5395`
- best TER checkpoint epoch: 53
- best TER avg prediction length: 2.4214
- best TER empty prediction ratio: 0.0524

## Conclusion

负结果。Transformer CTC 能学，但比 v002 差很多，主要问题是输出更短，deleteRate 上升到 0.4736。self-attention 主干没有带来质变。
