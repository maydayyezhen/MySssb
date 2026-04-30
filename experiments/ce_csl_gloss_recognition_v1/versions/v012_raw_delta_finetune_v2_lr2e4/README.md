# v012 raw_delta_finetune_v2_lr2e4

从 `v002_raw_delta` 的 `best_dev_ter.pt` 继续训练，模型和输入完全保持 v2 配置，只把学习率降到 `2e-4`，用于验证当前最佳 basin 里是否还能继续降低 TER。

## Train Entry

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v012_raw_delta_finetune_v2_lr2e4\train.py
```

## Tools

```powershell
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v012_raw_delta_finetune_v2_lr2e4\inspect_predictions.py
D:\MySssb\.venv\Scripts\python.exe experiments\ce_csl_gloss_recognition_v1\versions\v012_raw_delta_finetune_v2_lr2e4\analyze_token_frequency.py
```

## Config

- Source checkpoint: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta\best_dev_ter.pt`
- `feature_mode = "raw_delta"`
- `MODEL_INPUT_DIM = 332`
- `hidden_size = 256`
- `learning_rate = 2e-4`
- `epochs = 80`

## Outputs

- Checkpoints: `D:\CE-CSL\CE-CSL\processed\checkpoints\full_bilstm_ctc_raw_delta_finetune_v2_lr2e4`
- Logs: `D:\CE-CSL\CE-CSL\processed\logs\full_bilstm_ctc_raw_delta_finetune_v2_lr2e4`

## Result

- Source `v002_raw_delta` checkpoint:
  - `epoch = 59`
  - `dev_TER = 0.7018`
  - `dev_loss = 7.2732`
- Fine-tune stopped manually after epoch 24 because dev TER had rebounded.
- Best greedy checkpoint:
  - `best_dev_TER = 0.6940`
  - `best_dev_TER_epoch = 7`
  - `best_dev_TER_dev_loss = 7.4620`
  - `best_dev_loss = 7.2620`
  - `best_dev_loss_epoch = 1`
- Prediction diagnosis for best TER checkpoint:
  - `avgPredictionLength = 3.2136`
  - `avgReferenceLength = 4.5184`
  - `emptyPredictionRatio = 0.0019`
  - `tokenAccuracy = 0.3167`
  - `deleteRate = 0.2995`
  - `substituteRate = 0.3838`
  - `insertPerReferenceToken = 0.0107`
- Decode probes on this checkpoint:
  - Greedy + blank penalty `0.45` or `0.60`: `TER = 0.6910`
  - Beam search best observed: `beam_size=3`, `top_k=20`, blank penalty around `0.45-0.60`, token insert bonus around `0.1-0.3`
  - Best observed decode TER: `0.6889`

## Conclusion

- 这是目前最重要的训练版本：从 v2 最佳 checkpoint 低学习率续训，greedy TER 从 `0.7018` 降到 `0.6940`。
- 主要错误仍然是输出偏短，平均预测长度 `3.2136` 明显低于真实平均长度 `4.5184`。
- 最低 dev TER 不是靠继续训练得到，而是靠 v12 checkpoint 上的 blank/beam 解码校准得到：`0.6889`。
