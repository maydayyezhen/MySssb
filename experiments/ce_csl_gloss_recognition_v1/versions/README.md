# CE-CSL Gloss Recognition Versions

本目录按实验版本保存正式全量训练入口和版本专属评估/诊断工具。旧的 `scripts/05_train/train_full*.py` 已移动到这里；`scripts/05_train` 保留小样本训练和训练链路检查工具。

数据、checkpoint、log 仍然写到 `D:\CE-CSL\CE-CSL\processed` 下，不放进 Git。

| Version | Main change | Best dev TER | Best dev loss | Conclusion |
|---|---:|---:|---:|---|
| v001_raw | raw feature baseline | 0.7228 | 5.4695 | 被 raw_delta 超过 |
| v002_raw_delta | raw + delta baseline | 0.7018 | 5.3171 | 当前最佳 |
| v003_raw_delta_reg | 更强正则 + dev_loss scheduler | 0.7069 | 5.2157 | loss 更稳但 TER 不如 v2 |
| v004_raw_delta_ter_scheduler | dev_TER scheduler | 0.7039 | 5.3171 | 接近 v2 但未超过 |
| v005_raw_delta_hidden384 | hidden_size=384 | 0.8101 | 6.3493 | 明显变差 |
| v006_raw_delta_delta | raw + delta + delta_delta | 0.7379 | 5.6325 | 能学但不如 raw_delta |
| v007_raw_delta_lowfreq_sampler | raw_delta + lowfreq sampler | 0.7563 | 6.0332 | 未改善低频，且拉低整体 |
| v008_raw_delta_tcn_frontend | raw_delta + TCN frontend | 0.8775 | 6.4803 | 早停，明显落后 v2 |
| v009_raw_delta_presence_mask | raw_delta + hand presence mask | 0.8711 | 6.5103 | 早停，明显落后 v2 |
| v010_raw_delta_packed_lstm | raw_delta + packed BiLSTM | 0.8466 | 6.3419 | 早停，明显落后 v2 |
| v011_raw_delta_transformer_ctc | raw_delta + Transformer CTC | 0.7520 | 5.5395 | 能学但输出更短，不如 v2 |
| v012_raw_delta_finetune_v2_lr2e4 | raw_delta + fine-tune v2 best checkpoint | Running | Running | 低学习率续训 v2 最佳点 |
| v013_raw_delta_blankpenalty_finetune | raw_delta + blank-penalty fine-tune | Running | Running | 针对输出偏短和 deletion 做校准续训 |
| v014_raw_delta_finetune_swa | raw_delta + fine-tune with SWA averaging | Running | Running | 平滑 v12 早期 checkpoint 抖动 |
| v015_raw_delta_finetune_seed2026 | raw_delta + fine-tune seed 2026 | Running | Running | 检查低 LR 续训对 seed/batch order 的敏感性 |

当前最佳仍是 `v002_raw_delta`。v007 说明单纯低频样本重采样不够，下一步应回到 v2 配置上改输入表示或时序建模。

## Version-specific Tools

| Version | Tools |
|---|---|
| v001_raw | `inspect_predictions.py`, `analyze_token_frequency.py`, `evaluate_beam_search.py` |
| v002_raw_delta | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v007_raw_delta_lowfreq_sampler | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v008_raw_delta_tcn_frontend | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v009_raw_delta_presence_mask | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v010_raw_delta_packed_lstm | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v011_raw_delta_transformer_ctc | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v012_raw_delta_finetune_v2_lr2e4 | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v013_raw_delta_blankpenalty_finetune | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v014_raw_delta_finetune_swa | `inspect_predictions.py`, `analyze_token_frequency.py` |
| v015_raw_delta_finetune_seed2026 | `inspect_predictions.py`, `analyze_token_frequency.py` |
