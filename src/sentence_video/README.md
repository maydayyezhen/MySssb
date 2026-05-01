# sentence_video

Offline sentence-level video recognition.

Current route:

- `POST /api/sentence/recognize`

The service saves the uploaded mp4 under `tmp/sentence_video`, runs the copied
WLASL inference pipeline in `wlasl_pipeline/`, then normalizes the result for
Spring Boot. DeepSeek semantic correction stays in the Java service.

Configuration lives in `config.py`. Defaults can be overridden with environment
variables, for example:

- `SENTENCE_VIDEO_FEATURE_DIR`
- `SENTENCE_VIDEO_MODEL_DIR`
- `SENTENCE_VIDEO_TMP_ROOT`
- `SENTENCE_VIDEO_TIMEOUT_SEC`
- `SENTENCE_VIDEO_CONFIDENCE_THRESHOLD`

Copied pipeline files:

- `wlasl_pipeline/inference.py`: service-facing inference entrypoint
- `wlasl_pipeline/infer_wlasl_sentence_video.py`: copied segmenting logic
- `wlasl_pipeline/build_wlasl_features_20f_plus.py`: 232-dim feature extraction
- `wlasl_pipeline/build_wlasl_features_20f_base166.py`: 166-dim feature fallback
- `wlasl_pipeline/build_wlasl_features_20f_static194.py`: 194-dim feature fallback
