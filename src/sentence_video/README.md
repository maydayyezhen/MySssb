# sentence_video

Offline sentence-level video recognition.

Current route:

- `POST /api/sentence/recognize`

The service saves the uploaded mp4 under `tmp/sentence_video`, runs the copied
WLASL inference pipeline in `wlasl_pipeline/`, then normalizes the result for
Spring Boot. DeepSeek semantic correction stays in the Java service.

`runtime.py` caches the loaded Keras model and labels for the current
`feature_dir` / `model_dir` combination, avoiding one model load per request.

Configuration lives in `config.py`. Defaults can be overridden with environment
variables, for example:

- `SENTENCE_VIDEO_FEATURE_DIR`
- `SENTENCE_VIDEO_MODEL_DIR`
- `SENTENCE_VIDEO_TMP_ROOT`
- `SENTENCE_VIDEO_TIMEOUT_SEC`
- `SENTENCE_VIDEO_CONFIDENCE_THRESHOLD`
- `SENTENCE_VIDEO_MAX_UPLOAD_MB`
- `SENTENCE_VIDEO_KEEP_TMP`

Uploads are limited to `.mp4`, `.mov`, `.avi`, and `.mkv`. Temporary request
directories are deleted by default; set `SENTENCE_VIDEO_KEEP_TMP=true` to keep
them for debugging.

Copied pipeline files:

- `wlasl_pipeline/inference.py`: service-facing inference entrypoint
- `wlasl_pipeline/infer_wlasl_sentence_video.py`: copied segmenting logic
- `wlasl_pipeline/build_wlasl_features_20f_plus.py`: 232-dim feature extraction
- `wlasl_pipeline/build_wlasl_features_20f_base166.py`: 166-dim feature fallback
- `wlasl_pipeline/build_wlasl_features_20f_static194.py`: 194-dim feature fallback
