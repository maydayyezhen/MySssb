# -*- coding: utf-8 -*-
"""Runtime cache for sentence video recognition models."""

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from src.sentence_video.wlasl_pipeline.infer_wlasl_sentence_video import load_json


@dataclass
class SentenceModelRuntime:
    """Cached sentence recognition model runtime."""

    model: object
    labels: list[str]
    model_path: Path
    feature_dim: int


_cached_runtime: SentenceModelRuntime | None = None
_cached_feature_dir: Path | None = None
_cached_model_dir: Path | None = None


def resolve_sentence_model_path(model_dir: Path) -> Path:
    """Resolve the active sentence recognition model path."""
    model_path = model_dir / "best_wlasl_20f_plus_classifier.keras"

    if not model_path.exists():
        model_path = model_dir / "wlasl_20f_plus_classifier.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"WLASL 模型文件不存在：{model_path}")

    return model_path


def get_sentence_model_runtime(
    feature_dir: Path,
    model_dir: Path,
) -> SentenceModelRuntime:
    """Return cached sentence recognition runtime, loading it when needed."""
    global _cached_runtime
    global _cached_feature_dir
    global _cached_model_dir

    feature_dir = Path(feature_dir)
    model_dir = Path(model_dir)
    model_path = resolve_sentence_model_path(model_dir)

    should_reload = (
        _cached_runtime is None
        or _cached_feature_dir != feature_dir
        or _cached_model_dir != model_dir
        or _cached_runtime.model_path != model_path
    )

    if not should_reload:
        return _cached_runtime

    labels_payload = load_json(feature_dir / "labels.json")
    labels = labels_payload["labels"]

    model = tf.keras.models.load_model(model_path)
    feature_dim = int(model.input_shape[-1])

    _cached_runtime = SentenceModelRuntime(
        model=model,
        labels=labels,
        model_path=model_path,
        feature_dim=feature_dim,
    )
    _cached_feature_dir = feature_dir
    _cached_model_dir = model_dir

    print(f"[sentence_video] loaded model: {model_path}")
    print(f"[sentence_video] class_count: {len(labels)}")
    print(f"[sentence_video] feature_dim: {feature_dim}")

    return _cached_runtime
