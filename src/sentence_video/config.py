# -*- coding: utf-8 -*-
"""Configuration for sentence video recognition."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return Path(value).expanduser()


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class SentenceVideoConfig:
    """Runtime configuration for uploaded sentence-video recognition."""

    feature_dir: Path
    model_dir: Path
    tmp_root: Path
    inference_timeout_sec: int
    window_size: int
    stride: int
    confidence_threshold: float
    margin_threshold: float
    min_segment_windows: int
    min_segment_avg_confidence: float
    min_segment_max_confidence: float
    same_label_merge_gap: int
    nms_suppress_radius: int
    max_upload_mb: int
    keep_tmp: bool


def load_sentence_video_config() -> SentenceVideoConfig:
    """Load config from environment variables with project defaults."""
    return SentenceVideoConfig(
        feature_dir=_path_env(
            "SENTENCE_VIDEO_FEATURE_DIR",
            Path("D:/datasets/WLASL-mini-v2-25/features_20f_plus"),
        ),
        model_dir=_path_env(
            "SENTENCE_VIDEO_MODEL_DIR",
            Path("D:/datasets/WLASL-mini-v2-25/models_20f_plus"),
        ),
        tmp_root=_path_env(
            "SENTENCE_VIDEO_TMP_ROOT",
            PROJECT_ROOT / "tmp" / "sentence_video",
        ),
        inference_timeout_sec=_int_env("SENTENCE_VIDEO_TIMEOUT_SEC", 600),
        window_size=_int_env("SENTENCE_VIDEO_WINDOW_SIZE", 20),
        stride=_int_env("SENTENCE_VIDEO_STRIDE", 2),
        confidence_threshold=_float_env("SENTENCE_VIDEO_CONFIDENCE_THRESHOLD", 0.45),
        margin_threshold=_float_env("SENTENCE_VIDEO_MARGIN_THRESHOLD", 0.05),
        min_segment_windows=_int_env("SENTENCE_VIDEO_MIN_SEGMENT_WINDOWS", 2),
        min_segment_avg_confidence=_float_env(
            "SENTENCE_VIDEO_MIN_SEGMENT_AVG_CONFIDENCE",
            0.45,
        ),
        min_segment_max_confidence=_float_env(
            "SENTENCE_VIDEO_MIN_SEGMENT_MAX_CONFIDENCE",
            0.55,
        ),
        same_label_merge_gap=_int_env("SENTENCE_VIDEO_SAME_LABEL_MERGE_GAP", 8),
        nms_suppress_radius=_int_env("SENTENCE_VIDEO_NMS_SUPPRESS_RADIUS", 6),
        max_upload_mb=_int_env("SENTENCE_VIDEO_MAX_UPLOAD_MB", 50),
        keep_tmp=_bool_env("SENTENCE_VIDEO_KEEP_TMP", False),
    )


CONFIG = load_sentence_video_config()
