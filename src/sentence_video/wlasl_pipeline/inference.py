# -*- coding: utf-8 -*-
"""Service-facing WLASL sentence video inference entrypoint."""

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.sentence_video.wlasl_pipeline.infer_wlasl_sentence_video import (
    build_raw_segments,
    build_result_cache,
    build_windows,
    compare_sequence,
    filter_segments,
    get_video_info,
    load_json,
    make_dense_rows,
    nms_segments,
    parse_expected,
)


def recognize_wlasl_sentence_video(
    video_path: Path,
    feature_dir: Path,
    model_dir: Path,
    expected: str = "",
    window_size: int = 20,
    stride: int = 2,
    confidence_threshold: float = 0.45,
    margin_threshold: float = 0.05,
    min_segment_windows: int = 2,
    min_segment_avg_confidence: float = 0.45,
    min_segment_max_confidence: float = 0.55,
    same_label_merge_gap: int = 8,
    nms_suppress_radius: int = 6,
) -> dict[str, Any]:
    """Run WLASL sentence inference and return in-memory segment data."""
    labels_payload = load_json(feature_dir / "labels.json")
    labels = labels_payload["labels"]

    model_path = model_dir / "best_wlasl_20f_plus_classifier.keras"
    if not model_path.exists():
        model_path = model_dir / "wlasl_20f_plus_classifier.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"WLASL 模型文件不存在：{model_path}")

    model = tf.keras.models.load_model(model_path)
    feature_dim = int(model.input_shape[-1])

    video_info = get_video_info(video_path)
    frames, results, pose_flags, hand_flags = build_result_cache(video_path)

    X, window_meta = build_windows(
        results=results,
        window_size=window_size,
        stride=stride,
        feature_dim=feature_dim,
    )

    probs = model.predict(X, verbose=0)

    dense_rows = make_dense_rows(
        probs=probs,
        window_meta=window_meta,
        labels=labels,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )

    raw_segments = build_raw_segments(
        dense_rows=dense_rows,
        same_label_merge_gap=same_label_merge_gap,
    )

    filtered_segments = filter_segments(
        segments=raw_segments,
        min_segment_windows=min_segment_windows,
        min_segment_avg_confidence=min_segment_avg_confidence,
        min_segment_max_confidence=min_segment_max_confidence,
        blank_label="blank",
    )

    final_segments = nms_segments(
        segments=filtered_segments,
        suppress_radius=nms_suppress_radius,
    )

    detected_sequence = [str(segment["label"]) for segment in final_segments]
    expected_sequence = parse_expected(expected)
    comparison = compare_sequence(
        expected=expected_sequence,
        detected=detected_sequence,
    )

    payload = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "video_info": video_info,
        "window_size": window_size,
        "stride": stride,
        "thresholds": {
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "min_segment_windows": min_segment_windows,
            "min_segment_avg_confidence": min_segment_avg_confidence,
            "min_segment_max_confidence": min_segment_max_confidence,
            "same_label_merge_gap": same_label_merge_gap,
            "nms_suppress_radius": nms_suppress_radius,
        },
        "pose_ratio": round(sum(pose_flags) / len(pose_flags), 6),
        "any_hand_ratio": round(sum(hand_flags) / len(hand_flags), 6),
        "raw_segments": raw_segments,
        "filtered_segments": filtered_segments,
        "segments": final_segments,
        **comparison,
    }

    return {
        "payload": payload,
        "dense_rows": dense_rows,
        "windowCount": int(X.shape[0]),
        "frameCount": int(len(frames)),
        "featureDim": int(feature_dim),
        "probShape": list(np.asarray(probs).shape),
    }
