# -*- coding: utf-8 -*-
"""Service-facing WLASL sentence video inference entrypoint."""

from pathlib import Path
from typing import Any

import numpy as np

from src.sentence_video.runtime import get_sentence_model_runtime
from src.sentence_video.wlasl_pipeline.infer_wlasl_sentence_video import (
    build_raw_segments,
    build_result_cache,
    build_windows,
    compare_sequence,
    filter_segments,
    get_video_info,
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
    runtime = get_sentence_model_runtime(feature_dir, model_dir)
    model = runtime.model
    labels = runtime.labels
    model_path = runtime.model_path
    feature_dim = runtime.feature_dim

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
