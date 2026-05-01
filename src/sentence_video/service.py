# -*- coding: utf-8 -*-
"""Sentence video recognition orchestration."""

import asyncio
import time
import uuid
from typing import Any

from fastapi import HTTPException, UploadFile

from src.sentence_video.config import CONFIG
from src.sentence_video.schemas import (
    SegmentResult,
    SegmentTopKItem,
    SentenceRecognizeResponse,
)
from src.sentence_video.video_io import save_upload_file
from src.sentence_video.wlasl_pipeline.inference import recognize_wlasl_sentence_video
from src.sentence_video.zh_map import sequence_to_zh_text, to_zh


async def recognize_sentence_video(file: UploadFile) -> SentenceRecognizeResponse:
    """Save an uploaded video, run sentence inference, and normalize the response."""
    request_id = uuid.uuid4().hex
    work_dir = CONFIG.tmp_root / request_id
    input_path = work_dir / "input.mp4"

    started_at = time.perf_counter()
    await save_upload_file(file, input_path)

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="上传视频为空")

    try:
        inference_result = await asyncio.wait_for(
            asyncio.to_thread(
                recognize_wlasl_sentence_video,
                video_path=input_path,
                feature_dir=CONFIG.feature_dir,
                model_dir=CONFIG.model_dir,
                window_size=CONFIG.window_size,
                stride=CONFIG.stride,
                confidence_threshold=CONFIG.confidence_threshold,
                margin_threshold=CONFIG.margin_threshold,
                min_segment_windows=CONFIG.min_segment_windows,
                min_segment_avg_confidence=CONFIG.min_segment_avg_confidence,
                min_segment_max_confidence=CONFIG.min_segment_max_confidence,
                same_label_merge_gap=CONFIG.same_label_merge_gap,
                nms_suppress_radius=CONFIG.nms_suppress_radius,
            ),
            timeout=CONFIG.inference_timeout_sec,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="句子视频识别超时")
    except FileNotFoundError as exception:
        raise HTTPException(status_code=500, detail=str(exception))
    except RuntimeError as exception:
        raise HTTPException(status_code=500, detail=str(exception))
    except Exception as exception:
        raise HTTPException(
            status_code=500,
            detail=f"句子视频识别失败：{exception}",
        )

    payload = inference_result["payload"]
    dense_rows = inference_result["dense_rows"]
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return _build_response(payload, dense_rows, elapsed_ms)


def _build_response(
    payload: dict[str, Any],
    dense_rows: list[dict[str, str]],
    elapsed_ms: int,
) -> SentenceRecognizeResponse:
    segments = payload.get("segments") or []
    raw_sequence = [str(segment.get("label", "")) for segment in segments]
    raw_sequence = [label for label in raw_sequence if label]

    segment_results = [
        _build_segment_result(segment, index, dense_rows)
        for index, segment in enumerate(segments, start=1)
    ]

    return SentenceRecognizeResponse(
        status="recognized" if raw_sequence else "empty",
        mode="fast",
        rawSequence=raw_sequence,
        rawDisplayZh=[to_zh(label) for label in raw_sequence],
        rawTextZh=sequence_to_zh_text(raw_sequence),
        segmentTopK=segment_results,
        elapsedMs=elapsed_ms,
    )


def _build_segment_result(
    segment: dict[str, Any],
    index: int,
    dense_rows: list[dict[str, str]],
) -> SegmentResult:
    raw_label = str(segment.get("label", ""))
    return SegmentResult(
        segmentIndex=index,
        startFrame=_parse_int(segment.get("start_frame")),
        endFrame=_parse_int(segment.get("end_frame")),
        rawLabel=raw_label,
        rawLabelZh=to_zh(raw_label),
        avgConfidence=_round_optional(_parse_float(segment.get("avg_confidence"))),
        maxConfidence=_round_optional(_parse_float(segment.get("max_confidence"))),
        topK=_build_segment_top_k(segment, dense_rows),
    )


def _build_segment_top_k(
    segment: dict[str, Any],
    dense_rows: list[dict[str, str]],
    limit: int = 3,
) -> list[SegmentTopKItem]:
    buckets: dict[str, list[float]] = {}

    for row in dense_rows:
        if not _row_overlaps_segment(row, segment):
            continue

        for rank in range(1, limit + 1):
            label = row.get(f"top{rank}_label") or row.get(f"top{rank}")
            prob = _parse_float(row.get(f"top{rank}_prob"))
            if not label or prob is None:
                continue
            buckets.setdefault(str(label), []).append(prob)

    if not buckets:
        raw_label = str(segment.get("label", ""))
        if not raw_label:
            return []
        avg_confidence = _round_optional(_parse_float(segment.get("avg_confidence")))
        max_confidence = _round_optional(_parse_float(segment.get("max_confidence")))
        return [
            SegmentTopKItem(
                label=raw_label,
                labelZh=to_zh(raw_label),
                avgProb=avg_confidence,
                maxProb=max_confidence,
            )
        ]

    items = []
    for label, values in buckets.items():
        avg_prob = sum(values) / max(1, len(values))
        items.append(
            SegmentTopKItem(
                label=label,
                labelZh=to_zh(label),
                avgProb=round(avg_prob, 6),
                maxProb=round(max(values), 6),
            )
        )

    return sorted(
        items,
        key=lambda item: ((item.avgProb or 0.0), (item.maxProb or 0.0)),
        reverse=True,
    )[:limit]


def _row_overlaps_segment(row: dict[str, str], segment: dict[str, Any]) -> bool:
    row_start = _parse_int(row.get("start_frame"))
    row_end = _parse_int(row.get("end_frame"))
    segment_start = _parse_int(segment.get("start_frame"))
    segment_end = _parse_int(segment.get("end_frame"))

    if row_start is None or row_end is None:
        return False
    if segment_start is None or segment_end is None:
        return False

    return max(row_start, segment_start) <= min(row_end, segment_end)


def _parse_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)
