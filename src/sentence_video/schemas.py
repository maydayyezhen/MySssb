# -*- coding: utf-8 -*-
"""Sentence video recognition DTOs."""

from pydantic import BaseModel


class SegmentTopKItem(BaseModel):
    """A candidate gloss for one segment."""

    label: str
    labelZh: str
    avgProb: float | None = None
    maxProb: float | None = None
    hitCount: int | None = None


class SegmentResult(BaseModel):
    """A recognized segment in the uploaded sentence video."""

    segmentIndex: int
    startFrame: int | None = None
    endFrame: int | None = None
    rawLabel: str
    rawLabelZh: str
    avgConfidence: float | None = None
    maxConfidence: float | None = None
    topK: list[SegmentTopKItem]


class SentenceRecognizeResponse(BaseModel):
    """Sentence video recognition response."""

    status: str
    mode: str
    rawSequence: list[str]
    rawDisplayZh: list[str]
    rawTextZh: str
    segmentTopK: list[SegmentResult]
    elapsedMs: int
