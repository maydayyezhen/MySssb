# -*- coding: utf-8 -*-
"""Sentence video upload recognition route."""

from fastapi import APIRouter, File, UploadFile

from src.sentence_video.service import recognize_sentence_video


router = APIRouter(prefix="/api/sentence", tags=["sentence-video"])


@router.post("/recognize")
async def recognize_sentence(file: UploadFile = File(...)):
    """Recognize a multipart/form-data mp4 file as a gloss sequence."""
    return await recognize_sentence_video(file)
