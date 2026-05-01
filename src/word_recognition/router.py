# -*- coding: utf-8 -*-
"""Aggregate router for the word recognition service module."""

from fastapi import APIRouter

from src.word_recognition.dataset.router import router as dataset_router
from src.word_recognition.model_runtime.router import router as model_runtime_router
from src.word_recognition.realtime.router import router as realtime_router
from src.word_recognition.training.router import router as training_router


router = APIRouter()

router.include_router(realtime_router)
router.include_router(dataset_router)
router.include_router(training_router)
router.include_router(model_runtime_router)
