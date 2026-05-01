# -*- coding: utf-8 -*-
"""
HearBridge gesture recognition service entrypoint.

The app module only owns application creation, startup wiring, health checks,
and router registration. Business routes live in their feature modules.
"""

from fastapi import FastAPI

from src.sentence_video.router import router as sentence_video_router
from src.word_recognition.router import router as word_recognition_router
from src.word_recognition.utils.runtime_model_bootstrap import (
    bootstrap_runtime_model_from_backend,
)


app = FastAPI(title="HearBridge Gesture Service")


@app.on_event("startup")
async def startup_load_published_model():
    """Try to load the current published model without blocking service startup."""
    result = bootstrap_runtime_model_from_backend()
    print(f"[startup] runtime model bootstrap result: {result}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True}


app.include_router(word_recognition_router)
app.include_router(sentence_video_router)
