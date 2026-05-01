# -*- coding: utf-8 -*-
"""Runtime model management routes."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.word_recognition.config.gesture_config import LABEL_MAP_FILE_NAME, MODEL_FILE_NAME
from src.word_recognition.utils.runtime_model_registry import (
    get_runtime_model_info,
    reload_runtime_model,
)


router = APIRouter(prefix="/model", tags=["model-runtime"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_MODEL_ROOT = PROJECT_ROOT / "runtime_models"


class ModelReloadRequest(BaseModel):
    """Runtime model reload request."""

    modelPath: str
    labelMapPath: str
    versionName: str = ""


class ModelReloadFromUrlRequest(BaseModel):
    """Runtime model reload request using downloadable URLs."""

    modelUrl: str
    labelMapUrl: str
    versionName: str = ""


def sanitize_runtime_name(value: str) -> str:
    """Convert a model version name to a safe local cache directory name."""
    text = str(value or "").strip() or "current"
    return "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in text
    )


def download_url_to_file(url: str, target_path: Path) -> None:
    """Download a file from URL to a local runtime cache path."""
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="只支持 http / https 文件地址")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urlopen(url, timeout=60) as response:
            with target_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    except HTTPError as exception:
        raise HTTPException(
            status_code=502,
            detail=f"下载模型文件失败，HTTP 状态码：{exception.code}",
        )
    except URLError as exception:
        raise HTTPException(
            status_code=502,
            detail=f"下载模型文件失败，URL 不可访问：{exception}",
        )


@router.post("/reload")
async def reload_model(request: ModelReloadRequest):
    """Reload the model used by newly created realtime recognition sessions."""
    return reload_runtime_model(
        model_path=request.modelPath,
        label_map_path=request.labelMapPath,
        version_name=request.versionName,
    )


@router.post("/reload-from-url")
async def reload_model_from_url(request: ModelReloadFromUrlRequest):
    """Download model files from URLs and reload the runtime model."""
    version_name = request.versionName or "current"
    runtime_dir = RUNTIME_MODEL_ROOT / sanitize_runtime_name(version_name)

    local_model_path = runtime_dir / MODEL_FILE_NAME
    local_label_map_path = runtime_dir / LABEL_MAP_FILE_NAME

    download_url_to_file(request.modelUrl, local_model_path)
    download_url_to_file(request.labelMapUrl, local_label_map_path)

    return reload_runtime_model(
        model_path=str(local_model_path),
        label_map_path=str(local_label_map_path),
        version_name=version_name,
    )


@router.get("/current")
async def current_model():
    """Return current runtime model metadata."""
    return get_runtime_model_info()
