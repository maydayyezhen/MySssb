# -*- coding: utf-8 -*-
"""Training and artifact routes."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.word_recognition.config.gesture_config import LABEL_MAP_FILE_NAME, MODEL_FILE_NAME
from src.word_recognition.utils.raw_feature_converter import convert_raw_dataset_to_features
from src.word_recognition.utils.training_runner import run_training


router = APIRouter(tags=["training"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

ALLOWED_ARTIFACT_FILES = {
    MODEL_FILE_NAME,
    LABEL_MAP_FILE_NAME,
    "training_curve.png",
    "confusion_matrix.png",
    "eval_result.txt",
}

ARTIFACT_MEDIA_TYPES = {
    MODEL_FILE_NAME: "application/octet-stream",
    LABEL_MAP_FILE_NAME: "application/json",
    "training_curve.png": "image/png",
    "confusion_matrix.png": "image/png",
    "eval_result.txt": "text/plain; charset=utf-8",
}


def resolve_artifact_file(run_name: str, file_name: str) -> Path:
    """Resolve and validate a training artifact file path."""
    if not run_name or "/" in run_name or "\\" in run_name or ".." in run_name:
        raise HTTPException(status_code=400, detail="非法的训练运行名称")

    if file_name not in ALLOWED_ARTIFACT_FILES:
        raise HTTPException(status_code=400, detail=f"不允许访问的训练产物文件：{file_name}")

    artifacts_root = ARTIFACTS_ROOT.resolve()
    file_path = (artifacts_root / run_name / file_name).resolve()

    if not file_path.is_relative_to(artifacts_root):
        raise HTTPException(status_code=400, detail="非法的训练产物路径")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"训练产物不存在：{run_name}/{file_name}")

    return file_path


@router.post("/dataset/raw/convert-to-features")
async def convert_raw_to_features():
    """Convert raw phone dataset samples to model training features."""
    return convert_raw_dataset_to_features(PROJECT_ROOT)


@router.post("/model/train")
async def train_model():
    """Run one model training job."""
    return run_training(PROJECT_ROOT)


@router.get("/artifacts/{run_name}/{file_name}")
async def get_artifact_file(run_name: str, file_name: str):
    """Download an allowed training artifact file."""
    file_path = resolve_artifact_file(run_name, file_name)
    return FileResponse(
        path=str(file_path),
        media_type=ARTIFACT_MEDIA_TYPES.get(file_name, "application/octet-stream"),
        filename=file_name,
    )
