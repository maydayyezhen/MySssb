"""模型训练运行工具。

用于 FastAPI 接口调用训练流程。
第一版目标：
1. 读取 data_processed_arm_pose_10fps；
2. 训练 1D CNN；
3. 保存模型、label_map、训练曲线、混淆矩阵、评估结果；
4. 返回训练摘要给 Spring Boot / 管理端。
"""

import json
import time
from pathlib import Path
from typing import Dict

import matplotlib

# HTTP 接口环境下不要弹出绘图窗口。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import callbacks

from src.config.gesture_config import (
    DATA_DIR_NAME,
    MODEL_FILE_NAME,
    LABEL_MAP_FILE_NAME,
)
from src.train import build_model, evaluate_on_validation
from src.utils.dataset_loader import load_dataset, split_dataset_stratified


def run_training(project_root: Path) -> Dict:
    """执行一次模型训练。

    Args:
        project_root: 项目根目录。

    Returns:
        训练结果摘要。
    """
    start_time = time.time()

    data_root = project_root / DATA_DIR_NAME
    artifacts_root = project_root / "artifacts"
    run_name = time.strftime("train_%Y%m%d_%H%M%S")
    save_dir = artifacts_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y, label_map = load_dataset(data_root)
    X_train, y_train, X_val, y_val = split_dataset_stratified(X, y)

    input_shape = X_train.shape[1:]
    num_classes = len(label_map)

    model = build_model(input_shape, num_classes)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[early_stop],
        verbose=1,
    )

    train_acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

    final_train_accuracy = float(history.history[train_acc_key][-1])
    final_val_accuracy = float(history.history[val_acc_key][-1])
    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    save_training_curve_no_show(history, save_dir)

    # 复用已有评估函数。注意：原函数会保存 eval_result.txt 和 confusion_matrix.png。
    evaluate_on_validation(model, X_val, y_val, label_map, save_dir)

    model_path = save_dir / MODEL_FILE_NAME
    label_map_path = save_dir / LABEL_MAP_FILE_NAME

    model.save(model_path)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    duration_sec = round(time.time() - start_time, 2)

    return {
        "runName": run_name,
        "dataRoot": str(data_root),
        "artifactDir": str(save_dir),
        "modelPath": str(model_path),
        "labelMapPath": str(label_map_path),
        "trainingCurvePath": str(save_dir / "training_curve.png"),
        "confusionMatrixPath": str(save_dir / "confusion_matrix.png"),
        "evalResultPath": str(save_dir / "eval_result.txt"),
        "sampleCount": int(X.shape[0]),
        "trainSampleCount": int(X_train.shape[0]),
        "valSampleCount": int(X_val.shape[0]),
        "classCount": int(num_classes),
        "inputShape": list(input_shape),
        "epochsRan": int(len(history.history["loss"])),
        "finalTrainAccuracy": round(final_train_accuracy, 4),
        "finalValAccuracy": round(final_val_accuracy, 4),
        "finalTrainLoss": round(final_train_loss, 4),
        "finalValLoss": round(final_val_loss, 4),
        "durationSec": duration_sec,
        "labelMap": label_map,
        "message": "training completed",
    }


def save_training_curve_no_show(history, save_dir: Path) -> None:
    """保存训练曲线，不弹窗。"""
    train_acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history[train_acc_key], label="train_acc")
    plt.plot(history.history[val_acc_key], label="val_acc")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = save_dir / "training_curve.png"
    plt.savefig(save_path, dpi=200)
    plt.close()