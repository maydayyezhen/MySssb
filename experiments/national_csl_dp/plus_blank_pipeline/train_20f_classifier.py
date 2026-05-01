# -*- coding: utf-8 -*-
"""
NationalCSL-DP 20帧词语分类模型训练脚本。

输入：
D:/datasets/HearBridge-NationalCSL-mini/features_20f/
  X.npy
  y.npy
  label_map.json
  sample_index.csv
  feature_config.json

输出：
D:/datasets/HearBridge-NationalCSL-mini/models_20f/
  national_csl_20f_classifier.keras
  normalizer.npz
  train_config.json
  metrics.json
  confusion_matrix.csv

切分方式：
- 训练集：Participant_01 ~ Participant_08
- 验证集：Participant_09
- 测试集：Participant_10

说明：
本脚本用于训练一个轻量级 Conv1D + BiLSTM 词语分类模型。
输入形状为：
  [20, 166]
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


TRAIN_PARTICIPANTS = {
    "Participant_01",
    "Participant_02",
    "Participant_03",
    "Participant_04",
    "Participant_05",
    "Participant_06",
    "Participant_07",
    "Participant_08",
}

VAL_PARTICIPANTS = {
    "Participant_09",
}

TEST_PARTICIPANTS = {
    "Participant_10",
}


def load_json(path: Path) -> Dict:
    """
    读取 JSON 文件。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    """
    保存 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def read_sample_index(sample_index_csv: Path) -> List[Dict[str, str]]:
    """
    读取 sample_index.csv。

    注意：
    这里要求只保留 status=ok 且 label_id 有效的样本，
    其顺序应与 X.npy / y.npy 中样本顺序一致。
    """
    with sample_index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    valid_rows = [
        row for row in rows
        if row.get("status") == "ok" and str(row.get("label_id", "")).strip() != ""
    ]

    return valid_rows


def split_by_participant(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据参与者划分训练集、验证集、测试集索引。
    """
    train_indices = []
    val_indices = []
    test_indices = []

    for index, row in enumerate(rows):
        participant = row["participant"]

        if participant in TRAIN_PARTICIPANTS:
            train_indices.append(index)
        elif participant in VAL_PARTICIPANTS:
            val_indices.append(index)
        elif participant in TEST_PARTICIPANTS:
            test_indices.append(index)
        else:
            print(f"[警告] 未知参与者，默认忽略：{participant}")

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )


def compute_normalizer(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据训练集计算标准化参数。

    只使用训练集统计 mean/std，避免验证集和测试集信息泄漏。
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)

    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    应用标准化。
    """
    return ((X - mean) / std).astype(np.float32)


def build_model(input_shape: Tuple[int, int], class_count: int) -> keras.Model:
    """
    构建 20 帧词语分类模型。

    模型结构：
    - Conv1D 提取局部时序特征
    - BiLSTM 建模动作时序变化
    - Dense 分类
    """
    inputs = keras.Input(shape=input_shape, name="feature_sequence")

    x = layers.Conv1D(
        filters=96,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)

    x = layers.Bidirectional(
        layers.LSTM(
            units=64,
            return_sequences=False,
            dropout=0.25,
            recurrent_dropout=0.0,
        ),
        name="bilstm",
    )(x)

    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.35, name="dropout1")(x)

    outputs = layers.Dense(class_count, activation="softmax", name="class_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="national_csl_20f_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    return model


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> np.ndarray:
    """
    计算混淆矩阵。

    行：真实类别
    列：预测类别
    """
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id), int(pred_id)] += 1

    return matrix


def save_confusion_matrix_csv(
    output_path: Path,
    matrix: np.ndarray,
    id_to_label: Dict[int, str],
) -> None:
    """
    保存混淆矩阵 CSV。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_count = matrix.shape[0]
    headers = ["true\\pred"] + [id_to_label[i] for i in range(class_count)]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for true_id in range(class_count):
            writer.writerow([id_to_label[true_id]] + matrix[true_id].tolist())

    print(f"[完成] 已写出混淆矩阵：{output_path}")


def summarize_split(name: str, y: np.ndarray, id_to_label: Dict[int, str]) -> None:
    """
    打印某个数据划分中的类别分布。
    """
    print(f"\n========== {name} 类别分布 ==========")

    unique, counts = np.unique(y, return_counts=True)

    for label_id, count in zip(unique, counts):
        print(f"{id_to_label[int(label_id)]}: {int(count)}")


def train_classifier(
    feature_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
) -> None:
    """
    训练 20 帧词语分类模型。
    """
    X_path = feature_dir / "X.npy"
    y_path = feature_dir / "y.npy"
    label_map_path = feature_dir / "label_map.json"
    sample_index_path = feature_dir / "sample_index.csv"

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    label_map = load_json(label_map_path)

    rows = read_sample_index(sample_index_path)

    if len(rows) != len(X):
        raise RuntimeError(
            f"sample_index 有效样本数与 X 数量不一致：rows={len(rows)}, X={len(X)}"
        )

    class_count = len(label_map)
    id_to_label = {int(v): k for k, v in label_map.items()}

    train_idx, val_idx, test_idx = split_by_participant(rows)

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    print("========== 数据概览 ==========")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"class_count: {class_count}")
    print(f"train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    summarize_split("训练集", y_train, id_to_label)
    summarize_split("验证集", y_val, id_to_label)
    summarize_split("测试集", y_test, id_to_label)

    mean, std = compute_normalizer(X_train)

    X_train = apply_normalizer(X_train, mean, std)
    X_val = apply_normalizer(X_val, mean, std)
    X_test = apply_normalizer(X_test, mean, std)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "normalizer.npz",
        mean=mean,
        std=std,
    )
    print(f"[完成] 已保存标准化参数：{output_dir / 'normalizer.npz'}")

    model = build_model(
        input_shape=(X.shape[1], X.shape[2]),
        class_count=class_count,
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=30,
            restore_best_weights=True,
            mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=12,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_national_csl_20f_classifier.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = output_dir / "national_csl_20f_classifier.keras"
    model.save(final_model_path)
    print(f"[完成] 已保存最终模型：{final_model_path}")

    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    metric_names = model.metrics_names

    metrics = {
        name: float(value)
        for name, value in zip(metric_names, test_metrics)
    }

    print("\n========== 测试集结果 ==========")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    confusion_matrix = compute_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_count=class_count,
    )

    save_confusion_matrix_csv(
        output_path=output_dir / "confusion_matrix.csv",
        matrix=confusion_matrix,
        id_to_label=id_to_label,
    )

    train_config = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "input_shape": list(X.shape[1:]),
        "class_count": class_count,
        "label_map": label_map,
        "split": {
            "train_participants": sorted(TRAIN_PARTICIPANTS),
            "val_participants": sorted(VAL_PARTICIPANTS),
            "test_participants": sorted(TEST_PARTICIPANTS),
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "early_stopping_monitor": "val_accuracy",
        },
    }

    save_json(output_dir / "train_config.json", train_config)

    history_data = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }

    save_json(
        output_dir / "metrics.json",
        {
            "test_metrics": metrics,
            "history": history_data,
        },
    )


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_dir",
        required=True,
        help="features_20f 目录，例如 D:/datasets/HearBridge-NationalCSL-mini/features_20f",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="模型输出目录，例如 D:/datasets/HearBridge-NationalCSL-mini/models_20f",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="最大训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批大小",
    )

    args = parser.parse_args()

    train_classifier(
        feature_dir=Path(args.feature_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()