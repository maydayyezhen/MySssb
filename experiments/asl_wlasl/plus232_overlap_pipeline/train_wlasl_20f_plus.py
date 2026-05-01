# -*- coding: utf-8 -*-
"""
训练 WLASL-mini 20帧 plus 特征分类模型。

输入：
- D:/datasets/WLASL-mini/features_20f_plus/X.npy
- D:/datasets/WLASL-mini/features_20f_plus/y.npy
- D:/datasets/WLASL-mini/features_20f_plus/labels.json
- D:/datasets/WLASL-mini/features_20f_plus/sample_index.csv

输出：
- best_wlasl_20f_plus_classifier.keras
- wlasl_20f_plus_classifier.keras
- metrics.json
- train_config.json
- confusion_matrix.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_json(path: Path) -> Dict:
    """读取 JSON。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def write_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    labels: List[str],
) -> None:
    """写出混淆矩阵 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)

        for i, label in enumerate(labels):
            writer.writerow([label] + matrix[i].tolist())

    print(f"[完成] 已写出混淆矩阵：{path}")


def stratified_split(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按类别分层划分 train / val / test。

    每类样本很少，所以策略：
    - 每类至少留 1 个 test
    - 每类至少留 1 个 val
    - 其余 train
    """
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    labels = sorted(set(int(v) for v in y.tolist()))

    for label in labels:
        indices = np.where(y == label)[0].tolist()
        rng.shuffle(indices)

        n = len(indices)

        if n < 3:
            raise RuntimeError(f"类别 {label} 样本数过少：{n}")

        test_count = max(1, int(round(n * test_ratio)))
        val_count = max(1, int(round(n * val_ratio)))

        if test_count + val_count >= n:
            test_count = 1
            val_count = 1

        test_part = indices[:test_count]
        val_part = indices[test_count:test_count + val_count]
        train_part = indices[test_count + val_count:]

        test_indices.extend(test_part)
        val_indices.extend(val_part)
        train_indices.extend(train_part)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )


def build_model(
    input_shape: Tuple[int, int],
    class_count: int,
    dropout: float,
) -> tf.keras.Model:
    """
    构建轻量时序分类模型。

    数据量只有 120，不适合太大模型。
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Masking(mask_value=0.0)(inputs)

    x = layers.Bidirectional(
        layers.GRU(
            96,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
        )
    )(x)

    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.GRU(
            64,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=0.0,
        )
    )(x)

    x = layers.LayerNormalization()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(class_count, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=min(3, class_count),
                name="top3_accuracy",
            ),
        ],
    )

    return model


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """计算混淆矩阵。"""
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1

    return matrix


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        default="D:/datasets/WLASL-mini/features_20f_plus",
        help="特征目录",
    )
    parser.add_argument(
        "--model_dir",
        default="D:/datasets/WLASL-mini/models_20f_plus",
        help="模型输出目录",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--dropout", type=float, default=0.35)

    args = parser.parse_args()

    set_seed(args.seed)

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(feature_dir / "X.npy")
    y = np.load(feature_dir / "y.npy")

    labels_payload = load_json(feature_dir / "labels.json")
    labels = labels_payload["labels"]
    class_count = len(labels)

    print("========== WLASL 20f plus 训练 ==========")
    print(f"[信息] feature_dir：{feature_dir}")
    print(f"[信息] model_dir：{model_dir}")
    print(f"[信息] X shape：{X.shape}")
    print(f"[信息] y shape：{y.shape}")
    print(f"[信息] class_count：{class_count}")
    print(f"[信息] labels：{labels}")

    train_idx, val_idx, test_idx = stratified_split(
        y=y,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\n========== 数据划分 ==========")
    print(f"train：{X_train.shape}, {y_train.shape}")
    print(f"val：{X_val.shape}, {y_val.shape}")
    print(f"test：{X_test.shape}, {y_test.shape}")

    model = build_model(
        input_shape=(X.shape[1], X.shape[2]),
        class_count=class_count,
        dropout=args.dropout,
    )

    model.summary()

    best_model_path = model_dir / "best_wlasl_20f_plus_classifier.keras"
    final_model_path = model_dir / "wlasl_20f_plus_classifier.keras"

    cb = [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1,
    )

    model.save(final_model_path)
    print(f"[完成] 已保存最终模型：{final_model_path}")

    best_model = tf.keras.models.load_model(best_model_path)

    print("\n========== 测试集结果 ==========")
    test_result = best_model.evaluate(X_test, y_test, verbose=0)

    metric_names = best_model.metrics_names
    test_metrics = {
        name: float(value)
        for name, value in zip(metric_names, test_result)
    }

    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    probs = best_model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)

    confusion = compute_confusion_matrix(
        y_true=y_test,
        y_pred=pred,
        class_count=class_count,
    )

    write_confusion_matrix(
        model_dir / "confusion_matrix.csv",
        confusion,
        labels,
    )

    wrong_cases = []

    for i, sample_idx in enumerate(test_idx.tolist()):
        true_id = int(y_test[i])
        pred_id = int(pred[i])

        if true_id != pred_id:
            top3_ids = np.argsort(probs[i])[::-1][:min(3, class_count)]
            wrong_cases.append({
                "sample_index": int(sample_idx),
                "true": labels[true_id],
                "pred": labels[pred_id],
                "top3": [
                    {
                        "label": labels[int(label_id)],
                        "prob": float(probs[i][label_id]),
                    }
                    for label_id in top3_ids
                ],
            })

    save_json(
        model_dir / "metrics.json",
        {
            "test_metrics": test_metrics,
            "wrong_cases": wrong_cases,
            "train_count": int(len(train_idx)),
            "val_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
        },
    )

    save_json(
        model_dir / "train_config.json",
        {
            "feature_dir": str(feature_dir),
            "model_dir": str(model_dir),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "dropout": args.dropout,
            "X_shape": list(X.shape),
            "class_count": class_count,
            "labels": labels,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "history": {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
        },
    )

    print("\n========== 错误样本 ==========")

    if not wrong_cases:
        print("无错误样本")
    else:
        for item in wrong_cases:
            top3_text = " | ".join(
                f"{top['label']}({top['prob']:.4f})"
                for top in item["top3"]
            )
            print(
                f"sample={item['sample_index']} "
                f"真实={item['true']} "
                f"预测={item['pred']} "
                f"Top3={top3_text}"
            )


if __name__ == "__main__":
    main()