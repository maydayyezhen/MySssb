# -*- coding: utf-8 -*-
"""
训练 WLASL 20帧特征序列的 Conv1D 时序 CNN 分类模型。

适用输入：
- X.npy: shape = (N, 20, feature_dim)
- y.npy: shape = (N,)
- labels.json

设计目标：
1. 与原 BiGRU 模型使用同一份特征。
2. 可复用原模型的 train/val/test split，便于公平对比。
3. 输出格式尽量兼容现有 evaluate_wlasl_single_words.py 和 infer_wlasl_sentence_video.py。
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证可复现。"""
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


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def load_or_make_split(
    y: np.ndarray,
    split_config_path: Path,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载已有划分；如果没有，就生成一个简单分层划分。

    优先使用原 GRU 模型的 train_config.json，
    保证 CNN 和 GRU 在同一测试集上比较。
    """
    if split_config_path.exists():
        config = load_json(split_config_path)

        if all(key in config for key in ["train_indices", "val_indices", "test_indices"]):
            print(f"[信息] 使用已有 split：{split_config_path}")
            return (
                np.array(config["train_indices"], dtype=np.int64),
                np.array(config["val_indices"], dtype=np.int64),
                np.array(config["test_indices"], dtype=np.int64),
            )

    print("[警告] 未找到可用 split_config，使用随机分层划分。")

    from sklearn.model_selection import train_test_split

    all_indices = np.arange(len(y), dtype=np.int64)

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.22,
        random_state=seed,
        stratify=y,
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.18,
        random_state=seed,
        stratify=y[train_val_indices],
    )

    return train_indices, val_indices, test_indices


def compute_class_weight_dict(y_train: np.ndarray) -> Dict[int, float]:
    """计算类别权重，轻微缓解类别数量不均衡。"""
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_count = len(classes)

    result = {}

    for cls, count in zip(classes, counts):
        result[int(cls)] = float(total / (class_count * count))

    return result


def build_cnn_model(
    window_size: int,
    feature_dim: int,
    class_count: int,
    dropout_rate: float,
    learning_rate: float,
) -> tf.keras.Model:
    """
    构建 Conv1D 时序 CNN。

    输入：
    - (20, feature_dim)

    结构：
    - LayerNorm
    - 多层 Conv1D 提取短时动作模式
    - GlobalAveragePooling + GlobalMaxPooling
    - Dense 分类
    """
    inputs = tf.keras.Input(shape=(window_size, feature_dim), name="sequence_input")

    x = tf.keras.layers.LayerNormalization(name="input_layer_norm")(inputs)

    # 轻微噪声增强，防止小样本过拟合。
    x = tf.keras.layers.GaussianNoise(0.01, name="input_noise")(x)

    x = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)

    x = tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate, name="dropout2")(x)

    x = tf.keras.layers.Conv1D(
        filters=192,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv3",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)

    # dilation 让 CNN 看更宽一点的时间上下文。
    x = tf.keras.layers.Conv1D(
        filters=192,
        kernel_size=3,
        padding="same",
        dilation_rate=2,
        activation="relu",
        name="conv4_dilated",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate, name="dropout4")(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    x = tf.keras.layers.Concatenate(name="pool_concat")([avg_pool, max_pool])

    x = tf.keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dense_dropout")(x)

    outputs = tf.keras.layers.Dense(
        class_count,
        activation="softmax",
        name="classifier",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="wlasl_20f_conv1d_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    return model


def evaluate_and_write_outputs(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[str],
    output_dir: Path,
) -> Dict[str, object]:
    """评估模型并写出混淆矩阵、错误样本。"""
    loss, accuracy, top3_accuracy = model.evaluate(X_test, y_test, verbose=0)

    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)

    class_count = len(labels)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for true_id, pred_id in zip(y_test.tolist(), pred.tolist()):
        matrix[int(true_id), int(pred_id)] += 1

    matrix_rows = []

    for true_id, true_label in enumerate(labels):
        row = {"true\\pred": true_label}

        for pred_id, pred_label in enumerate(labels):
            row[pred_label] = int(matrix[true_id, pred_id])

        matrix_rows.append(row)

    write_csv(
        output_dir / "confusion_matrix.csv",
        matrix_rows,
        ["true\\pred"] + labels,
    )

    error_rows = []

    for i, (true_id, pred_id) in enumerate(zip(y_test.tolist(), pred.tolist())):
        if int(true_id) == int(pred_id):
            continue

        prob = probs[i]
        top_ids = np.argsort(prob)[::-1][:min(5, class_count)]

        row = {
            "test_row": i,
            "true_label": labels[int(true_id)],
            "pred_label": labels[int(pred_id)],
            "top1_prob": round(float(prob[int(pred_id)]), 6),
        }

        for rank, label_id in enumerate(top_ids, start=1):
            row[f"top{rank}_label"] = labels[int(label_id)]
            row[f"top{rank}_prob"] = round(float(prob[int(label_id)]), 6)

        error_rows.append(row)

    write_csv(
        output_dir / "error_samples.csv",
        error_rows,
        [
            "test_row",
            "true_label",
            "pred_label",
            "top1_prob",
            "top1_label",
            "top1_prob",
            "top2_label",
            "top2_prob",
            "top3_label",
            "top3_prob",
            "top4_label",
            "top4_prob",
            "top5_label",
            "top5_prob",
        ],
    )

    return {
        "loss": round(float(loss), 6),
        "accuracy": round(float(accuracy), 6),
        "top3_accuracy": round(float(top3_accuracy), 6),
        "error_count": len(error_rows),
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        required=True,
        help="特征目录，例如 D:/datasets/WLASL-mini-v2-25/features_20f_plus",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="模型输出目录",
    )
    parser.add_argument(
        "--split_config",
        default="",
        help="可选：复用已有 train_config.json",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dropout_rate", type=float, default=0.30)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

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
    window_size = int(X.shape[1])
    feature_dim = int(X.shape[2])

    if args.split_config.strip():
        split_config_path = Path(args.split_config)
    else:
        split_config_path = model_dir / "train_config.json"

    train_indices, val_indices, test_indices = load_or_make_split(
        y=y,
        split_config_path=split_config_path,
        seed=args.seed,
    )

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    class_weight = compute_class_weight_dict(y_train)

    print("========== WLASL Conv1D CNN 训练 ==========")
    print(f"[信息] feature_dir：{feature_dir}")
    print(f"[信息] model_dir：{model_dir}")
    print(f"[信息] X shape：{X.shape}")
    print(f"[信息] class_count：{class_count}")
    print(f"[信息] labels：{labels}")
    print(f"[信息] train/val/test：{len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"[信息] window_size：{window_size}")
    print(f"[信息] feature_dim：{feature_dim}")

    model = build_cnn_model(
        window_size=window_size,
        feature_dim=feature_dim,
        class_count=class_count,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )

    model.summary()

    best_model_path = model_dir / "best_wlasl_20f_plus_classifier.keras"
    final_model_path = model_dir / "wlasl_20f_plus_classifier.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
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
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(final_model_path)
    print(f"[完成] 已保存最终模型：{final_model_path}")

    # 使用 best 模型评估。
    if best_model_path.exists():
        eval_model = tf.keras.models.load_model(best_model_path)
    else:
        eval_model = model

    metrics = evaluate_and_write_outputs(
        model=eval_model,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        output_dir=model_dir,
    )

    save_json(
        model_dir / "metrics.json",
        metrics,
    )

    train_config = {
        "architecture": "conv1d_cnn",
        "feature_dir": str(feature_dir),
        "model_dir": str(model_dir),
        "labels": labels,
        "class_count": class_count,
        "input_shape": [window_size, feature_dim],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
    }

    save_json(
        model_dir / "train_config.json",
        train_config,
    )

    print("\n========== 测试集结果 ==========")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"top3_accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"error_count: {metrics['error_count']}")


if __name__ == "__main__":
    main()
