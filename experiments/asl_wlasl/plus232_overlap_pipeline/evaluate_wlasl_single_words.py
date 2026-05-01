# -*- coding: utf-8 -*-
"""
评估 WLASL-mini 单词级识别效果。

用途：
1. 判断问题到底来自单词级分类，还是来自拼接句子滑窗后处理。
2. 输出每个样本的 Top1 / Top2 / Top3 预测。
3. 输出每个类别的准确率。
4. 输出主要混淆对，例如 want -> today、school -> work。

输入：
- features_20f_plus/X.npy
- features_20f_plus/y.npy
- features_20f_plus/labels.json
- features_20f_plus/sample_index.csv
- models_20f_plus/best_wlasl_20f_plus_classifier.keras
- models_20f_plus/train_config.json

输出：
- single_word_predictions.csv
- class_summary.csv
- confusion_pairs.csv
- confusion_matrix.csv
- summary.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """读取 CSV。"""
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def choose_model_path(model_dir: Path, model_name: str) -> Path:
    """选择模型文件路径。"""
    if model_name:
        path = model_dir / model_name
        if not path.exists():
            raise FileNotFoundError(f"指定模型不存在：{path}")
        return path

    candidates = [
        model_dir / "best_wlasl_20f_plus_classifier.keras",
        model_dir / "wlasl_20f_plus_classifier.keras",
        model_dir / "best_national_csl_20f_classifier.keras",
        model_dir / "national_csl_20f_classifier.keras",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"模型目录中找不到可用模型：{model_dir}")


def load_selected_indices(
    mode: str,
    total_count: int,
    train_config_path: Path,
) -> np.ndarray:
    """
    根据 mode 选择评估样本。

    mode:
    - all：全部样本
    - train：训练集样本
    - val：验证集样本
    - test：测试集样本
    """
    if mode == "all":
        return np.arange(total_count, dtype=np.int64)

    if not train_config_path.exists():
        raise FileNotFoundError(
            f"mode={mode} 需要 train_config.json，但文件不存在：{train_config_path}"
        )

    config = load_json(train_config_path)

    key_map = {
        "train": "train_indices",
        "val": "val_indices",
        "test": "test_indices",
    }

    key = key_map.get(mode)

    if key is None:
        raise ValueError(f"未知 mode：{mode}")

    if key not in config:
        raise KeyError(f"train_config.json 中没有字段：{key}")

    return np.array(config[key], dtype=np.int64)


def get_index_row(index_rows: List[Dict[str, str]], sample_index: int) -> Dict[str, str]:
    """
    根据 sample_index 获取 sample_index.csv 中的行。
    """
    if not index_rows:
        return {}

    if sample_index < len(index_rows):
        return index_rows[sample_index]

    return {}


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """计算混淆矩阵。"""
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for true_id, pred_id in zip(y_true, y_pred):
        matrix[int(true_id), int(pred_id)] += 1

    return matrix


def write_confusion_matrix(path: Path, matrix: np.ndarray, labels: List[str]) -> None:
    """写出混淆矩阵 CSV。"""
    rows = []

    for true_id, true_label in enumerate(labels):
        row = {"true\\pred": true_label}

        for pred_id, pred_label in enumerate(labels):
            row[pred_label] = int(matrix[true_id, pred_id])

        rows.append(row)

    write_csv(
        path,
        rows,
        ["true\\pred"] + labels,
    )


def build_prediction_rows(
    selected_indices: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    labels: List[str],
    index_rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    """构建逐样本预测明细。"""
    rows = []

    class_count = len(labels)

    for row_pos, sample_index in enumerate(selected_indices.tolist()):
        true_id = int(y_true[row_pos])
        prob = probs[row_pos]
        sorted_ids = np.argsort(prob)[::-1]

        top_ids = sorted_ids[:min(5, class_count)]
        pred_id = int(top_ids[0])

        meta = get_index_row(index_rows, sample_index)

        output_row = {
            "sample_index": sample_index,
            "sample_id": meta.get("sample_id", ""),
            "source_type": meta.get("source_type", "word"),
            "source_label": meta.get("source_label", labels[true_id]),
            "true_label": labels[true_id],
            "pred_label": labels[pred_id],
            "correct": int(true_id == pred_id),
            "top1_label": labels[pred_id],
            "top1_prob": round(float(prob[pred_id]), 6),
            "local_path": meta.get("local_path", ""),
            "action_start": meta.get("action_start", ""),
            "action_end": meta.get("action_end", ""),
            "hand_frame_count": meta.get("hand_frame_count", ""),
            "pose_ratio": meta.get("pose_ratio", ""),
            "any_hand_ratio": meta.get("any_hand_ratio", ""),
        }

        for rank, label_id in enumerate(top_ids, start=1):
            output_row[f"top{rank}_label"] = labels[int(label_id)]
            output_row[f"top{rank}_prob"] = round(float(prob[int(label_id)]), 6)

        rows.append(output_row)

    return rows


def build_class_summary(
    prediction_rows: List[Dict[str, object]],
    labels: List[str],
) -> List[Dict[str, object]]:
    """构建每类准确率统计。"""
    rows = []

    for label in labels:
        label_rows = [
            row for row in prediction_rows
            if row["true_label"] == label
        ]

        support = len(label_rows)
        correct = sum(int(row["correct"]) for row in label_rows)
        accuracy = correct / support if support else 0.0

        wrong_rows = [
            row for row in label_rows
            if int(row["correct"]) == 0
        ]

        pred_counter: Dict[str, int] = {}

        for row in wrong_rows:
            pred = str(row["pred_label"])
            pred_counter[pred] = pred_counter.get(pred, 0) + 1

        confused_as = sorted(
            pred_counter.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        rows.append({
            "label": label,
            "support": support,
            "correct": correct,
            "wrong": support - correct,
            "accuracy": round(accuracy, 6),
            "most_confused_as": " ; ".join(
                f"{pred}:{count}"
                for pred, count in confused_as
            ),
        })

    return rows


def build_confusion_pairs(
    prediction_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """构建错误混淆对统计。"""
    counter: Dict[Tuple[str, str], Dict[str, object]] = {}

    for row in prediction_rows:
        if int(row["correct"]) == 1:
            continue

        true_label = str(row["true_label"])
        pred_label = str(row["pred_label"])
        key = (true_label, pred_label)

        if key not in counter:
            counter[key] = {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": 0,
                "sample_indices": [],
                "top1_probs": [],
            }

        counter[key]["count"] = int(counter[key]["count"]) + 1
        counter[key]["sample_indices"].append(str(row["sample_index"]))
        counter[key]["top1_probs"].append(float(row["top1_prob"]))

    rows = []

    for item in counter.values():
        probs = item["top1_probs"]

        rows.append({
            "true_label": item["true_label"],
            "pred_label": item["pred_label"],
            "count": item["count"],
            "avg_top1_prob": round(sum(probs) / len(probs), 6) if probs else 0.0,
            "sample_indices": " ".join(item["sample_indices"]),
        })

    return sorted(
        rows,
        key=lambda row: (int(row["count"]), float(row["avg_top1_prob"])),
        reverse=True,
    )


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
        help="模型目录",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini/single_word_eval",
        help="输出目录",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "train", "val", "test"],
        help="评估范围",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help="指定模型文件名；默认自动选择 best 模型",
    )

    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(feature_dir / "X.npy")
    y = np.load(feature_dir / "y.npy")

    labels_payload = load_json(feature_dir / "labels.json")
    labels = labels_payload["labels"]
    class_count = len(labels)

    sample_index_path = feature_dir / "sample_index.csv"
    index_rows = read_csv_rows(sample_index_path)

    model_path = choose_model_path(model_dir, args.model_name)
    train_config_path = model_dir / "train_config.json"

    selected_indices = load_selected_indices(
        mode=args.mode,
        total_count=X.shape[0],
        train_config_path=train_config_path,
    )

    X_selected = X[selected_indices]
    y_selected = y[selected_indices]

    print("========== WLASL 单词级识别诊断 ==========")
    print(f"[信息] feature_dir：{feature_dir}")
    print(f"[信息] model_dir：{model_dir}")
    print(f"[信息] model_path：{model_path}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] mode：{args.mode}")
    print(f"[信息] X shape：{X.shape}")
    print(f"[信息] selected X shape：{X_selected.shape}")
    print(f"[信息] class_count：{class_count}")
    print(f"[信息] labels：{labels}")

    model = tf.keras.models.load_model(model_path)

    probs = model.predict(X_selected, verbose=0)
    pred = np.argmax(probs, axis=1)

    correct_count = int(np.sum(pred == y_selected))
    total_count = int(len(y_selected))
    accuracy = correct_count / total_count if total_count else 0.0

    top3_correct = 0

    for i, true_id in enumerate(y_selected.tolist()):
        top3_ids = np.argsort(probs[i])[::-1][:min(3, class_count)]
        if int(true_id) in [int(x) for x in top3_ids]:
            top3_correct += 1

    top3_accuracy = top3_correct / total_count if total_count else 0.0

    prediction_rows = build_prediction_rows(
        selected_indices=selected_indices,
        y_true=y_selected,
        probs=probs,
        labels=labels,
        index_rows=index_rows,
    )

    class_summary_rows = build_class_summary(
        prediction_rows=prediction_rows,
        labels=labels,
    )

    confusion_pair_rows = build_confusion_pairs(
        prediction_rows=prediction_rows,
    )

    confusion_matrix = compute_confusion_matrix(
        y_true=y_selected,
        y_pred=pred,
        class_count=class_count,
    )

    write_csv(
        output_dir / "single_word_predictions.csv",
        prediction_rows,
        [
            "sample_index",
            "sample_id",
            "source_type",
            "source_label",
            "true_label",
            "pred_label",
            "correct",
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
            "local_path",
            "action_start",
            "action_end",
            "hand_frame_count",
            "pose_ratio",
            "any_hand_ratio",
        ],
    )

    write_csv(
        output_dir / "class_summary.csv",
        class_summary_rows,
        [
            "label",
            "support",
            "correct",
            "wrong",
            "accuracy",
            "most_confused_as",
        ],
    )

    write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_pair_rows,
        [
            "true_label",
            "pred_label",
            "count",
            "avg_top1_prob",
            "sample_indices",
        ],
    )

    write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        confusion_matrix,
        labels,
    )

    summary = {
        "mode": args.mode,
        "feature_dir": str(feature_dir),
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "sample_count": total_count,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 6),
        "top3_correct_count": int(top3_correct),
        "top3_accuracy": round(top3_accuracy, 6),
        "class_count": class_count,
        "labels": labels,
        "confusion_pairs": confusion_pair_rows,
        "class_summary": class_summary_rows,
    }

    save_json(output_dir / "summary.json", summary)

    print("\n========== 总体结果 ==========")
    print(f"样本数：{total_count}")
    print(f"Top1 正确数：{correct_count}")
    print(f"Top1 accuracy：{accuracy:.4f}")
    print(f"Top3 正确数：{top3_correct}")
    print(f"Top3 accuracy：{top3_accuracy:.4f}")

    print("\n========== 类别准确率 ==========")

    for row in class_summary_rows:
        print(
            f"{row['label']}: "
            f"{row['correct']}/{row['support']} "
            f"acc={row['accuracy']} "
            f"confused_as={row['most_confused_as']}"
        )

    print("\n========== 主要混淆对 ==========")

    if not confusion_pair_rows:
        print("无错误混淆")
    else:
        for row in confusion_pair_rows[:20]:
            print(
                f"{row['true_label']} -> {row['pred_label']} "
                f"count={row['count']} "
                f"avg_prob={row['avg_top1_prob']} "
                f"samples={row['sample_indices']}"
            )


if __name__ == "__main__":
    main()
