# -*- coding: utf-8 -*-
"""
NationalCSL-DP 20帧模型测试集 TopK 诊断脚本。

作用：
1. 加载训练好的 20 帧词语分类模型
2. 加载 X.npy / y.npy / sample_index.csv / label_map.json
3. 按参与者划分测试集，默认使用 Participant_10
4. 输出每个测试样本的 Top1 / Top3 预测结果
5. 生成 CSV 和 JSON 诊断文件

推荐运行：
python experiments/national_csl_dp/predict_20f_test_samples.py ^
  --feature_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f" ^
  --model_dir "D:/datasets/HearBridge-NationalCSL-mini/models_20f" ^
  --output_dir "D:/datasets/HearBridge-NationalCSL-mini/predict_20f"
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


DEFAULT_TRAIN_PARTICIPANTS = {
    "Participant_01",
    "Participant_02",
    "Participant_03",
    "Participant_04",
    "Participant_05",
    "Participant_06",
    "Participant_07",
    "Participant_08",
}

DEFAULT_VAL_PARTICIPANTS = {
    "Participant_09",
}

DEFAULT_TEST_PARTICIPANTS = {
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
    读取特征样本索引。

    只保留：
    1. status=ok
    2. label_id 非空

    注意：
    返回顺序必须和 X.npy / y.npy 中的样本顺序一致。
    """
    with sample_index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    valid_rows = [
        row for row in rows
        if row.get("status") == "ok" and str(row.get("label_id", "")).strip() != ""
    ]

    return valid_rows


def load_split_from_train_config(model_dir: Path) -> Tuple[set, set, set]:
    """
    尝试从 train_config.json 中读取训练/验证/测试参与者划分。

    如果不存在，则使用默认划分。
    """
    train_config_path = model_dir / "train_config.json"

    if not train_config_path.exists():
        return DEFAULT_TRAIN_PARTICIPANTS, DEFAULT_VAL_PARTICIPANTS, DEFAULT_TEST_PARTICIPANTS

    train_config = load_json(train_config_path)
    split = train_config.get("split", {})

    train_participants = set(split.get("train_participants", sorted(DEFAULT_TRAIN_PARTICIPANTS)))
    val_participants = set(split.get("val_participants", sorted(DEFAULT_VAL_PARTICIPANTS)))
    test_participants = set(split.get("test_participants", sorted(DEFAULT_TEST_PARTICIPANTS)))

    return train_participants, val_participants, test_participants


def select_indices_by_mode(
    rows: List[Dict[str, str]],
    mode: str,
    train_participants: set,
    val_participants: set,
    test_participants: set,
) -> np.ndarray:
    """
    根据 mode 选择样本索引。

    mode:
    - train
    - val
    - test
    - all
    """
    selected_indices = []

    for index, row in enumerate(rows):
        participant = row["participant"]

        if mode == "all":
            selected_indices.append(index)
        elif mode == "train" and participant in train_participants:
            selected_indices.append(index)
        elif mode == "val" and participant in val_participants:
            selected_indices.append(index)
        elif mode == "test" and participant in test_participants:
            selected_indices.append(index)

    return np.array(selected_indices, dtype=np.int64)


def apply_normalizer(X: np.ndarray, normalizer_path: Path) -> np.ndarray:
    """
    加载并应用训练阶段保存的标准化参数。
    """
    data = np.load(normalizer_path)
    mean = data["mean"]
    std = data["std"]

    std = np.where(std < 1e-6, 1.0, std)

    return ((X - mean) / std).astype(np.float32)


def get_topk_predictions(prob: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取 TopK 预测类别和概率。

    返回：
    - topk_ids: shape=(N, k)
    - topk_probs: shape=(N, k)
    """
    topk_ids = np.argsort(prob, axis=1)[:, -k:][:, ::-1]
    topk_probs = np.take_along_axis(prob, topk_ids, axis=1)

    return topk_ids, topk_probs


def write_predictions_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出预测明细 CSV。
    """
    if not rows:
        print("[警告] 没有预测结果可写出")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出预测明细：{output_path}")


def build_prediction_rows(
    selected_rows: List[Dict[str, str]],
    y_true: np.ndarray,
    topk_ids: np.ndarray,
    topk_probs: np.ndarray,
    id_to_label: Dict[int, str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    构造预测明细和汇总指标。
    """
    result_rows: List[Dict[str, object]] = []

    top1_correct_count = 0
    top3_correct_count = 0

    for i, row in enumerate(selected_rows):
        true_id = int(y_true[i])
        true_label = id_to_label[true_id]

        top_ids = [int(x) for x in topk_ids[i].tolist()]
        top_probs = [float(x) for x in topk_probs[i].tolist()]

        top_labels = [id_to_label[class_id] for class_id in top_ids]

        top1_hit = top_ids[0] == true_id
        top3_hit = true_id in top_ids

        top1_correct_count += int(top1_hit)
        top3_correct_count += int(top3_hit)

        result_rows.append({
            "sample_no": i + 1,
            "participant": row["participant"],
            "resource_id": row["resource_id"],
            "source_word": row["source_word"],
            "true_label": true_label,
            "true_id": true_id,

            "top1_label": top_labels[0],
            "top1_id": top_ids[0],
            "top1_prob": f"{top_probs[0]:.6f}",

            "top2_label": top_labels[1] if len(top_labels) > 1 else "",
            "top2_id": top_ids[1] if len(top_ids) > 1 else "",
            "top2_prob": f"{top_probs[1]:.6f}" if len(top_probs) > 1 else "",

            "top3_label": top_labels[2] if len(top_labels) > 2 else "",
            "top3_id": top_ids[2] if len(top_ids) > 2 else "",
            "top3_prob": f"{top_probs[2]:.6f}" if len(top_probs) > 2 else "",

            "top1_hit": int(top1_hit),
            "top3_hit": int(top3_hit),

            "raw_frame_count": row.get("raw_frame_count", ""),
            "used_frame_count": row.get("used_frame_count", ""),
            "frame_dir": row.get("frame_dir", ""),
        })

    total = len(result_rows)

    metrics = {
        "sample_count": total,
        "top1_correct_count": top1_correct_count,
        "top3_correct_count": top3_correct_count,
        "top1_accuracy": round(top1_correct_count / total, 6) if total else 0.0,
        "top3_accuracy": round(top3_correct_count / total, 6) if total else 0.0,
    }

    return result_rows, metrics


def predict_samples(
    feature_dir: Path,
    model_dir: Path,
    output_dir: Path,
    mode: str,
    top_k: int,
    model_name: str,
) -> None:
    """
    执行 TopK 预测诊断。
    """
    X_path = feature_dir / "X.npy"
    y_path = feature_dir / "y.npy"
    label_map_path = feature_dir / "label_map.json"
    sample_index_path = feature_dir / "sample_index.csv"
    normalizer_path = model_dir / "normalizer.npz"
    model_path = model_dir / model_name

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    label_map = load_json(label_map_path)
    id_to_label = {int(v): k for k, v in label_map.items()}

    rows = read_sample_index(sample_index_path)

    if len(rows) != len(X):
        raise RuntimeError(
            f"sample_index 有效样本数和 X 数量不一致：rows={len(rows)}, X={len(X)}"
        )

    train_participants, val_participants, test_participants = load_split_from_train_config(model_dir)

    selected_indices = select_indices_by_mode(
        rows=rows,
        mode=mode,
        train_participants=train_participants,
        val_participants=val_participants,
        test_participants=test_participants,
    )

    if len(selected_indices) == 0:
        raise RuntimeError(f"mode={mode} 没有选中任何样本，请检查参与者划分。")

    selected_rows = [rows[int(index)] for index in selected_indices]

    X_selected = X[selected_indices]
    y_selected = y[selected_indices]

    X_selected = apply_normalizer(X_selected, normalizer_path)

    print("========== TopK 预测诊断 ==========")
    print(f"[信息] feature_dir：{feature_dir}")
    print(f"[信息] model_dir：{model_dir}")
    print(f"[信息] model_path：{model_path}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] mode：{mode}")
    print(f"[信息] 样本数：{len(X_selected)}")
    print(f"[信息] X_selected shape：{X_selected.shape}")
    print(f"[信息] top_k：{top_k}")

    model = keras.models.load_model(model_path)
    prob = model.predict(X_selected, verbose=0)

    topk_ids, topk_probs = get_topk_predictions(prob, k=top_k)

    result_rows, metrics = build_prediction_rows(
        selected_rows=selected_rows,
        y_true=y_selected,
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        id_to_label=id_to_label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{mode}_top{top_k}_predictions.csv"
    json_path = output_dir / f"{mode}_top{top_k}_metrics.json"

    write_predictions_csv(csv_path, result_rows)

    metrics_payload = {
        "mode": mode,
        "top_k": top_k,
        "model_path": str(model_path),
        "feature_dir": str(feature_dir),
        "metrics": metrics,
    }

    save_json(json_path, metrics_payload)

    print("\n========== 预测汇总 ==========")
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

    print("\n========== 错误样本 ==========")
    wrong_rows = [row for row in result_rows if int(row["top1_hit"]) == 0]

    if not wrong_rows:
        print("[信息] 没有 Top1 错误样本。")
    else:
        for row in wrong_rows:
            print(
                f"真实={row['true_label']} "
                f"Top1={row['top1_label']}({row['top1_prob']}) "
                f"Top2={row['top2_label']}({row['top2_prob']}) "
                f"Top3={row['top3_label']}({row['top3_prob']}) "
                f"Top3命中={row['top3_hit']} "
                f"participant={row['participant']}"
            )


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_dir",
        required=True,
        help="features_20f 目录",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="models_20f 目录",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="预测结果输出目录",
    )
    parser.add_argument(
        "--mode",
        default="test",
        choices=["train", "val", "test", "all"],
        help="预测哪一部分数据，默认 test",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="输出 TopK，默认 3",
    )
    parser.add_argument(
        "--model_name",
        default="best_national_csl_20f_classifier.keras",
        help="模型文件名，默认使用验证集最佳模型",
    )

    args = parser.parse_args()

    predict_samples(
        feature_dir=Path(args.feature_dir),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        top_k=args.top_k,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()