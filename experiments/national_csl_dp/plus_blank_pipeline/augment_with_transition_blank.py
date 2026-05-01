# -*- coding: utf-8 -*-
"""
为 NationalCSL-DP 20帧特征数据集增加 blank / transition 类。

输入：
D:/datasets/HearBridge-NationalCSL-mini/features_20f/
  X.npy
  y.npy
  label_map.json
  sample_index.csv

输出：
D:/datasets/HearBridge-NationalCSL-mini/features_20f_blank/
  X.npy
  y.npy
  label_map.json
  sample_index.csv
  augment_config.json

blank 构造逻辑：
1. 从同一参与者的两个不同词样本中选 A / B
2. 取 A 的后半段 + B 的前半段，拼成 20 帧窗口
3. 将该混合窗口标记为 blank
4. 用于训练模型识别“词间过渡 / 非完整词窗口”
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_json(path: Path) -> Dict:
    """
    读取 JSON。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    """
    保存 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def read_sample_index(sample_index_csv: Path) -> List[Dict[str, str]]:
    """
    读取样本索引。
    """
    with sample_index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    return [
        row for row in rows
        if row.get("status") == "ok" and str(row.get("label_id", "")).strip() != ""
    ]


def write_sample_index(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出新的样本索引。

    注意：
    原始样本和 synthetic blank 样本字段不完全一致，
    所以这里需要收集所有行的字段并统一补空值。
    """
    if not rows:
        print("[警告] 没有样本索引可写出")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_fieldnames = [
        "resource_id",
        "label",
        "source_word",
        "participant",
        "view",
        "frame_dir",
        "raw_frame_count",
        "used_frame_count",
        "target_frames",
        "pose_present_count",
        "any_hand_present_count",
        "both_hands_present_count",
        "left_hand_present_count",
        "right_hand_present_count",
        "status",
        "reason",
        "label_id",
        "synthetic",
        "left_label",
        "left_resource_id",
        "right_label",
        "right_resource_id",
        "split_point",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = [
        key for key in preferred_fieldnames
        if key in all_keys
    ]

    extra_fieldnames = sorted([
        key for key in all_keys
        if key not in fieldnames
    ])

    fieldnames.extend(extra_fieldnames)

    normalized_rows = []

    for row in rows:
        normalized_row = {}
        for fieldname in fieldnames:
            normalized_row[fieldname] = row.get(fieldname, "")
        normalized_rows.append(normalized_row)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"[完成] 已写出样本索引：{output_path}")


def group_indices_by_participant(rows: List[Dict[str, str]]) -> Dict[str, List[int]]:
    """
    按参与者聚合样本索引。
    """
    result: Dict[str, List[int]] = {}

    for index, row in enumerate(rows):
        participant = row["participant"]
        result.setdefault(participant, []).append(index)

    return result


def build_transition_blank_sample(
    X: np.ndarray,
    left_index: int,
    right_index: int,
    split_point: int,
) -> np.ndarray:
    """
    构造一个 transition blank 样本。

    split_point 表示左词贡献多少帧：
    - 取 left 样本的最后 split_point 帧
    - 取 right 样本的前 20 - split_point 帧

    例如：
    split_point = 8
    blank = left[-8:] + right[:12]
    """
    left = X[left_index]
    right = X[right_index]

    window_size = left.shape[0]

    if right.shape[0] != window_size:
        raise ValueError("左右样本窗口长度不一致")

    left_part = left[-split_point:]
    right_part = right[:window_size - split_point]

    mixed = np.concatenate([left_part, right_part], axis=0)

    if mixed.shape != left.shape:
        raise RuntimeError(f"混合后 shape 异常：mixed={mixed.shape}, expected={left.shape}")

    return mixed.astype(np.float32)


def choose_transition_pairs_for_participant(
    rows: List[Dict[str, str]],
    participant_indices: List[int],
    blank_count: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    为某个参与者选择若干不同标签样本对。

    要求：
    1. left / right 来自同一参与者
    2. left_label != right_label
    3. 尽量随机覆盖更多组合
    """
    possible_pairs = []

    for left_index in participant_indices:
        for right_index in participant_indices:
            if left_index == right_index:
                continue

            left_label = rows[left_index]["label"]
            right_label = rows[right_index]["label"]

            if left_label == right_label:
                continue

            possible_pairs.append((left_index, right_index))

    rng.shuffle(possible_pairs)

    return possible_pairs[:blank_count]


def normalize_row_for_output(row: Dict[str, str]) -> Dict[str, object]:
    """
    将原始样本索引行转成可写出的普通 dict。

    保留原字段，方便训练脚本继续读取 participant / label_id / status。
    """
    return dict(row)


def augment_blank_dataset(
    feature_dir: Path,
    output_dir: Path,
    blank_label: str,
    blank_per_participant: int,
    min_split: int,
    max_split: int,
    seed: int,
) -> None:
    """
    构建带 blank 类的新特征数据集。
    """
    X = np.load(feature_dir / "X.npy").astype(np.float32)
    y = np.load(feature_dir / "y.npy").astype(np.int64)

    label_map = load_json(feature_dir / "label_map.json")
    rows = read_sample_index(feature_dir / "sample_index.csv")

    if len(rows) != len(X):
        raise RuntimeError(f"sample_index 与 X 数量不一致：rows={len(rows)}, X={len(X)}")

    if blank_label in label_map:
        raise RuntimeError(f"label_map 中已经存在 {blank_label}，请检查输入目录。")

    rng = random.Random(seed)

    new_label_map = dict(label_map)
    blank_id = max(int(v) for v in new_label_map.values()) + 1
    new_label_map[blank_label] = blank_id

    participant_to_indices = group_indices_by_participant(rows)

    X_list = [item for item in X]
    y_list = [int(item) for item in y]
    output_rows: List[Dict[str, object]] = [
        normalize_row_for_output(row) for row in rows
    ]

    generated_blank_count = 0

    print("========== 开始生成 transition blank 样本 ==========")
    print(f"[信息] 输入特征：{feature_dir}")
    print(f"[信息] 输出目录：{output_dir}")
    print(f"[信息] 原始样本数：{len(X)}")
    print(f"[信息] 原始类别数：{len(label_map)}")
    print(f"[信息] blank_label：{blank_label}")
    print(f"[信息] blank_id：{blank_id}")
    print(f"[信息] 每个参与者 blank 数：{blank_per_participant}")
    print(f"[信息] split范围：{min_split} ~ {max_split}")

    for participant, indices in sorted(participant_to_indices.items()):
        pairs = choose_transition_pairs_for_participant(
            rows=rows,
            participant_indices=indices,
            blank_count=blank_per_participant,
            rng=rng,
        )

        print(f"\n========== 处理 {participant} ==========")
        print(f"[信息] 可用原始样本数：{len(indices)}")
        print(f"[信息] 生成 blank 数：{len(pairs)}")

        for local_no, (left_index, right_index) in enumerate(pairs, start=1):
            split_point = rng.randint(min_split, max_split)

            blank_feature = build_transition_blank_sample(
                X=X,
                left_index=left_index,
                right_index=right_index,
                split_point=split_point,
            )

            left_row = rows[left_index]
            right_row = rows[right_index]

            X_list.append(blank_feature)
            y_list.append(blank_id)

            generated_blank_count += 1

            output_rows.append({
                "resource_id": f"blank_{participant}_{local_no:03d}",
                "label": blank_label,
                "source_word": f"{left_row['label']}->{right_row['label']}",
                "participant": participant,
                "view": left_row.get("view", "front"),
                "frame_dir": "",
                "raw_frame_count": int(blank_feature.shape[0]),
                "used_frame_count": int(blank_feature.shape[0]),
                "target_frames": int(blank_feature.shape[0]),
                "pose_present_count": "",
                "any_hand_present_count": "",
                "both_hands_present_count": "",
                "left_hand_present_count": "",
                "right_hand_present_count": "",
                "status": "ok",
                "reason": "synthetic_transition_blank",
                "label_id": blank_id,
                "synthetic": 1,
                "left_label": left_row["label"],
                "left_resource_id": left_row["resource_id"],
                "right_label": right_row["label"],
                "right_resource_id": right_row["resource_id"],
                "split_point": split_point,
            })

            print(
                f"[blank] {participant} "
                f"{left_row['label']}({left_row['resource_id']}) -> "
                f"{right_row['label']}({right_row['resource_id']}) "
                f"split={split_point}"
            )

    X_new = np.stack(X_list, axis=0).astype(np.float32)
    y_new = np.array(y_list, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X_new)
    np.save(output_dir / "y.npy", y_new)

    save_json(output_dir / "label_map.json", new_label_map)
    write_sample_index(output_dir / "sample_index.csv", output_rows)

    augment_config = {
        "source_feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "original_sample_count": int(len(X)),
        "generated_blank_count": int(generated_blank_count),
        "final_sample_count": int(len(X_new)),
        "original_class_count": int(len(label_map)),
        "final_class_count": int(len(new_label_map)),
        "blank_label": blank_label,
        "blank_id": int(blank_id),
        "blank_per_participant": int(blank_per_participant),
        "split_range": [int(min_split), int(max_split)],
        "seed": int(seed),
        "method": "tail_of_left_word + head_of_right_word",
    }

    save_json(output_dir / "augment_config.json", augment_config)

    print("\n========== blank 增强完成 ==========")
    print(f"[完成] X_new shape：{X_new.shape}")
    print(f"[完成] y_new shape：{y_new.shape}")
    print(f"[完成] 新类别数：{len(new_label_map)}")
    print(f"[完成] blank 样本数：{generated_blank_count}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", required=True, help="原始 features_20f 目录")
    parser.add_argument("--output_dir", required=True, help="输出 features_20f_blank 目录")
    parser.add_argument("--blank_label", default="blank", help="blank 类名称")
    parser.add_argument("--blank_per_participant", type=int, default=8, help="每个参与者生成多少个 blank 样本")
    parser.add_argument("--min_split", type=int, default=6, help="左词最少贡献帧数")
    parser.add_argument("--max_split", type=int, default=14, help="左词最多贡献帧数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    augment_blank_dataset(
        feature_dir=Path(args.feature_dir),
        output_dir=Path(args.output_dir),
        blank_label=args.blank_label,
        blank_per_participant=args.blank_per_participant,
        min_split=args.min_split,
        max_split=args.max_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()