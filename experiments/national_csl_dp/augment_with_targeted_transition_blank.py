# -*- coding: utf-8 -*-
"""
为 NationalCSL-DP 20帧特征数据集增加“定向 transition blank”。

目标：
1. 不再随机大量生成 blank
2. 只针对当前伪连续测试中容易误报的相邻词过渡生成 blank
3. 用少量、定向、硬负样本教模型：
   半个 A + 半个 B ≠ 某个真实词

默认过渡：
- 你 -> 今天
- 今天 -> 学习
- 朋友 -> 帮助
- 帮助 -> 我们
- 我们 -> 需要
- 需要 -> 帮助

输入：
D:/datasets/HearBridge-NationalCSL-mini/features_20f/
  X.npy
  y.npy
  label_map.json
  sample_index.csv

输出：
D:/datasets/HearBridge-NationalCSL-mini/features_20f_blank_targeted/
  X.npy
  y.npy
  label_map.json
  sample_index.csv
  augment_config.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_TRANSITIONS = [
    ("你", "今天"),
    ("今天", "学习"),
    ("朋友", "帮助"),
    ("帮助", "我们"),
    ("我们", "需要"),
    ("需要", "帮助"),
]


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
    读取 sample_index.csv。

    只保留 status=ok 且 label_id 有效的原始样本。
    """
    with sample_index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    return [
        row for row in rows
        if row.get("status") == "ok" and str(row.get("label_id", "")).strip() != ""
    ]


def write_sample_index(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出新的 sample_index.csv。

    原始样本和 synthetic blank 样本字段不完全一致，
    所以这里统一收集所有字段并补空值。
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

    fieldnames = [key for key in preferred_fieldnames if key in all_keys]
    fieldnames.extend(sorted([key for key in all_keys if key not in fieldnames]))

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


def parse_split_points(split_points_text: str) -> List[int]:
    """
    解析 split_points 参数。

    示例：
    --split_points "8,12"
    """
    points = []

    for item in split_points_text.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        points.append(int(item))

    if not points:
        raise ValueError("split_points 不能为空")

    return points


def parse_transitions(transitions_text: str) -> List[Tuple[str, str]]:
    """
    解析 transitions 参数。

    格式：
    "你->今天,今天->学习,朋友->帮助"
    """
    if not transitions_text.strip():
        return DEFAULT_TRANSITIONS

    transitions = []

    for item in transitions_text.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue

        if "->" not in item:
            raise ValueError(f"过渡格式错误：{item}，应为 A->B")

        left, right = item.split("->", 1)
        left = left.strip()
        right = right.strip()

        if not left or not right:
            raise ValueError(f"过渡格式错误：{item}，应为 A->B")

        transitions.append((left, right))

    return transitions


def find_sample_index(
    rows: List[Dict[str, str]],
    participant: str,
    label: str,
) -> int:
    """
    查找某个参与者下某个 label 的样本索引。

    对于“学习”这种有多个 resource_id 的 label，优先取第一个匹配项。
    """
    for index, row in enumerate(rows):
        if row["participant"] == participant and row["label"] == label:
            return index

    raise RuntimeError(f"找不到样本：participant={participant}, label={label}")


def get_participants(rows: List[Dict[str, str]]) -> List[str]:
    """
    获取所有参与者编号。
    """
    return sorted({row["participant"] for row in rows})


def build_transition_blank_sample(
    X: np.ndarray,
    left_index: int,
    right_index: int,
    split_point: int,
) -> np.ndarray:
    """
    构造一个 transition blank 样本。

    split_point 表示左词贡献多少帧：
    blank = left[-split_point:] + right[:20-split_point]
    """
    left = X[left_index]
    right = X[right_index]

    window_size = left.shape[0]

    if right.shape[0] != window_size:
        raise ValueError("左右样本窗口长度不一致")

    if split_point <= 0 or split_point >= window_size:
        raise ValueError(f"split_point 必须在 1 到 {window_size - 1} 之间，当前={split_point}")

    left_part = left[-split_point:]
    right_part = right[:window_size - split_point]

    mixed = np.concatenate([left_part, right_part], axis=0).astype(np.float32)

    if mixed.shape != left.shape:
        raise RuntimeError(f"混合后 shape 异常：mixed={mixed.shape}, expected={left.shape}")

    return mixed


def normalize_original_row(row: Dict[str, str]) -> Dict[str, object]:
    """
    将原始行转为普通 dict，方便和 synthetic 行合并。
    """
    result = dict(row)
    result.setdefault("synthetic", 0)
    result.setdefault("left_label", "")
    result.setdefault("left_resource_id", "")
    result.setdefault("right_label", "")
    result.setdefault("right_resource_id", "")
    result.setdefault("split_point", "")
    return result


def augment_targeted_blank_dataset(
    feature_dir: Path,
    output_dir: Path,
    blank_label: str,
    transitions: List[Tuple[str, str]],
    split_points: List[int],
) -> None:
    """
    构建带 targeted transition blank 的新特征集。
    """
    X = np.load(feature_dir / "X.npy").astype(np.float32)
    y = np.load(feature_dir / "y.npy").astype(np.int64)
    label_map = load_json(feature_dir / "label_map.json")
    rows = read_sample_index(feature_dir / "sample_index.csv")

    if len(rows) != len(X):
        raise RuntimeError(f"sample_index 与 X 数量不一致：rows={len(rows)}, X={len(X)}")

    if blank_label in label_map:
        raise RuntimeError(f"输入 label_map 中已经存在 {blank_label}，请换原始 features_20f 目录")

    new_label_map = dict(label_map)
    blank_id = max(int(v) for v in new_label_map.values()) + 1
    new_label_map[blank_label] = blank_id

    participants = get_participants(rows)

    X_list = [item for item in X]
    y_list = [int(item) for item in y]
    output_rows: List[Dict[str, object]] = [
        normalize_original_row(row) for row in rows
    ]

    generated_count = 0

    print("========== 开始生成 targeted transition blank ==========")
    print(f"[信息] 输入目录：{feature_dir}")
    print(f"[信息] 输出目录：{output_dir}")
    print(f"[信息] 原始样本数：{len(X)}")
    print(f"[信息] 原始类别数：{len(label_map)}")
    print(f"[信息] blank_label：{blank_label}")
    print(f"[信息] blank_id：{blank_id}")
    print(f"[信息] transitions：{transitions}")
    print(f"[信息] split_points：{split_points}")
    print(f"[信息] participants：{participants}")

    for participant in participants:
        print(f"\n========== 处理 {participant} ==========")

        local_count = 0

        for left_label, right_label in transitions:
            try:
                left_index = find_sample_index(
                    rows=rows,
                    participant=participant,
                    label=left_label,
                )
                right_index = find_sample_index(
                    rows=rows,
                    participant=participant,
                    label=right_label,
                )
            except RuntimeError as e:
                print(f"[跳过] {participant} {left_label}->{right_label}，原因：{e}")
                continue

            left_row = rows[left_index]
            right_row = rows[right_index]

            for split_point in split_points:
                blank_feature = build_transition_blank_sample(
                    X=X,
                    left_index=left_index,
                    right_index=right_index,
                    split_point=split_point,
                )

                X_list.append(blank_feature)
                y_list.append(blank_id)

                generated_count += 1
                local_count += 1

                output_rows.append({
                    "resource_id": f"blank_targeted_{participant}_{generated_count:04d}",
                    "label": blank_label,
                    "source_word": f"{left_label}->{right_label}",
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
                    "reason": "synthetic_targeted_transition_blank",
                    "label_id": blank_id,
                    "synthetic": 1,
                    "left_label": left_label,
                    "left_resource_id": left_row["resource_id"],
                    "right_label": right_label,
                    "right_resource_id": right_row["resource_id"],
                    "split_point": split_point,
                })

                print(
                    f"[blank] {left_label}({left_row['resource_id']}) -> "
                    f"{right_label}({right_row['resource_id']}) split={split_point}"
                )

        print(f"[信息] {participant} 生成 blank 数：{local_count}")

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
        "generated_blank_count": int(generated_count),
        "final_sample_count": int(len(X_new)),
        "original_class_count": int(len(label_map)),
        "final_class_count": int(len(new_label_map)),
        "blank_label": blank_label,
        "blank_id": int(blank_id),
        "transitions": [
            {"left": left, "right": right}
            for left, right in transitions
        ],
        "split_points": split_points,
        "method": "targeted_tail_of_left_word + head_of_right_word",
    }

    save_json(output_dir / "augment_config.json", augment_config)

    print("\n========== targeted blank 增强完成 ==========")
    print(f"[完成] X_new shape：{X_new.shape}")
    print(f"[完成] y_new shape：{y_new.shape}")
    print(f"[完成] 新类别数：{len(new_label_map)}")
    print(f"[完成] blank 样本数：{generated_count}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", required=True, help="原始 features_20f 目录")
    parser.add_argument("--output_dir", required=True, help="输出 features_20f_blank_targeted 目录")
    parser.add_argument("--blank_label", default="blank", help="blank 类名称")
    parser.add_argument(
        "--transitions",
        default="",
        help="定向过渡，格式：你->今天,今天->学习。为空则使用默认过渡。",
    )
    parser.add_argument(
        "--split_points",
        default="8,12",
        help="split 点，例如：8,12",
    )

    args = parser.parse_args()

    augment_targeted_blank_dataset(
        feature_dir=Path(args.feature_dir),
        output_dir=Path(args.output_dir),
        blank_label=args.blank_label,
        transitions=parse_transitions(args.transitions),
        split_points=parse_split_points(args.split_points),
    )


if __name__ == "__main__":
    main()