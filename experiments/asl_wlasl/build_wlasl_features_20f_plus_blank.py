# -*- coding: utf-8 -*-
"""
构建 WLASL-mini 的 20帧 plus + blank 特征。

核心思路：
1. word 样本：
   - 使用 MediaPipe any_hand=True 定位动作有效窗口
   - 从动作窗口重采样 20 帧
   - 标签为原始 ASL gloss

2. blank 样本：
   - 从每个视频的起手前 / 收手后空档段中抽取
   - 每个原始视频最多生成 1 个 blank 样本
   - 标签为 blank

输出：
- X.npy
- y.npy
- labels.json
- sample_index.csv
- config.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_wlasl_features_20f_plus import (
    read_csv_rows,
    write_csv,
    save_json,
    read_all_frames,
    process_frame,
    make_sample_indices,
    detect_action_window,
    extract_frame_feature,
)


def extract_features_from_indices(
    results: List[object],
    frame_indices: List[int],
) -> np.ndarray:
    """
    根据指定帧下标提取一个 20 帧序列特征。

    注意：
    每个样本内部 dynamic_delta 从第一帧重新开始。
    """
    features = []
    previous_dynamic_source = None

    for frame_index in frame_indices:
        frame_index = max(0, min(len(results) - 1, int(frame_index)))
        result = results[frame_index]

        feature, previous_dynamic_source = extract_frame_feature(
            result,
            previous_dynamic_source,
        )

        features.append(feature)

    return np.array(features, dtype=np.float32)


def collect_video_results(
    holistic,
    video_path: Path,
) -> Tuple[List[object], List[bool], List[bool]]:
    """
    读取视频并缓存每一帧的 MediaPipe 结果。
    """
    frames = read_all_frames(video_path)

    if not frames:
        raise RuntimeError(f"视频无法读取或为空：{video_path}")

    results = []
    hand_flags = []
    pose_flags = []

    for frame in frames:
        result = process_frame(holistic, frame)
        results.append(result)

        has_pose = result.pose_landmarks is not None
        has_left = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None

        pose_flags.append(bool(has_pose))
        hand_flags.append(bool(has_left or has_right))

    return results, hand_flags, pose_flags


def find_blank_segment(
    hand_flags: List[bool],
    min_blank_span: int,
    blank_exclude_padding: int,
) -> Optional[Tuple[int, int, str]]:
    """
    从起手前 / 收手后寻找 blank 段。

    返回：
    - start
    - end
    - blank_source: pre 或 post

    如果前后空档都太短，则返回 None。
    """
    total_frames = len(hand_flags)

    hand_indices = [
        index
        for index, flag in enumerate(hand_flags)
        if flag
    ]

    if not hand_indices:
        return None

    first_hand = hand_indices[0]
    last_hand = hand_indices[-1]

    candidates = []

    pre_start = 0
    pre_end = first_hand - blank_exclude_padding - 1

    if pre_end >= pre_start and (pre_end - pre_start + 1) >= min_blank_span:
        candidates.append((pre_start, pre_end, "pre"))

    post_start = last_hand + blank_exclude_padding + 1
    post_end = total_frames - 1

    if post_end >= post_start and (post_end - post_start + 1) >= min_blank_span:
        candidates.append((post_start, post_end, "post"))

    if not candidates:
        return None

    # 选择更长的空档段，减少误裁风险。
    candidates.sort(
        key=lambda item: item[1] - item[0] + 1,
        reverse=True,
    )

    return candidates[0]


def build_word_and_blank_for_sample(
    holistic,
    row: Dict[str, str],
    target_frames: int,
    padding: int,
    min_hand_frames: int,
    min_blank_span: int,
    blank_exclude_padding: int,
) -> Tuple[np.ndarray, Dict[str, object], Optional[np.ndarray], Optional[Dict[str, object]]]:
    """
    为单个原始视频构建：
    - 1 个 word 样本
    - 0 或 1 个 blank 样本
    """
    video_path = Path(row["local_path"])
    source_label = row["label"]
    sample_id = row["sample_id"]

    results, hand_flags, pose_flags = collect_video_results(
        holistic=holistic,
        video_path=video_path,
    )

    total_frames = len(results)

    word_start, word_end, hand_frame_count = detect_action_window(
        hand_flags=hand_flags,
        total_frames=total_frames,
        padding=padding,
        min_hand_frames=min_hand_frames,
    )

    word_indices = make_sample_indices(
        start=word_start,
        end=word_end,
        target_frames=target_frames,
    )

    word_X = extract_features_from_indices(
        results=results,
        frame_indices=word_indices,
    )

    word_meta = {
        "source_type": "word",
        "sample_id": sample_id,
        "source_label": source_label,
        "label": source_label,
        "local_path": row["local_path"],
        "total_frames": total_frames,
        "action_start": word_start,
        "action_end": word_end,
        "action_length": word_end - word_start + 1,
        "hand_frame_count": hand_frame_count,
        "pose_ratio": round(sum(pose_flags) / total_frames, 4),
        "any_hand_ratio": round(sum(hand_flags) / total_frames, 4),
        "sampled_indices": " ".join(str(x) for x in word_indices),
        "blank_source": "",
    }

    blank_segment = find_blank_segment(
        hand_flags=hand_flags,
        min_blank_span=min_blank_span,
        blank_exclude_padding=blank_exclude_padding,
    )

    if blank_segment is None:
        return word_X, word_meta, None, None

    blank_start, blank_end, blank_source = blank_segment

    blank_indices = make_sample_indices(
        start=blank_start,
        end=blank_end,
        target_frames=target_frames,
    )

    blank_X = extract_features_from_indices(
        results=results,
        frame_indices=blank_indices,
    )

    blank_meta = {
        "source_type": "blank",
        "sample_id": sample_id,
        "source_label": source_label,
        "label": "blank",
        "local_path": row["local_path"],
        "total_frames": total_frames,
        "action_start": blank_start,
        "action_end": blank_end,
        "action_length": blank_end - blank_start + 1,
        "hand_frame_count": 0,
        "pose_ratio": round(sum(pose_flags) / total_frames, 4),
        "any_hand_ratio": round(sum(hand_flags) / total_frames, 4),
        "sampled_indices": " ".join(str(x) for x in blank_indices),
        "blank_source": blank_source,
    }

    return word_X, word_meta, blank_X, blank_meta


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        default="D:/datasets/WLASL-mini/samples.csv",
        help="WLASL-mini samples.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini/features_20f_plus_blank",
        help="输出特征目录",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=20,
        help="每个样本重采样帧数",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="word 动作窗口前后扩展帧数",
    )
    parser.add_argument(
        "--min_hand_frames",
        type=int,
        default=3,
        help="至少多少帧检测到手才使用手部动作窗口",
    )
    parser.add_argument(
        "--min_blank_span",
        type=int,
        default=8,
        help="blank 段最少需要多少帧",
    )
    parser.add_argument(
        "--blank_exclude_padding",
        type=int,
        default=2,
        help="blank 段距离有手帧至少间隔多少帧",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多处理多少原始样本；0 表示全部",
    )

    args = parser.parse_args()

    samples_csv = Path(args.samples_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(samples_csv)

    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    word_labels = sorted(set(row["label"] for row in rows))
    labels = word_labels + ["blank"]

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    print("========== 构建 WLASL 20f plus blank 特征 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] 原始样本数：{len(rows)}")
    print(f"[信息] word 类别数：{len(word_labels)}")
    print(f"[信息] 总类别数：{len(labels)}")
    print(f"[信息] labels：{labels}")
    print(f"[信息] target_frames：{args.target_frames}")
    print(f"[信息] padding：{args.padding}")
    print(f"[信息] min_hand_frames：{args.min_hand_frames}")
    print(f"[信息] min_blank_span：{args.min_blank_span}")
    print(f"[信息] blank_exclude_padding：{args.blank_exclude_padding}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    index_rows: List[Dict[str, object]] = []

    blank_count = 0
    word_count = 0
    skipped_count = 0

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for index, row in enumerate(rows, start=1):
            try:
                word_X, word_meta, blank_X, blank_meta = build_word_and_blank_for_sample(
                    holistic=holistic,
                    row=row,
                    target_frames=args.target_frames,
                    padding=args.padding,
                    min_hand_frames=args.min_hand_frames,
                    min_blank_span=args.min_blank_span,
                    blank_exclude_padding=args.blank_exclude_padding,
                )

                X_list.append(word_X)
                y_list.append(label_to_id[word_meta["label"]])

                word_meta["sample_index"] = len(X_list) - 1
                word_meta["label_id"] = label_to_id[word_meta["label"]]
                index_rows.append(word_meta)
                word_count += 1

                blank_text = "no_blank"

                if blank_X is not None and blank_meta is not None:
                    X_list.append(blank_X)
                    y_list.append(label_to_id["blank"])

                    blank_meta["sample_index"] = len(X_list) - 1
                    blank_meta["label_id"] = label_to_id["blank"]
                    index_rows.append(blank_meta)
                    blank_count += 1

                    blank_text = f"blank={blank_meta['blank_source']}[{blank_meta['action_start']},{blank_meta['action_end']}]"

                print(
                    f"[{index}/{len(rows)}] {row['label']} {row['sample_id']} "
                    f"word=[{word_meta['action_start']},{word_meta['action_end']}] "
                    f"hand_frames={word_meta['hand_frame_count']} "
                    f"{blank_text}"
                )

            except Exception as e:
                skipped_count += 1
                print(
                    f"[跳过] {row.get('label')} {row.get('sample_id')} "
                    f"{row.get('local_path')} 原因：{e}"
                )

    if not X_list:
        raise RuntimeError("没有成功构建任何样本")

    X_all = np.stack(X_list, axis=0).astype(np.float32)
    y_all = np.array(y_list, dtype=np.int64)

    np.save(output_dir / "X.npy", X_all)
    np.save(output_dir / "y.npy", y_all)

    save_json(
        output_dir / "labels.json",
        {
            "labels": labels,
            "label_to_id": label_to_id,
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "blank_label": "blank",
        },
    )

    write_csv(
        output_dir / "sample_index.csv",
        index_rows,
        [
            "sample_index",
            "source_type",
            "sample_id",
            "source_label",
            "label",
            "label_id",
            "local_path",
            "total_frames",
            "action_start",
            "action_end",
            "action_length",
            "hand_frame_count",
            "pose_ratio",
            "any_hand_ratio",
            "sampled_indices",
            "blank_source",
        ],
    )

    label_counts = {}

    for label_id in y_all.tolist():
        label = labels[int(label_id)]
        label_counts[label] = label_counts.get(label, 0) + 1

    save_json(
        output_dir / "config.json",
        {
            "samples_csv": str(samples_csv),
            "target_frames": args.target_frames,
            "padding": args.padding,
            "min_hand_frames": args.min_hand_frames,
            "min_blank_span": args.min_blank_span,
            "blank_exclude_padding": args.blank_exclude_padding,
            "source_word_sample_count": len(rows),
            "word_feature_count": word_count,
            "blank_feature_count": blank_count,
            "skipped_count": skipped_count,
            "sample_count": int(X_all.shape[0]),
            "sequence_shape": list(X_all.shape),
            "feature_dim": int(X_all.shape[-1]),
            "class_count": len(labels),
            "labels": labels,
            "label_counts": label_counts,
            "feature_note": "232 dims = base166 + static_plus28 + dynamic_delta38; with blank class",
        },
    )

    print("\n========== 构建完成 ==========")
    print(f"[完成] X shape：{X_all.shape}")
    print(f"[完成] y shape：{y_all.shape}")
    print(f"[完成] word 样本数：{word_count}")
    print(f"[完成] blank 样本数：{blank_count}")
    print(f"[完成] skipped：{skipped_count}")
    print(f"[完成] 输出目录：{output_dir}")

    if blank_count < 10:
        print("[警告] blank 样本数偏少，后续可能需要降低 min_blank_span 或从拼接 gap 中补充 blank")


if __name__ == "__main__":
    main()
