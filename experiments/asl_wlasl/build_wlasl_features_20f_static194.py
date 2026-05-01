# -*- coding: utf-8 -*-
"""
从 WLASL-mini 视频中构建 20 帧 static194 特征。

目的：
在 base166 与 plus232 之间做折中实验。

static194 每帧特征：
- base166：
  - 左手 78 维
  - 右手 78 维
  - 基础身体 10 维
- static_plus 28 维：
  - 关键身体点相对肩中心位置
  - 左右手中心位置
  - 手部 bbox 尺寸
  - 左右手存在标记
  - 手腕到身体中心 / 手中心的距离

不包含：
- dynamic_delta 38 维
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    extract_hand_feature,
    get_pose_context,
    extract_pose_base_feature,
    extract_static_plus_feature,
)


def collect_video_results(holistic, video_path: Path) -> Tuple[List[object], List[bool], List[bool]]:
    """
    读取视频，并缓存每一帧 MediaPipe 结果。
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


def extract_frame_feature_static194(result) -> List[float]:
    """
    提取单帧 194 维静态增强特征。

    组成：
    - left hand 78
    - right hand 78
    - pose base 10
    - static plus 28
    """
    pose_ctx = get_pose_context(result)

    left_hand_feature = extract_hand_feature(result.left_hand_landmarks)
    right_hand_feature = extract_hand_feature(result.right_hand_landmarks)
    pose_base_feature = extract_pose_base_feature(result, pose_ctx)
    static_plus_feature = extract_static_plus_feature(result, pose_ctx)

    feature = (
        left_hand_feature
        + right_hand_feature
        + pose_base_feature
        + static_plus_feature
    )

    if len(feature) != 194:
        raise RuntimeError(f"static194 feature dim error: {len(feature)}")

    return feature


def extract_sequence_static194(
    results: List[object],
    frame_indices: List[int],
) -> np.ndarray:
    """
    根据帧下标提取一个样本的 20 帧 static194 特征。
    """
    features = []

    for frame_index in frame_indices:
        frame_index = max(0, min(len(results) - 1, int(frame_index)))
        result = results[frame_index]
        features.append(extract_frame_feature_static194(result))

    return np.array(features, dtype=np.float32)


def build_one_sample_features_static194(
    holistic,
    video_path: Path,
    target_frames: int,
    padding: int,
    min_hand_frames: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    构建单个视频样本的 static194 特征。
    """
    results, hand_flags, pose_flags = collect_video_results(
        holistic=holistic,
        video_path=video_path,
    )

    total_frames = len(results)

    start, end, hand_frame_count = detect_action_window(
        hand_flags=hand_flags,
        total_frames=total_frames,
        padding=padding,
        min_hand_frames=min_hand_frames,
    )

    sample_indices = make_sample_indices(
        start=start,
        end=end,
        target_frames=target_frames,
    )

    X = extract_sequence_static194(
        results=results,
        frame_indices=sample_indices,
    )

    meta = {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "action_start": start,
        "action_end": end,
        "action_length": end - start + 1,
        "hand_frame_count": hand_frame_count,
        "pose_ratio": round(sum(pose_flags) / total_frames, 4),
        "any_hand_ratio": round(sum(hand_flags) / total_frames, 4),
        "sampled_indices": " ".join(str(x) for x in sample_indices),
    }

    return X, meta


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
        default="D:/datasets/WLASL-mini/features_20f_static194",
        help="特征输出目录",
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
        help="动作窗口前后扩展帧数",
    )
    parser.add_argument(
        "--min_hand_frames",
        type=int,
        default=3,
        help="至少多少帧检测到手，才使用手部动作窗口",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多处理多少样本；0 表示全部",
    )

    args = parser.parse_args()

    samples_csv = Path(args.samples_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(samples_csv)

    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    labels = sorted(set(row["label"] for row in rows))
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    print("========== 开始构建 WLASL 20f static194 特征 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] 样本数：{len(rows)}")
    print(f"[信息] 类别数：{len(labels)}")
    print(f"[信息] labels：{labels}")
    print(f"[信息] target_frames：{args.target_frames}")
    print(f"[信息] padding：{args.padding}")
    print(f"[信息] min_hand_frames：{args.min_hand_frames}")

    X_list = []
    y_list = []
    index_rows = []

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
        for idx, row in enumerate(rows, start=1):
            label = row["label"]
            video_path = Path(row["local_path"])

            try:
                X, meta = build_one_sample_features_static194(
                    holistic=holistic,
                    video_path=video_path,
                    target_frames=args.target_frames,
                    padding=args.padding,
                    min_hand_frames=args.min_hand_frames,
                )

                X_list.append(X)
                y_list.append(label_to_id[label])

                index_row = {
                    "sample_index": len(X_list) - 1,
                    "sample_id": row["sample_id"],
                    "label": label,
                    "label_id": label_to_id[label],
                    "local_path": row["local_path"],
                    **meta,
                }

                index_rows.append(index_row)

                print(
                    f"[{idx}/{len(rows)}] {label} {row['sample_id']} "
                    f"frames={meta['total_frames']} "
                    f"window=[{meta['action_start']},{meta['action_end']}] "
                    f"hand_frames={meta['hand_frame_count']} "
                    f"pose={meta['pose_ratio']} "
                    f"hand={meta['any_hand_ratio']}"
                )

            except Exception as e:
                print(f"[跳过] {label} {row.get('sample_id')} {video_path} 原因：{e}")

    if not X_list:
        raise RuntimeError("没有成功构建任何样本特征")

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
        },
    )

    write_csv(
        output_dir / "sample_index.csv",
        index_rows,
        [
            "sample_index",
            "sample_id",
            "label",
            "label_id",
            "local_path",
            "video_path",
            "total_frames",
            "action_start",
            "action_end",
            "action_length",
            "hand_frame_count",
            "pose_ratio",
            "any_hand_ratio",
            "sampled_indices",
        ],
    )

    save_json(
        output_dir / "config.json",
        {
            "samples_csv": str(samples_csv),
            "target_frames": args.target_frames,
            "padding": args.padding,
            "min_hand_frames": args.min_hand_frames,
            "sample_count": int(X_all.shape[0]),
            "sequence_shape": list(X_all.shape),
            "feature_dim": int(X_all.shape[-1]),
            "class_count": len(labels),
            "labels": labels,
            "feature_note": "194 dims = base166 + static_plus28; no dynamic_delta",
        },
    )

    print("\n========== 构建完成 ==========")
    print(f"[完成] X shape：{X_all.shape}")
    print(f"[完成] y shape：{y_all.shape}")
    print(f"[完成] 输出目录：{output_dir}")


if __name__ == "__main__":
    main()
