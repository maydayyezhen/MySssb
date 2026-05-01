# -*- coding: utf-8 -*-
"""
从 WLASL-mini 视频中构建 20 帧 plus 特征。

核心策略：
1. 先扫描视频帧，检测 MediaPipe Pose / Hands。
2. 用 any_hand=True 的帧确定有效动作窗口。
3. 对有效动作窗口前后扩展若干帧。
4. 从有效窗口中重采样 target_frames 帧。
5. 每帧提取 232 维特征：
   - base 166维
   - static plus 28维
   - dynamic delta 38维

输入：
- D:/datasets/WLASL-mini/samples.csv

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
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """读取 CSV。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def read_all_frames(video_path: Path) -> List[np.ndarray]:
    """读取视频全部帧。"""
    cap = cv2.VideoCapture(str(video_path))

    frames: List[np.ndarray] = []

    if not cap.isOpened():
        return frames

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        frames.append(frame)

    cap.release()
    return frames


def process_frame(holistic, frame_bgr: np.ndarray):
    """执行 MediaPipe Holistic。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    result = holistic.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return result


def get_landmark_xyz(landmarks, index: int) -> Tuple[float, float, float]:
    """读取 landmark 的 xyz。"""
    lm = landmarks.landmark[index]
    return float(lm.x), float(lm.y), float(lm.z)


def get_landmark_xy(landmarks, index: int) -> Tuple[float, float]:
    """读取 landmark 的 xy。"""
    lm = landmarks.landmark[index]
    return float(lm.x), float(lm.y)


def safe_norm(value: float, eps: float = 1e-6) -> float:
    """避免除零。"""
    if abs(value) < eps:
        return eps
    return value


def distance_2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """二维距离。"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def angle_2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    计算夹角，返回归一化到 [0, 1] 的角度。
    b 是顶点。
    """
    v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cos_value = float(np.dot(v1, v2) / (n1 * n2))
    cos_value = max(-1.0, min(1.0, cos_value))

    angle = math.acos(cos_value)

    return angle / math.pi


def get_pose_context(result) -> Dict[str, object]:
    """
    获取 Pose 上下文。
    """
    pose = result.pose_landmarks

    if pose is None:
        return {
            "has_pose": False,
            "center": (0.5, 0.5),
            "shoulder_width": 1.0,
            "points": {},
        }

    mp_pose = mp.solutions.pose.PoseLandmark

    left_shoulder = get_landmark_xy(pose, mp_pose.LEFT_SHOULDER.value)
    right_shoulder = get_landmark_xy(pose, mp_pose.RIGHT_SHOULDER.value)

    center = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )

    shoulder_width = distance_2d(left_shoulder, right_shoulder)
    shoulder_width = safe_norm(shoulder_width)

    point_indices = {
        "nose": mp_pose.NOSE.value,
        "left_shoulder": mp_pose.LEFT_SHOULDER.value,
        "right_shoulder": mp_pose.RIGHT_SHOULDER.value,
        "left_elbow": mp_pose.LEFT_ELBOW.value,
        "right_elbow": mp_pose.RIGHT_ELBOW.value,
        "left_wrist": mp_pose.LEFT_WRIST.value,
        "right_wrist": mp_pose.RIGHT_WRIST.value,
        "left_hip": mp_pose.LEFT_HIP.value,
        "right_hip": mp_pose.RIGHT_HIP.value,
    }

    points = {}

    for name, idx in point_indices.items():
        points[name] = get_landmark_xy(pose, idx)

    return {
        "has_pose": True,
        "center": center,
        "shoulder_width": shoulder_width,
        "points": points,
    }


def normalize_point_xy(
    point: Tuple[float, float],
    center: Tuple[float, float],
    scale: float,
) -> Tuple[float, float]:
    """相对中心和尺度归一化。"""
    return (
        (point[0] - center[0]) / scale,
        (point[1] - center[1]) / scale,
    )


def extract_hand_feature(hand_landmarks) -> List[float]:
    """
    提取单只手 78 维特征：
    - 21 点 xyz 相对手腕归一化：63 维
    - 15 个手部关节角：15 维
    """
    if hand_landmarks is None:
        return [0.0] * 78

    wrist = get_landmark_xyz(hand_landmarks, 0)

    scale_candidates = []

    for idx in [5, 9, 13, 17]:
        point = get_landmark_xyz(hand_landmarks, idx)
        dist = math.sqrt(
            (point[0] - wrist[0]) ** 2
            + (point[1] - wrist[1]) ** 2
            + (point[2] - wrist[2]) ** 2
        )
        scale_candidates.append(dist)

    scale = max(scale_candidates) if scale_candidates else 1.0
    scale = safe_norm(scale)

    features: List[float] = []

    for idx in range(21):
        x, y, z = get_landmark_xyz(hand_landmarks, idx)
        features.extend([
            (x - wrist[0]) / scale,
            (y - wrist[1]) / scale,
            (z - wrist[2]) / scale,
        ])

    angle_triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20),
    ]

    for a, b, c in angle_triplets:
        pa = get_landmark_xy(hand_landmarks, a)
        pb = get_landmark_xy(hand_landmarks, b)
        pc = get_landmark_xy(hand_landmarks, c)
        features.append(angle_2d(pa, pb, pc))

    if len(features) != 78:
        raise RuntimeError(f"hand feature dim error: {len(features)}")

    return features


def hand_center(hand_landmarks) -> Tuple[float, float]:
    """计算手部中心。"""
    if hand_landmarks is None:
        return (0.0, 0.0)

    xs = []
    ys = []

    for idx in range(21):
        x, y = get_landmark_xy(hand_landmarks, idx)
        xs.append(x)
        ys.append(y)

    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hand_bbox_size(hand_landmarks) -> Tuple[float, float]:
    """计算手部 bbox 宽高。"""
    if hand_landmarks is None:
        return (0.0, 0.0)

    xs = []
    ys = []

    for idx in range(21):
        x, y = get_landmark_xy(hand_landmarks, idx)
        xs.append(x)
        ys.append(y)

    return (max(xs) - min(xs), max(ys) - min(ys))


def extract_pose_base_feature(result, pose_ctx: Dict[str, object]) -> List[float]:
    """
    提取 10 维身体基础特征：
    - 左右肘、左右腕相对肩中心坐标：8 维
    - 左右肘角度：2 维
    """
    center = pose_ctx["center"]
    scale = pose_ctx["shoulder_width"]
    points = pose_ctx["points"]

    if not pose_ctx["has_pose"]:
        return [0.0] * 10

    names = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]

    features: List[float] = []

    for name in names:
        nx, ny = normalize_point_xy(points[name], center, scale)
        features.extend([nx, ny])

    left_angle = angle_2d(
        points["left_shoulder"],
        points["left_elbow"],
        points["left_wrist"],
    )
    right_angle = angle_2d(
        points["right_shoulder"],
        points["right_elbow"],
        points["right_wrist"],
    )

    features.extend([left_angle, right_angle])

    if len(features) != 10:
        raise RuntimeError(f"pose base dim error: {len(features)}")

    return features


def extract_static_plus_feature(result, pose_ctx: Dict[str, object]) -> List[float]:
    """
    提取 28 维静态增强特征。
    """
    center = pose_ctx["center"]
    scale = pose_ctx["shoulder_width"]
    points = pose_ctx["points"]

    features: List[float] = []

    pose_names = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]

    if pose_ctx["has_pose"]:
        for name in pose_names:
            nx, ny = normalize_point_xy(points[name], center, scale)
            features.extend([nx, ny])
    else:
        features.extend([0.0] * 14)

    left_hand = result.left_hand_landmarks
    right_hand = result.right_hand_landmarks

    for hand in [left_hand, right_hand]:
        if hand is not None:
            cx, cy = hand_center(hand)
            nx, ny = normalize_point_xy((cx, cy), center, scale)
            features.extend([nx, ny])
        else:
            features.extend([0.0, 0.0])

    for hand in [left_hand, right_hand]:
        bw, bh = hand_bbox_size(hand)
        features.extend([bw / scale, bh / scale])

    features.extend([
        1.0 if left_hand is not None else 0.0,
        1.0 if right_hand is not None else 0.0,
    ])

    if pose_ctx["has_pose"]:
        for hand_name, wrist_name in [
            ("left", "left_wrist"),
            ("right", "right_wrist"),
        ]:
            wrist = points[wrist_name]
            wrist_to_center = distance_2d(wrist, center) / scale

            if hand_name == "left" and left_hand is not None:
                hc = hand_center(left_hand)
                wrist_to_hand = distance_2d(wrist, hc) / scale
            elif hand_name == "right" and right_hand is not None:
                hc = hand_center(right_hand)
                wrist_to_hand = distance_2d(wrist, hc) / scale
            else:
                wrist_to_hand = 0.0

            features.extend([wrist_to_center, wrist_to_hand])
    else:
        features.extend([0.0] * 4)

    if len(features) != 28:
        raise RuntimeError(f"static plus dim error: {len(features)}")

    return features


def extract_dynamic_source(result, pose_ctx: Dict[str, object]) -> List[float]:
    """
    构造 38 维动态源向量：
    19 个点，每个点 x/y，共 38 维。
    动态特征最终使用相邻帧差分。
    """
    center = pose_ctx["center"]
    scale = pose_ctx["shoulder_width"]
    points = pose_ctx["points"]

    source: List[float] = []

    pose_names = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]

    if pose_ctx["has_pose"]:
        for name in pose_names:
            nx, ny = normalize_point_xy(points[name], center, scale)
            source.extend([nx, ny])
    else:
        source.extend([0.0] * 18)

    hand_indices = [0, 4, 8, 12, 20]

    for hand in [result.left_hand_landmarks, result.right_hand_landmarks]:
        if hand is not None:
            for idx in hand_indices:
                point = get_landmark_xy(hand, idx)
                nx, ny = normalize_point_xy(point, center, scale)
                source.extend([nx, ny])
        else:
            source.extend([0.0] * 10)

    if len(source) != 38:
        raise RuntimeError(f"dynamic source dim error: {len(source)}")

    return source


def extract_frame_feature(result, previous_dynamic_source: Optional[List[float]]) -> Tuple[List[float], List[float]]:
    """
    提取单帧 232 维 plus 特征。
    """
    pose_ctx = get_pose_context(result)

    left_hand_feature = extract_hand_feature(result.left_hand_landmarks)
    right_hand_feature = extract_hand_feature(result.right_hand_landmarks)
    pose_base_feature = extract_pose_base_feature(result, pose_ctx)

    base_feature = left_hand_feature + right_hand_feature + pose_base_feature

    if len(base_feature) != 166:
        raise RuntimeError(f"base feature dim error: {len(base_feature)}")

    static_plus = extract_static_plus_feature(result, pose_ctx)

    dynamic_source = extract_dynamic_source(result, pose_ctx)

    if previous_dynamic_source is None:
        dynamic_delta = [0.0] * 38
    else:
        dynamic_delta = [
            dynamic_source[i] - previous_dynamic_source[i]
            for i in range(38)
        ]

    feature = base_feature + static_plus + dynamic_delta

    if len(feature) != 232:
        raise RuntimeError(f"frame feature dim error: {len(feature)}")

    return feature, dynamic_source


def make_sample_indices(start: int, end: int, target_frames: int) -> List[int]:
    """
    从 [start, end] 闭区间重采样 target_frames 个下标。
    """
    if end < start:
        return [start] * target_frames

    if start == end:
        return [start] * target_frames

    values = np.linspace(start, end, target_frames)
    return [int(round(v)) for v in values]


def detect_action_window(
    hand_flags: List[bool],
    total_frames: int,
    padding: int,
    min_hand_frames: int,
) -> Tuple[int, int, int]:
    """
    根据 any_hand=True 的帧确定动作窗口。
    返回：start, end, hand_frame_count
    """
    indices = [idx for idx, flag in enumerate(hand_flags) if flag]
    hand_frame_count = len(indices)

    if hand_frame_count >= min_hand_frames:
        start = max(0, indices[0] - padding)
        end = min(total_frames - 1, indices[-1] + padding)
        return start, end, hand_frame_count

    # 退化策略：取视频中间区域
    if total_frames <= 0:
        return 0, 0, hand_frame_count

    mid = total_frames // 2
    half = max(1, total_frames // 3)

    start = max(0, mid - half)
    end = min(total_frames - 1, mid + half)

    return start, end, hand_frame_count


def build_one_sample_features(
    holistic,
    video_path: Path,
    target_frames: int,
    padding: int,
    min_hand_frames: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    为单个视频构建特征。
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

        has_left = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None
        has_pose = result.pose_landmarks is not None

        hand_flags.append(bool(has_left or has_right))
        pose_flags.append(bool(has_pose))

    start, end, hand_frame_count = detect_action_window(
        hand_flags=hand_flags,
        total_frames=len(frames),
        padding=padding,
        min_hand_frames=min_hand_frames,
    )

    sample_indices = make_sample_indices(start, end, target_frames)

    features = []
    previous_dynamic_source = None

    for frame_index in sample_indices:
        frame_index = max(0, min(len(results) - 1, frame_index))
        result = results[frame_index]

        feature, previous_dynamic_source = extract_frame_feature(
            result,
            previous_dynamic_source,
        )

        features.append(feature)

    X = np.array(features, dtype=np.float32)

    meta = {
        "video_path": str(video_path),
        "total_frames": len(frames),
        "action_start": start,
        "action_end": end,
        "action_length": end - start + 1,
        "hand_frame_count": hand_frame_count,
        "pose_ratio": round(sum(pose_flags) / len(pose_flags), 4),
        "any_hand_ratio": round(sum(hand_flags) / len(hand_flags), 4),
        "sampled_indices": " ".join(str(x) for x in sample_indices),
    }

    return X, meta


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        default="D:/datasets/WLASL-mini/samples.csv",
        help="WLASL-mini samples.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini/features_20f_plus",
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
        help="至少多少帧检测到手，才使用手部窗口",
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

    print("========== 开始构建 WLASL 20f plus 特征 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] 样本数：{len(rows)}")
    print(f"[信息] 类别数：{len(labels)}")
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
                X, meta = build_one_sample_features(
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
            "feature_note": "232 dims = base166 + static_plus28 + dynamic_delta38",
        },
    )

    print("\n========== 构建完成 ==========")
    print(f"[完成] X shape：{X_all.shape}")
    print(f"[完成] y shape：{y_all.shape}")
    print(f"[完成] 输出目录：{output_dir}")


if __name__ == "__main__":
    main()
