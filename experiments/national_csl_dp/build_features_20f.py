# -*- coding: utf-8 -*-
"""
将 HearBridge-NationalCSL-mini 图片帧数据转换为 20 帧 MediaPipe 特征数据。

输入：
D:/datasets/HearBridge-NationalCSL-mini/
  raw_frames/
  samples.csv

输出：
D:/datasets/HearBridge-NationalCSL-mini/features_20f/
  X.npy
  y.npy
  label_map.json
  sample_index.csv
  feature_config.json

特征设计：
1. 每只手 78 维：
   - 21 个手部关键点，每点 3 维，相对手腕归一化坐标：21 * 3 = 63
   - 15 个手部关节角度特征：15
2. 双手共 156 维
3. Pose 上半身补充 10 维：
   - 左右腕、左右肘相对肩中心的 x/y：4 * 2 = 8
   - 左右肘角度：2
4. 总维度：156 + 10 = 166

注意：
NationalCSL-DP 是规范录制的 front 视角数据，这里不做镜像翻转，也不交换左右 Pose 点。
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


MP_POSE = mp.solutions.pose
MP_HANDS = mp.solutions.hands


# 每只手使用的角度点组合。
# 每个三元组表示：angle(a, b, c)，即以 b 为顶点计算夹角。
HAND_ANGLE_TRIPLES = [
    # 拇指
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),

    # 食指
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),

    # 中指
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),

    # 无名指
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),

    # 小指
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
]


def read_samples(samples_csv: Path) -> List[Dict[str, str]]:
    """
    读取样本索引文件。
    """
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_image_frames(frame_dir: Path) -> List[Path]:
    """
    获取一个样本目录下的所有图片帧。
    """
    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def read_image_bgr_unicode(image_path: Path) -> Optional[np.ndarray]:
    """
    读取可能包含中文路径的图片。

    Windows 下 cv2.imread(str(path)) 遇到中文路径可能返回 None，
    所以这里使用 np.fromfile + cv2.imdecode。
    """
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)

        if image_bytes.size == 0:
            return None

        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[错误] 图片读取异常：{image_path}，原因：{e}")
        return None


def calc_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    计算三维空间中 a-b-c 的夹角，并归一化到 0~1。

    返回值：
    0 表示 0 度，1 表示 180 度。
    """
    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)

    if ba_norm < 1e-6 or bc_norm < 1e-6:
        return 0.0

    cosine = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cosine = max(-1.0, min(1.0, cosine))

    angle = math.acos(cosine)
    return angle / math.pi


def extract_single_hand_feature(hand_landmarks) -> np.ndarray:
    """
    提取单只手 78 维特征。

    结构：
    1. 21 个手部关键点，以 wrist 为原点做相对坐标归一化：63 维
    2. 15 个关节角度：15 维
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    # 以手腕为原点
    wrist = points[0].copy()
    relative_points = points - wrist

    # 用手掌尺度归一化，减少人与人之间距离/大小差异
    # 这里使用 wrist 到 middle_mcp 的距离，太小时兜底为 1。
    palm_scale = np.linalg.norm(points[9] - points[0])
    if palm_scale < 1e-6:
        palm_scale = 1.0

    normalized_points = relative_points / palm_scale
    landmark_feature = normalized_points.reshape(-1)

    angle_features = []
    for a, b, c in HAND_ANGLE_TRIPLES:
        angle_features.append(calc_angle_3d(points[a], points[b], points[c]))

    return np.concatenate(
        [landmark_feature, np.array(angle_features, dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def empty_hand_feature() -> np.ndarray:
    """
    返回一只手缺失时的 78 维零特征。
    """
    return np.zeros((78,), dtype=np.float32)


def get_pose_point(pose_landmarks, index: int) -> np.ndarray:
    """
    获取 Pose 中指定点的 x/y/z 坐标。
    """
    lm = pose_landmarks.landmark[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_pose_feature(pose_landmarks) -> np.ndarray:
    """
    提取 Pose 上半身补充 10 维特征。

    使用点：
    - 左肩、右肩
    - 左肘、右肘
    - 左腕、右腕

    特征：
    - 左腕、右腕、左肘、右肘相对肩中心的 x/y：8 维
    - 左肘角度、右肘角度：2 维
    """
    if pose_landmarks is None:
        return np.zeros((10,), dtype=np.float32)

    left_shoulder = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.RIGHT_SHOULDER.value)
    left_elbow = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.LEFT_ELBOW.value)
    right_elbow = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.RIGHT_ELBOW.value)
    left_wrist = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.LEFT_WRIST.value)
    right_wrist = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.RIGHT_WRIST.value)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    key_points = [
        left_wrist,
        right_wrist,
        left_elbow,
        right_elbow,
    ]

    relative_xy = []
    for point in key_points:
        relative = (point[:2] - shoulder_center[:2]) / shoulder_width
        relative_xy.extend([float(relative[0]), float(relative[1])])

    left_elbow_angle = calc_angle_3d(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calc_angle_3d(right_shoulder, right_elbow, right_wrist)

    return np.array(
        relative_xy + [left_elbow_angle, right_elbow_angle],
        dtype=np.float32,
    )


def extract_frame_feature(
    image_bgr: np.ndarray,
    pose_model,
    hands_model,
    swap_handedness: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    对单帧图片提取 166 维特征。

    返回：
    - feature: shape=(166,)
    - stats: 检测状态信息
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    pose_result = pose_model.process(image_rgb)
    hands_result = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    left_hand_feature = empty_hand_feature()
    right_hand_feature = empty_hand_feature()

    left_present = False
    right_present = False

    if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
        for hand_landmarks, handedness in zip(
            hands_result.multi_hand_landmarks,
            hands_result.multi_handedness,
        ):
            label = handedness.classification[0].label

            if swap_handedness:
                if label == "Left":
                    label = "Right"
                elif label == "Right":
                    label = "Left"

            hand_feature = extract_single_hand_feature(hand_landmarks)

            # NationalCSL-DP 不做镜像翻转，直接按 MediaPipe handedness 分槽。
            if label == "Left":
                left_hand_feature = hand_feature
                left_present = True
            elif label == "Right":
                right_hand_feature = hand_feature
                right_present = True

    pose_feature = extract_pose_feature(pose_result.pose_landmarks)

    feature = np.concatenate(
        [left_hand_feature, right_hand_feature, pose_feature],
        axis=0,
    ).astype(np.float32)

    stats = {
        "pose_present": pose_result.pose_landmarks is not None,
        "left_hand_present": left_present,
        "right_hand_present": right_present,
        "any_hand_present": left_present or right_present,
        "both_hands_present": left_present and right_present,
        "feature_dim": int(feature.shape[0]),
    }

    return feature, stats


def temporal_resample_features(features: np.ndarray, target_frames: int) -> np.ndarray:
    """
    将任意长度的特征序列重采样为固定帧数。

    输入：
    features: shape=(T, D)

    输出：
    shape=(target_frames, D)

    说明：
    - T < target_frames：通过均匀索引重复部分帧
    - T > target_frames：通过均匀索引压缩部分帧
    """
    if features.ndim != 2:
        raise ValueError(f"features 必须是二维数组，当前 shape={features.shape}")

    source_frames = features.shape[0]

    if source_frames <= 0:
        raise ValueError("空特征序列无法重采样")

    if source_frames == target_frames:
        return features.astype(np.float32)

    indices = np.linspace(0, source_frames - 1, target_frames)
    indices = np.round(indices).astype(np.int64)
    indices = np.clip(indices, 0, source_frames - 1)

    return features[indices].astype(np.float32)


def build_label_map(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """
    根据样本中的 label 构建 label -> id 映射。

    注意：
    学习1-1 / 学习1-2 在 samples.csv 中已经统一为 label=学习。
    """
    labels = sorted({row["label"] for row in rows if row.get("status") == "ok"})
    return {label: index for index, label in enumerate(labels)}


def process_one_sample(
    row: Dict[str, str],
    pose_model,
    hands_model,
    target_frames: int,
    min_required_frames: int,
    swap_handedness: bool,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """
    处理单个样本目录：

    1. 读取图片帧
    2. 每帧提取 MediaPipe 特征
    3. 重采样到 target_frames
    """
    frame_dir = Path(row["frame_dir"])
    image_paths = list_image_frames(frame_dir)

    result_info = {
        "resource_id": row["resource_id"],
        "label": row["label"],
        "source_word": row["source_word"],
        "participant": row["participant"],
        "view": row["view"],
        "frame_dir": str(frame_dir),
        "raw_frame_count": len(image_paths),
        "used_frame_count": 0,
        "target_frames": target_frames,
        "pose_present_count": 0,
        "any_hand_present_count": 0,
        "both_hands_present_count": 0,
        "left_hand_present_count": 0,
        "right_hand_present_count": 0,
        "status": "ok",
        "reason": "",
    }

    if len(image_paths) < min_required_frames:
        result_info["status"] = "skipped"
        result_info["reason"] = f"frame_count < {min_required_frames}"
        return None, result_info

    frame_features = []

    for image_path in image_paths:
        image_bgr = read_image_bgr_unicode(image_path)

        if image_bgr is None:
            continue

        feature, stats = extract_frame_feature(
            image_bgr=image_bgr,
            pose_model=pose_model,
            hands_model=hands_model,
            swap_handedness=swap_handedness,
        )

        frame_features.append(feature)

        result_info["pose_present_count"] += int(stats["pose_present"])
        result_info["any_hand_present_count"] += int(stats["any_hand_present"])
        result_info["both_hands_present_count"] += int(stats["both_hands_present"])
        result_info["left_hand_present_count"] += int(stats["left_hand_present"])
        result_info["right_hand_present_count"] += int(stats["right_hand_present"])

    if len(frame_features) < min_required_frames:
        result_info["status"] = "skipped"
        result_info["reason"] = f"valid_feature_frames < {min_required_frames}"
        result_info["used_frame_count"] = len(frame_features)
        return None, result_info

    feature_array = np.stack(frame_features, axis=0).astype(np.float32)
    fixed_feature_array = temporal_resample_features(
        features=feature_array,
        target_frames=target_frames,
    )

    result_info["used_frame_count"] = len(frame_features)

    return fixed_feature_array, result_info


def write_sample_index(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出特征样本索引 CSV。
    """
    if not rows:
        print("[警告] 没有样本索引可写出")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出样本索引：{output_path}")


def save_json(output_path: Path, data: Dict) -> None:
    """
    保存 JSON 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{output_path}")


def build_features(
    samples_csv: Path,
    output_dir: Path,
    target_frames: int,
    min_required_frames: int,
    max_samples: Optional[int],
    min_detection_confidence: float,
    swap_handedness: bool,
) -> None:
    """
    构建 20 帧特征数据集。
    """
    rows = read_samples(samples_csv)
    rows = [row for row in rows if row.get("status") == "ok"]

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    label_map = build_label_map(rows)

    X_list = []
    y_list = []
    index_rows = []

    print("========== 开始构建 NationalCSL-DP 20 帧特征 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] 样本数：{len(rows)}")
    print(f"[信息] 类别数：{len(label_map)}")
    print(f"[信息] label_map：{label_map}")
    print(f"[信息] target_frames：{target_frames}")
    print(f"[信息] min_required_frames：{min_required_frames}")

    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
    ) as pose_model, MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as hands_model:

        for index, row in enumerate(rows, start=1):
            fixed_features, result_info = process_one_sample(
                row=row,
                pose_model=pose_model,
                hands_model=hands_model,
                target_frames=target_frames,
                min_required_frames=min_required_frames,
                swap_handedness=swap_handedness,
            )

            label = row["label"]

            if fixed_features is not None:
                X_list.append(fixed_features)
                y_list.append(label_map[label])
                result_info["label_id"] = label_map[label]
            else:
                result_info["label_id"] = ""

            index_rows.append(result_info)

            print(
                f"[{index}/{len(rows)}] "
                f"{row['label']} {row['participant']} "
                f"raw={result_info['raw_frame_count']} "
                f"used={result_info['used_frame_count']} "
                f"status={result_info['status']} "
                f"{result_info['reason']}"
            )

    if not X_list:
        raise RuntimeError("没有成功生成任何特征样本，请检查输入数据。")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)

    save_json(output_dir / "label_map.json", label_map)

    feature_config = {
        "dataset": "HearBridge-NationalCSL-mini",
        "source": "NationalCSL-DP",
        "target_frames": target_frames,
        "feature_dim": int(X.shape[-1]),
        "feature_shape": list(X.shape),
        "class_count": len(label_map),
        "sample_count": int(X.shape[0]),
        "min_required_frames": min_required_frames,
        "mediapipe": {
            "pose": True,
            "hands": True,
            "static_image_mode": True,
            "min_detection_confidence": min_detection_confidence,
        },
        "preprocess": {
            "mirror_input": False,
            "swap_pose_lr": False,
            "mirror_pose_x": False,
            "view": "front",
            "temporal_resample": "uniform_round_indices",
        },
        "feature_layout": {
            "left_hand": "0:78",
            "right_hand": "78:156",
            "pose_upper_body": "156:166",
            "total_dim": 166,
        },
    }

    save_json(output_dir / "feature_config.json", feature_config)
    write_sample_index(output_dir / "sample_index.csv", index_rows)

    print("\n========== 构建完成 ==========")
    print(f"[完成] X shape：{X.shape}")
    print(f"[完成] y shape：{y.shape}")
    print(f"[完成] 输出目录：{output_dir}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples_csv",
        required=True,
        help="小数据集 samples.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="特征输出目录，例如 D:/datasets/HearBridge-NationalCSL-mini/features_20f",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=20,
        help="统一重采样后的帧数",
    )
    parser.add_argument(
        "--min_required_frames",
        type=int,
        default=8,
        help="少于该帧数的样本丢弃",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多处理多少个样本；0 表示全量处理",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe 检测阈值",
    )
    parser.add_argument(
        "--swap_handedness",
        action="store_true",
        help="是否交换 MediaPipe Hands 的 Left/Right 标签。NationalCSL-DP 这类非自拍镜像数据建议开启。",
    )

    args = parser.parse_args()

    build_features(
        samples_csv=Path(args.samples_csv),
        output_dir=Path(args.output_dir),
        target_frames=args.target_frames,
        min_required_frames=args.min_required_frames,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        min_detection_confidence=args.min_detection_confidence,
        swap_handedness=args.swap_handedness,
    )


if __name__ == "__main__":
    main()