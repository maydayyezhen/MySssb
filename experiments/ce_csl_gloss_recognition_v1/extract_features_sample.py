"""
CE-CSL Feature V1 少量样本特征提取测试脚本

作用：
1. 读取 processed/train.jsonl、dev.jsonl、test.jsonl 中少量样本。
2. 按 TARGET_FPS = 10 从视频中抽帧。
3. 不翻转视频帧。
4. 使用 MediaPipe Holistic 提取 pose / left hand / right hand。
5. 按 FEATURE_SPEC.md 中定义的 CE-CSL Feature V1 生成 166 维单帧特征。
6. 保存为 .npy 文件，形状为 T × 166。

本脚本只用于少量样本测试，不做全量提取，不训练模型。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# =========================
# 1. 路径与基础配置
# =========================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"

OUTPUT_DIR = PROCESSED_DIR / "features_sample"

TARGET_FPS = 10.0
TARGET_WIDTH = 960
MIRROR_INPUT = False

# 先只抽少量样本测试
SPLIT_LIMITS = {
    "train": 3,
    "dev": 1,
    "test": 1,
}


# =========================
# 2. Pose 点位编号
# =========================

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16


# =========================
# 3. 基础工具函数
# =========================

def read_jsonl(path: Path, limit: int) -> List[Dict]:
    """
    读取 jsonl 文件中的前 limit 条样本。
    """
    samples: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(samples) >= limit:
                break

            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def resize_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    等比例缩放图像到 TARGET_WIDTH。
    """
    height, width = frame_bgr.shape[:2]

    if width <= TARGET_WIDTH:
        return frame_bgr

    scale = TARGET_WIDTH / width
    new_height = int(height * scale)

    return cv2.resize(frame_bgr, (TARGET_WIDTH, new_height))


def distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算二维欧氏距离。
    """
    return float(np.linalg.norm(a[:2] - b[:2]))


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    计算三点夹角，b 为夹角顶点。

    返回值归一化到 0~1。
    """
    v1 = a - b
    v2 = c - b

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cos_value = float(np.dot(v1, v2) / (norm1 * norm2))
    cos_value = max(-1.0, min(1.0, cos_value))

    angle = math.acos(cos_value)

    return float(angle / math.pi)


def landmark_to_img_point(landmark, width: int, height: int) -> np.ndarray:
    """
    将 MediaPipe normalized landmark 还原到当前处理帧的图像坐标系。

    x_img = x * width
    y_img = y * height
    z_img = z * width
    """
    return np.array(
        [
            landmark.x * width,
            landmark.y * height,
            landmark.z * width,
        ],
        dtype=np.float32,
    )


# =========================
# 4. 手部特征：78 维
# =========================

def hand_landmarks_to_points(hand_landmarks, width: int, height: int) -> np.ndarray | None:
    """
    将一只手的 21 个 landmark 转成图像坐标数组。

    返回：
        shape = 21 × 3
    """
    if hand_landmarks is None:
        return None

    points = [
        landmark_to_img_point(landmark, width, height)
        for landmark in hand_landmarks.landmark
    ]

    return np.stack(points, axis=0).astype(np.float32)


def normalize_hand_points(points: np.ndarray | None) -> np.ndarray:
    """
    手部 21 点相对化与尺度归一化。

    规则：
    1. 以 WRIST，也就是 0 号点，为原点。
    2. 使用掌长和掌宽估计 hand_scale。
    3. 输出 21 × 3 的局部归一化坐标。
    """
    if points is None:
        return np.zeros((21, 3), dtype=np.float32)

    wrist = points[0]

    palm_length = distance_2d(points[0], points[9])
    palm_width = distance_2d(points[5], points[17])

    hand_scale = (palm_length + palm_width) / 2.0

    if hand_scale < 1e-6:
        hand_scale = 1.0

    normalized = (points - wrist) / hand_scale

    return normalized.astype(np.float32)


def extract_finger_angles(normalized_points: np.ndarray | None) -> np.ndarray:
    """
    提取 15 个手指关节角度。

    每根手指取 3 个角度。
    """
    if normalized_points is None:
        return np.zeros((15,), dtype=np.float32)

    triplets = [
        # thumb
        (0, 1, 2), (1, 2, 3), (2, 3, 4),

        # index
        (0, 5, 6), (5, 6, 7), (6, 7, 8),

        # middle
        (0, 9, 10), (9, 10, 11), (10, 11, 12),

        # ring
        (0, 13, 14), (13, 14, 15), (14, 15, 16),

        # pinky
        (0, 17, 18), (17, 18, 19), (18, 19, 20),
    ]

    angles = [
        angle_between_points(
            normalized_points[a],
            normalized_points[b],
            normalized_points[c],
        )
        for a, b, c in triplets
    ]

    return np.array(angles, dtype=np.float32)


def extract_hand_78(hand_landmarks, width: int, height: int) -> np.ndarray:
    """
    提取单只手的 78 维特征。

    结构：
    - 63 维局部归一化坐标
    - 15 维手指角度
    """
    points = hand_landmarks_to_points(hand_landmarks, width, height)

    if points is None:
        return np.zeros((78,), dtype=np.float32)

    normalized_points = normalize_hand_points(points)

    coords_63 = normalized_points.reshape(-1)
    angles_15 = extract_finger_angles(normalized_points)

    feature = np.concatenate([coords_63, angles_15], axis=0).astype(np.float32)

    if feature.shape[0] != 78:
        raise ValueError(f"手部特征维度错误，期望 78，实际 {feature.shape[0]}")

    return feature


# =========================
# 5. Pose 手臂特征：8 + 2 维
# =========================

def pose_landmarks_to_points(pose_landmarks, width: int, height: int) -> np.ndarray | None:
    """
    将 Pose landmarks 转成图像坐标数组。

    返回：
        shape = 33 × 3
    """
    if pose_landmarks is None:
        return None

    points = [
        landmark_to_img_point(landmark, width, height)
        for landmark in pose_landmarks.landmark
    ]

    return np.stack(points, axis=0).astype(np.float32)


def extract_arm_position_8_and_elbow_angle_2(pose_landmarks, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Pose 中提取：
    - 手臂位置 8 维
    - 肘角 2 维
    """
    points = pose_landmarks_to_points(pose_landmarks, width, height)

    if points is None:
        return (
            np.zeros((8,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        )

    left_shoulder = points[POSE_LEFT_SHOULDER]
    right_shoulder = points[POSE_RIGHT_SHOULDER]
    left_elbow = points[POSE_LEFT_ELBOW]
    right_elbow = points[POSE_RIGHT_ELBOW]
    left_wrist = points[POSE_LEFT_WRIST]
    right_wrist = points[POSE_RIGHT_WRIST]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = distance_2d(left_shoulder, right_shoulder)

    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    left_elbow_xy = (left_elbow[:2] - shoulder_center[:2]) / shoulder_width
    right_elbow_xy = (right_elbow[:2] - shoulder_center[:2]) / shoulder_width
    left_wrist_xy = (left_wrist[:2] - shoulder_center[:2]) / shoulder_width
    right_wrist_xy = (right_wrist[:2] - shoulder_center[:2]) / shoulder_width

    arm_position_8 = np.concatenate(
        [
            left_elbow_xy,
            right_elbow_xy,
            left_wrist_xy,
            right_wrist_xy,
        ],
        axis=0,
    ).astype(np.float32)

    left_elbow_angle = angle_between_points(
        left_shoulder,
        left_elbow,
        left_wrist,
    )

    right_elbow_angle = angle_between_points(
        right_shoulder,
        right_elbow,
        right_wrist,
    )

    elbow_angle_2 = np.array(
        [left_elbow_angle, right_elbow_angle],
        dtype=np.float32,
    )

    return arm_position_8, elbow_angle_2


# =========================
# 6. 单帧 166 维特征
# =========================

def holistic_results_to_feature_166(results, width: int, height: int) -> np.ndarray:
    """
    将 MediaPipe Holistic 单帧检测结果转成 CE-CSL Feature V1 的 166 维特征。
    """
    left_hand_78 = extract_hand_78(
        results.left_hand_landmarks,
        width,
        height,
    )

    right_hand_78 = extract_hand_78(
        results.right_hand_landmarks,
        width,
        height,
    )

    arm_position_8, elbow_angle_2 = extract_arm_position_8_and_elbow_angle_2(
        results.pose_landmarks,
        width,
        height,
    )

    feature = np.concatenate(
        [
            left_hand_78,
            right_hand_78,
            arm_position_8,
            elbow_angle_2,
        ],
        axis=0,
    ).astype(np.float32)

    if feature.shape[0] != 166:
        raise ValueError(f"单帧特征维度错误，期望 166，实际 {feature.shape[0]}")

    return feature


# =========================
# 7. 视频采样与特征提取
# =========================

def build_sample_indices(frame_count: int, fps: float) -> List[int]:
    """
    按 TARGET_FPS 构建采样帧编号。

    不按固定帧间隔，而是按时间采样。
    """
    if frame_count <= 0:
        return []

    if fps <= 0:
        fps = 30.0

    duration = frame_count / fps
    sample_times = np.arange(0.0, duration, 1.0 / TARGET_FPS)

    indices = []

    for time_sec in sample_times:
        index = int(round(time_sec * fps))
        index = max(0, min(index, frame_count - 1))
        indices.append(index)

    # 去重并保持顺序
    result = []
    seen = set()

    for index in indices:
        if index not in seen:
            result.append(index)
            seen.add(index)

    return result


def extract_video_feature(sample: Dict, holistic) -> np.ndarray:
    """
    提取单个视频的 T × 166 特征。
    """
    video_path = DATASET_ROOT / sample["videoPath"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    sample_indices = build_sample_indices(frame_count, fps)

    features = []

    for frame_index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame_bgr = cap.read()

        if not success or frame_bgr is None:
            continue

        frame_bgr = resize_frame(frame_bgr)

        if MIRROR_INPUT:
            frame_bgr = cv2.flip(frame_bgr, 1)

        height, width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        feature_166 = holistic_results_to_feature_166(results, width, height)
        features.append(feature_166)

    cap.release()

    if not features:
        raise RuntimeError(f"没有提取到有效特征：{video_path}")

    return np.stack(features, axis=0).astype(np.float32)


# =========================
# 8. 主流程
# =========================

def main() -> None:
    """
    主入口。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== CE-CSL Feature V1 少量样本提取开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("输出目录:", OUTPUT_DIR)
    print("TARGET_FPS:", TARGET_FPS)
    print("TARGET_WIDTH:", TARGET_WIDTH)
    print("MIRROR_INPUT:", MIRROR_INPUT)

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for split, limit in SPLIT_LIMITS.items():
            manifest_path = PROCESSED_DIR / f"{split}.jsonl"

            if not manifest_path.exists():
                raise FileNotFoundError(f"找不到 manifest 文件：{manifest_path}")

            samples = read_jsonl(manifest_path, limit=limit)

            split_output_dir = OUTPUT_DIR / split
            split_output_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "#" * 80)
            print(f"处理 {split}，样本数: {len(samples)}")
            print("#" * 80)

            for sample in samples:
                feature = extract_video_feature(sample, holistic)

                output_path = split_output_dir / f"{sample['sampleId']}.npy"
                np.save(output_path, feature)

                print(
                    f"[保存] {sample['sampleId']} "
                    f"shape={feature.shape} "
                    f"path={output_path}"
                )

    print("\n===== CE-CSL Feature V1 少量样本提取结束 =====")


if __name__ == "__main__":
    main()