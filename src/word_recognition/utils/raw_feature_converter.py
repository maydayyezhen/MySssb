"""raw MediaPipe 样本转 feature 样本工具。

输入：
dataset_raw_phone_10fps/{label}/sample_001.npz

输出：
data_processed_arm_pose_10fps/{label}/sample_001.npy

每个输出样本形状：
(30, 166)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.word_recognition.config.gesture_config import (
    WINDOW_SIZE,
    FEATURE_DIM,
    DATA_DIR_NAME,
    RAW_PHONE_DATA_DIR_NAME,
)


# MediaPipe Pose 常用点位索引。
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_ELBOW = 13
POSE_RIGHT_ELBOW = 14
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16


def convert_raw_dataset_to_features(project_root: Path) -> Dict:
    """转换整个 raw dataset。

    Args:
        project_root: 项目根目录。

    Returns:
        转换结果摘要。
    """
    raw_root = project_root / RAW_PHONE_DATA_DIR_NAME
    feature_root = project_root / DATA_DIR_NAME

    feature_root.mkdir(parents=True, exist_ok=True)

    scanned_count = 0
    converted_count = 0
    skipped_count = 0
    failed_count = 0
    failed_items: List[Dict] = []

    if not raw_root.exists():
        return {
            "rawRoot": str(raw_root),
            "featureRoot": str(feature_root),
            "scannedCount": 0,
            "convertedCount": 0,
            "skippedCount": 0,
            "failedCount": 0,
            "failedItems": [],
            "message": "raw dataset directory does not exist",
        }

    for label_dir in sorted(raw_root.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        output_label_dir = feature_root / label
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for raw_file in sorted(label_dir.glob("sample_*.npz")):
            scanned_count += 1

            output_file = output_label_dir / f"{raw_file.stem}.npy"

            try:
                sample = convert_single_raw_sample(raw_file)

                if sample.shape != (WINDOW_SIZE, FEATURE_DIM):
                    skipped_count += 1
                    failed_items.append({
                        "path": str(raw_file),
                        "reason": f"特征形状异常：{sample.shape}",
                    })
                    continue

                np.save(output_file, sample.astype(np.float32))
                converted_count += 1

            except Exception as ex:
                failed_count += 1
                failed_items.append({
                    "path": str(raw_file),
                    "reason": str(ex),
                })

    return {
        "rawRoot": str(raw_root),
        "featureRoot": str(feature_root),
        "scannedCount": scanned_count,
        "convertedCount": converted_count,
        "skippedCount": skipped_count,
        "failedCount": failed_count,
        "failedItems": failed_items,
        "message": "raw to feature conversion completed",
    }


def convert_single_raw_sample(raw_file: Path) -> np.ndarray:
    """转换单个 raw npz 样本。

    Args:
        raw_file: raw npz 文件路径。

    Returns:
        shape=(30, 166) 的 feature 样本。
    """
    with np.load(raw_file, allow_pickle=True) as data:
        hand_world = data["hand_world_landmarks_xyz"]
        hand_scores = data["hand_scores"]
        hand_present = data["hand_present"]
        pose_xyzc = data["pose_landmarks_xyzc"]
        pose_present = data["pose_present"]

        frame_count = hand_world.shape[0]
        if frame_count != WINDOW_SIZE:
            raise ValueError(f"样本帧数异常：{frame_count}，期望：{WINDOW_SIZE}")

        rows = []

        for frame_index in range(WINDOW_SIZE):
            row = build_feature_row_from_raw_frame(
                hand_world_frame=hand_world[frame_index],
                hand_scores_frame=hand_scores[frame_index],
                hand_present_frame=hand_present[frame_index],
                pose_xyzc_frame=pose_xyzc[frame_index],
                pose_present_frame=pose_present[frame_index],
            )

            if row.shape[0] != FEATURE_DIM:
                raise ValueError(f"第 {frame_index} 帧特征维度异常：{row.shape}")

            rows.append(row)

        return np.stack(rows, axis=0).astype(np.float32)


def build_feature_row_from_raw_frame(
    hand_world_frame: np.ndarray,
    hand_scores_frame: np.ndarray,
    hand_present_frame: np.ndarray,
    pose_xyzc_frame: np.ndarray,
    pose_present_frame: np.ndarray,
) -> np.ndarray:
    """从单帧 raw 数据构造 166 维特征。

    特征结构：
    1. left_hand_78
    2. right_hand_78
    3. left_wrist_rel 2
    4. right_wrist_rel 2
    5. left_elbow_rel 2
    6. right_elbow_rel 2
    7. left_elbow_angle 1
    8. right_elbow_angle 1

    合计：
    78 + 78 + 2 + 2 + 2 + 2 + 1 + 1 = 166
    """
    left_hand_78 = np.zeros(78, dtype=np.float32)
    right_hand_78 = np.zeros(78, dtype=np.float32)

    # 当前 raw 保存约定：
    # slot0 = MediaPipe Left
    # slot1 = MediaPipe Right
    if is_hand_present(hand_present_frame, 0):
        left_hand_78 = build_hand_78_from_points(hand_world_frame[0])

    if is_hand_present(hand_present_frame, 1):
        right_hand_78 = build_hand_78_from_points(hand_world_frame[1])

    pose_parts = build_pose_parts(pose_xyzc_frame, pose_present_frame)

    row = np.concatenate([
        left_hand_78,
        right_hand_78,
        pose_parts["left_wrist_rel"],
        pose_parts["right_wrist_rel"],
        pose_parts["left_elbow_rel"],
        pose_parts["right_elbow_rel"],
        np.array([
            pose_parts["left_elbow_angle"],
            pose_parts["right_elbow_angle"],
        ], dtype=np.float32),
    ]).astype(np.float32)

    return row


def is_hand_present(hand_present_frame: np.ndarray, slot_index: int) -> bool:
    """判断某个手部 slot 是否存在。"""
    if hand_present_frame is None:
        return False

    arr = np.asarray(hand_present_frame)

    if arr.ndim == 0:
        return bool(arr)

    if slot_index >= arr.shape[0]:
        return False

    return float(arr[slot_index]) > 0.0


def build_hand_78_from_points(points_array: np.ndarray) -> np.ndarray:
    """从单只手 21×3 world landmarks 构造 78 维局部手型特征。"""
    points_array = np.asarray(points_array, dtype=np.float32)

    if points_array.shape != (21, 3):
        return np.zeros(78, dtype=np.float32)

    scale = np.linalg.norm(points_array[9] - points_array[0])
    if scale < 1e-6:
        scale = 1e-6

    points_array = points_array / scale

    parent_indices = [
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19,
    ]
    child_indices = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
    ]

    bone_vectors = points_array[child_indices] - points_array[parent_indices]
    norms = np.linalg.norm(bone_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-6, norms)
    bone_vectors = bone_vectors / norms

    angle_parent_indices = [
        0, 1, 2,
        4, 5, 6,
        8, 9, 10,
        12, 13, 14,
        16, 17, 18,
    ]
    angle_child_indices = [
        1, 2, 3,
        5, 6, 7,
        9, 10, 11,
        13, 14, 15,
        17, 18, 19,
    ]

    angle_array = np.sum(
        bone_vectors[angle_parent_indices] * bone_vectors[angle_child_indices],
        axis=1
    ).astype(np.float32)

    angle_array = np.clip(angle_array, -1.0, 1.0)

    coord_feature = points_array.flatten().astype(np.float32)
    return np.concatenate([coord_feature, angle_array]).astype(np.float32)


def build_pose_parts(pose_xyzc_frame: np.ndarray, pose_present_frame: np.ndarray) -> Dict[str, np.ndarray | float]:
    """构造 Pose 相关的 10 维辅助特征。"""
    zero2 = np.zeros(2, dtype=np.float32)

    parts = {
        "left_wrist_rel": zero2.copy(),
        "right_wrist_rel": zero2.copy(),
        "left_elbow_rel": zero2.copy(),
        "right_elbow_rel": zero2.copy(),
        "left_elbow_angle": 0.0,
        "right_elbow_angle": 0.0,
    }

    pose_xyzc_frame = np.asarray(pose_xyzc_frame, dtype=np.float32)

    if pose_xyzc_frame.ndim != 2 or pose_xyzc_frame.shape[0] <= POSE_RIGHT_WRIST:
        return parts

    if not is_pose_available(pose_xyzc_frame, pose_present_frame, POSE_LEFT_SHOULDER):
        return parts

    if not is_pose_available(pose_xyzc_frame, pose_present_frame, POSE_RIGHT_SHOULDER):
        return parts

    left_shoulder = pose_xyzc_frame[POSE_LEFT_SHOULDER, :2]
    right_shoulder = pose_xyzc_frame[POSE_RIGHT_SHOULDER, :2]

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))

    if shoulder_width < 1e-6:
        return parts

    parts["left_wrist_rel"] = normalize_pose_point(
        pose_xyzc_frame,
        pose_present_frame,
        POSE_LEFT_WRIST,
        shoulder_center,
        shoulder_width,
    )
    parts["right_wrist_rel"] = normalize_pose_point(
        pose_xyzc_frame,
        pose_present_frame,
        POSE_RIGHT_WRIST,
        shoulder_center,
        shoulder_width,
    )
    parts["left_elbow_rel"] = normalize_pose_point(
        pose_xyzc_frame,
        pose_present_frame,
        POSE_LEFT_ELBOW,
        shoulder_center,
        shoulder_width,
    )
    parts["right_elbow_rel"] = normalize_pose_point(
        pose_xyzc_frame,
        pose_present_frame,
        POSE_RIGHT_ELBOW,
        shoulder_center,
        shoulder_width,
    )

    if all(is_pose_available(pose_xyzc_frame, pose_present_frame, idx) for idx in [
        POSE_LEFT_SHOULDER,
        POSE_LEFT_ELBOW,
        POSE_LEFT_WRIST,
    ]):
        parts["left_elbow_angle"] = calc_elbow_angle_cos(
            pose_xyzc_frame[POSE_LEFT_SHOULDER, :2],
            pose_xyzc_frame[POSE_LEFT_ELBOW, :2],
            pose_xyzc_frame[POSE_LEFT_WRIST, :2],
        )

    if all(is_pose_available(pose_xyzc_frame, pose_present_frame, idx) for idx in [
        POSE_RIGHT_SHOULDER,
        POSE_RIGHT_ELBOW,
        POSE_RIGHT_WRIST,
    ]):
        parts["right_elbow_angle"] = calc_elbow_angle_cos(
            pose_xyzc_frame[POSE_RIGHT_SHOULDER, :2],
            pose_xyzc_frame[POSE_RIGHT_ELBOW, :2],
            pose_xyzc_frame[POSE_RIGHT_WRIST, :2],
        )

    return parts


def is_pose_available(
    pose_xyzc_frame: np.ndarray,
    pose_present_frame: np.ndarray,
    landmark_index: int,
) -> bool:
    """判断 Pose 点位是否可用。"""
    if landmark_index >= pose_xyzc_frame.shape[0]:
        return False

    if pose_present_frame is not None:
        arr = np.asarray(pose_present_frame)
        if arr.ndim == 1 and landmark_index < arr.shape[0]:
            return float(arr[landmark_index]) > 0.0
        if arr.ndim == 0:
            return bool(arr)

    # 兜底：使用第 4 维置信度 / 可见性。
    if pose_xyzc_frame.shape[1] >= 4:
        return float(pose_xyzc_frame[landmark_index, 3]) > 0.0

    return True


def normalize_pose_point(
    pose_xyzc_frame: np.ndarray,
    pose_present_frame: np.ndarray,
    landmark_index: int,
    shoulder_center: np.ndarray,
    shoulder_width: float,
) -> np.ndarray:
    """把 Pose 点转换为相对肩中心、按肩宽归一化的坐标。"""
    if not is_pose_available(pose_xyzc_frame, pose_present_frame, landmark_index):
        return np.zeros(2, dtype=np.float32)

    point_xy = pose_xyzc_frame[landmark_index, :2]
    return ((point_xy - shoulder_center) / max(shoulder_width, 1e-6)).astype(np.float32)


def calc_elbow_angle_cos(
    shoulder_xy: np.ndarray,
    elbow_xy: np.ndarray,
    wrist_xy: np.ndarray,
) -> float:
    """计算肩-肘-腕夹角余弦。"""
    upper = shoulder_xy - elbow_xy
    lower = wrist_xy - elbow_xy

    upper_norm = np.linalg.norm(upper)
    lower_norm = np.linalg.norm(lower)

    if upper_norm < 1e-6 or lower_norm < 1e-6:
        return 0.0

    cos_value = float(np.dot(upper, lower) / (upper_norm * lower_norm))
    return float(np.clip(cos_value, -1.0, 1.0))
