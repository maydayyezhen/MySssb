"""把 raw MediaPipe dataset 转换成模型训练用的 166 维特征。"""

from pathlib import Path

import numpy as np

from src.config.gesture_config import (
    RAW_PHONE_DATA_DIR_NAME,
    DATA_DIR_NAME,
    EXPECTED_SAMPLE_SHAPE,
    POSE_VISIBILITY_THRESHOLD,
    SWAP_POSE_LR,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / RAW_PHONE_DATA_DIR_NAME
FEATURE_ROOT = PROJECT_ROOT / DATA_DIR_NAME

# MediaPipe Pose 索引
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_ELBOW_INDEX = 13
RIGHT_ELBOW_INDEX = 14
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16

PARENT_INDICES = [
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
]
CHILD_INDICES = [
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16,
    17, 18, 19, 20
]
ANGLE_PARENT_INDICES = [
    0, 1, 2,
    4, 5, 6,
    8, 9, 10,
    12, 13, 14,
    16, 17, 18
]
ANGLE_CHILD_INDICES = [
    1, 2, 3,
    5, 6, 7,
    9, 10, 11,
    13, 14, 15,
    17, 18, 19
]


def get_pose_indices():
    """返回规范化后的 Pose 语义点位索引。"""
    return {
        "left_shoulder": LEFT_SHOULDER_INDEX,
        "right_shoulder": RIGHT_SHOULDER_INDEX,
        "left_elbow": LEFT_ELBOW_INDEX,
        "right_elbow": RIGHT_ELBOW_INDEX,
        "left_wrist": LEFT_WRIST_INDEX,
        "right_wrist": RIGHT_WRIST_INDEX,
    }


def build_hand_78_from_world_points(points_array: np.ndarray) -> np.ndarray:
    """从单手 21×3 world landmarks 构造 78 维手型特征。"""
    if points_array.shape != (21, 3):
        raise ValueError(f"手部 world landmarks shape 异常：{points_array.shape}")

    if np.allclose(points_array, 0.0):
        return np.zeros(78, dtype=np.float32)

    points = points_array.astype(np.float32).copy()

    scale = np.linalg.norm(points[9] - points[0])
    if scale < 1e-6:
        scale = 1e-6

    points = points / scale

    bone_vectors = points[CHILD_INDICES] - points[PARENT_INDICES]
    norms = np.linalg.norm(bone_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-6, norms)
    bone_vectors = bone_vectors / norms

    angle_array = np.sum(
        bone_vectors[ANGLE_PARENT_INDICES] * bone_vectors[ANGLE_CHILD_INDICES],
        axis=1
    ).astype(np.float32)
    angle_array = np.clip(angle_array, -1.0, 1.0)

    return np.concatenate([points.flatten(), angle_array]).astype(np.float32)


def pose_point_xy(pose_frame: np.ndarray, index: int) -> np.ndarray:
    return pose_frame[index, 0:2].astype(np.float32)


def pose_visible(pose_frame: np.ndarray, index: int) -> bool:
    return float(pose_frame[index, 3]) >= POSE_VISIBILITY_THRESHOLD


def normalize_to_shoulders(point_xy: np.ndarray,
                           shoulder_center: np.ndarray,
                           shoulder_width: float) -> np.ndarray:
    safe_width = max(float(shoulder_width), 1e-6)
    return ((point_xy - shoulder_center) / safe_width).astype(np.float32)


def calc_elbow_angle_cos(shoulder_xy: np.ndarray,
                         elbow_xy: np.ndarray,
                         wrist_xy: np.ndarray) -> float:
    upper = shoulder_xy - elbow_xy
    lower = wrist_xy - elbow_xy

    upper_norm = np.linalg.norm(upper)
    lower_norm = np.linalg.norm(lower)

    if upper_norm < 1e-6 or lower_norm < 1e-6:
        return 0.0

    cos_value = float(np.dot(upper, lower) / (upper_norm * lower_norm))
    return float(np.clip(cos_value, -1.0, 1.0))


def build_feature_sample_from_raw_npz(raw_path: Path) -> np.ndarray:
    """从一个 raw npz 样本构造 (30,166) 特征。"""
    data = np.load(raw_path, allow_pickle=True)

    hand_world = data["hand_world_landmarks_xyz"].astype(np.float32)
    hand_present = data["hand_present"].astype(np.float32)
    pose = data["pose_landmarks_xyzc"].astype(np.float32)

    pose_indices = get_pose_indices()

    rows = []
    frame_count = hand_world.shape[0]

    for frame_index in range(frame_count):
        left_hand_78 = (
            build_hand_78_from_world_points(hand_world[frame_index, 0])
            if hand_present[frame_index, 0] > 0.5
            else np.zeros(78, dtype=np.float32)
        )

        right_hand_78 = (
            build_hand_78_from_world_points(hand_world[frame_index, 1])
            if hand_present[frame_index, 1] > 0.5
            else np.zeros(78, dtype=np.float32)
        )

        # pose_landmarks_xyzc 在采集保存阶段已经完成规范化。
        # 这里不要再次 normalize_mirrored_pose_xyzc，否则会二次翻转。
        pose_frame = pose[frame_index]

        left_wrist_rel = np.zeros(2, dtype=np.float32)
        right_wrist_rel = np.zeros(2, dtype=np.float32)
        left_elbow_rel = np.zeros(2, dtype=np.float32)
        right_elbow_rel = np.zeros(2, dtype=np.float32)
        left_elbow_angle = 0.0
        right_elbow_angle = 0.0

        left_shoulder_idx = pose_indices["left_shoulder"]
        right_shoulder_idx = pose_indices["right_shoulder"]
        left_elbow_idx = pose_indices["left_elbow"]
        right_elbow_idx = pose_indices["right_elbow"]
        left_wrist_idx = pose_indices["left_wrist"]
        right_wrist_idx = pose_indices["right_wrist"]

        if pose_visible(pose_frame, left_shoulder_idx) and pose_visible(pose_frame, right_shoulder_idx):
            left_shoulder = pose_point_xy(pose_frame, left_shoulder_idx)
            right_shoulder = pose_point_xy(pose_frame, right_shoulder_idx)
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))

            if pose_visible(pose_frame, left_wrist_idx):
                left_wrist = pose_point_xy(pose_frame, left_wrist_idx)
                left_wrist_rel = normalize_to_shoulders(left_wrist, shoulder_center, shoulder_width)

            if pose_visible(pose_frame, right_wrist_idx):
                right_wrist = pose_point_xy(pose_frame, right_wrist_idx)
                right_wrist_rel = normalize_to_shoulders(right_wrist, shoulder_center, shoulder_width)

            if pose_visible(pose_frame, left_elbow_idx):
                left_elbow = pose_point_xy(pose_frame, left_elbow_idx)
                left_elbow_rel = normalize_to_shoulders(left_elbow, shoulder_center, shoulder_width)

            if pose_visible(pose_frame, right_elbow_idx):
                right_elbow = pose_point_xy(pose_frame, right_elbow_idx)
                right_elbow_rel = normalize_to_shoulders(right_elbow, shoulder_center, shoulder_width)

            if pose_visible(pose_frame, left_elbow_idx) and pose_visible(pose_frame, left_wrist_idx):
                left_elbow_angle = calc_elbow_angle_cos(
                    left_shoulder,
                    pose_point_xy(pose_frame, left_elbow_idx),
                    pose_point_xy(pose_frame, left_wrist_idx),
                )

            if pose_visible(pose_frame, right_elbow_idx) and pose_visible(pose_frame, right_wrist_idx):
                right_elbow_angle = calc_elbow_angle_cos(
                    right_shoulder,
                    pose_point_xy(pose_frame, right_elbow_idx),
                    pose_point_xy(pose_frame, right_wrist_idx),
                )

        row = np.concatenate([
            left_hand_78,
            right_hand_78,
            left_wrist_rel,
            right_wrist_rel,
            left_elbow_rel,
            right_elbow_rel,
            np.array([left_elbow_angle, right_elbow_angle], dtype=np.float32)
        ]).astype(np.float32)

        rows.append(row)

    sample = np.stack(rows, axis=0).astype(np.float32)

    if sample.shape != EXPECTED_SAMPLE_SHAPE:
        raise ValueError(f"转换后的特征 shape 异常：{raw_path}, shape={sample.shape}, expected={EXPECTED_SAMPLE_SHAPE}")

    return sample


def convert_all():
    if not RAW_ROOT.exists():
        raise ValueError(f"raw 数据目录不存在：{RAW_ROOT}")

    class_dirs = sorted([item for item in RAW_ROOT.iterdir() if item.is_dir()])
    if not class_dirs:
        raise ValueError(f"raw 数据目录下没有标签文件夹：{RAW_ROOT}")

    converted_count = 0

    for class_dir in class_dirs:
        label = class_dir.name
        output_dir = FEATURE_ROOT / label
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(class_dir.glob("sample_*.npz"))
        print(f"转换标签 [{label}]，raw 样本数：{len(raw_files)}")

        for raw_file in raw_files:
            try:
                feature_sample = build_feature_sample_from_raw_npz(raw_file)
            except Exception as error:
                print(f"跳过转换失败样本：{raw_file}, error={error}")
                continue

            output_path = output_dir / f"{raw_file.stem}.npy"
            np.save(output_path, feature_sample)
            converted_count += 1

    print(f"转换完成，有效样本数：{converted_count}")
    print(f"输出目录：{FEATURE_ROOT}")


if __name__ == "__main__":
    convert_all()