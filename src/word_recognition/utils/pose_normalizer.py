"""Pose 坐标规范化工具。

用途：
- 前端发送镜像自拍图时，MediaPipe Pose 的身体左右和坐标会处在镜像空间。
- 本工具把 Pose 关键点转换回项目内部使用的非镜像身体坐标系。
"""

import numpy as np

# MediaPipe Pose 左右对称点位索引
POSE_LEFT_RIGHT_PAIRS = [
    (1, 4),    # left_eye_inner ↔ right_eye_inner
    (2, 5),    # left_eye ↔ right_eye
    (3, 6),    # left_eye_outer ↔ right_eye_outer
    (7, 8),    # left_ear ↔ right_ear
    (9, 10),   # mouth_left ↔ mouth_right
    (11, 12),  # left_shoulder ↔ right_shoulder
    (13, 14),  # left_elbow ↔ right_elbow
    (15, 16),  # left_wrist ↔ right_wrist
    (17, 18),  # left_pinky ↔ right_pinky
    (19, 20),  # left_index ↔ right_index
    (21, 22),  # left_thumb ↔ right_thumb
    (23, 24),  # left_hip ↔ right_hip
    (25, 26),  # left_knee ↔ right_knee
    (27, 28),  # left_ankle ↔ right_ankle
    (29, 30),  # left_heel ↔ right_heel
    (31, 32),  # left_foot_index ↔ right_foot_index
]


def normalize_mirrored_pose_xyzc(pose_frame: np.ndarray) -> np.ndarray:
    """把镜像图上的 Pose 关键点转换为非镜像身体坐标系。

    参数：
        pose_frame: shape = (33, 4)，字段为 x, y, z, visibility。

    返回：
        normalized: shape = (33, 4)。

    处理：
        1. x 坐标水平翻转：x = 1 - x
        2. 左右身体点交换：LEFT_* ↔ RIGHT_*
    """
    normalized = pose_frame.astype(np.float32).copy()

    # 1. 坐标水平翻转
    normalized[:, 0] = 1.0 - normalized[:, 0]

    # 2. 左右点位交换
    for left_index, right_index in POSE_LEFT_RIGHT_PAIRS:
        left_copy = normalized[left_index].copy()
        normalized[left_index] = normalized[right_index]
        normalized[right_index] = left_copy

    return normalized