import numpy as np


def build_frame_feature(hand_world_landmarks):
    """使用 world landmarks 构造最终单帧特征，返回形状为 (78,) 的数组。"""

    # 1. 提取 21 个 world landmarks，组成 (21, 3)
    points = []
    for landmark in hand_world_landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])

    points_array = np.array(points, dtype=np.float32)

    # 2. 做尺度归一化：使用 0 -> 9 的距离作为 scale
    scale = np.linalg.norm(points_array[9] - points_array[0])
    if scale < 1e-6:
        scale = 1e-6

    points_array = points_array / scale

    # 3. 构造骨骼向量
    parent_indices = [
        0, 1, 2, 3,
        0, 5, 6, 7,
        0, 9, 10, 11,
        0, 13, 14, 15,
        0, 17, 18, 19
    ]
    child_indices = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20
    ]

    bone_vectors = points_array[child_indices] - points_array[parent_indices]

    norms = np.linalg.norm(bone_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-6, norms)
    bone_vectors = bone_vectors / norms

    # 4. 计算 15 个角度余弦值
    angle_parent_indices = [
        0, 1, 2,
        4, 5, 6,
        8, 9, 10,
        12, 13, 14,
        16, 17, 18
    ]
    angle_child_indices = [
        1, 2, 3,
        5, 6, 7,
        9, 10, 11,
        13, 14, 15,
        17, 18, 19
    ]

    angle_array = np.sum(
        bone_vectors[angle_parent_indices] * bone_vectors[angle_child_indices],
        axis=1
    ).astype(np.float32)

    angle_array = np.clip(angle_array, -1.0, 1.0)

    # 5. 拼接成最终 78 维特征
    coord_feature = points_array.flatten().astype(np.float32)
    frame_feature = np.concatenate([coord_feature, angle_array]).astype(np.float32)

    return frame_feature

def extract_palm_center(hand_landmarks):
    """从 image landmarks 中提取掌心中心坐标，返回形状为 (2,) 的数组。"""
    palm_indices = [0, 5, 9, 13, 17]

    points = []
    for idx in palm_indices:
        landmark = hand_landmarks.landmark[idx]
        points.append([landmark.x, landmark.y])

    points_array = np.array(points, dtype=np.float32)
    palm_center = np.mean(points_array, axis=0)

    return palm_center.astype(np.float32)

def extract_palm_scale(hand_landmarks):
    """从 image landmarks 中提取掌部尺度，返回一个标量。"""
    p0 = hand_landmarks.landmark[0]
    p9 = hand_landmarks.landmark[9]

    p0_xy = np.array([p0.x, p0.y], dtype=np.float32)
    p9_xy = np.array([p9.x, p9.y], dtype=np.float32)

    scale = np.linalg.norm(p9_xy - p0_xy)
    return max(float(scale), 1e-6)