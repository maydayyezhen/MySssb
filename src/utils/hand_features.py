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

# ========= 双手统一特征配置 =========

SINGLE_HAND_BASE_DIM = 78
SINGLE_HAND_FINAL_DIM = 80
TWO_HAND_FEATURE_DIM = 162


def normalize_handedness(label: str, swap_handedness: bool = False) -> str:
    """根据需要修正 MediaPipe 返回的左右手标签。"""
    if not swap_handedness:
        return label

    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def extract_two_hand_frame_parts(results, swap_handedness: bool = False):
    """从 MediaPipe 结果中提取单帧双手中间特征。

    返回值说明：
    - left_base: 左手 78 维结构特征，不存在则全 0
    - right_base: 右手 78 维结构特征，不存在则全 0
    - left_center/right_center: 掌心中心，不存在则 None
    - left_scale/right_scale: 掌部尺度，不存在则 None

    注意：
    这里只构造 78 维静态特征。
    每只手额外的 2 维 motion 要等 30 帧窗口齐了以后再统一计算。
    """
    if (
        not results.multi_hand_landmarks
        or not results.multi_hand_world_landmarks
        or not results.multi_handedness
    ):
        return None

    frame_parts = {
        "left_base": np.zeros(SINGLE_HAND_BASE_DIM, dtype=np.float32),
        "right_base": np.zeros(SINGLE_HAND_BASE_DIM, dtype=np.float32),
        "left_center": None,
        "right_center": None,
        "left_scale": None,
        "right_scale": None,
        "left_score": -1.0,
        "right_score": -1.0,
    }

    hand_count = min(
        len(results.multi_hand_landmarks),
        len(results.multi_hand_world_landmarks),
        len(results.multi_handedness)
    )

    for index in range(hand_count):
        hand_landmarks = results.multi_hand_landmarks[index]
        hand_world_landmarks = results.multi_hand_world_landmarks[index]
        handedness = results.multi_handedness[index].classification[0]

        label = normalize_handedness(handedness.label, swap_handedness)
        score = float(handedness.score)

        base_feature = build_frame_feature(hand_world_landmarks)
        palm_center = extract_palm_center(hand_landmarks)
        palm_scale = extract_palm_scale(hand_landmarks)

        if label == "Left":
            if score > frame_parts["left_score"]:
                frame_parts["left_base"] = base_feature
                frame_parts["left_center"] = palm_center
                frame_parts["left_scale"] = palm_scale
                frame_parts["left_score"] = score

        elif label == "Right":
            if score > frame_parts["right_score"]:
                frame_parts["right_base"] = base_feature
                frame_parts["right_center"] = palm_center
                frame_parts["right_scale"] = palm_scale
                frame_parts["right_score"] = score

    return frame_parts


def _build_hand_motion(frame_parts_list, center_key: str, scale_key: str):
    """根据 30 帧掌心中心构造某只手的 2 维运动特征。"""
    motion = np.zeros((len(frame_parts_list), 2), dtype=np.float32)

    reference_center = None
    reference_scale = 1e-6

    for frame_parts in frame_parts_list:
        center = frame_parts[center_key]
        scale = frame_parts[scale_key]

        if center is not None:
            reference_center = center
            reference_scale = max(float(scale), 1e-6)
            break

    if reference_center is None:
        return motion

    for index, frame_parts in enumerate(frame_parts_list):
        center = frame_parts[center_key]
        if center is not None:
            motion[index] = (center - reference_center) / reference_scale

    return motion


def _build_two_hand_relation(frame_parts_list):
    """构造双手相对位置特征：dx, dy。

    dx/dy 语义：
    右手掌心 - 左手掌心，并按掌部尺度归一化。

    如果当前帧只有一只手，则 dx/dy = 0。
    """
    relation = np.zeros((len(frame_parts_list), 2), dtype=np.float32)

    for index, frame_parts in enumerate(frame_parts_list):
        left_center = frame_parts["left_center"]
        right_center = frame_parts["right_center"]

        if left_center is None or right_center is None:
            continue

        left_scale = frame_parts["left_scale"]
        right_scale = frame_parts["right_scale"]
        scale = max(float(left_scale), float(right_scale), 1e-6)

        relation[index] = (right_center - left_center) / scale

    return relation


def build_two_hand_sample(frame_parts_list):
    """把 30 帧双手中间特征拼成最终样本，形状为 (30, 162)。

    最终格式：
    left_hand_80 + right_hand_80 + relation_2

    其中：
    left_hand_80 = left_base_78 + left_motion_2
    right_hand_80 = right_base_78 + right_motion_2
    relation_2 = dx + dy
    """
    if len(frame_parts_list) == 0:
        raise ValueError("frame_parts_list 不能为空。")

    left_base = np.stack(
        [frame_parts["left_base"] for frame_parts in frame_parts_list],
        axis=0
    ).astype(np.float32)

    right_base = np.stack(
        [frame_parts["right_base"] for frame_parts in frame_parts_list],
        axis=0
    ).astype(np.float32)

    left_motion = _build_hand_motion(frame_parts_list, "left_center", "left_scale")
    right_motion = _build_hand_motion(frame_parts_list, "right_center", "right_scale")
    relation = _build_two_hand_relation(frame_parts_list)

    left_feature = np.concatenate([left_base, left_motion], axis=1)
    right_feature = np.concatenate([right_base, right_motion], axis=1)

    sample = np.concatenate(
        [left_feature, right_feature, relation],
        axis=1
    ).astype(np.float32)

    if sample.shape[1] != TWO_HAND_FEATURE_DIM:
        raise ValueError(f"双手特征维度异常：{sample.shape}")

    return sample