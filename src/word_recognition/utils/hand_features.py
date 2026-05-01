import numpy as np

try:
    from src.word_recognition.config.gesture_config import (
        FEATURE_DIM,
        POSE_VISIBILITY_THRESHOLD,
        SWAP_HANDEDNESS,
        SWAP_POSE_LR,
        MIRROR_POSE_X,
    )
except ImportError:
    from word_recognition.config.gesture_config import (
        FEATURE_DIM,
        POSE_VISIBILITY_THRESHOLD,
        SWAP_HANDEDNESS,
        SWAP_POSE_LR,
        MIRROR_POSE_X
    )


# =========================
# 单手局部 3D 手型特征
# =========================


def _pose_xy_for_feature(pose_landmarks, landmark_enum):
    """读取用于特征构造的 Pose xy 坐标。"""
    lm = pose_landmarks.landmark[landmark_enum.value]
    x = float(lm.x)
    y = float(lm.y)

    if MIRROR_POSE_X:
        x = 1.0 - x

    return np.array([x, y], dtype=np.float32)

def build_frame_feature(hand_world_landmarks):
    """使用 Hands world landmarks 构造单只手 78 维局部手型特征。

    组成：
    - 21 个 world landmarks × xyz = 63
    - 15 个手指骨骼角度余弦 = 15
    - 合计 78
    """

    # 1. 提取 21 个 world landmarks，组成 (21, 3)
    points = []
    for landmark in hand_world_landmarks.landmark:
        points.append([landmark.x, landmark.y, landmark.z])

    points_array = np.array(points, dtype=np.float32)

    # 2. 做手部尺度归一化：使用 wrist(0) -> middle_mcp(9) 的距离作为 scale
    scale = np.linalg.norm(points_array[9] - points_array[0])
    if scale < 1e-6:
        scale = 1e-6

    points_array = points_array / scale

    # 3. 构造 20 根骨骼向量
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

    # 4. 计算 15 个相邻骨骼夹角余弦
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

    # 5. 拼接成 78 维
    coord_feature = points_array.flatten().astype(np.float32)
    frame_feature = np.concatenate([coord_feature, angle_array]).astype(np.float32)

    return frame_feature


def normalize_handedness(label: str, swap_handedness: bool = False) -> str:
    """根据需要修正 MediaPipe 返回的左右手标签。"""
    if not swap_handedness:
        return label

    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def resolve_pose_lr(PoseLandmark, swap_pose_lr: bool = SWAP_POSE_LR):
    """根据配置决定 Pose 的左右语义映射。"""
    if not swap_pose_lr:
        return {
            "LEFT_SHOULDER": PoseLandmark.LEFT_SHOULDER,
            "RIGHT_SHOULDER": PoseLandmark.RIGHT_SHOULDER,
            "LEFT_ELBOW": PoseLandmark.LEFT_ELBOW,
            "RIGHT_ELBOW": PoseLandmark.RIGHT_ELBOW,
            "LEFT_WRIST": PoseLandmark.LEFT_WRIST,
            "RIGHT_WRIST": PoseLandmark.RIGHT_WRIST,
        }

    return {
        "LEFT_SHOULDER": PoseLandmark.RIGHT_SHOULDER,
        "RIGHT_SHOULDER": PoseLandmark.LEFT_SHOULDER,
        "LEFT_ELBOW": PoseLandmark.RIGHT_ELBOW,
        "RIGHT_ELBOW": PoseLandmark.LEFT_ELBOW,
        "LEFT_WRIST": PoseLandmark.RIGHT_WRIST,
        "RIGHT_WRIST": PoseLandmark.LEFT_WRIST,
    }


# =========================
# Pose 身体坐标系辅助函数
# =========================


def _landmark_visible(pose_landmarks, landmark_enum, threshold: float = POSE_VISIBILITY_THRESHOLD) -> bool:
    """判断 Pose 某个关键点是否可用。"""
    lm = pose_landmarks.landmark[landmark_enum.value]
    return getattr(lm, "visibility", 1.0) >= threshold


def _pose_xy(pose_landmarks, landmark_enum) -> np.ndarray:
    """读取 Pose 关键点 xy 坐标。"""
    lm = pose_landmarks.landmark[landmark_enum.value]
    return np.array([lm.x, lm.y], dtype=np.float32)


def _normalize_point_to_shoulder(point_xy: np.ndarray,
                                 shoulder_center: np.ndarray,
                                 shoulder_width: float) -> np.ndarray:
    """把某个点转换成相对两肩中心、按肩宽归一化的身体坐标。"""
    safe_width = max(float(shoulder_width), 1e-6)
    return ((point_xy - shoulder_center) / safe_width).astype(np.float32)


def _calc_elbow_angle_cos(shoulder_xy: np.ndarray,
                          elbow_xy: np.ndarray,
                          wrist_xy: np.ndarray) -> float:
    """计算肩-肘-腕夹角余弦。"""
    upper = shoulder_xy - elbow_xy
    lower = wrist_xy - elbow_xy

    upper_norm = np.linalg.norm(upper)
    lower_norm = np.linalg.norm(lower)

    if upper_norm < 1e-6 or lower_norm < 1e-6:
        return 0.0

    cos_value = float(np.dot(upper, lower) / (upper_norm * lower_norm))
    return float(np.clip(cos_value, -1.0, 1.0))


# =========================
# 新版 166 维特征
# =========================


def extract_arm_pose_frame_parts(hand_results,
                                 pose_results,
                                 mp_pose,
                                 swap_handedness: bool = SWAP_HANDEDNESS,
                                 swap_pose_lr: bool = SWAP_POSE_LR):
    """从单帧 MediaPipe Hands + Pose 结果中提取 166 维特征所需的中间数据。

    如果 Pose 左右肩不可用，返回 None。
    左右肩是身体坐标系的基础，没有肩膀就不采这一帧。
    """
    if pose_results is None or pose_results.pose_landmarks is None:
        return None

    pose_landmarks = pose_results.pose_landmarks
    PoseLandmark = mp_pose.PoseLandmark

    pose_lr = resolve_pose_lr(PoseLandmark, swap_pose_lr)
    LEFT_SHOULDER = pose_lr["LEFT_SHOULDER"]
    RIGHT_SHOULDER = pose_lr["RIGHT_SHOULDER"]
    LEFT_ELBOW = pose_lr["LEFT_ELBOW"]
    RIGHT_ELBOW = pose_lr["RIGHT_ELBOW"]
    LEFT_WRIST = pose_lr["LEFT_WRIST"]
    RIGHT_WRIST = pose_lr["RIGHT_WRIST"]

    # 1. 左右肩必须可用
    required_shoulders = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
    ]

    for landmark_enum in required_shoulders:
        if not _landmark_visible(pose_landmarks, landmark_enum):
            return None

    left_shoulder = _pose_xy_for_feature(pose_landmarks,LEFT_SHOULDER)
    right_shoulder = _pose_xy_for_feature(pose_landmarks,RIGHT_SHOULDER)

    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
    if shoulder_width < 1e-6:
        return None

    frame_parts = {
        "left_hand_78": np.zeros(78, dtype=np.float32),
        "right_hand_78": np.zeros(78, dtype=np.float32),
        "left_score": -1.0,
        "right_score": -1.0,
        "left_wrist_rel": np.zeros(2, dtype=np.float32),
        "right_wrist_rel": np.zeros(2, dtype=np.float32),
        "left_elbow_rel": np.zeros(2, dtype=np.float32),
        "right_elbow_rel": np.zeros(2, dtype=np.float32),
        "left_elbow_angle": 0.0,
        "right_elbow_angle": 0.0,
    }

    # 2. Hands：左右手 78 维精细局部手型
    if (
        hand_results is not None
        and hand_results.multi_hand_world_landmarks
        and hand_results.multi_handedness
    ):
        hand_count = min(
            len(hand_results.multi_hand_world_landmarks),
            len(hand_results.multi_handedness)
        )

        for index in range(hand_count):
            hand_world_landmarks = hand_results.multi_hand_world_landmarks[index]
            handedness = hand_results.multi_handedness[index].classification[0]
            label = normalize_handedness(handedness.label, swap_handedness)
            score = float(handedness.score)
            hand_feature = build_frame_feature(hand_world_landmarks)

            if label == "Left" and score > frame_parts["left_score"]:
                frame_parts["left_hand_78"] = hand_feature
                frame_parts["left_score"] = score
            elif label == "Right" and score > frame_parts["right_score"]:
                frame_parts["right_hand_78"] = hand_feature
                frame_parts["right_score"] = score

    # 3. Pose：左右 wrist / elbow 相对肩膀中心的位置，并按肩宽归一化
    pose_pairs = [
        ("left_wrist_rel", LEFT_WRIST),
        ("right_wrist_rel", RIGHT_WRIST),
        ("left_elbow_rel", LEFT_ELBOW),
        ("right_elbow_rel", RIGHT_ELBOW),
    ]

    for key, landmark_enum in pose_pairs:
        if _landmark_visible(pose_landmarks, landmark_enum):
            point_xy = _pose_xy_for_feature(pose_landmarks, landmark_enum)
            frame_parts[key] = _normalize_point_to_shoulder(
                point_xy,
                shoulder_center,
                shoulder_width
            )

    # 4. Pose：左肘角度
    left_arm_points = [
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
    ]
    if all(_landmark_visible(pose_landmarks, item) for item in left_arm_points):
        left_elbow = _pose_xy_for_feature(pose_landmarks, LEFT_ELBOW)
        left_wrist = _pose_xy_for_feature(pose_landmarks, LEFT_WRIST)
        frame_parts["left_elbow_angle"] = _calc_elbow_angle_cos(
            left_shoulder,
            left_elbow,
            left_wrist
        )

    # 5. Pose：右肘角度
    right_arm_points = [
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
    ]
    if all(_landmark_visible(pose_landmarks, item) for item in right_arm_points):
        right_elbow = _pose_xy_for_feature(pose_landmarks, RIGHT_ELBOW)
        right_wrist = _pose_xy_for_feature(pose_landmarks, RIGHT_WRIST)
        frame_parts["right_elbow_angle"] = _calc_elbow_angle_cos(
            right_shoulder,
            right_elbow,
            right_wrist
        )

    return frame_parts


def build_arm_pose_sample(frame_parts_list):
    """把 30 帧中间特征拼成最终模型输入，形状为 (30, 166)。"""
    if len(frame_parts_list) == 0:
        raise ValueError("frame_parts_list 不能为空。")

    rows = []
    for frame_parts in frame_parts_list:
        row = np.concatenate([
            frame_parts["left_hand_78"],
            frame_parts["right_hand_78"],
            frame_parts["left_wrist_rel"],
            frame_parts["right_wrist_rel"],
            frame_parts["left_elbow_rel"],
            frame_parts["right_elbow_rel"],
            np.array([
                frame_parts["left_elbow_angle"],
                frame_parts["right_elbow_angle"],
            ], dtype=np.float32)
        ]).astype(np.float32)

        if row.shape[0] != FEATURE_DIM:
            raise ValueError(f"单帧特征维度异常：{row.shape}")

        rows.append(row)

    sample = np.stack(rows, axis=0).astype(np.float32)
    return sample
