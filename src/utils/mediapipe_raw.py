"""MediaPipe 原始结果提取工具。

职责：
- 把 MediaPipe Hands / Pose 的结果转换成固定 shape 的 numpy 数组
- 固定 left / right 手槽位
- 不计算最终模型特征
- 不做 166 维拼接
"""

import numpy as np

try:
    from src.config.gesture_config import (
        SWAP_HANDEDNESS,
        SWAP_POSE_LR,
        MIRROR_POSE_X,
        POSE_VISIBILITY_THRESHOLD,
        RAW_REQUIRE_SHOULDERS,
        RAW_REQUIRE_HAND,
    )
    from src.utils.pose_normalizer import normalize_mirrored_pose_xyzc
except ImportError:
    from config.gesture_config import (
        SWAP_HANDEDNESS,
        SWAP_POSE_LR,
        MIRROR_POSE_X,
        POSE_VISIBILITY_THRESHOLD,
        RAW_REQUIRE_SHOULDERS,
        RAW_REQUIRE_HAND,
    )
    from utils.pose_normalizer import normalize_mirrored_pose_xyzc


def normalize_handedness(label: str, swap_handedness: bool = SWAP_HANDEDNESS) -> str:
    """根据配置修正 Hands 返回的左右手标签。"""
    if not swap_handedness:
        return label

    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def resolve_pose_lr(PoseLandmark, swap_pose_lr: bool = SWAP_POSE_LR):
    """根据配置决定 Pose 左右点位是否交换。"""
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


def _pose_landmark_visible(pose_landmarks, landmark_enum, threshold: float = POSE_VISIBILITY_THRESHOLD) -> bool:
    """判断 Pose 某个关键点是否可见。"""
    lm = pose_landmarks.landmark[landmark_enum.value]
    return getattr(lm, "visibility", 1.0) >= threshold


def _has_required_shoulders(pose_results, mp_pose, swap_pose_lr: bool = SWAP_POSE_LR) -> bool:
    """判断左右肩是否满足采集要求。"""
    if pose_results is None or pose_results.pose_landmarks is None:
        return False

    pose_lr = resolve_pose_lr(mp_pose.PoseLandmark, swap_pose_lr)
    pose_landmarks = pose_results.pose_landmarks

    return (
        _pose_landmark_visible(pose_landmarks, pose_lr["LEFT_SHOULDER"])
        and _pose_landmark_visible(pose_landmarks, pose_lr["RIGHT_SHOULDER"])
    )


def extract_raw_mediapipe_frame(hand_results,
                                pose_results,
                                mp_pose,
                                timestamp_ms: float,
                                frame_width: int,
                                frame_height: int,
                                swap_handedness: bool = SWAP_HANDEDNESS,
                                swap_pose_lr: bool = SWAP_POSE_LR):
    """提取单帧 MediaPipe 原始数据。

    返回：
        dict 或 None。

    如果当前帧不满足 raw 采集要求，则返回 None。
    """

    hand_landmarks_xyzn = np.zeros((2, 21, 4), dtype=np.float32)
    hand_world_landmarks_xyz = np.zeros((2, 21, 3), dtype=np.float32)
    hand_scores = np.zeros((2,), dtype=np.float32)
    hand_present = np.zeros((2,), dtype=np.float32)

    pose_landmarks_xyzc = np.zeros((33, 4), dtype=np.float32)
    pose_present = np.array(0, dtype=np.float32)

    # 1. Hands 原始结果
    if (
        hand_results is not None
        and hand_results.multi_hand_landmarks
        and hand_results.multi_hand_world_landmarks
        and hand_results.multi_handedness
    ):
        hand_count = min(
            len(hand_results.multi_hand_landmarks),
            len(hand_results.multi_hand_world_landmarks),
            len(hand_results.multi_handedness),
        )

        for index in range(hand_count):
            hand_landmarks = hand_results.multi_hand_landmarks[index]
            hand_world_landmarks = hand_results.multi_hand_world_landmarks[index]
            handedness = hand_results.multi_handedness[index].classification[0]

            label = normalize_handedness(handedness.label, swap_handedness)
            score = float(handedness.score)

            if label == "Left":
                slot = 0
            elif label == "Right":
                slot = 1
            else:
                continue

            if score <= hand_scores[slot]:
                continue

            hand_scores[slot] = score
            hand_present[slot] = 1.0

            for lm_index, lm in enumerate(hand_landmarks.landmark):
                hand_landmarks_xyzn[slot, lm_index, 0] = float(lm.x)
                hand_landmarks_xyzn[slot, lm_index, 1] = float(lm.y)
                hand_landmarks_xyzn[slot, lm_index, 2] = float(lm.z)
                hand_landmarks_xyzn[slot, lm_index, 3] = score

            for lm_index, lm in enumerate(hand_world_landmarks.landmark):
                hand_world_landmarks_xyz[slot, lm_index, 0] = float(lm.x)
                hand_world_landmarks_xyz[slot, lm_index, 1] = float(lm.y)
                hand_world_landmarks_xyz[slot, lm_index, 2] = float(lm.z)

    # 2. Pose 原始结果
    pose_landmarks_xyzc_raw = pose_landmarks_xyzc.copy()

    if pose_results is not None and pose_results.pose_landmarks is not None:
        pose_present = np.array(1, dtype=np.float32)

        for lm_index, lm in enumerate(pose_results.pose_landmarks.landmark):
            pose_landmarks_xyzc[lm_index, 0] = float(lm.x)
            pose_landmarks_xyzc[lm_index, 1] = float(lm.y)
            pose_landmarks_xyzc[lm_index, 2] = float(lm.z)
            pose_landmarks_xyzc[lm_index, 3] = float(getattr(lm, "visibility", 1.0))

        # 注意：必须等 33 个点全部填完后，再复制 raw。
        pose_landmarks_xyzc_raw = pose_landmarks_xyzc.copy()

        # 前端发送镜像自拍图时，保存前把 Pose 规范化成项目内部真实身体坐标系。
        # 只允许执行一次，不能放在 for 循环里。
        if MIRROR_POSE_X:
            pose_landmarks_xyzc = normalize_mirrored_pose_xyzc(pose_landmarks_xyzc)

    # 3. 有效帧判定
    if RAW_REQUIRE_HAND and float(np.sum(hand_present)) <= 0:
        return None

    if RAW_REQUIRE_SHOULDERS and not _has_required_shoulders(pose_results, mp_pose, swap_pose_lr):
        return None

    return {
        "hand_landmarks_xyzn": hand_landmarks_xyzn,
        "hand_world_landmarks_xyz": hand_world_landmarks_xyz,
        "hand_scores": hand_scores,
        "hand_present": hand_present,

        # 注意：这里保存的是已经规范化后的 Pose。
        "pose_landmarks_xyzc": pose_landmarks_xyzc,

        # 可选保留原始 Pose，方便以后排查。
        "pose_landmarks_xyzc_raw": pose_landmarks_xyzc_raw,

        "pose_present": pose_present,
        "timestamp_ms": np.array(timestamp_ms, dtype=np.float64),
        "frame_width_height": np.array([frame_width, frame_height], dtype=np.int32),
        "pose_normalized": np.array(1 if MIRROR_POSE_X else 0, dtype=np.int32),
    }