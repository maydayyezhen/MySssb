# -*- coding: utf-8 -*-
"""
将 HearBridge-NationalCSL-mini 图片帧数据转换为增强版 20 帧 MediaPipe 特征数据。

输入：
D:/datasets/HearBridge-NationalCSL-mini/
  raw_frames/
  samples.csv

输出示例：
D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus/
  X.npy
  y.npy
  label_map.json
  sample_index.csv
  feature_config.json

增强版特征设计：

一、原始基础特征 166 维：
1. 左手 78 维：
   - 21 个手部关键点相对手腕归一化坐标：63
   - 15 个手部关节角度：15
2. 右手 78 维：
   - 同上
3. Pose 上半身 10 维：
   - 左右腕、左右肘相对肩中心坐标：8
   - 左右肘角度：2

二、新增静态关系特征 28 维：
1. 左手相对身体位置 10 维：
   - palm_center / wrist / index_tip / thumb_tip / middle_tip 相对肩中心 xy
2. 右手相对身体位置 10 维：
   - 同上
3. 双手关系 4 维：
   - 左右 palm 距离
   - 左右 wrist 距离
   - 左右 palm x 差
   - 左右 palm y 差
4. 手到身体关键点距离 4 维：
   - 左 palm 到 nose
   - 右 palm 到 nose
   - 左 palm 到 shoulder_center
   - 右 palm 到 shoulder_center

三、新增动态差分特征 38 维：
1. Pose 10 维的逐帧差分
2. 静态关系特征 28 维的逐帧差分

最终每帧维度：
166 + 28 + 38 = 232

注意：
1. 默认不交换 handedness，保持当前最佳主线一致。
2. 如果后续要做对照，可加 --swap_handedness。
3. 本脚本不覆盖原 build_features_20f.py。
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
    读取 samples.csv。
    """
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_image_frames(frame_dir: Path) -> List[Path]:
    """
    获取样本目录下的所有图片帧。
    """
    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def read_image_bgr_unicode(image_path: Path) -> Optional[np.ndarray]:
    """
    读取可能包含中文路径的图片。

    Windows 下 cv2.imread(str(path)) 遇到中文路径可能返回 None，
    所以使用 np.fromfile + cv2.imdecode。
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


def empty_hand_feature() -> np.ndarray:
    """
    返回单手缺失时的 78 维零特征。
    """
    return np.zeros((78,), dtype=np.float32)


def empty_hand_relation_feature() -> np.ndarray:
    """
    返回单手相对身体位置缺失时的 10 维零特征。
    """
    return np.zeros((10,), dtype=np.float32)


def extract_single_hand_feature(hand_landmarks) -> np.ndarray:
    """
    提取单只手 78 维基础特征。

    结构：
    1. 21 个手部关键点，以 wrist 为原点做相对坐标归一化：63 维
    2. 15 个关节角度：15 维
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    wrist = points[0].copy()
    relative_points = points - wrist

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


def get_pose_point(pose_landmarks, index: int) -> np.ndarray:
    """
    获取 Pose 中指定点的 x/y/z 坐标。
    """
    lm = pose_landmarks.landmark[index]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_pose_context(pose_landmarks) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    提取 Pose 上半身 10 维特征，并返回用于增强特征的身体参考点。

    返回：
    - pose_feature: 10维
    - context: shoulder_center / shoulder_width / nose 等参考点
    """
    zero_context = {
        "pose_present": False,
        "left_shoulder": np.zeros((3,), dtype=np.float32),
        "right_shoulder": np.zeros((3,), dtype=np.float32),
        "left_elbow": np.zeros((3,), dtype=np.float32),
        "right_elbow": np.zeros((3,), dtype=np.float32),
        "left_wrist": np.zeros((3,), dtype=np.float32),
        "right_wrist": np.zeros((3,), dtype=np.float32),
        "nose": np.zeros((3,), dtype=np.float32),
        "shoulder_center": np.zeros((3,), dtype=np.float32),
        "shoulder_width": np.array(1.0, dtype=np.float32),
    }

    if pose_landmarks is None:
        return np.zeros((10,), dtype=np.float32), zero_context

    left_shoulder = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.RIGHT_SHOULDER.value)
    left_elbow = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.LEFT_ELBOW.value)
    right_elbow = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.RIGHT_ELBOW.value)
    left_wrist = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.LEFT_WRIST.value)
    right_wrist = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.RIGHT_WRIST.value)
    nose = get_pose_point(pose_landmarks, MP_POSE.PoseLandmark.NOSE.value)

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

    pose_feature = np.array(
        relative_xy + [left_elbow_angle, right_elbow_angle],
        dtype=np.float32,
    )

    context = {
        "pose_present": True,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "nose": nose,
        "shoulder_center": shoulder_center,
        "shoulder_width": np.array(shoulder_width, dtype=np.float32),
    }

    return pose_feature, context


def hand_points_to_array(hand_landmarks) -> np.ndarray:
    """
    将 MediaPipe hand_landmarks 转为 shape=(21, 3) 的数组。
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def calc_relative_xy_to_body(
    point: np.ndarray,
    shoulder_center: np.ndarray,
    shoulder_width: float,
) -> List[float]:
    """
    计算某个手部点相对肩中心的 xy 坐标，并除以肩宽。
    """
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    relative = (point[:2] - shoulder_center[:2]) / shoulder_width

    return [float(relative[0]), float(relative[1])]


def extract_hand_relation_feature(
    hand_landmarks,
    pose_context: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    提取单只手相对身体的 10 维位置特征。

    使用点：
    - palm_center：0,5,9,13,17 平均
    - wrist：0
    - index_tip：8
    - thumb_tip：4
    - middle_tip：12

    每个点相对肩中心 xy，除以肩宽。
    5 个点 * 2 = 10维。

    同时返回 hand_context，用于双手关系和手到鼻子距离。
    """
    if hand_landmarks is None or not bool(pose_context.get("pose_present", False)):
        return empty_hand_relation_feature(), {
            "present": False,
            "palm_center": np.zeros((3,), dtype=np.float32),
            "wrist": np.zeros((3,), dtype=np.float32),
        }

    points = hand_points_to_array(hand_landmarks)

    palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
    wrist = points[0]
    index_tip = points[8]
    thumb_tip = points[4]
    middle_tip = points[12]

    shoulder_center = pose_context["shoulder_center"]
    shoulder_width = float(pose_context["shoulder_width"])

    selected_points = [
        palm_center,
        wrist,
        index_tip,
        thumb_tip,
        middle_tip,
    ]

    features: List[float] = []

    for point in selected_points:
        features.extend(
            calc_relative_xy_to_body(
                point=point,
                shoulder_center=shoulder_center,
                shoulder_width=shoulder_width,
            )
        )

    hand_context = {
        "present": True,
        "palm_center": palm_center,
        "wrist": wrist,
    }

    return np.array(features, dtype=np.float32), hand_context


def calc_xy_distance_normalized(
    a: np.ndarray,
    b: np.ndarray,
    scale: float,
) -> float:
    """
    计算 xy 距离并按 scale 归一化。
    """
    if scale < 1e-6:
        scale = 1.0

    return float(np.linalg.norm(a[:2] - b[:2]) / scale)


def extract_inter_hand_and_body_distance_feature(
    left_hand_context: Dict[str, np.ndarray],
    right_hand_context: Dict[str, np.ndarray],
    pose_context: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    提取 8 维关系特征：

    双手关系 4 维：
    1. 左右 palm 距离
    2. 左右 wrist 距离
    3. 左右 palm x 差 / 肩宽
    4. 左右 palm y 差 / 肩宽

    手到身体关键点距离 4 维：
    5. 左 palm 到 nose
    6. 右 palm 到 nose
    7. 左 palm 到 shoulder_center
    8. 右 palm 到 shoulder_center
    """
    if not bool(pose_context.get("pose_present", False)):
        return np.zeros((8,), dtype=np.float32)

    shoulder_width = float(pose_context["shoulder_width"])
    shoulder_center = pose_context["shoulder_center"]
    nose = pose_context["nose"]

    left_present = bool(left_hand_context.get("present", False))
    right_present = bool(right_hand_context.get("present", False))

    left_palm = left_hand_context.get("palm_center", np.zeros((3,), dtype=np.float32))
    right_palm = right_hand_context.get("palm_center", np.zeros((3,), dtype=np.float32))

    left_wrist = left_hand_context.get("wrist", np.zeros((3,), dtype=np.float32))
    right_wrist = right_hand_context.get("wrist", np.zeros((3,), dtype=np.float32))

    if left_present and right_present:
        palm_distance = calc_xy_distance_normalized(left_palm, right_palm, shoulder_width)
        wrist_distance = calc_xy_distance_normalized(left_wrist, right_wrist, shoulder_width)
        palm_dx = float((left_palm[0] - right_palm[0]) / shoulder_width)
        palm_dy = float((left_palm[1] - right_palm[1]) / shoulder_width)
    else:
        palm_distance = 0.0
        wrist_distance = 0.0
        palm_dx = 0.0
        palm_dy = 0.0

    left_palm_to_nose = (
        calc_xy_distance_normalized(left_palm, nose, shoulder_width)
        if left_present else 0.0
    )
    right_palm_to_nose = (
        calc_xy_distance_normalized(right_palm, nose, shoulder_width)
        if right_present else 0.0
    )
    left_palm_to_center = (
        calc_xy_distance_normalized(left_palm, shoulder_center, shoulder_width)
        if left_present else 0.0
    )
    right_palm_to_center = (
        calc_xy_distance_normalized(right_palm, shoulder_center, shoulder_width)
        if right_present else 0.0
    )

    return np.array(
        [
            palm_distance,
            wrist_distance,
            palm_dx,
            palm_dy,
            left_palm_to_nose,
            right_palm_to_nose,
            left_palm_to_center,
            right_palm_to_center,
        ],
        dtype=np.float32,
    )


def extract_frame_feature_plus(
    image_bgr: np.ndarray,
    pose_model,
    hands_model,
    swap_handedness: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    对单帧图片提取增强版静态特征。

    静态输出维度：
    - 基础 166
    - 新增静态关系 28
    合计 194

    动态差分在序列重采样后统一追加。
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    pose_result = pose_model.process(image_rgb)
    hands_result = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    left_hand_feature = empty_hand_feature()
    right_hand_feature = empty_hand_feature()

    left_hand_relation_feature = empty_hand_relation_feature()
    right_hand_relation_feature = empty_hand_relation_feature()

    left_hand_landmarks = None
    right_hand_landmarks = None

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

            if label == "Left":
                left_hand_feature = hand_feature
                left_hand_landmarks = hand_landmarks
                left_present = True
            elif label == "Right":
                right_hand_feature = hand_feature
                right_hand_landmarks = hand_landmarks
                right_present = True

    pose_feature, pose_context = extract_pose_context(pose_result.pose_landmarks)

    left_hand_relation_feature, left_hand_context = extract_hand_relation_feature(
        hand_landmarks=left_hand_landmarks,
        pose_context=pose_context,
    )
    right_hand_relation_feature, right_hand_context = extract_hand_relation_feature(
        hand_landmarks=right_hand_landmarks,
        pose_context=pose_context,
    )

    inter_hand_and_body_feature = extract_inter_hand_and_body_distance_feature(
        left_hand_context=left_hand_context,
        right_hand_context=right_hand_context,
        pose_context=pose_context,
    )

    relation_feature = np.concatenate(
        [
            left_hand_relation_feature,
            right_hand_relation_feature,
            inter_hand_and_body_feature,
        ],
        axis=0,
    ).astype(np.float32)

    static_feature = np.concatenate(
        [
            left_hand_feature,
            right_hand_feature,
            pose_feature,
            relation_feature,
        ],
        axis=0,
    ).astype(np.float32)

    stats = {
        "pose_present": pose_result.pose_landmarks is not None,
        "left_hand_present": left_present,
        "right_hand_present": right_present,
        "any_hand_present": left_present or right_present,
        "both_hands_present": left_present and right_present,
        "static_feature_dim": int(static_feature.shape[0]),
    }

    return static_feature, stats


def temporal_resample_features(features: np.ndarray, target_frames: int) -> np.ndarray:
    """
    将任意长度的特征序列重采样为固定帧数。

    输入：
    features: shape=(T, D)

    输出：
    shape=(target_frames, D)
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


def append_dynamic_features(fixed_static_features: np.ndarray) -> np.ndarray:
    """
    在重采样后的静态特征后追加动态差分特征。

    输入：
    fixed_static_features: shape=(T, 194)

    动态特征：
    - Pose 10 维：索引 156:166
    - relation 28 维：索引 166:194

    动态维度：10 + 28 = 38

    输出：
    shape=(T, 232)
    """
    if fixed_static_features.ndim != 2:
        raise ValueError(f"fixed_static_features 必须是二维数组，当前 shape={fixed_static_features.shape}")

    pose_slice = fixed_static_features[:, 156:166]
    relation_slice = fixed_static_features[:, 166:194]

    dynamic_source = np.concatenate(
        [pose_slice, relation_slice],
        axis=1,
    ).astype(np.float32)

    delta = np.zeros_like(dynamic_source, dtype=np.float32)
    delta[1:] = dynamic_source[1:] - dynamic_source[:-1]

    final_features = np.concatenate(
        [fixed_static_features, delta],
        axis=1,
    ).astype(np.float32)

    return final_features


def build_label_map(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """
    根据样本 label 构建 label -> id 映射。
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
    2. 每帧提取增强静态特征
    3. 重采样到 target_frames
    4. 追加动态差分特征
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

        feature, stats = extract_frame_feature_plus(
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

    static_feature_array = np.stack(frame_features, axis=0).astype(np.float32)

    fixed_static_features = temporal_resample_features(
        features=static_feature_array,
        target_frames=target_frames,
    )

    final_features = append_dynamic_features(fixed_static_features)

    result_info["used_frame_count"] = len(frame_features)

    return final_features, result_info


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


def build_features_plus(
    samples_csv: Path,
    output_dir: Path,
    target_frames: int,
    min_required_frames: int,
    max_samples: Optional[int],
    min_detection_confidence: float,
    swap_handedness: bool,
) -> None:
    """
    构建增强版 20 帧特征数据集。
    """
    rows = read_samples(samples_csv)
    rows = [row for row in rows if row.get("status") == "ok"]

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    label_map = build_label_map(rows)

    X_list = []
    y_list = []
    index_rows = []

    print("========== 开始构建 NationalCSL-DP 20帧增强特征 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] 样本数：{len(rows)}")
    print(f"[信息] 类别数：{len(label_map)}")
    print(f"[信息] label_map：{label_map}")
    print(f"[信息] target_frames：{target_frames}")
    print(f"[信息] min_required_frames：{min_required_frames}")
    print(f"[信息] swap_handedness：{swap_handedness}")

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
            features, result_info = process_one_sample(
                row=row,
                pose_model=pose_model,
                hands_model=hands_model,
                target_frames=target_frames,
                min_required_frames=min_required_frames,
                swap_handedness=swap_handedness,
            )

            label = row["label"]

            if features is not None:
                X_list.append(features)
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
            "swap_handedness": swap_handedness,
            "swap_pose_lr": False,
            "mirror_pose_x": False,
            "view": "front",
            "temporal_resample": "uniform_round_indices",
        },
        "feature_layout": {
            "base_166": {
                "left_hand": "0:78",
                "right_hand": "78:156",
                "pose_upper_body": "156:166",
            },
            "static_relation_28": {
                "left_hand_body_relation": "166:176",
                "right_hand_body_relation": "176:186",
                "inter_hand_and_body_distance": "186:194",
            },
            "dynamic_38": {
                "delta_pose_upper_body": "194:204",
                "delta_static_relation": "204:232",
            },
            "total_dim": int(X.shape[-1]),
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
        help="增强特征输出目录，例如 D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus",
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
        help="是否交换 MediaPipe Hands 的 Left/Right 标签。默认关闭，保持当前最佳主线一致。",
    )

    args = parser.parse_args()

    build_features_plus(
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