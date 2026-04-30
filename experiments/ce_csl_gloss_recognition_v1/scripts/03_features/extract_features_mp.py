"""
CE-CSL Feature V1 多进程特征提取脚本

作用：
1. 读取 processed/train.jsonl、dev.jsonl、test.jsonl。
2. 按 TARGET_FPS 从视频中按时间抽帧。
3. 使用 MediaPipe Holistic 提取 pose / left hand / right hand。
4. 按 FEATURE_SPEC.md 中定义的 CE-CSL Feature V1 生成 166 维单帧特征。
5. 保存为 .npy 文件，形状为 T × 166。
6. 支持多进程。
7. 支持已存在 .npy 自动跳过。
8. 支持 limit 参数，只提取每个 split 前 N 条，方便测速。

示例：
    # 测试 train 前 100 条，单进程
    python experiments/ce_csl_gloss_recognition_v1/extract_features_mp.py --splits train --limit 100 --workers 1 --output-subdir features_bench_w1

    # 测试 train 前 100 条，2 进程
    python experiments/ce_csl_gloss_recognition_v1/extract_features_mp.py --splits train --limit 100 --workers 2 --output-subdir features_bench_w2

    # 全量提取，2 进程
    python experiments/ce_csl_gloss_recognition_v1/extract_features_mp.py --splits train dev test --limit 0 --workers 2 --output-subdir features
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as multiprocessing
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# 1. 全局配置
# =========================================================

# CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# processed 目录
PROCESSED_DIR = DATASET_ROOT / "processed"

# Pose 点位编号：左肩
POSE_LEFT_SHOULDER = 11

# Pose 点位编号：右肩
POSE_RIGHT_SHOULDER = 12

# Pose 点位编号：左肘
POSE_LEFT_ELBOW = 13

# Pose 点位编号：右肘
POSE_RIGHT_ELBOW = 14

# Pose 点位编号：左腕
POSE_LEFT_WRIST = 15

# Pose 点位编号：右腕
POSE_RIGHT_WRIST = 16

# worker 进程内的全局配置
WORKER_CONFIG: Optional[Dict] = None

# worker 进程内独立创建的 MediaPipe Holistic 实例
WORKER_HOLISTIC = None


# =========================================================
# 2. 基础工具函数
# =========================================================

def read_jsonl(path: Path, limit: int = 0) -> List[Dict]:
    """
    读取 jsonl manifest 文件。

    Args:
        path: jsonl 文件路径。
        limit: 读取条数。0 表示读取全部。

    Returns:
        样本列表。
    """
    samples: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if limit > 0 and len(samples) >= limit:
                break

            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def resize_frame(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    """
    等比例缩放视频帧。

    Args:
        frame_bgr: OpenCV 读取到的 BGR 图像。
        target_width: 目标宽度。如果原图宽度小于等于目标宽度，则不缩放。

    Returns:
        缩放后的 BGR 图像。
    """
    height, width = frame_bgr.shape[:2]

    if width <= target_width:
        return frame_bgr

    scale = target_width / width
    new_height = int(height * scale)

    return cv2.resize(frame_bgr, (target_width, new_height))


def distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个三维点在 x/y 平面上的二维欧氏距离。

    Args:
        a: 第一个点，形状为 3。
        b: 第二个点，形状为 3。

    Returns:
        二维距离。
    """
    return float(np.linalg.norm(a[:2] - b[:2]))


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    计算三点夹角，b 为夹角顶点。

    Args:
        a: 第一个点。
        b: 夹角顶点。
        c: 第三个点。

    Returns:
        归一化到 0~1 的角度值。
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

    Args:
        landmark: MediaPipe landmark。
        width: 当前处理帧宽度。
        height: 当前处理帧高度。

    Returns:
        图像坐标点，形状为 3。
    """
    return np.array(
        [
            landmark.x * width,
            landmark.y * height,
            landmark.z * width,
        ],
        dtype=np.float32,
    )


# =========================================================
# 3. 手部特征：78 维
# =========================================================

def hand_landmarks_to_points(hand_landmarks, width: int, height: int) -> Optional[np.ndarray]:
    """
    将一只手的 21 个 landmark 转成图像坐标数组。

    Args:
        hand_landmarks: MediaPipe hand landmarks。
        width: 当前处理帧宽度。
        height: 当前处理帧高度。

    Returns:
        检测到手时返回 21 × 3 数组，否则返回 None。
    """
    if hand_landmarks is None:
        return None

    points = [
        landmark_to_img_point(landmark, width, height)
        for landmark in hand_landmarks.landmark
    ]

    return np.stack(points, axis=0).astype(np.float32)


def normalize_hand_points(points: Optional[np.ndarray]) -> np.ndarray:
    """
    对手部 21 点进行相对化与尺度归一化。

    Args:
        points: 21 × 3 的图像坐标点。

    Returns:
        21 × 3 的局部归一化坐标。
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


def extract_finger_angles(normalized_points: Optional[np.ndarray]) -> np.ndarray:
    """
    提取 15 个手指关节角度。

    Args:
        normalized_points: 21 × 3 的局部归一化坐标。

    Returns:
        15 维角度特征。
    """
    if normalized_points is None:
        return np.zeros((15,), dtype=np.float32)

    triplets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
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

    Args:
        hand_landmarks: MediaPipe hand landmarks。
        width: 当前处理帧宽度。
        height: 当前处理帧高度。

    Returns:
        78 维手部特征。
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


# =========================================================
# 4. Pose 手臂特征：8 + 2 维
# =========================================================

def pose_landmarks_to_points(pose_landmarks, width: int, height: int) -> Optional[np.ndarray]:
    """
    将 Pose landmarks 转成图像坐标数组。

    Args:
        pose_landmarks: MediaPipe pose landmarks。
        width: 当前处理帧宽度。
        height: 当前处理帧高度。

    Returns:
        检测到 Pose 时返回 33 × 3 数组，否则返回 None。
    """
    if pose_landmarks is None:
        return None

    points = [
        landmark_to_img_point(landmark, width, height)
        for landmark in pose_landmarks.landmark
    ]

    return np.stack(points, axis=0).astype(np.float32)


def extract_arm_position_8_and_elbow_angle_2(
    pose_landmarks,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Pose 中提取手臂位置 8 维和肘角 2 维。

    Args:
        pose_landmarks: MediaPipe pose landmarks。
        width: 当前处理帧宽度。
        height: 当前处理帧高度。

    Returns:
        arm_position_8 和 elbow_angle_2。
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

    left_elbow_angle = angle_between_points(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_between_points(right_shoulder, right_elbow, right_wrist)

    elbow_angle_2 = np.array(
        [left_elbow_angle, right_elbow_angle],
        dtype=np.float32,
    )

    return arm_position_8, elbow_angle_2


# =========================================================
# 5. 单帧 166 维特征
# =========================================================

def holistic_results_to_feature_166(results, width: int, height: int) -> np.ndarray:
    """
    将 MediaPipe Holistic 单帧检测结果转成 166 维特征。

    Args:
        results: MediaPipe Holistic 检测结果。
        width: 当前处理帧宽度。
        height: 当前处理帧高度。

    Returns:
        166 维单帧特征。
    """
    left_hand_78 = extract_hand_78(results.left_hand_landmarks, width, height)
    right_hand_78 = extract_hand_78(results.right_hand_landmarks, width, height)

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


# =========================================================
# 6. 视频采样与特征提取
# =========================================================

def build_sample_indices(frame_count: int, fps: float, target_fps: float) -> List[int]:
    """
    按目标 FPS 构建采样帧编号。

    Args:
        frame_count: 视频总帧数。
        fps: 视频原始 FPS。
        target_fps: 目标采样 FPS。

    Returns:
        采样帧编号列表。
    """
    if frame_count <= 0:
        return []

    if fps <= 0:
        fps = 30.0

    duration = frame_count / fps
    sample_times = np.arange(0.0, duration, 1.0 / target_fps)

    indices = []

    for time_sec in sample_times:
        index = int(round(time_sec * fps))
        index = max(0, min(index, frame_count - 1))
        indices.append(index)

    result = []
    seen = set()

    for index in indices:
        if index not in seen:
            result.append(index)
            seen.add(index)

    return result


def extract_video_feature(sample: Dict) -> np.ndarray:
    """
    提取单个视频的 T × 166 特征。

    Args:
        sample: manifest 中的一条样本。

    Returns:
        T × 166 特征矩阵。
    """
    if WORKER_CONFIG is None:
        raise RuntimeError("WORKER_CONFIG 未初始化")

    if WORKER_HOLISTIC is None:
        raise RuntimeError("WORKER_HOLISTIC 未初始化")

    dataset_root = Path(WORKER_CONFIG["dataset_root"])
    target_fps = float(WORKER_CONFIG["target_fps"])
    target_width = int(WORKER_CONFIG["target_width"])
    mirror_input = bool(WORKER_CONFIG["mirror_input"])

    video_path = dataset_root / sample["videoPath"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    sample_indices = build_sample_indices(frame_count, fps, target_fps)
    sample_index_set = set(sample_indices)

    features = []
    frame_index = 0

    while True:
        success, frame_bgr = cap.read()

        if not success or frame_bgr is None:
            break

        if frame_index not in sample_index_set:
            frame_index += 1
            continue

        frame_bgr = resize_frame(frame_bgr, target_width)

        if mirror_input:
            frame_bgr = cv2.flip(frame_bgr, 1)

        height, width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = WORKER_HOLISTIC.process(frame_rgb)

        feature_166 = holistic_results_to_feature_166(results, width, height)
        features.append(feature_166)

        frame_index += 1

    cap.release()

    if not features:
        raise RuntimeError(f"没有提取到有效特征：{video_path}")

    return np.stack(features, axis=0).astype(np.float32)


# =========================================================
# 7. 多进程 worker
# =========================================================

def init_worker(config: Dict) -> None:
    """
    初始化 worker 进程。

    每个 worker 独立创建自己的 MediaPipe Holistic 实例，避免跨进程共享模型对象。

    Args:
        config: worker 配置。
    """
    global WORKER_CONFIG
    global WORKER_HOLISTIC

    WORKER_CONFIG = config

    mp_holistic = mp.solutions.holistic

    WORKER_HOLISTIC = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def process_one_sample(sample: Dict) -> Dict:
    """
    处理单个样本。

    Args:
        sample: manifest 样本。

    Returns:
        处理结果字典。
    """
    if WORKER_CONFIG is None:
        raise RuntimeError("WORKER_CONFIG 未初始化")

    output_root = Path(WORKER_CONFIG["output_root"])
    overwrite = bool(WORKER_CONFIG["overwrite"])

    split = sample["split"]
    sample_id = sample["sampleId"]

    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sample_id}.npy"

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "sampleId": sample_id,
            "split": split,
            "shape": None,
            "seconds": 0.0,
            "message": "already exists",
        }

    start_time = time.perf_counter()

    try:
        feature = extract_video_feature(sample)

        temp_path = output_dir / f"{sample_id}.tmp.npy"
        np.save(temp_path, feature)
        temp_path.replace(output_path)

        elapsed = time.perf_counter() - start_time

        return {
            "status": "ok",
            "sampleId": sample_id,
            "split": split,
            "shape": list(feature.shape),
            "seconds": elapsed,
            "message": str(output_path),
        }

    except Exception as exc:
        elapsed = time.perf_counter() - start_time

        return {
            "status": "error",
            "sampleId": sample_id,
            "split": split,
            "shape": None,
            "seconds": elapsed,
            "message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# =========================================================
# 8. 任务构建与主流程
# =========================================================

def build_tasks(splits: List[str], limit: int) -> List[Dict]:
    """
    根据 split 和 limit 构建待处理任务。

    Args:
        splits: 要处理的 split 列表。
        limit: 每个 split 读取条数，0 表示全部。

    Returns:
        样本任务列表。
    """
    tasks: List[Dict] = []

    for split in splits:
        manifest_path = PROCESSED_DIR / f"{split}.jsonl"

        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 manifest 文件：{manifest_path}")

        samples = read_jsonl(manifest_path, limit=limit)
        tasks.extend(samples)

    return tasks


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        命令行参数对象。
    """
    parser = argparse.ArgumentParser(
        description="CE-CSL Feature V1 多进程特征提取脚本"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="要处理的数据划分，可选 train dev test",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="每个 split 处理前 N 条；0 表示全部",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker 进程数，建议先测试 1、2、4",
    )

    parser.add_argument(
        "--output-subdir",
        type=str,
        default="features",
        help="输出到 processed 下的子目录名",
    )

    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="目标采样 FPS",
    )

    parser.add_argument(
        "--target-width",
        type=int,
        default=960,
        help="送入 MediaPipe 前等比例缩放到的目标宽度",
    )

    parser.add_argument(
        "--mirror-input",
        action="store_true",
        help="是否水平翻转输入帧。CE-CSL 当前默认不翻转",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="是否覆盖已存在的 .npy 文件",
    )

    return parser.parse_args()


def main() -> None:
    """
    主入口。
    """
    args = parse_args()

    output_root = PROCESSED_DIR / args.output_subdir
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.splits, args.limit)

    config = {
        "dataset_root": str(DATASET_ROOT),
        "output_root": str(output_root),
        "target_fps": args.target_fps,
        "target_width": args.target_width,
        "mirror_input": args.mirror_input,
        "overwrite": args.overwrite,
    }

    print("===== CE-CSL Feature V1 多进程特征提取开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("输出目录:", output_root)
    print("splits:", args.splits)
    print("limit:", args.limit)
    print("workers:", args.workers)
    print("target_fps:", args.target_fps)
    print("target_width:", args.target_width)
    print("mirror_input:", args.mirror_input)
    print("overwrite:", args.overwrite)
    print("任务总数:", len(tasks))

    start_time = time.perf_counter()

    ok_count = 0
    skipped_count = 0
    error_count = 0
    total_extract_seconds = 0.0

    results = []

    if args.workers <= 1:
        init_worker(config)

        for index, task in enumerate(tasks, start=1):
            result = process_one_sample(task)
            results.append(result)

            if result["status"] == "ok":
                ok_count += 1
                total_extract_seconds += result["seconds"]
            elif result["status"] == "skipped":
                skipped_count += 1
            else:
                error_count += 1

            print_progress(index, len(tasks), result, start_time, ok_count, skipped_count, error_count)

    else:
        context = multiprocessing.get_context("spawn")

        with context.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(config,),
        ) as pool:
            for index, result in enumerate(pool.imap_unordered(process_one_sample, tasks, chunksize=1), start=1):
                results.append(result)

                if result["status"] == "ok":
                    ok_count += 1
                    total_extract_seconds += result["seconds"]
                elif result["status"] == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1

                print_progress(index, len(tasks), result, start_time, ok_count, skipped_count, error_count)

    elapsed_total = time.perf_counter() - start_time

    error_results = [result for result in results if result["status"] == "error"]

    print("\n===== CE-CSL Feature V1 多进程特征提取结束 =====")
    print("任务总数:", len(tasks))
    print("成功:", ok_count)
    print("跳过:", skipped_count)
    print("失败:", error_count)
    print("总耗时秒:", round(elapsed_total, 2))
    print("总耗时分钟:", round(elapsed_total / 60.0, 2))

    if ok_count > 0:
        print("平均每个成功样本耗时秒:", round(total_extract_seconds / ok_count, 3))

    if len(tasks) > 0:
        print("端到端平均每个任务耗时秒:", round(elapsed_total / len(tasks), 3))

    if error_results:
        error_log_path = output_root / "errors.jsonl"

        with error_log_path.open("w", encoding="utf-8") as file:
            for result in error_results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")

        print("错误日志:", error_log_path)


def print_progress(
    index: int,
    total: int,
    result: Dict,
    start_time: float,
    ok_count: int,
    skipped_count: int,
    error_count: int,
) -> None:
    """
    打印处理进度。

    Args:
        index: 当前任务序号。
        total: 总任务数。
        result: 当前任务结果。
        start_time: 总开始时间。
        ok_count: 成功数。
        skipped_count: 跳过数。
        error_count: 失败数。
    """
    elapsed = time.perf_counter() - start_time
    speed = index / elapsed if elapsed > 0 else 0.0
    remaining = total - index
    eta_seconds = remaining / speed if speed > 0 else 0.0

    status = result["status"]
    sample_id = result["sampleId"]
    shape = result["shape"]

    print(
        f"[{index}/{total}] "
        f"{status.upper()} "
        f"{sample_id} "
        f"shape={shape} "
        f"ok={ok_count} skip={skipped_count} err={error_count} "
        f"elapsed={elapsed/60.0:.2f}min "
        f"eta={eta_seconds/60.0:.2f}min"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()