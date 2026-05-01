"""raw MediaPipe dataset 扫描工具。

用于管理端样本同步：
1. 扫描 dataset_raw_phone_10fps 目录；
2. 读取每个 .npz 样本；
3. 计算帧数、时长、FPS、hand_present 比例、pose_present 比例；
4. 输出 Spring Boot 可同步入库的样本摘要。
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def scan_raw_dataset(root_dir: Path) -> List[Dict]:
    """扫描 raw dataset 根目录。

    目录结构预期：
    dataset_raw_phone_10fps/
      hello/
        sample_001.npz
        sample_002.npz
      thanks/
        sample_001.npz

    Args:
        root_dir: raw dataset 根目录。

    Returns:
        样本摘要列表。
    """
    if not root_dir.exists():
        return []

    if not root_dir.is_dir():
        return []

    samples: List[Dict] = []

    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name

        for sample_file in sorted(label_dir.glob("sample_*.npz")):
            summary = build_sample_summary(root_dir, label, sample_file)
            if summary is not None:
                samples.append(summary)

    return samples


def build_sample_summary(root_dir: Path, label: str, sample_file: Path) -> Optional[Dict]:
    """读取单个 npz 文件并生成样本摘要。

    Args:
        root_dir: raw dataset 根目录。
        label: 样本标签，也是 resourceCode 第一版默认值。
        sample_file: 样本 npz 文件路径。

    Returns:
        样本摘要；读取失败时返回 None。
    """
    try:
        with np.load(sample_file, allow_pickle=True) as data:
            hand_present = data.get("hand_present")
            pose_present = data.get("pose_present")
            timestamps_relative_ms = data.get("timestamps_relative_ms")
            pose_normalized = data.get("pose_normalized")
            label_value = data.get("label")

            frame_count = infer_frame_count(data, hand_present, pose_present)
            duration_ms = infer_duration_ms(timestamps_relative_ms, frame_count)
            fps = infer_fps(frame_count, duration_ms)

            hand_present_ratio = infer_present_ratio(hand_present)
            pose_present_ratio = infer_present_ratio(pose_present)
            pose_normalized_value = bool(int(pose_normalized)) if pose_normalized is not None else False

            actual_label = normalize_np_label(label_value, label)
            quality_status, quality_message = evaluate_quality(
                frame_count=frame_count,
                hand_present_ratio=hand_present_ratio,
                pose_present_ratio=pose_present_ratio,
                pose_normalized=pose_normalized_value,
            )

            relative_path = sample_file.relative_to(root_dir.parent).as_posix()
            sample_code = build_sample_code(actual_label, sample_file)

            return {
                "sampleCode": sample_code,
                "resourceCode": actual_label,
                "label": actual_label,
                "rawFilePath": relative_path,
                "frameCount": frame_count,
                "durationMs": duration_ms,
                "fps": fps,
                "handPresentRatio": hand_present_ratio,
                "posePresentRatio": pose_present_ratio,
                "poseNormalized": pose_normalized_value,
                "qualityStatus": quality_status,
                "qualityMessage": quality_message,
            }
    except Exception as ex:
        print(f"[raw_dataset_scanner] scan failed: {sample_file}, error={ex}")
        return None


def build_sample_code(label: str, sample_file: Path) -> str:
    """构造样本唯一编码。

    Args:
        label: 标签。
        sample_file: 样本文件。

    Returns:
        样本唯一编码。
    """
    return f"{label}_{sample_file.stem}"


def infer_frame_count(data, hand_present, pose_present) -> int:
    """推断样本帧数。"""
    if hand_present is not None:
        return int(hand_present.shape[0])

    if pose_present is not None:
        return int(pose_present.shape[0])

    for key in data.files:
        value = data.get(key)
        if hasattr(value, "shape") and len(value.shape) > 0:
            return int(value.shape[0])

    return 0


def infer_duration_ms(timestamps_relative_ms, frame_count: int) -> int:
    """推断样本时长。"""
    if timestamps_relative_ms is not None and len(timestamps_relative_ms) > 0:
        return int(round(float(timestamps_relative_ms[-1])))

    if frame_count <= 1:
        return 0

    return int(round((frame_count - 1) * 100))


def infer_fps(frame_count: int, duration_ms: int) -> float:
    """推断样本 FPS。"""
    if frame_count <= 1 or duration_ms <= 0:
        return 0.0

    return round(frame_count * 1000.0 / duration_ms, 2)


def infer_present_ratio(present_array) -> float:
    """计算 present 比例。

    hand_present / pose_present 通常是 shape=(frames,) 或 shape=(frames, ...)
    这里统一按每帧是否存在有效值计算。
    """
    if present_array is None:
        return 0.0

    arr = np.asarray(present_array)

    if arr.ndim == 1:
        frame_present = arr > 0
    else:
        frame_present = np.any(arr > 0, axis=tuple(range(1, arr.ndim)))

    if frame_present.size == 0:
        return 0.0

    return round(float(np.mean(frame_present)), 4)


def normalize_np_label(label_value, fallback: str) -> str:
    """从 npz 中读取 label，并归一化为字符串。"""
    if label_value is None:
        return fallback

    try:
        if isinstance(label_value, np.ndarray):
            return str(label_value.item())
        return str(label_value)
    except Exception:
        return fallback


def evaluate_quality(
    frame_count: int,
    hand_present_ratio: float,
    pose_present_ratio: float,
    pose_normalized: bool,
):
    """评估样本质量。

    第一版规则保持简单，避免过度设计。
    """
    if frame_count < 15:
        return "BAD", "样本帧数过少"

    if hand_present_ratio < 0.5:
        return "BAD", "手部关键点缺失过多"

    if frame_count < 20:
        return "WARNING", "有效帧偏少，建议补充采集"

    if hand_present_ratio < 0.8:
        return "WARNING", "手部检测比例偏低"

    if pose_present_ratio < 0.6:
        return "WARNING", "人体姿态检测比例偏低"

    if not pose_normalized:
        return "WARNING", "Pose 未归一化"

    return "GOOD", "样本质量良好"