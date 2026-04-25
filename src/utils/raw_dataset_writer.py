"""raw MediaPipe dataset 保存工具。"""

from pathlib import Path
from typing import Dict, List

import numpy as np


def get_next_sample_index(save_dir: Path) -> int:
    """获取下一个样本编号。"""
    existing_files = sorted(save_dir.glob("sample_*.npz"))
    if not existing_files:
        return 1

    max_index = 0
    for file in existing_files:
        stem = file.stem
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            max_index = max(max_index, int(parts[1]))

    return max_index + 1


def save_raw_sample(root_dir: Path, label: str, frames: List[Dict]) -> Path:
    """保存一个 raw MediaPipe 样本。

    frames 必须是长度为 30 的 raw frame dict 列表。
    """
    if len(frames) == 0:
        raise ValueError("frames 不能为空。")

    save_dir = root_dir / label
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_index = get_next_sample_index(save_dir)
    save_path = save_dir / f"sample_{sample_index:03d}.npz"

    hand_landmarks_xyzn = np.stack(
        [frame["hand_landmarks_xyzn"] for frame in frames],
        axis=0
    ).astype(np.float32)

    hand_world_landmarks_xyz = np.stack(
        [frame["hand_world_landmarks_xyz"] for frame in frames],
        axis=0
    ).astype(np.float32)

    hand_scores = np.stack(
        [frame["hand_scores"] for frame in frames],
        axis=0
    ).astype(np.float32)

    hand_present = np.stack(
        [frame["hand_present"] for frame in frames],
        axis=0
    ).astype(np.float32)

    pose_landmarks_xyzc = np.stack(
        [frame["pose_landmarks_xyzc"] for frame in frames],
        axis=0
    ).astype(np.float32)

    pose_present = np.stack(
        [frame["pose_present"] for frame in frames],
        axis=0
    ).astype(np.float32)

    timestamps_ms = np.stack(
        [frame["timestamp_ms"] for frame in frames],
        axis=0
    ).astype(np.float32)

    frame_width_height = frames[0]["frame_width_height"].astype(np.int32)

    np.savez_compressed(
        save_path,
        hand_landmarks_xyzn=hand_landmarks_xyzn,
        hand_world_landmarks_xyz=hand_world_landmarks_xyz,
        hand_scores=hand_scores,
        hand_present=hand_present,
        pose_landmarks_xyzc=pose_landmarks_xyzc,
        pose_present=pose_present,
        timestamps_ms=timestamps_ms,
        frame_width_height=frame_width_height,
        label=np.array(label),
    )

    return save_path