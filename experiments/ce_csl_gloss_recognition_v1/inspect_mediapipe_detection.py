"""
CE-CSL MediaPipe 检测稳定性检查脚本

作用：
1. 读取 processed/train.jsonl、dev.jsonl、test.jsonl 中少量样本。
2. 用 OpenCV 读取视频帧。
3. 每隔若干帧抽样一次，送入 MediaPipe Holistic。
4. 统计手部与人体姿态检测率。

本脚本只检查检测效果，不保存特征，不训练模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp


# 改成你的 CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"

# 每个 split 检查几个样本
SAMPLES_PER_SPLIT = 3

# 每隔多少帧检测一次。30fps 下，每 3 帧约等于 10fps
FRAME_STEP = 3

# 为了提速，检测前把视频宽度缩放到这个尺寸
TARGET_WIDTH = 960


def read_jsonl(path: Path, limit: int) -> List[Dict]:
    """
    读取 jsonl 文件中的前 limit 条样本。

    Args:
        path: jsonl 文件路径。
        limit: 读取数量。

    Returns:
        样本列表。
    """
    samples: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if len(samples) >= limit:
                break

            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def resize_frame(frame):
    """
    按指定宽度等比例缩放视频帧。

    Args:
        frame: OpenCV 读取到的 BGR 图像。

    Returns:
        缩放后的图像。
    """
    height, width = frame.shape[:2]

    if width <= TARGET_WIDTH:
        return frame

    scale = TARGET_WIDTH / width
    new_height = int(height * scale)

    return cv2.resize(frame, (TARGET_WIDTH, new_height))


def inspect_one_video(sample: Dict, holistic) -> None:
    """
    检查单个视频的 MediaPipe 检测情况。

    Args:
        sample: manifest 中的一条样本。
        holistic: MediaPipe Holistic 实例。
    """
    video_path = DATASET_ROOT / sample["videoPath"]

    print("=" * 80)
    print("样本 ID:", sample["sampleId"])
    print("视频路径:", video_path)
    print("中文句子:", sample["chinese"])
    print("Gloss:", "/".join(sample["gloss"]))

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("结果: 视频无法打开")
        return

    total_sampled = 0
    pose_detected = 0
    left_hand_detected = 0
    right_hand_detected = 0
    both_hands_detected = 0
    any_hand_detected = 0

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        frame = resize_frame(frame)

        # OpenCV 是 BGR，MediaPipe 需要 RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb)

        has_pose = results.pose_landmarks is not None
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None

        total_sampled += 1

        if has_pose:
            pose_detected += 1

        if has_left:
            left_hand_detected += 1

        if has_right:
            right_hand_detected += 1

        if has_left or has_right:
            any_hand_detected += 1

        if has_left and has_right:
            both_hands_detected += 1

        frame_index += 1

    cap.release()

    if total_sampled == 0:
        print("结果: 没有抽到有效帧")
        return

    print("抽样帧数:", total_sampled)
    print("Pose 检测率:", round(pose_detected / total_sampled, 3))
    print("左手检测率:", round(left_hand_detected / total_sampled, 3))
    print("右手检测率:", round(right_hand_detected / total_sampled, 3))
    print("任意手检测率:", round(any_hand_detected / total_sampled, 3))
    print("双手同时检测率:", round(both_hands_detected / total_sampled, 3))


def main() -> None:
    """
    主流程。
    """
    print("===== CE-CSL MediaPipe 检测检查开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("processed 目录:", PROCESSED_DIR)

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for split in ["train", "dev", "test"]:
            jsonl_path = PROCESSED_DIR / f"{split}.jsonl"

            print("\n" + "#" * 80)
            print(f"检查 {split}.jsonl 前 {SAMPLES_PER_SPLIT} 条样本")
            print("#" * 80)

            samples = read_jsonl(jsonl_path, limit=SAMPLES_PER_SPLIT)

            for sample in samples:
                inspect_one_video(sample, holistic)

    print("\n===== CE-CSL MediaPipe 检测检查结束 =====")


if __name__ == "__main__":
    main()