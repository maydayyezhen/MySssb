# -*- coding: utf-8 -*-
"""
检查 MediaPipe Hands handedness 与 Pose wrist 是否一致。

判断逻辑：
1. 对每帧运行 MediaPipe Pose + Hands
2. 获取 Pose LEFT_WRIST / RIGHT_WRIST
3. 对每只 Hands，取 hand_landmarks[0] 作为手腕点
4. 判断该手腕更靠近 Pose LEFT_WRIST 还是 Pose RIGHT_WRIST
5. 比较 Hands 输出的 Left/Right 与最近 Pose wrist 是否一致

如果大量出现：
Hands Left 更靠近 Pose RIGHT_WRIST
Hands Right 更靠近 Pose LEFT_WRIST

说明当前数据需要 swap_handedness=True。
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import mediapipe as mp
import numpy as np


MP_HANDS = mp.solutions.hands
MP_POSE = mp.solutions.pose


def read_image_bgr_unicode(image_path: Path) -> Optional[np.ndarray]:
    """
    读取可能包含中文路径的图片。
    """
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        if image_bytes.size == 0:
            return None
        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[错误] 读取图片失败：{image_path}，原因：{e}")
        return None


def list_image_frames(frame_dir: Path) -> List[Path]:
    """
    获取图片帧列表。
    """
    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def sample_evenly(items: List[Path], max_count: int) -> List[Path]:
    """
    均匀抽帧。
    """
    if len(items) <= max_count:
        return items

    indices = [
        round(i * (len(items) - 1) / (max_count - 1))
        for i in range(max_count)
    ]

    return [items[index] for index in indices]


def calc_dist_xy(a, b) -> float:
    """
    计算归一化坐标下的 xy 距离。
    """
    dx = float(a.x - b.x)
    dy = float(a.y - b.y)
    return float((dx * dx + dy * dy) ** 0.5)


def check_one_frame(image_path: Path, pose_model, hands_model) -> List[Dict[str, object]]:
    """
    检查单帧中的 Hands handedness 是否和 Pose wrist 对得上。
    """
    image_bgr = read_image_bgr_unicode(image_path)
    if image_bgr is None:
        return []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    pose_result = pose_model.process(image_rgb)
    hands_result = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    if pose_result.pose_landmarks is None:
        return []

    if not hands_result.multi_hand_landmarks or not hands_result.multi_handedness:
        return []

    pose_left_wrist = pose_result.pose_landmarks.landmark[
        MP_POSE.PoseLandmark.LEFT_WRIST.value
    ]
    pose_right_wrist = pose_result.pose_landmarks.landmark[
        MP_POSE.PoseLandmark.RIGHT_WRIST.value
    ]

    rows = []

    for hand_index, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
        if hand_index >= len(hands_result.multi_handedness):
            continue

        hand_wrist = hand_landmarks.landmark[0]
        hand_label = hands_result.multi_handedness[hand_index].classification[0].label
        hand_score = hands_result.multi_handedness[hand_index].classification[0].score

        dist_to_pose_left = calc_dist_xy(hand_wrist, pose_left_wrist)
        dist_to_pose_right = calc_dist_xy(hand_wrist, pose_right_wrist)

        nearest_pose_side = "Left" if dist_to_pose_left < dist_to_pose_right else "Right"
        is_consistent = hand_label == nearest_pose_side

        rows.append({
            "frame": image_path.name,
            "hand_label": hand_label,
            "hand_score": f"{hand_score:.4f}",
            "nearest_pose_side": nearest_pose_side,
            "dist_to_pose_left": f"{dist_to_pose_left:.6f}",
            "dist_to_pose_right": f"{dist_to_pose_right:.6f}",
            "is_consistent": int(is_consistent),
        })

    return rows


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写 CSV。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("[警告] 没有可写出的检查结果")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出明细：{output_path}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_dir",
        required=True,
        help="样本帧目录，例如 D:/datasets/HearBridge-NationalCSL-mini/raw_frames/你__1925/Participant_10",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=15,
        help="最多检查多少帧",
    )

    args = parser.parse_args()

    frame_dir = Path(args.frame_dir)
    image_paths = sample_evenly(list_image_frames(frame_dir), args.max_frames)

    all_rows = []

    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose_model, MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands_model:

        for image_path in image_paths:
            rows = check_one_frame(
                image_path=image_path,
                pose_model=pose_model,
                hands_model=hands_model,
            )
            all_rows.extend(rows)

    write_csv(Path(args.output_csv), all_rows)

    total = len(all_rows)
    consistent = sum(int(row["is_consistent"]) for row in all_rows)
    inconsistent = total - consistent

    print("\n========== handedness 一致性统计 ==========")
    print(f"检查手数量：{total}")
    print(f"一致数量：{consistent}")
    print(f"不一致数量：{inconsistent}")

    if total > 0:
        print(f"一致比例：{consistent / total:.4f}")
        print(f"不一致比例：{inconsistent / total:.4f}")

    if inconsistent > consistent:
        print("\n[结论] 大概率需要 swap_handedness=True")
    else:
        print("\n[结论] 大概率不需要交换 handedness")


if __name__ == "__main__":
    main()