# -*- coding: utf-8 -*-
"""
检查 WLASL-mini 视频的 MediaPipe 可见性。

输入：
- D:/datasets/WLASL-mini/samples.csv

输出：
- mediapipe_visibility_by_sample.csv
- mediapipe_visibility_overall.json
- mediapipe_overlay_preview.jpg

检查指标：
- pose_ratio：抽样帧中检测到 Pose 的比例
- any_hand_ratio：抽样帧中至少检测到一只手的比例
- both_hand_ratio：抽样帧中同时检测到两只手的比例
- avg_hand_count：平均每帧检测到的手数量

用途：
在正式提特征和训练前，确认 WLASL 视频是否适合当前 MediaPipe 关键点方案。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    读取 samples.csv。
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """
    写出 CSV。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def save_json(path: Path, data: Dict[str, object]) -> None:
    """
    写出 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def get_video_info(video_path: Path) -> Dict[str, object]:
    """
    获取视频基础信息。
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "opened": False,
            "frame_width": 0,
            "frame_height": 0,
            "fps": 0.0,
            "total_frame_count": 0,
        }

    info = {
        "opened": True,
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "total_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    cap.release()
    return info


def make_sample_indices(total_frames: int, frames_per_sample: int) -> List[int]:
    """
    生成抽帧下标。

    例如总帧 50，抽 6 帧，会在首尾之间均匀取样。
    """
    if total_frames <= 0:
        return []

    if total_frames <= frames_per_sample:
        return list(range(total_frames))

    indices = np.linspace(0, total_frames - 1, frames_per_sample)
    return sorted(set(int(round(x)) for x in indices))


def read_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    """
    读取指定帧。
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def process_frame(holistic, frame_bgr: np.ndarray):
    """
    对单帧执行 MediaPipe Holistic。
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    result = holistic.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return result


def draw_overlay(frame_bgr: np.ndarray, result, title: str) -> np.ndarray:
    """
    绘制 MediaPipe 检测结果预览。
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    image = frame_bgr.copy()

    if result.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
        )

    if result.left_hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            result.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

    if result.right_hand_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            result.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )

    cv2.putText(
        image,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def resize_for_grid(image: np.ndarray, width: int = 260, height: int = 190) -> np.ndarray:
    """
    缩放到预览网格尺寸。
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_preview_grid(images: List[np.ndarray], cols: int = 4) -> Optional[np.ndarray]:
    """
    拼接预览网格。
    """
    if not images:
        return None

    rows = []

    for start in range(0, len(images), cols):
        chunk = images[start:start + cols]

        while len(chunk) < cols:
            blank = np.zeros_like(images[0])
            chunk.append(blank)

        rows.append(np.hstack(chunk))

    return np.vstack(rows)


def check_one_video(
    holistic,
    sample: Dict[str, str],
    frames_per_sample: int,
    preview_images: List[np.ndarray],
    max_preview_images: int,
) -> Dict[str, object]:
    """
    检查单个视频样本。
    """
    video_path = Path(sample["local_path"])
    label = sample["label"]
    sample_id = sample["sample_id"]

    video_info = get_video_info(video_path)

    if not video_info["opened"]:
        return {
            "sample_id": sample_id,
            "label": label,
            "local_path": str(video_path),
            "opened": 0,
            "checked_frames": 0,
            "pose_ratio": 0.0,
            "any_hand_ratio": 0.0,
            "both_hand_ratio": 0.0,
            "left_hand_ratio": 0.0,
            "right_hand_ratio": 0.0,
            "avg_hand_count": 0.0,
            "frame_width": 0,
            "frame_height": 0,
            "fps": 0.0,
            "total_frame_count": 0,
        }

    total_frames = int(video_info["total_frame_count"])
    indices = make_sample_indices(total_frames, frames_per_sample)

    pose_count = 0
    any_hand_count = 0
    both_hand_count = 0
    left_hand_count = 0
    right_hand_count = 0
    hand_count_sum = 0
    checked_frames = 0

    for frame_index in indices:
        frame = read_frame(video_path, frame_index)

        if frame is None:
            continue

        result = process_frame(holistic, frame)

        has_pose = result.pose_landmarks is not None
        has_left = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None

        hand_count = int(has_left) + int(has_right)

        checked_frames += 1
        pose_count += int(has_pose)
        left_hand_count += int(has_left)
        right_hand_count += int(has_right)
        any_hand_count += int(hand_count > 0)
        both_hand_count += int(hand_count == 2)
        hand_count_sum += hand_count

        if len(preview_images) < max_preview_images:
            title = f"{label}_{sample_id}_f{frame_index}"
            overlay = draw_overlay(frame, result, title)
            preview_images.append(resize_for_grid(overlay))

    if checked_frames == 0:
        checked_frames = 1

    return {
        "sample_id": sample_id,
        "label": label,
        "local_path": str(video_path),
        "opened": 1,
        "checked_frames": checked_frames,
        "pose_ratio": round(pose_count / checked_frames, 4),
        "any_hand_ratio": round(any_hand_count / checked_frames, 4),
        "both_hand_ratio": round(both_hand_count / checked_frames, 4),
        "left_hand_ratio": round(left_hand_count / checked_frames, 4),
        "right_hand_ratio": round(right_hand_count / checked_frames, 4),
        "avg_hand_count": round(hand_count_sum / checked_frames, 4),
        "frame_width": video_info["frame_width"],
        "frame_height": video_info["frame_height"],
        "fps": round(float(video_info["fps"]), 4),
        "total_frame_count": video_info["total_frame_count"],
    }


def compute_overall(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """
    汇总总体统计。
    """
    valid_rows = [row for row in rows if int(row["opened"]) == 1]

    if not valid_rows:
        return {
            "sample_count": 0,
            "avg_pose_ratio": 0.0,
            "avg_any_hand_ratio": 0.0,
            "avg_both_hand_ratio": 0.0,
            "avg_left_hand_ratio": 0.0,
            "avg_right_hand_ratio": 0.0,
            "avg_hand_count": 0.0,
        }

    def avg(key: str) -> float:
        return round(
            sum(float(row[key]) for row in valid_rows) / len(valid_rows),
            4,
        )

    return {
        "sample_count": len(valid_rows),
        "avg_pose_ratio": avg("pose_ratio"),
        "avg_any_hand_ratio": avg("any_hand_ratio"),
        "avg_both_hand_ratio": avg("both_hand_ratio"),
        "avg_left_hand_ratio": avg("left_hand_ratio"),
        "avg_right_hand_ratio": avg("right_hand_ratio"),
        "avg_hand_count": avg("avg_hand_count"),
        "decision_hint": {
            "good": "pose_ratio >= 0.85 且 any_hand_ratio >= 0.75，基本可以继续提特征训练",
            "warning": "如果 any_hand_ratio 偏低，优先检查手部是否太小、出画或遮挡",
        },
    }


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        default="D:/datasets/WLASL-mini/samples.csv",
        help="WLASL-mini samples.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini/mediapipe_check",
        help="输出目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最多检查多少个样本；0 表示全部",
    )
    parser.add_argument(
        "--frames_per_sample",
        type=int,
        default=6,
        help="每个视频抽多少帧检查",
    )
    parser.add_argument(
        "--max_preview_images",
        type=int,
        default=40,
        help="最多输出多少张预览图",
    )

    args = parser.parse_args()

    samples_csv = Path(args.samples_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_csv_rows(samples_csv)

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    print("========== 开始检查 WLASL MediaPipe 可见性 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] 样本数：{len(samples)}")
    print(f"[信息] 每个样本抽帧数：{args.frames_per_sample}")

    rows: List[Dict[str, object]] = []
    preview_images: List[np.ndarray] = []

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for index, sample in enumerate(samples, start=1):
            row = check_one_video(
                holistic=holistic,
                sample=sample,
                frames_per_sample=args.frames_per_sample,
                preview_images=preview_images,
                max_preview_images=args.max_preview_images,
            )

            rows.append(row)

            print(
                f"[{index}/{len(samples)}] "
                f"{row['label']} {row['sample_id']} "
                f"pose={row['pose_ratio']} "
                f"any_hand={row['any_hand_ratio']} "
                f"both_hand={row['both_hand_ratio']} "
                f"avg_hand={row['avg_hand_count']}"
            )

    fieldnames = [
        "sample_id",
        "label",
        "local_path",
        "opened",
        "checked_frames",
        "pose_ratio",
        "any_hand_ratio",
        "both_hand_ratio",
        "left_hand_ratio",
        "right_hand_ratio",
        "avg_hand_count",
        "frame_width",
        "frame_height",
        "fps",
        "total_frame_count",
    ]

    write_csv(
        output_dir / "mediapipe_visibility_by_sample.csv",
        rows,
        fieldnames,
    )

    overall = compute_overall(rows)

    save_json(
        output_dir / "mediapipe_visibility_overall.json",
        overall,
    )

    preview_grid = make_preview_grid(preview_images)

    if preview_grid is not None:
        preview_path = output_dir / "mediapipe_overlay_preview.jpg"
        cv2.imwrite(str(preview_path), preview_grid)
        print(f"[完成] 已生成预览图：{preview_path}")

    print("\n========== 总体统计 ==========")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()