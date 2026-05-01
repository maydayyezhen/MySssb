# -*- coding: utf-8 -*-
"""
将 WLASL-mini 单词视频裁剪动作有效段后，拼接成一句更接近连续输入的演示视频。

核心逻辑：
1. 每个单词视频先用 MediaPipe Holistic 扫描每帧。
2. 找到 any_hand=True 的第一帧和最后一帧。
3. 前后保留 trim_padding 帧，裁出动作窗口。
4. 保持比例缩放 + padding 到统一输出尺寸。
5. 单词之间只保留很短 gap，避免 isolated word 空档过长。

注意：
这仍然不是天然连续手语，只是“裁剪后的标准单词视频拼接演示输入”。
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """读取 CSV。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_json(path: Path, payload: Dict) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def parse_sentence(sentence: str) -> List[str]:
    """解析逗号分隔 gloss。"""
    labels = [
        item.strip().lower()
        for item in sentence.replace("，", ",").split(",")
        if item.strip()
    ]

    if not labels:
        raise ValueError("sentence 不能为空，例如：friend,meet,today")

    return labels


def group_samples_by_label(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """按 label 分组。"""
    grouped: Dict[str, List[Dict[str, str]]] = {}

    for row in rows:
        label = row["label"].strip().lower()
        grouped.setdefault(label, []).append(row)

    return grouped


def choose_sample(
    candidates: List[Dict[str, str]],
    policy: str,
    rng: random.Random,
) -> Dict[str, str]:
    """选择某个 label 的一个视频样本。"""
    if not candidates:
        raise ValueError("候选样本为空")

    if policy == "first":
        return candidates[0]

    if policy == "random":
        return rng.choice(candidates)

    if policy == "largest":
        def size_value(row: Dict[str, str]) -> int:
            try:
                return int(float(row.get("size_bytes", "0") or 0))
            except Exception:
                path = Path(row["local_path"])
                return path.stat().st_size if path.exists() else 0

        return sorted(candidates, key=size_value, reverse=True)[0]

    raise ValueError(f"未知 sample_policy：{policy}")


def read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """读取视频全部帧。"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 25.0

    frames: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"视频没有可读帧：{video_path}")

    return frames, fps


def process_frame(holistic, frame_bgr: np.ndarray):
    """MediaPipe Holistic 处理单帧。"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    result = holistic.process(frame_rgb)
    frame_rgb.flags.writeable = True
    return result


def detect_active_window(
    holistic,
    frames: List[np.ndarray],
    trim_padding: int,
    min_hand_frames: int,
) -> Tuple[int, int, int]:
    """
    根据 any_hand=True 定位动作窗口。

    返回：
    - start
    - end
    - hand_frame_count

    如果检测到手的帧太少，则保留全视频，避免误裁。
    """
    hand_indices: List[int] = []

    for index, frame in enumerate(frames):
        result = process_frame(holistic, frame)

        has_left = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None

        if has_left or has_right:
            hand_indices.append(index)

    if len(hand_indices) < min_hand_frames:
        return 0, len(frames) - 1, len(hand_indices)

    start = max(0, hand_indices[0] - trim_padding)
    end = min(len(frames) - 1, hand_indices[-1] + trim_padding)

    if end <= start:
        return 0, len(frames) - 1, len(hand_indices)

    return start, end, len(hand_indices)


def resize_with_padding(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """保持比例缩放 + padding，不拉伸。"""
    height, width = frame.shape[:2]

    scale = min(target_width / width, target_height / height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x0 = (target_width - new_width) // 2
    y0 = (target_height - new_height) // 2

    canvas[y0:y0 + new_height, x0:x0 + new_width] = resized

    return canvas


def make_repeat_frames(
    frame: Optional[np.ndarray],
    count: int,
    target_width: int,
    target_height: int,
) -> List[np.ndarray]:
    """重复上一帧作为短暂停顿。"""
    if count <= 0:
        return []

    if frame is None:
        blank = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        return [blank.copy() for _ in range(count)]

    return [frame.copy() for _ in range(count)]


def compose_trimmed_sentence_video(
    samples_csv: Path,
    output_path: Path,
    sentence: str,
    sample_policy: str,
    seed: int,
    fps: float,
    target_width: int,
    target_height: int,
    trim_padding: int,
    min_hand_frames: int,
    gap_frames: int,
    tail_frames: int,
) -> None:
    """裁剪并拼接演示句子视频。"""
    labels = parse_sentence(sentence)

    rows = read_csv_rows(samples_csv)
    grouped = group_samples_by_label(rows)

    rng = random.Random(seed)

    selected_rows: List[Dict[str, str]] = []

    for label in labels:
        if label not in grouped:
            raise RuntimeError(f"找不到 label={label}，可用标签：{', '.join(sorted(grouped.keys()))}")

        selected_rows.append(
            choose_sample(
                candidates=grouped[label],
                policy=sample_policy,
                rng=rng,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (target_width, target_height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output_path}")

    print("========== 开始拼接裁剪版 WLASL 演示视频 ==========")
    print(f"[信息] sentence：{' '.join(labels)}")
    print(f"[信息] output_path：{output_path}")
    print(f"[信息] sample_policy：{sample_policy}")
    print(f"[信息] output_size：{target_width}x{target_height}")
    print(f"[信息] fps：{fps}")
    print(f"[信息] trim_padding：{trim_padding}")
    print(f"[信息] min_hand_frames：{min_hand_frames}")
    print(f"[信息] gap_frames：{gap_frames}")
    print(f"[信息] tail_frames：{tail_frames}")

    mp_holistic = mp.solutions.holistic

    manifest_segments = []
    output_frame_index = 0
    previous_frame: Optional[np.ndarray] = None

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for word_index, row in enumerate(selected_rows):
            label = row["label"]
            sample_id = row["sample_id"]
            video_path = Path(row["local_path"])

            frames, source_fps = read_video_frames(video_path)

            start, end, hand_frame_count = detect_active_window(
                holistic=holistic,
                frames=frames,
                trim_padding=trim_padding,
                min_hand_frames=min_hand_frames,
            )

            trimmed_frames = frames[start:end + 1]

            if word_index > 0:
                for gap_frame in make_repeat_frames(
                    frame=previous_frame,
                    count=gap_frames,
                    target_width=target_width,
                    target_height=target_height,
                ):
                    writer.write(gap_frame)
                    output_frame_index += 1

            segment_start = output_frame_index

            print(
                f"[词] {label} {sample_id} <- {video_path.name} "
                f"raw_frames={len(frames)} trim=[{start},{end}] "
                f"trim_frames={len(trimmed_frames)} hand_frames={hand_frame_count}"
            )

            for frame in trimmed_frames:
                out_frame = resize_with_padding(
                    frame,
                    target_width=target_width,
                    target_height=target_height,
                )

                writer.write(out_frame)

                previous_frame = out_frame
                output_frame_index += 1

            segment_end = output_frame_index - 1

            manifest_segments.append({
                "label": label,
                "sample_id": sample_id,
                "source_path": str(video_path),
                "source_fps": source_fps,
                "raw_frame_count": len(frames),
                "trim_start": start,
                "trim_end": end,
                "trim_frame_count": len(trimmed_frames),
                "hand_frame_count": hand_frame_count,
                "output_start_frame": segment_start,
                "output_end_frame": segment_end,
            })

    for tail_frame in make_repeat_frames(
        frame=previous_frame,
        count=tail_frames,
        target_width=target_width,
        target_height=target_height,
    ):
        writer.write(tail_frame)
        output_frame_index += 1

    writer.release()

    manifest_path = output_path.with_suffix(".json")

    save_json(
        manifest_path,
        {
            "sentence": labels,
            "output_path": str(output_path),
            "fps": fps,
            "target_width": target_width,
            "target_height": target_height,
            "sample_policy": sample_policy,
            "trim_padding": trim_padding,
            "min_hand_frames": min_hand_frames,
            "gap_frames": gap_frames,
            "tail_frames": tail_frames,
            "total_output_frames": output_frame_index,
            "segments": manifest_segments,
        },
    )

    print("\n========== 拼接完成 ==========")
    print(f"[完成] 输出视频：{output_path}")
    print(f"[完成] 输出帧数：{output_frame_index}")
    print(f"[完成] manifest：{manifest_path}")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        default="D:/datasets/WLASL-mini/samples.csv",
    )
    parser.add_argument(
        "--sentence",
        required=True,
        help="逗号分隔 gloss，例如：friend,meet,today",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出 mp4 路径",
    )
    parser.add_argument(
        "--sample_policy",
        default="largest",
        choices=["largest", "first", "random"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--target_width", type=int, default=320)
    parser.add_argument("--target_height", type=int, default=240)
    parser.add_argument("--trim_padding", type=int, default=4)
    parser.add_argument("--min_hand_frames", type=int, default=3)
    parser.add_argument("--gap_frames", type=int, default=2)
    parser.add_argument("--tail_frames", type=int, default=6)

    args = parser.parse_args()

    compose_trimmed_sentence_video(
        samples_csv=Path(args.samples_csv),
        output_path=Path(args.output_path),
        sentence=args.sentence,
        sample_policy=args.sample_policy,
        seed=args.seed,
        fps=args.fps,
        target_width=args.target_width,
        target_height=args.target_height,
        trim_padding=args.trim_padding,
        min_hand_frames=args.min_hand_frames,
        gap_frames=args.gap_frames,
        tail_frames=args.tail_frames,
    )


if __name__ == "__main__":
    main()
