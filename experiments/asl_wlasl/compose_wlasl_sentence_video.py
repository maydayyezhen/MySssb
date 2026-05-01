# -*- coding: utf-8 -*-
"""
将 WLASL-mini 的单词视频拼接成一句演示视频。

用途：
1. 用 WLASL 单词视频构造可复现的“句子级”演示输入。
2. 后续手机端可以上传这个 mp4，验证“上传视频 -> 识别结果返回”的闭环。
3. 支持不同分辨率 / 不同宽高比视频，统一输出时保持比例缩放 + padding，不拉伸。

输入：
- D:/datasets/WLASL-mini/samples.csv
- sentence: 例如 friend,meet,today

输出：
- demo mp4
- manifest json
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
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
    """解析逗号分隔的英文 gloss 序列。"""
    labels = [
        item.strip().lower()
        for item in sentence.replace("，", ",").split(",")
        if item.strip()
    ]

    if not labels:
        raise ValueError("sentence 不能为空，例如：friend,meet,today")

    return labels


def group_samples_by_label(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """按 label 分组样本。"""
    result: Dict[str, List[Dict[str, str]]] = {}

    for row in rows:
        label = row["label"].strip().lower()
        result.setdefault(label, []).append(row)

    return result


def choose_sample(
    candidates: List[Dict[str, str]],
    policy: str,
    rng: random.Random,
) -> Dict[str, str]:
    """
    从某个 label 的候选视频里选一个。

    largest:
        优先选文件更大的视频，通常质量/时长更稳定。
    first:
        选第一个。
    random:
        随机选一个。
    """
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
    """读取视频所有帧，并返回原 fps。"""
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


def resize_with_padding(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """
    保持比例缩放 + padding 到固定输出尺寸。

    不做强行拉伸，避免破坏人体和手部比例。
    """
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


def draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    """
    在帧左上角画 label。

    默认不建议开启，因为识别模型不需要文字。
    只用于做展示视频。
    """
    image = frame.copy()

    cv2.putText(
        image,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return image


def make_gap_frames(
    previous_frame: Optional[np.ndarray],
    gap_frames: int,
    mode: str,
    target_width: int,
    target_height: int,
) -> List[np.ndarray]:
    """
    生成单词之间的间隔帧。

    repeat_last:
        重复上一个单词最后一帧，更自然。
    black:
        黑屏间隔，分割明显，但不适合识别模型。
    none:
        不加间隔。
    """
    if gap_frames <= 0 or mode == "none":
        return []

    if mode == "black" or previous_frame is None:
        blank = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        return [blank.copy() for _ in range(gap_frames)]

    if mode == "repeat_last":
        return [previous_frame.copy() for _ in range(gap_frames)]

    raise ValueError(f"未知 gap_mode：{mode}")


def compose_sentence_video(
    samples_csv: Path,
    output_path: Path,
    sentence: str,
    sample_policy: str,
    seed: int,
    fps: float,
    target_width: int,
    target_height: int,
    gap_frames: int,
    gap_mode: str,
    tail_frames: int,
    draw_label_flag: bool,
) -> None:
    """拼接句子视频。"""
    labels = parse_sentence(sentence)

    rows = read_csv_rows(samples_csv)
    grouped = group_samples_by_label(rows)

    rng = random.Random(seed)

    selected_rows = []

    for label in labels:
        if label not in grouped:
            available = sorted(grouped.keys())
            raise RuntimeError(
                f"找不到 label={label}。可用标签：{', '.join(available)}"
            )

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

    manifest_segments = []
    output_frame_index = 0
    previous_frame: Optional[np.ndarray] = None

    print("========== 开始拼接 WLASL 演示视频 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] sentence：{' '.join(labels)}")
    print(f"[信息] output_path：{output_path}")
    print(f"[信息] sample_policy：{sample_policy}")
    print(f"[信息] output fps：{fps}")
    print(f"[信息] output size：{target_width}x{target_height}")
    print(f"[信息] gap：{gap_frames} frames, mode={gap_mode}")
    print(f"[信息] tail：{tail_frames} frames")

    for word_index, row in enumerate(selected_rows):
        label = row["label"]
        video_path = Path(row["local_path"])

        raw_frames, source_fps = read_video_frames(video_path)

        # 单词之间加 gap。
        if word_index > 0:
            for gap_frame in make_gap_frames(
                previous_frame=previous_frame,
                gap_frames=gap_frames,
                mode=gap_mode,
                target_width=target_width,
                target_height=target_height,
            ):
                writer.write(gap_frame)
                output_frame_index += 1

        segment_start = output_frame_index

        print(
            f"[词] {label} <- {video_path.name} "
            f"frames={len(raw_frames)} source_fps={source_fps:.2f}"
        )

        for frame in raw_frames:
            out_frame = resize_with_padding(
                frame,
                target_width=target_width,
                target_height=target_height,
            )

            if draw_label_flag:
                out_frame = draw_label(out_frame, label)

            writer.write(out_frame)
            previous_frame = out_frame
            output_frame_index += 1

        segment_end = output_frame_index - 1

        manifest_segments.append({
            "label": label,
            "sample_id": row["sample_id"],
            "source_path": str(video_path),
            "source_frame_count": len(raw_frames),
            "source_fps": source_fps,
            "output_start_frame": segment_start,
            "output_end_frame": segment_end,
        })

    # 句尾 tail。
    for tail_frame in make_gap_frames(
        previous_frame=previous_frame,
        gap_frames=tail_frames,
        mode="repeat_last",
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
            "gap_frames": gap_frames,
            "gap_mode": gap_mode,
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
        help="WLASL-mini samples.csv 路径",
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
        help="每个词选择哪个样本视频",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--target_width", type=int, default=320)
    parser.add_argument("--target_height", type=int, default=240)
    parser.add_argument("--gap_frames", type=int, default=6)
    parser.add_argument(
        "--gap_mode",
        default="repeat_last",
        choices=["repeat_last", "black", "none"],
    )
    parser.add_argument("--tail_frames", type=int, default=10)
    parser.add_argument(
        "--draw_label",
        action="store_true",
        help="是否在视频上绘制词标签；识别演示默认不要开启",
    )

    args = parser.parse_args()

    compose_sentence_video(
        samples_csv=Path(args.samples_csv),
        output_path=Path(args.output_path),
        sentence=args.sentence,
        sample_policy=args.sample_policy,
        seed=args.seed,
        fps=args.fps,
        target_width=args.target_width,
        target_height=args.target_height,
        gap_frames=args.gap_frames,
        gap_mode=args.gap_mode,
        tail_frames=args.tail_frames,
        draw_label_flag=args.draw_label,
    )


if __name__ == "__main__":
    main()
