# -*- coding: utf-8 -*-
"""
HearBridge-NationalCSL-mini 样本预览脚本。

作用：
1. 读取小数据集 samples.csv
2. 每个样本抽取若干关键帧
3. 生成 contact sheet 预览图
4. 用于检查 NationalCSL-DP 图片方向、裁剪、手部可见性和动作连续性
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont


def read_samples(samples_csv: Path) -> List[Dict[str, str]]:
    """
    读取 samples.csv 样本索引。
    """
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_image_frames(frame_dir: Path) -> List[Path]:
    """
    读取某个样本目录下的图片帧。
    """
    image_paths = sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])
    return image_paths


def sample_key_frames(image_paths: List[Path], max_frames: int) -> List[Path]:
    """
    从完整帧序列中均匀抽取若干关键帧。
    """
    if len(image_paths) <= max_frames:
        return image_paths

    if max_frames <= 1:
        return [image_paths[0]]

    indices = [
        round(i * (len(image_paths) - 1) / (max_frames - 1))
        for i in range(max_frames)
    ]

    return [image_paths[index] for index in indices]


def load_font(font_size: int = 18):
    """
    尝试加载中文字体；失败时使用默认字体。
    """
    candidate_fonts = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for font_path in candidate_fonts:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def build_contact_sheet(
    selected_rows: List[Dict[str, str]],
    output_path: Path,
    max_frames_per_sample: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    """
    生成预览拼图。
    """
    font = load_font(18)

    row_title_height = 32
    gap = 8

    sheet_width = max_frames_per_sample * (thumb_width + gap) + gap
    sheet_height = len(selected_rows) * (thumb_height + row_title_height + gap) + gap

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    y = gap

    for row in selected_rows:
        label = row["label"]
        resource_id = row["resource_id"]
        participant = row["participant"]
        frame_count = row["frame_count"]
        frame_dir = Path(row["frame_dir"])

        title = f"{label}__{resource_id} / {participant} / 帧数={frame_count}"
        draw.text((gap, y), title, fill="black", font=font)

        image_paths = list_image_frames(frame_dir)
        sampled_paths = sample_key_frames(image_paths, max_frames_per_sample)

        x = gap
        image_y = y + row_title_height

        for image_path in sampled_paths:
            try:
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((thumb_width, thumb_height))

                canvas = Image.new("RGB", (thumb_width, thumb_height), "white")
                paste_x = (thumb_width - img.width) // 2
                paste_y = (thumb_height - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))

                sheet.paste(canvas, (x, image_y))
            except Exception as e:
                draw.text((x, image_y + 20), f"读取失败\n{e}", fill="red", font=font)

            x += thumb_width + gap

        y += thumb_height + row_title_height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"[完成] 已生成预览图：{output_path}")


def select_preview_rows(rows: List[Dict[str, str]], max_samples: int) -> List[Dict[str, str]]:
    """
    从样本中挑选用于预览的记录。

    策略：
    每个标签优先选 Participant_01；如果不足，再补其他样本。
    """
    selected = []
    seen_labels = set()

    for row in rows:
        if row.get("status") != "ok":
            continue

        label = row["label"]

        if row["participant"] == "Participant_01" and label not in seen_labels:
            selected.append(row)
            seen_labels.add(label)

        if len(selected) >= max_samples:
            return selected

    for row in rows:
        if row.get("status") != "ok":
            continue

        if row not in selected:
            selected.append(row)

        if len(selected) >= max_samples:
            return selected

    return selected


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
        "--output_path",
        required=True,
        help="输出预览图路径，例如 D:/datasets/HearBridge-NationalCSL-mini/preview/contact_sheet.jpg",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=15,
        help="最多预览多少个样本",
    )
    parser.add_argument(
        "--max_frames_per_sample",
        type=int,
        default=8,
        help="每个样本最多展示多少帧",
    )
    parser.add_argument(
        "--thumb_width",
        type=int,
        default=160,
        help="单帧缩略图宽度",
    )
    parser.add_argument(
        "--thumb_height",
        type=int,
        default=160,
        help="单帧缩略图高度",
    )

    args = parser.parse_args()

    rows = read_samples(Path(args.samples_csv))
    selected_rows = select_preview_rows(rows, args.max_samples)

    build_contact_sheet(
        selected_rows=selected_rows,
        output_path=Path(args.output_path),
        max_frames_per_sample=args.max_frames_per_sample,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )


if __name__ == "__main__":
    main()