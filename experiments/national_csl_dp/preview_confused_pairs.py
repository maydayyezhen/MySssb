# -*- coding: utf-8 -*-
"""
NationalCSL-DP 混淆样本对比预览脚本。

作用：
1. 按指定参与者和标签目录生成对比拼图
2. 用于检查模型高置信错判是否来自动作相似、样本异常、左右手问题或特征不足
3. 默认对比：
   - 你 vs 需要
   - 我 vs 老师
   - 对不起 vs 学习

使用示例：
python experiments/national_csl_dp/preview_confused_pairs.py ^
  --raw_frames_root "D:/datasets/HearBridge-NationalCSL-mini/raw_frames" ^
  --output_path "D:/datasets/HearBridge-NationalCSL-mini/preview/confused_pairs_p10.jpg" ^
  --participant "Participant_10"
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


# 默认要检查的混淆对
DEFAULT_PAIRS = [
    ("你__1925", "需要__5610"),
    ("我__1928", "老师__1597"),
    ("对不起__5311", "学习__3701"),
    ("对不起__5311", "学习__4462"),
]


def load_font(font_size: int = 22) -> ImageFont.FreeTypeFont:
    """
    加载中文字体。

    Windows 下优先使用微软雅黑 / 黑体。
    如果加载失败，则使用 PIL 默认字体。
    """
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)

    return ImageFont.load_default()


def list_frame_images(frame_dir: Path) -> List[Path]:
    """
    获取某个样本目录下的图片帧。
    """
    if not frame_dir.exists():
        return []

    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def sample_evenly(items: List[Path], max_count: int) -> List[Path]:
    """
    从帧序列中均匀抽取若干帧。
    """
    if len(items) <= max_count:
        return items

    if max_count <= 1:
        return [items[0]]

    indices = [
        round(i * (len(items) - 1) / (max_count - 1))
        for i in range(max_count)
    ]

    return [items[index] for index in indices]


def read_image_safe(image_path: Path) -> Optional[Image.Image]:
    """
    安全读取图片。

    PIL 对中文路径比较稳，这里直接使用 Image.open。
    """
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[警告] 图片读取失败：{image_path}，原因：{e}")
        return None


def draw_sample_row(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    row_y: int,
    title: str,
    frame_paths: List[Path],
    font: ImageFont.FreeTypeFont,
    thumb_width: int,
    thumb_height: int,
    gap: int,
    title_height: int,
) -> None:
    """
    在拼图中绘制一行样本帧。
    """
    draw.text((gap, row_y), title, fill="black", font=font)

    x = gap
    image_y = row_y + title_height

    for frame_path in frame_paths:
        image = read_image_safe(frame_path)

        if image is None:
            x += thumb_width + gap
            continue

        image.thumbnail((thumb_width, thumb_height))

        canvas = Image.new("RGB", (thumb_width, thumb_height), "white")
        paste_x = (thumb_width - image.width) // 2
        paste_y = (thumb_height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y))

        sheet.paste(canvas, (x, image_y))

        # 在缩略图左上角标一下帧名，方便看顺序
        draw.text(
            (x + 4, image_y + 4),
            frame_path.stem,
            fill="black",
            font=font,
        )

        x += thumb_width + gap


def build_confused_pairs_sheet(
    raw_frames_root: Path,
    output_path: Path,
    participant: str,
    pairs: List[Tuple[str, str]],
    max_frames_per_sample: int,
    thumb_width: int,
    thumb_height: int,
) -> None:
    """
    生成混淆样本对比拼图。
    """
    font = load_font(20)

    gap = 10
    title_height = 34
    pair_gap = 24

    rows_per_pair = 2
    row_height = title_height + thumb_height + gap
    pair_height = rows_per_pair * row_height + pair_gap

    sheet_width = max_frames_per_sample * (thumb_width + gap) + gap
    sheet_height = len(pairs) * pair_height + gap

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    y = gap

    for left_label_dir, right_label_dir in pairs:
        for label_dir in [left_label_dir, right_label_dir]:
            sample_dir = raw_frames_root / label_dir / participant
            frame_paths = list_frame_images(sample_dir)
            selected_frames = sample_evenly(frame_paths, max_frames_per_sample)

            title = f"{label_dir} / {participant} / 帧数={len(frame_paths)}"
            if not frame_paths:
                title += " / 未找到"

            draw_sample_row(
                sheet=sheet,
                draw=draw,
                row_y=y,
                title=title,
                frame_paths=selected_frames,
                font=font,
                thumb_width=thumb_width,
                thumb_height=thumb_height,
                gap=gap,
                title_height=title_height,
            )

            y += row_height

        # 画一条分割线
        line_y = y + 2
        draw.line((gap, line_y, sheet_width - gap, line_y), fill=(180, 180, 180), width=2)

        y += pair_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)

    print(f"[完成] 已生成混淆对比预览图：{output_path}")


def parse_pair_args(pair_args: Optional[List[str]]) -> List[Tuple[str, str]]:
    """
    解析命令行传入的 pair 参数。

    格式：
    --pair "你__1925,需要__5610" --pair "我__1928,老师__1597"
    """
    if not pair_args:
        return DEFAULT_PAIRS

    pairs = []

    for item in pair_args:
        parts = [part.strip() for part in item.replace("，", ",").split(",")]

        if len(parts) != 2:
            raise ValueError(f"pair 参数格式错误：{item}，应为：标签目录A,标签目录B")

        pairs.append((parts[0], parts[1]))

    return pairs


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_frames_root",
        required=True,
        help="raw_frames 根目录，例如 D:/datasets/HearBridge-NationalCSL-mini/raw_frames",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出预览图路径，例如 D:/datasets/HearBridge-NationalCSL-mini/preview/confused_pairs_p10.jpg",
    )
    parser.add_argument(
        "--participant",
        default="Participant_10",
        help="要查看的参与者，例如 Participant_10",
    )
    parser.add_argument(
        "--max_frames_per_sample",
        type=int,
        default=10,
        help="每个样本最多展示多少帧",
    )
    parser.add_argument(
        "--thumb_width",
        type=int,
        default=150,
        help="缩略图宽度",
    )
    parser.add_argument(
        "--thumb_height",
        type=int,
        default=180,
        help="缩略图高度",
    )
    parser.add_argument(
        "--pair",
        action="append",
        help="自定义混淆对，格式：标签目录A,标签目录B。可重复传入。",
    )

    args = parser.parse_args()

    pairs = parse_pair_args(args.pair)

    build_confused_pairs_sheet(
        raw_frames_root=Path(args.raw_frames_root),
        output_path=Path(args.output_path),
        participant=args.participant,
        pairs=pairs,
        max_frames_per_sample=args.max_frames_per_sample,
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )


if __name__ == "__main__":
    main()