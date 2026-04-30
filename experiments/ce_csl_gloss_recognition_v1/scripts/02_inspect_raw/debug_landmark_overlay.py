"""
CE-CSL MediaPipe 点位完整标注调试脚本

作用：
1. 从 processed/train.jsonl 中读取少量样本。
2. 读取对应视频。
3. 按指定时间点抽帧。
4. 对帧做水平翻转，用于模拟手机前置摄像头镜像输入。
5. 使用 MediaPipe Holistic 检测 Pose / Left Hand / Right Hand。
6. 在图片上完整标注每个点位编号与文字。
7. 保存调试图片到 processed/debug_landmark_overlay。

注意：
- 本脚本只用于人工检查点位和翻转方向。
- 不提取训练特征。
- 不训练模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# 1. 路径配置
# =========================

# CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# manifest 目录
PROCESSED_DIR = DATASET_ROOT / "processed"

# 调试图片输出目录
OUTPUT_DIR = PROCESSED_DIR / "debug_landmark_overlay"

# 使用哪个 split 做调试：train / dev / test
SPLIT = "train"

# 调试前几个样本
SAMPLE_LIMIT = 3

# 每个视频抽几帧。这里用比例，不同长度视频都适用
FRAME_RATIOS = [0.25, 0.5, 0.75]

# 是否在送入 MediaPipe 前水平翻转
MIRROR_INPUT = False

# 为了提速，送入 MediaPipe 前缩放到指定宽度
# 注意：保存调试图也是缩放后的图，不影响判断点位是否正确
TARGET_WIDTH = 960


# =========================
# 2. MediaPipe Pose 点位名称
# =========================

POSE_LANDMARK_NAMES = {
    0: "NOSE",
    1: "L_EYE_IN",
    2: "L_EYE",
    3: "L_EYE_OUT",
    4: "R_EYE_IN",
    5: "R_EYE",
    6: "R_EYE_OUT",
    7: "L_EAR",
    8: "R_EAR",
    9: "MOUTH_L",
    10: "MOUTH_R",
    11: "L_SHOULDER",
    12: "R_SHOULDER",
    13: "L_ELBOW",
    14: "R_ELBOW",
    15: "L_WRIST",
    16: "R_WRIST",
    17: "L_PINKY",
    18: "R_PINKY",
    19: "L_INDEX",
    20: "R_INDEX",
    21: "L_THUMB",
    22: "R_THUMB",
    23: "L_HIP",
    24: "R_HIP",
    25: "L_KNEE",
    26: "R_KNEE",
    27: "L_ANKLE",
    28: "R_ANKLE",
    29: "L_HEEL",
    30: "R_HEEL",
    31: "L_FOOT_INDEX",
    32: "R_FOOT_INDEX",
}


# =========================
# 3. 字体工具
# =========================

def load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    加载中文字体。

    Windows 下优先使用微软雅黑。如果失败，则使用 PIL 默认字体。
    默认字体可能无法显示中文，但英文编号仍可正常显示。

    Args:
        size: 字体大小。

    Returns:
        PIL 字体对象。
    """
    candidate_fonts = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]

    for font_path in candidate_fonts:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


# =========================
# 4. 数据读取
# =========================

def read_jsonl(path: Path, limit: int) -> List[Dict]:
    """
    读取 jsonl 文件前 limit 条样本。

    Args:
        path: jsonl 文件路径。
        limit: 读取条数。

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


def resize_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    等比例缩放图像到 TARGET_WIDTH。

    Args:
        frame_bgr: OpenCV BGR 图像。

    Returns:
        缩放后的 BGR 图像。
    """
    height, width = frame_bgr.shape[:2]

    if width <= TARGET_WIDTH:
        return frame_bgr

    scale = TARGET_WIDTH / width
    new_height = int(height * scale)

    return cv2.resize(frame_bgr, (TARGET_WIDTH, new_height))


def read_frame_by_index(video_path: Path, frame_index: int) -> np.ndarray | None:
    """
    读取指定帧。

    Args:
        video_path: 视频路径。
        frame_index: 帧编号。

    Returns:
        成功时返回 BGR 图像，失败时返回 None。
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return frame


def get_video_frame_count(video_path: Path) -> int:
    """
    获取视频帧数。

    Args:
        video_path: 视频路径。

    Returns:
        视频总帧数。
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return frame_count


# =========================
# 5. 绘制工具
# =========================

def normalized_to_pixel(landmark, width: int, height: int) -> Tuple[int, int]:
    """
    将 MediaPipe 归一化坐标转换为像素坐标。

    Args:
        landmark: MediaPipe landmark。
        width: 图像宽度。
        height: 图像高度。

    Returns:
        像素坐标 (x, y)。
    """
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def draw_text_with_bg(
    draw: ImageDraw.ImageDraw,
    position: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int],
) -> None:
    """
    绘制带背景的文字，避免文字被画面淹没。

    Args:
        draw: PIL 绘图对象。
        position: 文字左上角坐标。
        text: 文字内容。
        font: 字体对象。
        text_color: 文字颜色。
        bg_color: 背景颜色。
    """
    x, y = position

    bbox = draw.textbbox((x, y), text, font=font)
    padding = 2

    bg_box = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )

    draw.rectangle(bg_box, fill=bg_color)
    draw.text((x, y), text, font=font, fill=text_color)


def draw_pose_landmarks(
    image: Image.Image,
    pose_landmarks,
    font: ImageFont.ImageFont,
) -> None:
    """
    绘制 Pose 33 个点位及文字标签。

    Args:
        image: PIL 图像。
        pose_landmarks: MediaPipe Pose landmarks。
        font: 字体对象。
    """
    if pose_landmarks is None:
        return

    draw = ImageDraw.Draw(image)
    width, height = image.size

    point_color = (30, 144, 255)
    text_color = (255, 255, 255)
    bg_color = (0, 64, 128)

    for index, landmark in enumerate(pose_landmarks.landmark):
        x, y = normalized_to_pixel(landmark, width, height)

        radius = 4
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=point_color,
            outline=(255, 255, 255),
        )

        name = POSE_LANDMARK_NAMES.get(index, f"P{index}")
        label = f"P{index:02d}_{name}"

        draw_text_with_bg(
            draw=draw,
            position=(x + 5, y + 5),
            text=label,
            font=font,
            text_color=text_color,
            bg_color=bg_color,
        )


def draw_hand_landmarks(
    image: Image.Image,
    hand_landmarks,
    prefix: str,
    font: ImageFont.ImageFont,
) -> None:
    """
    绘制单只手 21 个点位及文字标签。

    Args:
        image: PIL 图像。
        hand_landmarks: MediaPipe Hand landmarks。
        prefix: L 或 R，用于区分左手/右手。
        font: 字体对象。
    """
    if hand_landmarks is None:
        return

    draw = ImageDraw.Draw(image)
    width, height = image.size

    if prefix == "L":
        point_color = (0, 220, 120)
        bg_color = (0, 96, 48)
    else:
        point_color = (255, 128, 0)
        bg_color = (128, 64, 0)

    text_color = (255, 255, 255)

    for index, landmark in enumerate(hand_landmarks.landmark):
        x, y = normalized_to_pixel(landmark, width, height)

        radius = 5
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=point_color,
            outline=(255, 255, 255),
        )

        label = f"{prefix}{index:02d}"

        draw_text_with_bg(
            draw=draw,
            position=(x + 5, y - 18),
            text=label,
            font=font,
            text_color=text_color,
            bg_color=bg_color,
        )


def draw_header_info(
    image: Image.Image,
    sample: Dict,
    frame_index: int,
    mirror_input: bool,
    font: ImageFont.ImageFont,
) -> None:
    """
    在图片顶部绘制样本信息。

    Args:
        image: PIL 图像。
        sample: 样本信息。
        frame_index: 当前帧编号。
        mirror_input: 是否做了水平翻转。
        font: 字体对象。
    """
    draw = ImageDraw.Draw(image)

    lines = [
        f"sampleId: {sample['sampleId']}    split: {sample['split']}    translator: {sample['translator']}",
        f"frame: {frame_index}    mirror_input: {mirror_input}",
        f"Chinese: {sample['chinese']}",
        f"Gloss: {' / '.join(sample['gloss'])}",
        "Pose: P00~P32    Left hand: L00~L20    Right hand: R00~R20",
    ]

    x = 10
    y = 10

    for line in lines:
        draw_text_with_bg(
            draw=draw,
            position=(x, y),
            text=line,
            font=font,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )
        y += 24


def cv_bgr_to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    """
    OpenCV BGR 图像转 PIL RGB 图像。

    Args:
        frame_bgr: OpenCV BGR 图像。

    Returns:
        PIL RGB 图像。
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def pil_rgb_to_cv_bgr(image: Image.Image) -> np.ndarray:
    """
    PIL RGB 图像转 OpenCV BGR 图像。

    Args:
        image: PIL RGB 图像。

    Returns:
        OpenCV BGR 图像。
    """
    frame_rgb = np.array(image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


# =========================
# 6. 主处理逻辑
# =========================

def process_sample(sample: Dict, holistic, font: ImageFont.ImageFont) -> None:
    """
    处理单条样本，抽帧并保存带点位标注的图片。

    Args:
        sample: manifest 样本。
        holistic: MediaPipe Holistic 实例。
        font: 字体对象。
    """
    video_path = DATASET_ROOT / sample["videoPath"]

    frame_count = get_video_frame_count(video_path)

    if frame_count <= 0:
        print(f"[跳过] 无法读取帧数: {video_path}")
        return

    for ratio in FRAME_RATIOS:
        frame_index = int(frame_count * ratio)
        frame_index = max(0, min(frame_index, frame_count - 1))

        frame_bgr = read_frame_by_index(video_path, frame_index)

        if frame_bgr is None:
            print(f"[跳过] 无法读取帧: {sample['sampleId']} frame={frame_index}")
            continue

        frame_bgr = resize_frame(frame_bgr)

        if MIRROR_INPUT:
            frame_bgr = cv2.flip(frame_bgr, 1)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        image = cv_bgr_to_pil_rgb(frame_bgr)

        draw_header_info(
            image=image,
            sample=sample,
            frame_index=frame_index,
            mirror_input=MIRROR_INPUT,
            font=font,
        )

        draw_pose_landmarks(
            image=image,
            pose_landmarks=results.pose_landmarks,
            font=font,
        )

        draw_hand_landmarks(
            image=image,
            hand_landmarks=results.left_hand_landmarks,
            prefix="L",
            font=font,
        )

        draw_hand_landmarks(
            image=image,
            hand_landmarks=results.right_hand_landmarks,
            prefix="R",
            font=font,
        )

        output_name = f"{sample['sampleId']}_frame_{frame_index:04d}_overlay.jpg"
        output_path = OUTPUT_DIR / output_name

        image.save(output_path, quality=95)

        print(f"[保存] {output_path}")


def main() -> None:
    """
    主入口。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = PROCESSED_DIR / f"{SPLIT}.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest 文件：{manifest_path}")

    samples = read_jsonl(manifest_path, limit=SAMPLE_LIMIT)

    font = load_font(size=14)

    mp_holistic = mp.solutions.holistic

    print("===== CE-CSL MediaPipe 点位完整标注调试开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("manifest:", manifest_path)
    print("输出目录:", OUTPUT_DIR)
    print("MIRROR_INPUT:", MIRROR_INPUT)
    print("SAMPLE_LIMIT:", SAMPLE_LIMIT)
    print("FRAME_RATIOS:", FRAME_RATIOS)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for sample in samples:
            process_sample(sample, holistic, font)

    print("===== CE-CSL MediaPipe 点位完整标注调试结束 =====")


if __name__ == "__main__":
    main()