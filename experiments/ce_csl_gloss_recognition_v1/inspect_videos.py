"""
CE-CSL 视频读取检查脚本

作用：
1. 从 processed/train.jsonl、dev.jsonl、test.jsonl 中读取少量样本。
2. 根据 videoPath 找到对应 mp4。
3. 用 OpenCV 检查视频能否打开。
4. 输出帧数、FPS、分辨率、时长等信息。

本脚本只做视频检查，不做特征提取。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2


# TODO：改成你的 CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# manifest 输出目录
PROCESSED_DIR = DATASET_ROOT / "processed"


def read_jsonl(path: Path, limit: int = 5) -> List[Dict]:
    """
    读取 jsonl 文件中的前 limit 条样本。

    Args:
        path: jsonl 文件路径。
        limit: 读取数量。

    Returns:
        样本字典列表。
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


def inspect_video(sample: Dict) -> None:
    """
    检查单个视频文件。

    Args:
        sample: manifest 中的一条样本。
    """
    video_path = DATASET_ROOT / sample["videoPath"]

    print("=" * 80)
    print("样本 ID:", sample["sampleId"])
    print("split:", sample["split"])
    print("视频路径:", video_path)
    print("中文句子:", sample["chinese"])
    print("Gloss:", "/".join(sample["gloss"]))

    if not video_path.exists():
        print("结果: 视频文件不存在")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("结果: OpenCV 打不开该视频")
        cap.release()
        return

    # 视频帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 视频 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 视频宽高
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频时长，单位秒
    duration = frame_count / fps if fps and fps > 0 else 0

    # 尝试读取第一帧
    success, frame = cap.read()

    print("结果: 可以正常打开")
    print("帧数:", frame_count)
    print("FPS:", round(fps, 2))
    print("分辨率:", f"{width} x {height}")
    print("时长秒:", round(duration, 2))
    print("第一帧读取:", "成功" if success else "失败")

    if success and frame is not None:
        print("第一帧 shape:", frame.shape)

    cap.release()


def main() -> None:
    """
    主流程。
    """
    print("===== CE-CSL 视频读取检查开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("processed 目录:", PROCESSED_DIR)

    for split in ["train", "dev", "test"]:
        jsonl_path = PROCESSED_DIR / f"{split}.jsonl"

        if not jsonl_path.exists():
            print(f"找不到 manifest 文件: {jsonl_path}")
            continue

        print("\n" + "#" * 80)
        print(f"检查 {split}.jsonl 前 5 条样本")
        print("#" * 80)

        samples = read_jsonl(jsonl_path, limit=5)

        for sample in samples:
            inspect_video(sample)

    print("\n===== CE-CSL 视频读取检查结束 =====")


if __name__ == "__main__":
    main()