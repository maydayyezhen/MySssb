"""
CE-CSL 全量视频规格统计脚本

作用：
1. 扫描 processed/all.jsonl 中的全部视频。
2. 统计分辨率、FPS、帧数、时长分布。
3. 检查数据集中是否存在不同分辨率或异常视频。
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import cv2


DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
MANIFEST_PATH = DATASET_ROOT / "processed" / "all.jsonl"


def read_all_samples():
    samples = []

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def main() -> None:
    samples = read_all_samples()

    resolution_counter = Counter()
    fps_counter = Counter()

    frame_counts = []
    durations = []
    failed_samples = []

    for index, sample in enumerate(samples, start=1):
        video_path = DATASET_ROOT / sample["videoPath"]

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            failed_samples.append(sample["sampleId"])
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        resolution_counter[(width, height)] += 1
        fps_counter[round(fps, 2)] += 1

        frame_counts.append(frame_count)

        if fps and fps > 0:
            durations.append(frame_count / fps)

        if index % 500 == 0:
            print(f"已检查 {index}/{len(samples)}")

    print("\n===== CE-CSL 全量视频规格统计 =====")
    print("样本总数:", len(samples))
    print("失败视频数:", len(failed_samples))

    print("\n===== 分辨率分布 =====")
    for (width, height), count in resolution_counter.most_common():
        print(f"{width} x {height}: {count}")

    print("\n===== FPS 分布 =====")
    for fps, count in fps_counter.most_common():
        print(f"{fps}: {count}")

    print("\n===== 帧数统计 =====")
    print("最短帧数:", min(frame_counts))
    print("最长帧数:", max(frame_counts))
    print("平均帧数:", round(sum(frame_counts) / len(frame_counts), 2))

    print("\n===== 时长统计，单位秒 =====")
    print("最短时长:", round(min(durations), 2))
    print("最长时长:", round(max(durations), 2))
    print("平均时长:", round(sum(durations) / len(durations), 2))

    if failed_samples:
        print("\n===== 打不开的视频样本 =====")
        for sample_id in failed_samples[:20]:
            print(sample_id)


if __name__ == "__main__":
    main()