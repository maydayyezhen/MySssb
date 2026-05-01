# -*- coding: utf-8 -*-
"""
从 Hugging Face 的 Voxel51/WLASL 数据集中抽取 ASL 小词表视频。

输入：
- D:/datasets/WLASL_HF_meta/samples.json

输出：
- D:/datasets/WLASL-mini/
  - videos/<label>/*.mp4
  - samples.csv
  - label_counts.csv
  - config.json
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download


DEFAULT_LABELS = [
    "you",
    "want",
    "work",
    "go",
    "today",
    "learn",
    "help",
    "friend",
    "teacher",
    "school",
    "please",
    "sorry",
    "meet",
]


def parse_labels(labels_text: str) -> List[str]:
    """解析逗号分隔的标签列表。"""
    labels = [
        item.strip().lower()
        for item in labels_text.replace("，", ",").split(",")
        if item.strip()
    ]
    return labels if labels else DEFAULT_LABELS


def load_samples(samples_json: Path) -> List[Dict]:
    """读取 WLASL Hugging Face samples.json。"""
    with samples_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["samples"]


def get_sample_label(sample: Dict) -> str:
    """读取样本标签。"""
    return str(sample["gloss"]["label"]).lower()


def get_video_id(filepath: str) -> str:
    """从 data/data_0/00335.mp4 中提取 00335。"""
    return Path(filepath).stem


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出：{path}")


def save_json(path: Path, data: Dict) -> None:
    """写出 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出：{path}")


def build_wlasl_mini(
    samples_json: Path,
    output_root: Path,
    labels: List[str],
    max_per_label: int,
    overwrite: bool,
) -> None:
    """构建 WLASL 小词表视频数据集。"""
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples(samples_json)

    selected_by_label: Dict[str, List[Dict]] = {label: [] for label in labels}
    label_set = set(labels)

    for sample in all_samples:
        label = get_sample_label(sample)

        if label not in label_set:
            continue

        if max_per_label > 0 and len(selected_by_label[label]) >= max_per_label:
            continue

        selected_by_label[label].append(sample)

    print("========== WLASL 小词表抽取计划 ==========")
    print(f"[信息] samples_json：{samples_json}")
    print(f"[信息] output_root：{output_root}")
    print(f"[信息] labels：{', '.join(labels)}")
    print(f"[信息] max_per_label：{max_per_label}")

    label_count_rows = []
    sample_rows = []

    for label in labels:
        samples = selected_by_label[label]
        label_count_rows.append({"label": label, "count": len(samples)})
        print(f"[标签] {label}: {len(samples)}")

    for label in labels:
        samples = selected_by_label[label]

        for index, sample in enumerate(samples, start=1):
            source_filepath = sample["filepath"]
            video_id = get_video_id(source_filepath)

            print(f"[下载] {label} {index}/{len(samples)} {source_filepath}")

            cached_path = hf_hub_download(
                repo_id="Voxel51/WLASL",
                repo_type="dataset",
                filename=source_filepath,
            )

            label_dir = output_root / "videos" / label
            label_dir.mkdir(parents=True, exist_ok=True)

            output_filename = f"{label}_{video_id}.mp4"
            output_path = label_dir / output_filename

            shutil.copy2(cached_path, output_path)

            metadata = sample.get("metadata", {})

            sample_rows.append({
                "sample_id": video_id,
                "label": label,
                "source_filepath": source_filepath,
                "local_path": str(output_path),
                "frame_width": metadata.get("frame_width", ""),
                "frame_height": metadata.get("frame_height", ""),
                "frame_rate": metadata.get("frame_rate", ""),
                "total_frame_count": metadata.get("total_frame_count", ""),
                "duration": metadata.get("duration", ""),
                "size_bytes": metadata.get("size_bytes", ""),
            })

    write_csv(
        output_root / "label_counts.csv",
        label_count_rows,
        ["label", "count"],
    )

    write_csv(
        output_root / "samples.csv",
        sample_rows,
        [
            "sample_id",
            "label",
            "source_filepath",
            "local_path",
            "frame_width",
            "frame_height",
            "frame_rate",
            "total_frame_count",
            "duration",
            "size_bytes",
        ],
    )

    save_json(
        output_root / "config.json",
        {
            "source": "Voxel51/WLASL",
            "samples_json": str(samples_json),
            "labels": labels,
            "max_per_label": max_per_label,
            "sample_count": len(sample_rows),
            "label_counts": label_count_rows,
        },
    )

    print("\n========== 抽取完成 ==========")
    print(f"[完成] 样本数：{len(sample_rows)}")
    print(f"[完成] 输出目录：{output_root}")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_json",
        default="D:/datasets/WLASL_HF_meta/samples.json",
        help="WLASL Hugging Face samples.json 路径",
    )
    parser.add_argument(
        "--output_root",
        default="D:/datasets/WLASL-mini",
        help="小词表输出目录",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="逗号分隔标签列表",
    )
    parser.add_argument(
        "--max_per_label",
        type=int,
        default=0,
        help="每个标签最多下载多少个；0 表示全部",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出目录",
    )

    args = parser.parse_args()

    build_wlasl_mini(
        samples_json=Path(args.samples_json),
        output_root=Path(args.output_root),
        labels=parse_labels(args.labels),
        max_per_label=args.max_per_label,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
