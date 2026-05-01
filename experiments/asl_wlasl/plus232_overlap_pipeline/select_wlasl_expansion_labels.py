# -*- coding: utf-8 -*-
"""
从 WLASL HuggingFace samples.json 中统计候选词样本数，
并生成扩展词表推荐结果。

输入：
- D:/datasets/WLASL_HF_meta/samples.json

输出：
- D:/datasets/WLASL-mini-v2-25/label_selection/candidate_label_counts.csv
- D:/datasets/WLASL-mini-v2-25/label_selection/recommended_labels.txt
- D:/datasets/WLASL-mini-v2-25/label_selection/recommended_labels.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


DEFAULT_CANDIDATES = [
    # 原 13 词
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

    # 扩展常用词
    "good",
    "bad",
    "yes",
    "no",
    "what",
    "who",
    "why",
    "home",
    "family",
    "deaf",
    "language",
    "tomorrow",
    "need",
    "name",
    "stop",
    "student",
    "mother",
    "father",
    "brother",
    "sister",
    "doctor",
    "computer",
    "phone",
]


def load_samples(samples_json: Path) -> List[Dict]:
    """读取 WLASL samples.json。"""
    with samples_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]

    if isinstance(payload, list):
        return payload

    raise RuntimeError("samples_json 格式不符合预期")


def get_label(sample: Dict) -> str:
    """获取样本 gloss label。"""
    return str(sample["gloss"]["label"]).strip().lower()


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def save_json(path: Path, payload: Dict) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_json",
        default="D:/datasets/WLASL_HF_meta/samples.json",
        help="WLASL HuggingFace samples.json 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini-v2-25/label_selection",
        help="输出目录",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=7,
        help="推荐词最少样本数",
    )
    parser.add_argument(
        "--max_labels",
        type=int,
        default=25,
        help="最多推荐多少个词",
    )
    parser.add_argument(
        "--candidates",
        default="",
        help="可选，自定义逗号分隔候选词；为空则使用默认候选词",
    )

    args = parser.parse_args()

    samples_json = Path(args.samples_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(samples_json)
    counter = Counter(get_label(sample) for sample in samples)

    if args.candidates.strip():
        candidates = [
            item.strip().lower()
            for item in args.candidates.replace("，", ",").split(",")
            if item.strip()
        ]
    else:
        candidates = DEFAULT_CANDIDATES

    rows = []

    for label in candidates:
        count = int(counter.get(label, 0))

        if count >= args.min_count:
            status = "recommended"
        elif count >= 3:
            status = "low_count"
        else:
            status = "not_enough"

        rows.append({
            "label": label,
            "count": count,
            "status": status,
        })

    recommended = [
        row["label"]
        for row in rows
        if row["status"] == "recommended"
    ][:args.max_labels]

    low_count = [
        row["label"]
        for row in rows
        if row["status"] == "low_count"
    ]

    missing = [
        row["label"]
        for row in rows
        if row["status"] == "not_enough"
    ]

    write_csv(
        output_dir / "candidate_label_counts.csv",
        rows,
        ["label", "count", "status"],
    )

    labels_text = ",".join(recommended)

    (output_dir / "recommended_labels.txt").write_text(
        labels_text,
        encoding="utf-8",
    )

    save_json(
        output_dir / "recommended_labels.json",
        {
            "min_count": args.min_count,
            "max_labels": args.max_labels,
            "recommended_count": len(recommended),
            "recommended_labels": recommended,
            "recommended_labels_csv": labels_text,
            "low_count_labels": low_count,
            "missing_or_too_few_labels": missing,
            "candidate_rows": rows,
        },
    )

    print("\n========== WLASL 扩展词表筛选完成 ==========")
    print(f"候选词数：{len(candidates)}")
    print(f"推荐词数：{len(recommended)}")
    print(f"推荐词表：{labels_text}")

    print("\n========== 低样本词 ==========")
    print(", ".join(low_count) if low_count else "无")

    print("\n========== 样本不足/不存在 ==========")
    print(", ".join(missing) if missing else "无")


if __name__ == "__main__":
    main()
