# -*- coding: utf-8 -*-
"""
批量评估 NationalCSL-DP plus_blank_pipeline 的伪连续句子识别效果。

作用：
1. 批量调用 pseudo_sentence_stream_test.py
2. 读取每条句子的 segments.json
3. 汇总 expected_sequence / detected_sequence / exact_match
4. 输出 pseudo_sentence_eval_report.csv 和 summary.json

本脚本用于把“感觉效果怎么样”变成可写进论文/报告的表格。
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_SENTENCES = [
    "我们,需要,帮助",
    "朋友,帮助,我们",
    "你,今天,学习",
    "我,需要,帮助",
    "你们,今天,学习",
    "老师,今天,学习",
    "朋友,今天,需要,帮助",
    "我们,今天,学习",
    "你们,需要,帮助",
    "朋友,明天,再见",
]


def parse_sentence(sentence: str) -> List[str]:
    """
    解析句子。
    """
    return [
        item.strip()
        for item in sentence.replace("，", ",").split(",")
        if item.strip()
    ]


def load_sentences(sentence_file: str) -> List[str]:
    """
    从文件读取句子列表。

    文件格式：
    每行一个句子，例如：
    我们,需要,帮助
    朋友,帮助,我们
    """
    if not sentence_file:
        return DEFAULT_SENTENCES

    path = Path(sentence_file)

    sentences = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            sentences.append(line)

    if not sentences:
        raise RuntimeError(f"句子文件为空：{path}")

    return sentences


def calc_sequence_diff(expected: List[str], detected: List[str]) -> Dict[str, str]:
    """
    简单计算漏词、多词和位置错误。

    说明：
    这里不是复杂编辑距离，只给论文表格够用的直观分析。
    """
    expected_set = set(expected)
    detected_set = set(detected)

    missing_words = [word for word in expected if word not in detected_set]
    extra_words = [word for word in detected if word not in expected_set]

    position_errors = []

    min_len = min(len(expected), len(detected))

    for index in range(min_len):
        if expected[index] != detected[index]:
            position_errors.append(
                f"{index + 1}:{expected[index]}->{detected[index]}"
            )

    if len(expected) != len(detected):
        position_errors.append(f"length:{len(expected)}->{len(detected)}")

    return {
        "missing_words": " ".join(missing_words),
        "extra_words": " ".join(extra_words),
        "position_errors": " | ".join(position_errors),
    }


def run_one_sentence(
    script_path: Path,
    feature_dir: Path,
    model_dir: Path,
    output_dir: Path,
    sentence: str,
    participant: str,
    confidence_threshold: float,
    margin_threshold: float,
    min_segment_avg_confidence: float,
    min_segment_max_confidence: float,
    nms_suppress_radius: int,
    stable_frames: int,
    blank_end_frames: int,
    same_label_merge_gap: int,
    tail_frames: int,
    tail_mode: str,
) -> Dict[str, object]:
    """
    执行单条句子测试，并读取结果 JSON。
    """
    labels = parse_sentence(sentence)
    safe_sentence_name = "_".join(labels)
    result_json_path = output_dir / f"{participant}_{safe_sentence_name}_segments.json"

    command = [
        sys.executable,
        str(script_path),
        "--feature_dir", str(feature_dir),
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--sentence", sentence,
        "--participant", participant,
        "--gap_frames", "0",
        "--gap_mode", "none",
        "--tail_frames", str(tail_frames),
        "--tail_mode", tail_mode,
        "--confidence_threshold", str(confidence_threshold),
        "--margin_threshold", str(margin_threshold),
        "--min_segment_avg_confidence", str(min_segment_avg_confidence),
        "--min_segment_max_confidence", str(min_segment_max_confidence),
        "--nms_suppress_radius", str(nms_suppress_radius),
        "--stable_frames", str(stable_frames),
        "--blank_end_frames", str(blank_end_frames),
        "--same_label_merge_gap", str(same_label_merge_gap),
    ]

    print(f"\n========== 测试句子：{' '.join(labels)} ==========")

    subprocess.run(command, check=True)

    if not result_json_path.exists():
        raise RuntimeError(f"结果 JSON 不存在：{result_json_path}")

    with result_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    expected = payload.get("expected_sequence", labels)
    detected = payload.get("detected_sequence", [])
    exact_match = bool(payload.get("exact_match", False))

    diff = calc_sequence_diff(expected, detected)

    return {
        "sentence": " ".join(labels),
        "expected_sequence": " ".join(expected),
        "detected_sequence": " ".join(detected),
        "exact_match": int(exact_match),
        "expected_len": len(expected),
        "detected_len": len(detected),
        "missing_words": diff["missing_words"],
        "extra_words": diff["extra_words"],
        "position_errors": diff["position_errors"],
        "result_json": str(result_json_path),
    }


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出 CSV 报告。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sentence",
        "expected_sequence",
        "detected_sequence",
        "exact_match",
        "expected_len",
        "detected_len",
        "missing_words",
        "extra_words",
        "position_errors",
        "result_json",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV 报告：{output_path}")


def save_json(output_path: Path, data: Dict[str, object]) -> None:
    """
    写出 JSON。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{output_path}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        default="D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus_blank",
    )
    parser.add_argument(
        "--model_dir",
        default="D:/datasets/HearBridge-NationalCSL-mini/models_20f_plus_blank",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/HearBridge-NationalCSL-mini/pseudo_sentence_20f_plus_batch_eval",
    )
    parser.add_argument(
        "--sentence_file",
        default="",
        help="可选：每行一个句子的 txt 文件；为空则使用内置测试句",
    )
    parser.add_argument(
        "--participant",
        default="Participant_10",
    )

    parser.add_argument("--confidence_threshold", type=float, default=0.80)
    parser.add_argument("--margin_threshold", type=float, default=0.15)
    parser.add_argument("--min_segment_avg_confidence", type=float, default=0.75)
    parser.add_argument("--min_segment_max_confidence", type=float, default=0.85)
    parser.add_argument("--nms_suppress_radius", type=int, default=10)
    parser.add_argument("--stable_frames", type=int, default=1)
    parser.add_argument("--blank_end_frames", type=int, default=3)
    parser.add_argument("--same_label_merge_gap", type=int, default=8)
    parser.add_argument("--tail_frames", type=int, default=12)
    parser.add_argument("--tail_mode", default="repeat_last", choices=["repeat_last", "zero", "none"])

    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / "pseudo_sentence_stream_test.py"

    if not script_path.exists():
        raise RuntimeError(f"找不到 pseudo_sentence_stream_test.py：{script_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(args.sentence_file)

    rows = []

    for sentence in sentences:
        row = run_one_sentence(
            script_path=script_path,
            feature_dir=Path(args.feature_dir),
            model_dir=Path(args.model_dir),
            output_dir=output_dir,
            sentence=sentence,
            participant=args.participant,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            min_segment_avg_confidence=args.min_segment_avg_confidence,
            min_segment_max_confidence=args.min_segment_max_confidence,
            nms_suppress_radius=args.nms_suppress_radius,
            stable_frames=args.stable_frames,
            blank_end_frames=args.blank_end_frames,
            same_label_merge_gap=args.same_label_merge_gap,
            tail_frames=args.tail_frames,
            tail_mode=args.tail_mode,
        )
        rows.append(row)

    report_csv = output_dir / "pseudo_sentence_eval_report.csv"
    write_csv(report_csv, rows)

    total = len(rows)
    exact_count = sum(int(row["exact_match"]) for row in rows)
    exact_rate = exact_count / total if total else 0.0

    summary = {
        "total": total,
        "exact_match_count": exact_count,
        "exact_match_rate": round(exact_rate, 6),
        "participant": args.participant,
        "feature_dir": args.feature_dir,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "settings": {
            "confidence_threshold": args.confidence_threshold,
            "margin_threshold": args.margin_threshold,
            "min_segment_avg_confidence": args.min_segment_avg_confidence,
            "min_segment_max_confidence": args.min_segment_max_confidence,
            "nms_suppress_radius": args.nms_suppress_radius,
            "stable_frames": args.stable_frames,
            "blank_end_frames": args.blank_end_frames,
            "same_label_merge_gap": args.same_label_merge_gap,
            "tail_frames": args.tail_frames,
            "tail_mode": args.tail_mode,
        },
    }

    save_json(output_dir / "summary.json", summary)

    print("\n========== 批量评估完成 ==========")
    print(f"总句子数：{total}")
    print(f"完全匹配数：{exact_count}")
    print(f"完全匹配率：{exact_rate:.4f}")
    print(f"报告：{report_csv}")


if __name__ == "__main__":
    main()