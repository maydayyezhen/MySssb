# -*- coding: utf-8 -*-
"""
批量评估 WLASL 拼接演示视频的识别效果。

功能：
1. 批量调用 infer_wlasl_sentence_video.py
2. 读取每条视频的 segments.json
3. 汇总 expected / detected / exact_match
4. 输出 demo_sentence_eval_report.csv 和 summary.json
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_CASES = [
    {
        "name": "friend_meet_today",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/friend_meet_today_trimmed.mp4",
        "expected": "friend,meet,today",
    },
    {
        "name": "sorry_teacher",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/sorry_teacher_trimmed.mp4",
        "expected": "sorry,teacher",
    },
    {
        "name": "please_help",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/please_help_trimmed.mp4",
        "expected": "please,help",
    },
    {
        "name": "you_want_work",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/you_want_work_trimmed.mp4",
        "expected": "you,want,work",
    },
    {
        "name": "teacher_help_learn",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/teacher_help_learn_trimmed.mp4",
        "expected": "teacher,help,learn",
    },
    {
        "name": "friend_meet",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/friend_meet_trimmed.mp4",
        "expected": "friend,meet",
    },
    {
        "name": "please_teacher",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/please_teacher_trimmed.mp4",
        "expected": "please,teacher",
    },
    {
        "name": "you_want_help",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/you_want_help_trimmed.mp4",
        "expected": "you,want,help",
    },
    {
        "name": "sorry_friend",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/sorry_friend_trimmed.mp4",
        "expected": "sorry,friend",
    },
    {
        "name": "today_learn",
        "video_path": "D:/datasets/WLASL-mini/demo_videos_trimmed/today_learn_trimmed.mp4",
        "expected": "today,learn",
    },
]


def parse_sequence(text: str) -> List[str]:
    """解析逗号分隔序列。"""
    return [
        item.strip().lower()
        for item in text.replace("，", ",").split(",")
        if item.strip()
    ]


def load_json(path: Path) -> Dict:
    """读取 JSON。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def sequence_diff(expected: List[str], detected: List[str]) -> Dict[str, str]:
    """生成简单错误分析。"""
    expected_set = set(expected)
    detected_set = set(detected)

    missing = [word for word in expected if word not in detected_set]
    extra = [word for word in detected if word not in expected_set]

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
        "missing_words": " ".join(missing),
        "extra_words": " ".join(extra),
        "position_errors": " | ".join(position_errors),
    }


def segment_summary(segments: List[Dict[str, object]]) -> str:
    """压缩词段信息，便于放入 CSV。"""
    parts = []

    for seg in segments:
        parts.append(
            f"{seg.get('label')}[{seg.get('start_frame')}-{seg.get('end_frame')}]"
            f" avg={seg.get('avg_confidence')} max={seg.get('max_confidence')}"
        )

    return " ; ".join(parts)


def run_one_case(
    infer_script: Path,
    case: Dict[str, str],
    feature_dir: Path,
    model_dir: Path,
    output_root: Path,
    python_exe: str,
    window_size: int,
    stride: int,
    confidence_threshold: float,
    margin_threshold: float,
    min_segment_windows: int,
    min_segment_avg_confidence: float,
    min_segment_max_confidence: float,
    same_label_merge_gap: int,
    nms_suppress_radius: int,
) -> Dict[str, object]:
    """执行单条 demo 视频推理。"""
    name = case["name"]
    video_path = Path(case["video_path"])
    expected = case["expected"]

    if not video_path.exists():
        print(f"[跳过] 视频不存在：{video_path}")
        expected_sequence = parse_sequence(expected)
        diff = sequence_diff(expected_sequence, [])

        return {
            "name": name,
            "video_path": str(video_path),
            "expected_sequence": " ".join(expected_sequence),
            "detected_sequence": "",
            "exact_match": 0,
            "status": "missing_video",
            "missing_words": diff["missing_words"],
            "extra_words": diff["extra_words"],
            "position_errors": diff["position_errors"],
            "segment_count": 0,
            "segments": "",
            "segments_json": "",
        }

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_exe,
        str(infer_script),
        "--video_path", str(video_path),
        "--feature_dir", str(feature_dir),
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--expected", expected,
        "--window_size", str(window_size),
        "--stride", str(stride),
        "--confidence_threshold", str(confidence_threshold),
        "--margin_threshold", str(margin_threshold),
        "--min_segment_windows", str(min_segment_windows),
        "--min_segment_avg_confidence", str(min_segment_avg_confidence),
        "--min_segment_max_confidence", str(min_segment_max_confidence),
        "--same_label_merge_gap", str(same_label_merge_gap),
        "--nms_suppress_radius", str(nms_suppress_radius),
    ]

    print(f"\n========== 推理 demo：{name} ==========")
    subprocess.run(command, check=True)

    segments_json = output_dir / f"{video_path.stem}_segments.json"

    if not segments_json.exists():
        raise RuntimeError(f"推理结果不存在：{segments_json}")

    payload = load_json(segments_json)

    expected_sequence = payload.get("expected_sequence", parse_sequence(expected))
    detected_sequence = payload.get("detected_sequence", [])
    exact_match = bool(payload.get("exact_match", False))
    segments = payload.get("segments", [])

    diff = sequence_diff(expected_sequence, detected_sequence)

    return {
        "name": name,
        "video_path": str(video_path),
        "expected_sequence": " ".join(expected_sequence),
        "detected_sequence": " ".join(detected_sequence),
        "exact_match": int(exact_match),
        "status": "ok",
        "missing_words": diff["missing_words"],
        "extra_words": diff["extra_words"],
        "position_errors": diff["position_errors"],
        "segment_count": len(segments),
        "segments": segment_summary(segments),
        "segments_json": str(segments_json),
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        default="D:/datasets/WLASL-mini/features_20f_plus",
    )
    parser.add_argument(
        "--model_dir",
        default="D:/datasets/WLASL-mini/models_20f_plus",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini/demo_eval",
    )
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)

    # 当前已经验证可用的 v2 后处理参数
    parser.add_argument("--confidence_threshold", type=float, default=0.45)
    parser.add_argument("--margin_threshold", type=float, default=0.05)
    parser.add_argument("--min_segment_windows", type=int, default=2)
    parser.add_argument("--min_segment_avg_confidence", type=float, default=0.75)
    parser.add_argument("--min_segment_max_confidence", type=float, default=0.85)
    parser.add_argument("--same_label_merge_gap", type=int, default=8)
    parser.add_argument("--nms_suppress_radius", type=int, default=6)

    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    infer_script = current_dir / "infer_wlasl_sentence_video.py"

    if not infer_script.exists():
        raise RuntimeError(f"找不到推理脚本：{infer_script}")

    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for case in DEFAULT_CASES:
        row = run_one_case(
            infer_script=infer_script,
            case=case,
            feature_dir=feature_dir,
            model_dir=model_dir,
            output_root=output_root,
            python_exe=sys.executable,
            window_size=args.window_size,
            stride=args.stride,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            min_segment_windows=args.min_segment_windows,
            min_segment_avg_confidence=args.min_segment_avg_confidence,
            min_segment_max_confidence=args.min_segment_max_confidence,
            same_label_merge_gap=args.same_label_merge_gap,
            nms_suppress_radius=args.nms_suppress_radius,
        )
        rows.append(row)

    report_path = output_root / "demo_sentence_eval_report.csv"

    write_csv(
        report_path,
        rows,
        [
            "name",
            "video_path",
            "expected_sequence",
            "detected_sequence",
            "exact_match",
            "status",
            "missing_words",
            "extra_words",
            "position_errors",
            "segment_count",
            "segments",
            "segments_json",
        ],
    )

    valid_rows = [row for row in rows if row["status"] == "ok"]
    total = len(valid_rows)
    exact_count = sum(int(row["exact_match"]) for row in valid_rows)
    exact_rate = exact_count / total if total else 0.0

    summary = {
        "total": total,
        "exact_match_count": exact_count,
        "exact_match_rate": round(exact_rate, 6),
        "all_case_count": len(rows),
        "missing_video_count": sum(1 for row in rows if row["status"] == "missing_video"),
        "settings": {
            "window_size": args.window_size,
            "stride": args.stride,
            "confidence_threshold": args.confidence_threshold,
            "margin_threshold": args.margin_threshold,
            "min_segment_windows": args.min_segment_windows,
            "min_segment_avg_confidence": args.min_segment_avg_confidence,
            "min_segment_max_confidence": args.min_segment_max_confidence,
            "same_label_merge_gap": args.same_label_merge_gap,
            "nms_suppress_radius": args.nms_suppress_radius,
        },
        "rows": rows,
    }

    save_json(output_root / "summary.json", summary)

    print("\n========== Demo 批量评估完成 ==========")
    print(f"有效样例数：{total}")
    print(f"完全匹配数：{exact_count}")
    print(f"完全匹配率：{exact_rate:.4f}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
