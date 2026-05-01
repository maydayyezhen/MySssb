# -*- coding: utf-8 -*-
"""
构建并评估 WLASL-mini-v2-25 的语义重排测试句子。

目的：
不是只看 Top1 完全匹配率，而是生成：
- rawSequence：模型原始词序列
- expectedSequence：实验用真实句子
- segmentTopK：每个词段的 Top3 候选
- llmInput：不包含 expected 的 LLM 语义重排输入

后续可把 llmInput 交给大语言模型，只允许它在 TopK 候选里选择，
判断能否把 rawSequence 修正成更自然的句子。
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_CASES = [
    {"name": "friend_meet_today", "sentence": "friend,meet,today"},
    {"name": "please_help_friend", "sentence": "please,help,friend"},
    {"name": "teacher_help_you", "sentence": "teacher,help,you"},
    {"name": "you_want_help", "sentence": "you,want,help"},
    {"name": "you_want_work_tomorrow", "sentence": "you,want,work,tomorrow"},
    {"name": "you_go_school_today", "sentence": "you,go,school,today"},
    {"name": "what_you_want", "sentence": "what,you,want"},
    {"name": "who_help_teacher", "sentence": "who,help,teacher"},
    {"name": "why_you_sorry", "sentence": "why,you,sorry"},
    {"name": "deaf_friend_learn_language", "sentence": "deaf,friend,learn,language"},
    {"name": "family_go_home", "sentence": "family,go,home"},
    {"name": "no_work_today", "sentence": "no,work,today"},
    {"name": "please_meet_teacher", "sentence": "please,meet,teacher"},
    {"name": "yes_you_go_school", "sentence": "yes,you,go,school"},
]


def parse_sequence(text: str) -> List[str]:
    """解析逗号分隔的 gloss 序列。"""
    return [
        item.strip().lower()
        for item in text.replace("，", ",").split(",")
        if item.strip()
    ]


def read_json(path: Path) -> Dict:
    """读取 JSON。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """读取 CSV。"""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def get_float(value: object, default: float = 0.0) -> float:
    """安全转换 float。"""
    try:
        return float(value)
    except Exception:
        return default


def get_int(value: object, default: int = 0) -> int:
    """安全转换 int。"""
    try:
        return int(float(value))
    except Exception:
        return default


def aggregate_segment_topk(
    dense_rows: List[Dict[str, str]],
    segment: Dict[str, object],
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """
    根据 dense prediction CSV 为一个最终词段聚合 TopK。

    做法：
    - 找 center_frame 落在该 segment [start_frame, end_frame] 内的窗口
    - 收集每个窗口的 top1/top2/top3
    - 对同一 label 的概率求平均，并按平均概率排序
    """
    start_frame = get_int(segment.get("start_frame"))
    end_frame = get_int(segment.get("end_frame"))

    selected_rows = []

    for row in dense_rows:
        center = get_int(row.get("center_frame"))

        if start_frame <= center <= end_frame:
            selected_rows.append(row)

    score_map: Dict[str, List[float]] = {}

    for row in selected_rows:
        candidates = [
            (
                row.get("top1_label") or row.get("top1"),
                row.get("top1_prob"),
            ),
            (
                row.get("top2_label") or row.get("top2"),
                row.get("top2_prob"),
            ),
            (
                row.get("top3_label"),
                row.get("top3_prob"),
            ),
        ]

        for label, prob in candidates:
            if not label:
                continue

            label = str(label).strip()

            if not label:
                continue

            score_map.setdefault(label, []).append(get_float(prob))

    items = []

    for label, probs in score_map.items():
        if not probs:
            continue

        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)
        hit_count = len(probs)

        items.append({
            "label": label,
            "avgProb": round(avg_prob, 6),
            "maxProb": round(max_prob, 6),
            "hitCount": hit_count,
        })

    items.sort(
        key=lambda item: (float(item["avgProb"]), float(item["maxProb"]), int(item["hitCount"])),
        reverse=True,
    )

    return items[:top_k]


def build_llm_input(
    vocabulary: List[str],
    raw_sequence: List[str],
    segment_topk: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    构建给 LLM 的输入，不包含 expectedSequence。
    """
    return {
        "task": "semantic_rerank_sign_recognition",
        "rules": [
            "Only choose words from each segment's topK candidates.",
            "Do not invent words outside the candidate lists.",
            "Prefer a semantically natural short English gloss sequence.",
            "Keep the order of segments unless an extra segment is clearly unnatural.",
            "Return correctedSequence, correctionApplied, and reason."
        ],
        "vocabulary": vocabulary,
        "rawSequence": raw_sequence,
        "segments": segment_topk,
    }


def make_prompt_text(case_name: str, llm_input: Dict[str, object]) -> str:
    """
    生成可复制给 LLM 的文本提示。
    """
    return (
        f"CASE: {case_name}\n"
        f"You are a semantic reranker for a sign-language recognition system.\n"
        f"The recognizer outputs one raw gloss sequence and TopK candidates for each segment.\n"
        f"Your job is to choose the most natural corrected gloss sequence.\n"
        f"Rules:\n"
        f"1. Only choose words from each segment's TopK candidates.\n"
        f"2. Do not invent words outside the candidate lists.\n"
        f"3. You may remove an obviously extra segment, but explain why.\n"
        f"4. Return JSON only.\n\n"
        f"INPUT:\n"
        f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}\n"
    )


def run_case(
    case: Dict[str, str],
    compose_script: Path,
    infer_script: Path,
    samples_csv: Path,
    feature_dir: Path,
    model_dir: Path,
    video_dir: Path,
    output_root: Path,
    python_exe: str,
    trim_padding: int,
    gap_frames: int,
    tail_frames: int,
    window_size: int,
    stride: int,
    confidence_threshold: float,
    margin_threshold: float,
    min_segment_windows: int,
    min_segment_avg_confidence: float,
    min_segment_max_confidence: float,
    same_label_merge_gap: int,
    nms_suppress_radius: int,
    vocabulary: List[str],
) -> Dict[str, object]:
    """生成并评估一个句子样例。"""
    name = case["name"]
    sentence = case["sentence"]
    expected_sequence = parse_sequence(sentence)

    video_path = video_dir / f"{name}_trimmed.mp4"
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== CASE: {name} ==========")
    print(f"[句子] {sentence}")

    compose_cmd = [
        python_exe,
        str(compose_script),
        "--samples_csv", str(samples_csv),
        "--sentence", sentence,
        "--output_path", str(video_path),
        "--sample_policy", "largest",
        "--trim_padding", str(trim_padding),
        "--gap_frames", str(gap_frames),
        "--tail_frames", str(tail_frames),
    ]

    subprocess.run(compose_cmd, check=True)

    infer_cmd = [
        python_exe,
        str(infer_script),
        "--video_path", str(video_path),
        "--feature_dir", str(feature_dir),
        "--model_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--expected", sentence,
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

    subprocess.run(infer_cmd, check=True)

    segments_json_path = output_dir / f"{video_path.stem}_segments.json"
    dense_csv_path = output_dir / f"{video_path.stem}_dense_predictions.csv"

    payload = read_json(segments_json_path)
    dense_rows = read_csv_rows(dense_csv_path)

    raw_sequence = payload.get("detected_sequence", [])
    exact_match = bool(payload.get("exact_match", False))
    segments = payload.get("segments", [])

    segment_topk = []

    for index, segment in enumerate(segments, start=1):
        topk = aggregate_segment_topk(
            dense_rows=dense_rows,
            segment=segment,
            top_k=3,
        )

        segment_topk.append({
            "segmentIndex": index,
            "startFrame": segment.get("start_frame"),
            "endFrame": segment.get("end_frame"),
            "rawLabel": segment.get("label"),
            "avgConfidence": segment.get("avg_confidence"),
            "maxConfidence": segment.get("max_confidence"),
            "topK": topk,
        })

    llm_input = build_llm_input(
        vocabulary=vocabulary,
        raw_sequence=raw_sequence,
        segment_topk=segment_topk,
    )

    result = {
        "name": name,
        "sentence": sentence,
        "videoPath": str(video_path),
        "expectedSequence": expected_sequence,
        "rawSequence": raw_sequence,
        "exactMatch": exact_match,
        "segmentCount": len(segments),
        "segmentTopK": segment_topk,
        "llmInput": llm_input,
        "segmentsJson": str(segments_json_path),
        "denseCsv": str(dense_csv_path),
    }

    save_json(output_dir / f"{name}_semantic_case.json", result)

    return result


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        default="D:/datasets/WLASL-mini-v2-25/samples.csv",
    )
    parser.add_argument(
        "--feature_dir",
        default="D:/datasets/WLASL-mini-v2-25/features_20f_plus",
    )
    parser.add_argument(
        "--model_dir",
        default="D:/datasets/WLASL-mini-v2-25/models_20f_plus",
    )
    parser.add_argument(
        "--video_dir",
        default="D:/datasets/WLASL-mini-v2-25/demo_videos_semantic",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini-v2-25/demo_eval_semantic",
    )

    # 拼接参数
    parser.add_argument("--trim_padding", type=int, default=4)
    parser.add_argument("--gap_frames", type=int, default=2)
    parser.add_argument("--tail_frames", type=int, default=6)

    # 语义重排实验用稍宽松阈值，保留更多候选段。
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.35)
    parser.add_argument("--margin_threshold", type=float, default=0.0)
    parser.add_argument("--min_segment_windows", type=int, default=2)
    parser.add_argument("--min_segment_avg_confidence", type=float, default=0.55)
    parser.add_argument("--min_segment_max_confidence", type=float, default=0.70)
    parser.add_argument("--same_label_merge_gap", type=int, default=8)
    parser.add_argument("--nms_suppress_radius", type=int, default=6)

    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    compose_script = current_dir / "compose_wlasl_sentence_video_trimmed.py"
    infer_script = current_dir / "infer_wlasl_sentence_video.py"

    samples_csv = Path(args.samples_csv)
    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    video_dir = Path(args.video_dir)
    output_root = Path(args.output_dir)

    video_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    labels_payload = read_json(feature_dir / "labels.json")
    vocabulary = labels_payload["labels"]

    all_results = []
    prompt_chunks = []

    for case in DEFAULT_CASES:
        result = run_case(
            case=case,
            compose_script=compose_script,
            infer_script=infer_script,
            samples_csv=samples_csv,
            feature_dir=feature_dir,
            model_dir=model_dir,
            video_dir=video_dir,
            output_root=output_root,
            python_exe=sys.executable,
            trim_padding=args.trim_padding,
            gap_frames=args.gap_frames,
            tail_frames=args.tail_frames,
            window_size=args.window_size,
            stride=args.stride,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            min_segment_windows=args.min_segment_windows,
            min_segment_avg_confidence=args.min_segment_avg_confidence,
            min_segment_max_confidence=args.min_segment_max_confidence,
            same_label_merge_gap=args.same_label_merge_gap,
            nms_suppress_radius=args.nms_suppress_radius,
            vocabulary=vocabulary,
        )

        all_results.append(result)
        prompt_chunks.append(make_prompt_text(result["name"], result["llmInput"]))

    report_rows = []

    for item in all_results:
        report_rows.append({
            "name": item["name"],
            "expected": " ".join(item["expectedSequence"]),
            "raw": " ".join(item["rawSequence"]),
            "exact_match": int(item["exactMatch"]),
            "segment_count": item["segmentCount"],
            "semantic_case_json": str(output_root / item["name"] / f"{item['name']}_semantic_case.json"),
        })

    write_csv(
        output_root / "semantic_sentence_report.csv",
        report_rows,
        [
            "name",
            "expected",
            "raw",
            "exact_match",
            "segment_count",
            "semantic_case_json",
        ],
    )

    summary = {
        "caseCount": len(all_results),
        "exactMatchCount": sum(1 for item in all_results if item["exactMatch"]),
        "exactMatchRate": round(
            sum(1 for item in all_results if item["exactMatch"]) / max(1, len(all_results)),
            6,
        ),
        "settings": {
            "feature_dir": str(feature_dir),
            "model_dir": str(model_dir),
            "trim_padding": args.trim_padding,
            "gap_frames": args.gap_frames,
            "tail_frames": args.tail_frames,
            "window_size": args.window_size,
            "stride": args.stride,
            "confidence_threshold": args.confidence_threshold,
            "margin_threshold": args.margin_threshold,
            "min_segment_avg_confidence": args.min_segment_avg_confidence,
            "min_segment_max_confidence": args.min_segment_max_confidence,
        },
        "results": all_results,
    }

    save_json(output_root / "semantic_sentence_summary.json", summary)

    prompt_path = output_root / "llm_semantic_rerank_prompts.txt"
    prompt_path.write_text(
        "\n\n" + ("=" * 80) + "\n\n".join(prompt_chunks),
        encoding="utf-8",
    )

    print(f"[完成] 已写出 LLM 提示词：{prompt_path}")

    print("\n========== v2-25 语义重排测试完成 ==========")
    print(f"样例数：{len(all_results)}")
    print(f"原始完全匹配数：{summary['exactMatchCount']}")
    print(f"原始完全匹配率：{summary['exactMatchRate']:.4f}")
    print(f"报告：{output_root / 'semantic_sentence_report.csv'}")
    print(f"汇总：{output_root / 'semantic_sentence_summary.json'}")
    print(f"LLM提示词：{prompt_path}")


if __name__ == "__main__":
    main()
