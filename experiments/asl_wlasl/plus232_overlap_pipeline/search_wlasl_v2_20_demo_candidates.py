# -*- coding: utf-8 -*-
"""
自动筛选 WLASL-mini-v2-20 句子 Demo 候选。

本脚本只负责编排实验：
1. 调用 compose_wlasl_sentence_video_trimmed.py 生成拼接视频；
2. 调用 infer_wlasl_sentence_video.py 做滑窗识别；
3. 读取每个 case 的 segments.json；
4. 统计 exact_match / deletion_only_match；
5. 输出完整报告和每个句子的最佳候选。

注意：
- 不重写 MediaPipe / 特征提取 / 模型推理底层逻辑；
- 不修改模型文件、训练脚本、FastAPI、Java 后端或 HarmonyOS 端。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ALLOWED_LABELS = {
    "bad",
    "deaf",
    "family",
    "friend",
    "go",
    "help",
    "home",
    "language",
    "learn",
    "meet",
    "no",
    "please",
    "school",
    "sorry",
    "teacher",
    "today",
    "tomorrow",
    "why",
    "work",
    "you",
}

REMOVED_LABELS = {"good", "want", "what", "who", "yes"}

DEFAULT_SENTENCES = [
    "deaf,learn,language",
    "teacher,school,today",
    "please,help,teacher",
    "sorry,friend,today",
    "go,work,tomorrow",
    "friend,meet,today",
    "family,help,friend",
    "you,help,teacher",
    "you,learn,today",
    "please,teacher,today",
    "sorry,teacher,tomorrow",
    "deaf,teacher,help",
    "school,learn,today",
    "go,school,tomorrow",
    "friend,teacher,school",
]

DEFAULT_SAMPLE_POLICIES = ["random"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 7, 11, 13, 17, 23, 42]
DEFAULT_TRIM_PADDING_VALUES = [0]
DEFAULT_GAP_FRAMES_VALUES = [0]
DEFAULT_TAIL_FRAMES_VALUES = [0]
DEFAULT_BASELINE_POLICIES = ["first", "largest"]

PREFERRED_MAINLINE_SENTENCES = {
    "deaf,learn,language",
    "teacher,school,today",
    "please,help,teacher",
    "sorry,friend,today",
    "deaf,teacher,help",
    "school,learn,today",
    "you,help,teacher",
}

CAUTIOUS_LABELS = {"no", "home", "meet"}

FIELDNAMES = [
    "case_name",
    "sentence",
    "expected_sequence",
    "detected_sequence",
    "exact_match",
    "deletion_only_match",
    "missing_words",
    "extra_words",
    "segment_count",
    "expected_count",
    "sample_policy",
    "seed",
    "trim_padding",
    "gap_frames",
    "tail_frames",
    "video_path",
    "segments_json",
    "status",
    "error_message",
    "avg_segment_confidence",
    "min_segment_confidence",
    "max_segment_confidence",
    "total_frame_count",
]


@dataclass(frozen=True)
class SearchCase:
    """单个搜索 case 的配置。"""

    sentence: str
    sample_policy: str
    seed: int
    trim_padding: int
    gap_frames: int
    tail_frames: int

    @property
    def expected_sequence(self) -> List[str]:
        """期望 gloss 序列。"""
        return parse_sentence(self.sentence)

    @property
    def case_name(self) -> str:
        """可读 case 名称，用于目录和视频文件命名。"""
        sentence_slug = "_".join(self.expected_sequence)
        return (
            f"{sentence_slug}__{self.sample_policy}__seed{self.seed}"
            f"__tp{self.trim_padding}_gap{self.gap_frames}_tail{self.tail_frames}"
        )


def parse_sentence(sentence: str) -> List[str]:
    """解析逗号分隔 gloss，并统一为小写。"""
    return [
        item.strip().lower()
        for item in sentence.replace("，", ",").split(",")
        if item.strip()
    ]


def validate_sentence(sentence: str) -> None:
    """校验候选句子只包含 20 词版标签。"""
    labels = parse_sentence(sentence)
    if not 2 <= len(labels) <= 3:
        raise ValueError(f"候选句子只允许 2 到 3 个词：{sentence}")

    invalid = [label for label in labels if label not in ALLOWED_LABELS]
    removed = [label for label in labels if label in REMOVED_LABELS]

    if invalid or removed:
        raise ValueError(
            f"候选句子包含非法标签：sentence={sentence}, "
            f"invalid={invalid}, removed={removed}"
        )


def is_subsequence(expected: List[str], detected: List[str]) -> bool:
    """
    判断 expected 是否是 detected 的有序子序列。

    如果是，说明 detected 可以通过只删除多余词变成 expected。
    """
    cursor = 0

    for label in detected:
        if cursor < len(expected) and label == expected[cursor]:
            cursor += 1

    return cursor == len(expected)


def sequence_diff(expected: List[str], detected: List[str]) -> Tuple[List[str], List[str]]:
    """
    计算 missing_words 和 extra_words。

    采用贪心有序匹配：
    - 匹配到的词视为保留；
    - detected 中未匹配的词视为 extra；
    - expected 中未匹配的词视为 missing。
    """
    expected_index = 0
    matched_expected = [False for _ in expected]
    matched_detected = [False for _ in detected]

    for detected_index, label in enumerate(detected):
        if expected_index < len(expected) and label == expected[expected_index]:
            matched_expected[expected_index] = True
            matched_detected[detected_index] = True
            expected_index += 1

    missing_words = [
        label
        for index, label in enumerate(expected)
        if not matched_expected[index]
    ]
    extra_words = [
        label
        for index, label in enumerate(detected)
        if not matched_detected[index]
    ]

    return missing_words, extra_words


def load_json(path: Path) -> Dict:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: object) -> None:
    """写出 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    """写出 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_cases(
    sentences: Sequence[str],
    include_baselines: bool,
) -> List[SearchCase]:
    """根据默认搜索空间生成所有 case。"""
    cases: List[SearchCase] = []

    for sentence in sentences:
        validate_sentence(sentence)

        if include_baselines:
            for policy in DEFAULT_BASELINE_POLICIES:
                cases.append(
                    SearchCase(
                        sentence=sentence,
                        sample_policy=policy,
                        seed=42,
                        trim_padding=0,
                        gap_frames=0,
                        tail_frames=0,
                    )
                )

        for policy in DEFAULT_SAMPLE_POLICIES:
            for seed in DEFAULT_SEEDS:
                for trim_padding in DEFAULT_TRIM_PADDING_VALUES:
                    for gap_frames in DEFAULT_GAP_FRAMES_VALUES:
                        for tail_frames in DEFAULT_TAIL_FRAMES_VALUES:
                            cases.append(
                                SearchCase(
                                    sentence=sentence,
                                    sample_policy=policy,
                                    seed=seed,
                                    trim_padding=trim_padding,
                                    gap_frames=gap_frames,
                                    tail_frames=tail_frames,
                                )
                            )

    return cases


def run_command(command: Sequence[str]) -> None:
    """
    执行子进程命令。

    子脚本输出保留到控制台，便于观察 MediaPipe / TensorFlow 进度。
    """
    subprocess.run(command, check=True)


def run_compose(
    compose_script: Path,
    case: SearchCase,
    samples_csv: Path,
    video_path: Path,
    python_exe: str,
) -> None:
    """调用现有拼接脚本生成句子视频。"""
    command = [
        python_exe,
        str(compose_script),
        "--samples_csv",
        str(samples_csv),
        "--sentence",
        case.sentence,
        "--output_path",
        str(video_path),
        "--sample_policy",
        case.sample_policy,
        "--seed",
        str(case.seed),
        "--trim_padding",
        str(case.trim_padding),
        "--gap_frames",
        str(case.gap_frames),
        "--tail_frames",
        str(case.tail_frames),
    ]

    run_command(command)


def run_infer(
    infer_script: Path,
    case: SearchCase,
    video_path: Path,
    feature_dir: Path,
    model_dir: Path,
    output_dir: Path,
    python_exe: str,
) -> None:
    """调用现有推理脚本识别句子视频。"""
    command = [
        python_exe,
        str(infer_script),
        "--video_path",
        str(video_path),
        "--feature_dir",
        str(feature_dir),
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--expected",
        case.sentence,
        "--window_size",
        "20",
        "--stride",
        "2",
        "--confidence_threshold",
        "0.45",
        "--margin_threshold",
        "0.05",
        "--min_segment_windows",
        "2",
        "--min_segment_avg_confidence",
        "0.75",
        "--min_segment_max_confidence",
        "0.85",
        "--same_label_merge_gap",
        "8",
        "--nms_suppress_radius",
        "6",
    ]

    run_command(command)


def confidence_stats(segments: List[Dict[str, object]]) -> Tuple[float, float, float]:
    """统计最终词段平均 / 最低 / 最高平均置信度。"""
    values = []

    for segment in segments:
        try:
            values.append(float(segment.get("avg_confidence", 0.0)))
        except Exception:
            values.append(0.0)

    if not values:
        return 0.0, 0.0, 0.0

    return (
        round(sum(values) / len(values), 6),
        round(min(values), 6),
        round(max(values), 6),
    )


def make_failed_row(
    case: SearchCase,
    video_path: Path,
    case_output_dir: Path,
    error: Exception,
) -> Dict[str, object]:
    """构造失败 case 的报告行。"""
    expected = case.expected_sequence
    return {
        "case_name": case.case_name,
        "sentence": case.sentence,
        "expected_sequence": " ".join(expected),
        "detected_sequence": "",
        "exact_match": 0,
        "deletion_only_match": 0,
        "missing_words": " ".join(expected),
        "extra_words": "",
        "segment_count": 0,
        "expected_count": len(expected),
        "sample_policy": case.sample_policy,
        "seed": case.seed,
        "trim_padding": case.trim_padding,
        "gap_frames": case.gap_frames,
        "tail_frames": case.tail_frames,
        "video_path": str(video_path),
        "segments_json": str(case_output_dir / f"{video_path.stem}_segments.json"),
        "status": "failed",
        "error_message": str(error),
        "avg_segment_confidence": 0.0,
        "min_segment_confidence": 0.0,
        "max_segment_confidence": 0.0,
        "total_frame_count": 0,
    }


def make_success_row(
    case: SearchCase,
    video_path: Path,
    segments_json: Path,
    payload: Dict,
) -> Dict[str, object]:
    """根据推理 JSON 构造成功 case 的报告行。"""
    expected = [
        str(item).strip().lower()
        for item in payload.get("expected_sequence", case.expected_sequence)
        if str(item).strip()
    ]
    detected = [
        str(item).strip().lower()
        for item in payload.get("detected_sequence", [])
        if str(item).strip()
    ]

    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        segments = []

    missing_words, extra_words = sequence_diff(expected, detected)
    avg_conf, min_conf, max_conf = confidence_stats(segments)
    exact_match = expected == detected
    deletion_only_match = is_subsequence(expected, detected)
    video_info = payload.get("video_info", {})

    if not isinstance(video_info, dict):
        video_info = {}

    return {
        "case_name": case.case_name,
        "sentence": case.sentence,
        "expected_sequence": " ".join(expected),
        "detected_sequence": " ".join(detected),
        "exact_match": int(exact_match),
        "deletion_only_match": int(deletion_only_match),
        "missing_words": " ".join(missing_words),
        "extra_words": " ".join(extra_words),
        "segment_count": len(segments),
        "expected_count": len(expected),
        "sample_policy": case.sample_policy,
        "seed": case.seed,
        "trim_padding": case.trim_padding,
        "gap_frames": case.gap_frames,
        "tail_frames": case.tail_frames,
        "video_path": str(video_path),
        "segments_json": str(segments_json),
        "status": "ok",
        "error_message": "",
        "avg_segment_confidence": avg_conf,
        "min_segment_confidence": min_conf,
        "max_segment_confidence": max_conf,
        "total_frame_count": int(video_info.get("total_frame_count", 0) or 0),
    }


def run_one_case(
    case: SearchCase,
    compose_script: Path,
    infer_script: Path,
    samples_csv: Path,
    feature_dir: Path,
    model_dir: Path,
    video_output_dir: Path,
    cases_output_dir: Path,
    python_exe: str,
    reuse_existing: bool = False,
) -> Dict[str, object]:
    """执行单个 case：拼接、推理、读取结果。"""
    video_path = video_output_dir / f"{case.case_name}.mp4"
    case_output_dir = cases_output_dir / case.case_name
    segments_json = case_output_dir / f"{video_path.stem}_segments.json"

    try:
        if reuse_existing and video_path.exists() and segments_json.exists():
            payload = load_json(segments_json)
            row = make_success_row(
                case=case,
                video_path=video_path,
                segments_json=segments_json,
                payload=payload,
            )
            print(
                "[case reused] "
                f"{row['case_name']} | status={row['status']} | "
                f"expected={row['expected_sequence']} | detected={row['detected_sequence']} | "
                f"exact={row['exact_match']} | deletion_only={row['deletion_only_match']} | "
                f"extra={row['extra_words']} | missing={row['missing_words']}"
            )
            return row

        run_compose(
            compose_script=compose_script,
            case=case,
            samples_csv=samples_csv,
            video_path=video_path,
            python_exe=python_exe,
        )
        run_infer(
            infer_script=infer_script,
            case=case,
            video_path=video_path,
            feature_dir=feature_dir,
            model_dir=model_dir,
            output_dir=case_output_dir,
            python_exe=python_exe,
        )

        if not segments_json.exists():
            raise RuntimeError(f"推理结果不存在：{segments_json}")

        payload = load_json(segments_json)
        row = make_success_row(
            case=case,
            video_path=video_path,
            segments_json=segments_json,
            payload=payload,
        )

        print(
            "[case] "
            f"{row['case_name']} | status={row['status']} | "
            f"expected={row['expected_sequence']} | detected={row['detected_sequence']} | "
            f"exact={row['exact_match']} | deletion_only={row['deletion_only_match']} | "
            f"extra={row['extra_words']} | missing={row['missing_words']}"
        )
        return row
    except Exception as error:
        row = make_failed_row(
            case=case,
            video_path=video_path,
            case_output_dir=case_output_dir,
            error=error,
        )
        print(f"[case failed] {case.case_name}: {error}")
        return row


def score_case(row: Dict[str, object]) -> Tuple[object, ...]:
    """
    候选排序分数。

    排序优先级：
    1. exact_match = true；
    2. deletion_only_match = true；
    3. segment_count 越接近 expected_count 越好；
    4. extra_words 越少越好；
    5. avg_segment_confidence 越高越好；
    6. total_frame_count 适中优先。
    """
    segment_count = int(row.get("segment_count", 0) or 0)
    expected_count = int(row.get("expected_count", 0) or 0)
    extra_count = len(str(row.get("extra_words", "")).split())
    frame_count = int(row.get("total_frame_count", 0) or 0)
    confidence = float(row.get("avg_segment_confidence", 0.0) or 0.0)

    return (
        int(row.get("exact_match", 0) or 0),
        int(row.get("deletion_only_match", 0) or 0),
        -abs(segment_count - expected_count),
        -extra_count,
        confidence,
        -abs(frame_count - 120),
    )


def best_per_sentence(
    rows: Iterable[Dict[str, object]],
    match_key: str,
) -> List[Dict[str, object]]:
    """每个 sentence 选择一个最佳匹配 case。"""
    best: Dict[str, Dict[str, object]] = {}

    for row in rows:
        if row.get("status") != "ok":
            continue
        if int(row.get(match_key, 0) or 0) != 1:
            continue

        sentence = str(row["sentence"])
        if sentence not in best or score_case(row) > score_case(best[sentence]):
            best[sentence] = row

    return sorted(best.values(), key=score_case, reverse=True)


def as_int(value: object, default: int = 0) -> int:
    """Best-effort integer conversion for report sorting."""
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion for report sorting."""
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def word_count(value: object) -> int:
    """Count blank-separated labels in a report field."""
    return len(str(value or "").split())


def md_escape(value: object) -> str:
    """Escape pipe characters so values are safe in Markdown tables."""
    return str(value if value is not None else "").replace("|", "\\|")


def summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, int]:
    """Build aggregate stats for the Markdown report."""
    success_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    exact_rows = [row for row in success_rows if as_int(row.get("exact_match")) == 1]
    deletion_rows = [
        row for row in success_rows
        if as_int(row.get("deletion_only_match")) == 1
    ]

    return {
        "total": len(rows),
        "success": len(success_rows),
        "failed": len(failed_rows),
        "exact": len(exact_rows),
        "deletion_only": len(deletion_rows),
    }


def recommendation_score(row: Dict[str, object]) -> Tuple[object, ...]:
    """Rank demo candidates for the graduation-demo mainline."""
    sentence = str(row.get("sentence", ""))
    labels = set(parse_sentence(sentence))
    frame_count = as_int(row.get("total_frame_count"))
    confidence = as_float(row.get("avg_segment_confidence"))

    return (
        as_int(row.get("exact_match")),
        as_int(row.get("deletion_only_match")),
        -word_count(row.get("missing_words")),
        -word_count(row.get("extra_words")),
        int(sentence in PREFERRED_MAINLINE_SENTENCES),
        -int(bool(labels & CAUTIOUS_LABELS)),
        confidence,
        -abs(frame_count - 120),
    )


def recommended_mainline_rows(rows: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
    """Pick strong deletion-only-or-better candidates for the mainline report."""
    valid_rows = [
        row for row in rows
        if row.get("status") == "ok"
        and as_int(row.get("deletion_only_match")) == 1
        and word_count(row.get("missing_words")) == 0
    ]

    best: Dict[str, Dict[str, object]] = {}
    for row in valid_rows:
        sentence = str(row.get("sentence", ""))
        if sentence not in best or recommendation_score(row) > recommendation_score(best[sentence]):
            best[sentence] = row

    selected = sorted(best.values(), key=recommendation_score, reverse=True)[:limit]
    selected_cases = {str(row.get("case_name", "")) for row in selected}

    if len(selected) < limit:
        for row in sorted(valid_rows, key=recommendation_score, reverse=True):
            case_name = str(row.get("case_name", ""))
            if case_name in selected_cases:
                continue

            selected.append(row)
            selected_cases.add(case_name)
            if len(selected) >= limit:
                break

    return selected


def sentence_stats(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate report rows by sentence."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sentence", "")), []).append(row)

    stats: List[Dict[str, object]] = []
    for sentence, group in grouped.items():
        labels = set(parse_sentence(sentence))
        exact = sum(1 for row in group if as_int(row.get("exact_match")) == 1)
        deletion = sum(1 for row in group if as_int(row.get("deletion_only_match")) == 1)
        failed = sum(1 for row in group if row.get("status") != "ok")

        hints: List[str] = []
        if exact == 0:
            hints.append("no exact")
        if deletion == 0:
            hints.append("no deletion-only")
        if failed:
            hints.append(f"{failed} failed")
        if labels & CAUTIOUS_LABELS:
            hints.append("contains cautious label")

        stats.append(
            {
                "sentence": sentence,
                "total": len(group),
                "exact": exact,
                "deletion_only": deletion,
                "failed": failed,
                "hint": "; ".join(hints) if hints else "stable so far",
            }
        )

    return sorted(
        stats,
        key=lambda item: (
            as_int(item["exact"]),
            as_int(item["deletion_only"]),
            -as_int(item["failed"]),
        ),
        reverse=True,
    )


def append_candidate_table(
    lines: List[str],
    rows: Sequence[Dict[str, object]],
    title: str,
) -> None:
    """Append a compact candidate table to the Markdown report."""
    lines.append(f"## {title}")
    lines.append("")
    if not rows:
        lines.append("No candidates found.")
        lines.append("")
        return

    lines.append(
        "| sentence | case_name | detected_sequence | exact | deletion_only | extra_words | avg_conf | frames | video_path | segments_json |"
    )
    lines.append(
        "|---|---|---|---:|---:|---|---:|---:|---|---|"
    )
    for row in rows:
        lines.append(
            "| "
            f"{md_escape(row.get('sentence'))} | "
            f"{md_escape(row.get('case_name'))} | "
            f"{md_escape(row.get('detected_sequence'))} | "
            f"{as_int(row.get('exact_match'))} | "
            f"{as_int(row.get('deletion_only_match'))} | "
            f"{md_escape(row.get('extra_words'))} | "
            f"{as_float(row.get('avg_segment_confidence')):.6f} | "
            f"{as_int(row.get('total_frame_count'))} | "
            f"{md_escape(row.get('video_path'))} | "
            f"{md_escape(row.get('segments_json'))} |"
        )
    lines.append("")


def write_mainline_markdown_report(
    path: Path,
    rows: List[Dict[str, object]],
    best_exact_rows: List[Dict[str, object]],
    best_deletion_rows: List[Dict[str, object]],
) -> None:
    """Write the stage report requested by the v2-20 demo search task."""
    stats = summarize_rows(rows)
    recommended_rows = recommended_mainline_rows(rows, limit=5)
    unstable_rows = [
        item for item in sentence_stats(rows)
        if item["hint"] != "stable so far"
    ]
    next_step = (
        "Stop here for integration rehearsal: at least 4 exact demos were found."
        if stats["exact"] >= 4
        else "Continue the full search: fewer than 4 exact demos were found in this stage."
    )

    lines: List[str] = [
        "# WLASL-mini-v2-20 Demo Mainline Candidates",
        "",
        "## Search Scope",
        "",
        f"- total_cases: {stats['total']}",
        f"- success_cases: {stats['success']}",
        f"- failed_cases: {stats['failed']}",
        f"- exact_match_cases: {stats['exact']}",
        f"- deletion_only_match_cases: {stats['deletion_only']}",
        "",
    ]

    append_candidate_table(lines, best_exact_rows, "Best Exact Match Per Sentence")
    append_candidate_table(lines, best_deletion_rows, "Best Deletion-Only Match Per Sentence")
    append_candidate_table(lines, recommended_rows, "Recommended Mainline Demo Top 5")

    lines.append("## Unstable Sentence Hints")
    lines.append("")
    if unstable_rows:
        lines.append("| sentence | total | exact | deletion_only | failed | hint |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for item in unstable_rows:
            lines.append(
                "| "
                f"{md_escape(item['sentence'])} | "
                f"{item['total']} | "
                f"{item['exact']} | "
                f"{item['deletion_only']} | "
                f"{item['failed']} | "
                f"{md_escape(item['hint'])} |"
            )
    else:
        lines.append("No unstable sentences detected in this stage.")
    lines.append("")

    lines.append("## Next Step Suggestion")
    lines.append("")
    lines.append(next_step)
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(rows: List[Dict[str, object]]) -> None:
    """打印控制台最终摘要。"""
    total = len(rows)
    success_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    exact_rows = [row for row in success_rows if int(row.get("exact_match", 0) or 0) == 1]
    deletion_rows = [
        row for row in success_rows
        if int(row.get("deletion_only_match", 0) or 0) == 1
    ]

    print("\n========== WLASL v2-20 Demo 搜索摘要 ==========")
    print(f"总 case 数：{total}")
    print(f"成功执行 case 数：{len(success_rows)}")
    print(f"失败 case 数：{len(failed_rows)}")
    print(f"exact_match 数量：{len(exact_rows)}")
    print(f"deletion_only_match 数量：{len(deletion_rows)}")

    print("\n========== 每个 sentence 的最佳 case ==========")
    for row in best_per_sentence(success_rows, "deletion_only_match"):
        print(
            f"{row['sentence']} -> {row['case_name']} | "
            f"detected={row['detected_sequence']} | "
            f"exact={row['exact_match']} | deletion_only={row['deletion_only_match']}"
        )

    print("\n========== 推荐优先演示 top 5 demo ==========")
    top_rows = sorted(success_rows, key=score_case, reverse=True)[:5]
    for index, row in enumerate(top_rows, start=1):
        print(
            f"{index}. {row['case_name']} | "
            f"expected={row['expected_sequence']} | detected={row['detected_sequence']} | "
            f"exact={row['exact_match']} | deletion_only={row['deletion_only_match']} | "
            f"avg={row['avg_segment_confidence']}"
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples_csv",
        default="D:/datasets/WLASL-mini-v2-20/samples.csv",
    )
    parser.add_argument(
        "--feature_dir",
        default="D:/datasets/WLASL-mini-v2-20/features_20f_plus",
    )
    parser.add_argument(
        "--model_dir",
        default="D:/datasets/WLASL-mini-v2-20/models_20f_plus",
    )
    parser.add_argument(
        "--video_output_dir",
        default="D:/datasets/WLASL-mini-v2-20/demo_videos_trimmed/search_20",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini-v2-20/demo_search_20",
    )
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--only_sentence", default="")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--no_baselines", action="store_true")
    parser.add_argument("--reuse_existing", action="store_true")

    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    compose_script = script_dir / "compose_wlasl_sentence_video_trimmed.py"
    infer_script = script_dir / "infer_wlasl_sentence_video.py"

    if not compose_script.exists():
        raise RuntimeError(f"找不到拼接脚本：{compose_script}")
    if not infer_script.exists():
        raise RuntimeError(f"找不到推理脚本：{infer_script}")

    samples_csv = Path(args.samples_csv)
    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    video_output_dir = Path(args.video_output_dir)
    output_dir = Path(args.output_dir)
    cases_output_dir = output_dir / "cases"

    video_output_dir.mkdir(parents=True, exist_ok=True)
    cases_output_dir.mkdir(parents=True, exist_ok=True)

    sentences = DEFAULT_SENTENCES
    if args.only_sentence.strip():
        validate_sentence(args.only_sentence)
        sentences = [args.only_sentence.strip()]

    cases = build_cases(
        sentences=sentences,
        include_baselines=not args.no_baselines,
    )

    if args.max_cases and args.max_cases > 0:
        cases = cases[:args.max_cases]

    print("========== WLASL-mini-v2-20 Demo 候选搜索 ==========")
    print(f"[信息] samples_csv：{samples_csv}")
    print(f"[信息] feature_dir：{feature_dir}")
    print(f"[信息] model_dir：{model_dir}")
    print(f"[信息] video_output_dir：{video_output_dir}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] python_exe：{args.python_exe}")
    print(f"[信息] case 数：{len(cases)}")

    rows: List[Dict[str, object]] = []

    for index, case in enumerate(cases, start=1):
        print(f"\n========== case {index}/{len(cases)}：{case.case_name} ==========")
        row = run_one_case(
            case=case,
            compose_script=compose_script,
            infer_script=infer_script,
            samples_csv=samples_csv,
            feature_dir=feature_dir,
            model_dir=model_dir,
            video_output_dir=video_output_dir,
            cases_output_dir=cases_output_dir,
            python_exe=args.python_exe,
            reuse_existing=args.reuse_existing,
        )
        rows.append(row)

    report_csv = output_dir / "demo_candidate_report.csv"
    report_json = output_dir / "demo_candidate_report.json"
    best_exact_csv = output_dir / "best_exact_matches.csv"
    best_deletion_csv = output_dir / "best_deletion_only_matches.csv"
    mainline_markdown = output_dir / "demo_mainline_candidates.md"
    best_exact_rows = best_per_sentence(rows, "exact_match")
    best_deletion_rows = best_per_sentence(rows, "deletion_only_match")

    write_csv(report_csv, rows, FIELDNAMES)
    save_json(report_json, rows)
    write_csv(best_exact_csv, best_exact_rows, FIELDNAMES)
    write_csv(best_deletion_csv, best_deletion_rows, FIELDNAMES)
    write_mainline_markdown_report(
        mainline_markdown,
        rows,
        best_exact_rows,
        best_deletion_rows,
    )

    print_summary(rows)

    print("\n========== 输出文件 ==========")
    print(f"完整 CSV：{report_csv}")
    print(f"完整 JSON：{report_json}")
    print(f"最佳 exact：{best_exact_csv}")
    print(f"最佳 deletion-only：{best_deletion_csv}")
    print(f"mainline Markdown: {mainline_markdown}")


if __name__ == "__main__":
    main()
