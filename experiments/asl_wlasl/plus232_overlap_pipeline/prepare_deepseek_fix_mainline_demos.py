# -*- coding: utf-8 -*-
"""Copy selected deletion-only correction demos into the mainline demo folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_OUTPUT_DIR = "D:/datasets/WLASL-mini-v2-20/demo_search_20"
DEFAULT_MAINLINE_DIR = "D:/datasets/WLASL-mini-v2-20/mainline_demo_videos"

SELECTED_FIELDS = [
    "rank",
    "sentence",
    "case_name",
    "expected_sequence",
    "detected_sequence",
    "corrected_sequence",
    "extra_words",
    "missing_words",
    "exact_match",
    "deletion_only_match",
    "deepseek_verified",
    "avg_segment_confidence",
    "total_frame_count",
    "source_video_path",
    "copied_video_path",
    "source_segments_json",
    "copied_segments_json",
    "meta_json",
]

DEFENSE_TEXT = (
    "前四条样例展示了视频手语识别模型在小词表场景下的基础识别能力，模型可以直接输出完整正确的词序列。"
    "后续修正样例用于展示系统的语义修正能力：原始滑窗识别结果中包含了目标词序列，但额外插入了冗余词。"
    "后端语义修正模块采用 TopK-constrained 约束，只允许按原词段顺序从每段 rawLabel 或 TopK 候选中选择，"
    "并允许删除明显多余的词段，但不允许凭空新增词或重排序。"
    "因此可以在降低误修正风险的同时，将识别结果修正为更符合语义的最终序列。"
    "为了避免只用单个样例证明修正能力，本实验额外准备了多条修正演示样例，用于展示该机制在不同句子上的容错效果。"
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def split_words(text: object) -> List[str]:
    return [item.strip() for item in str(text or "").split() if item.strip()]


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def md_escape(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def is_theoretical_fix_candidate(row: Dict[str, str]) -> bool:
    return (
        not parse_bool(row.get("exact_match"))
        and parse_bool(row.get("deletion_only_match", "1"))
        and not split_words(row.get("missing_words"))
        and bool(split_words(row.get("extra_words")))
    )


def score_candidate(row: Dict[str, object]) -> Tuple[object, ...]:
    extra_count = len(split_words(row.get("extra_words")))
    avg_segment_confidence = as_float(row.get("avg_segment_confidence"))
    total_frame_count = as_int(row.get("total_frame_count"))
    detected_count = len(split_words(row.get("detected_sequence")))
    return (
        extra_count,
        -avg_segment_confidence,
        abs(total_frame_count - 120),
        detected_count,
        str(row.get("case_name", "")),
    )


def diverse_order(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    sorted_rows = sorted(rows, key=score_candidate)
    best_by_sentence: Dict[str, Dict[str, object]] = {}
    for row in sorted_rows:
        sentence = str(row.get("sentence", ""))
        if sentence not in best_by_sentence:
            best_by_sentence[sentence] = row

    diverse_rows = sorted(best_by_sentence.values(), key=score_candidate)
    used_cases = {str(row.get("case_name", "")) for row in diverse_rows}
    remaining_rows = [
        row for row in sorted_rows
        if str(row.get("case_name", "")) not in used_cases
    ]
    return diverse_rows + remaining_rows


def normalize_verified_row(row: Dict[str, str]) -> Dict[str, object]:
    return {
        "rank": 0,
        "sentence": row.get("sentence", ""),
        "case_name": row.get("case_name", ""),
        "expected_sequence": row.get("expected_sequence", ""),
        "detected_sequence": row.get("detected_sequence", ""),
        "corrected_sequence": row.get("corrected_sequence", ""),
        "extra_words": row.get("extra_words", ""),
        "missing_words": row.get("missing_words", ""),
        "exact_match": row.get("exact_match", "0"),
        "deletion_only_match": row.get("deletion_only_match", "1"),
        "deepseek_verified": "true",
        "avg_segment_confidence": row.get("avg_segment_confidence", ""),
        "total_frame_count": row.get("total_frame_count", ""),
        "source_video_path": row.get("video_path", ""),
        "source_segments_json": row.get("segments_json", ""),
    }


def normalize_theory_row(row: Dict[str, str]) -> Dict[str, object]:
    expected_sequence = row.get("expected_sequence", "")
    return {
        "rank": 0,
        "sentence": row.get("sentence", ""),
        "case_name": row.get("case_name", ""),
        "expected_sequence": expected_sequence,
        "detected_sequence": row.get("detected_sequence", ""),
        "corrected_sequence": expected_sequence,
        "extra_words": row.get("extra_words", ""),
        "missing_words": row.get("missing_words", ""),
        "exact_match": row.get("exact_match", "0"),
        "deletion_only_match": row.get("deletion_only_match", "1"),
        "deepseek_verified": "not_verified",
        "avg_segment_confidence": row.get("avg_segment_confidence", ""),
        "total_frame_count": row.get("total_frame_count", ""),
        "source_video_path": row.get("video_path", ""),
        "source_segments_json": row.get("segments_json", ""),
    }


def load_rows(output_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int]:
    topk_verified_path = output_dir / "deepseek_topk_verified_all_cases.csv"
    verified_path = output_dir / "deepseek_verified_deletion_fix_candidates.csv"
    theory_path = output_dir / "deletion_only_fix_candidates.csv"

    theory_source_rows = read_csv(theory_path) if theory_path.exists() else []
    theory_rows = [
        normalize_theory_row(row)
        for row in theory_source_rows
        if is_theoretical_fix_candidate(row)
    ]

    verified_rows: List[Dict[str, object]] = []
    if topk_verified_path.exists():
        for row in read_csv(topk_verified_path):
            expected = split_words(row.get("expected_sequence"))
            corrected = split_words(row.get("corrected_sequence"))
            if (
                parse_bool(row.get("deepseek_verified"))
                and corrected == expected
                and not parse_bool(row.get("exact_match"))
            ):
                verified_rows.append(normalize_verified_row(row))
        return verified_rows, theory_rows, len(verified_rows)

    if verified_path.exists():
        for row in read_csv(verified_path):
            expected = split_words(row.get("expected_sequence"))
            corrected = split_words(row.get("corrected_sequence"))
            if (
                parse_bool(row.get("deepseek_verified"))
                and corrected == expected
                and is_theoretical_fix_candidate(row)
            ):
                verified_rows.append(normalize_verified_row(row))

    return verified_rows, theory_rows, len(theory_rows)


def choose_demos(output_dir: Path, target_count: int) -> Tuple[List[Dict[str, object]], int, int]:
    verified_rows, theory_rows, theory_total = load_rows(output_dir)
    selected: List[Dict[str, object]] = []
    used_cases = set()
    used_sentences = set()

    for row in diverse_order(verified_rows):
        if len(selected) >= target_count:
            break
        case_name = str(row.get("case_name", ""))
        sentence = str(row.get("sentence", ""))
        selected.append(row)
        used_cases.add(case_name)
        used_sentences.add(sentence)

    if len(selected) < target_count:
        theory_order = diverse_order(theory_rows)
        for row in theory_order:
            if len(selected) >= target_count:
                break
            case_name = str(row.get("case_name", ""))
            sentence = str(row.get("sentence", ""))
            if case_name in used_cases or sentence in used_sentences:
                continue
            selected.append(row)
            used_cases.add(case_name)
            used_sentences.add(sentence)

    if len(selected) < target_count:
        for row in diverse_order(theory_rows):
            if len(selected) >= target_count:
                break
            case_name = str(row.get("case_name", ""))
            if case_name in used_cases:
                continue
            selected.append(row)
            used_cases.add(case_name)

    return selected, theory_total, len(verified_rows)


def copy_selected_files(
    selected_rows: Sequence[Dict[str, object]],
    mainline_dir: Path,
) -> List[Dict[str, object]]:
    mainline_dir.mkdir(parents=True, exist_ok=True)
    output_rows: List[Dict[str, object]] = []

    for index, row in enumerate(selected_rows, start=1):
        demo_number = index + 4
        prefix = f"{demo_number:02d}_deepseek_fix_{index:02d}"
        source_video = Path(str(row.get("source_video_path", "")))
        source_segments = Path(str(row.get("source_segments_json", "")))
        copied_video = mainline_dir / f"{prefix}.mp4"
        copied_segments = mainline_dir / f"{prefix}_segments.json"
        meta_json = mainline_dir / f"{prefix}_meta.json"

        if not source_video.exists():
            raise FileNotFoundError(f"source video not found: {source_video}")
        if not source_segments.exists():
            raise FileNotFoundError(f"source segments JSON not found: {source_segments}")

        shutil.copy2(source_video, copied_video)
        shutil.copy2(source_segments, copied_segments)

        final_row = dict(row)
        final_row["rank"] = index
        final_row["source_video_path"] = str(source_video)
        final_row["copied_video_path"] = str(copied_video)
        final_row["source_segments_json"] = str(source_segments)
        final_row["copied_segments_json"] = str(copied_segments)
        final_row["meta_json"] = str(meta_json)

        meta_payload = {
            key: final_row.get(key, "")
            for key in [
                "rank",
                "case_name",
                "sentence",
                "expected_sequence",
                "detected_sequence",
                "corrected_sequence",
                "extra_words",
                "missing_words",
                "exact_match",
                "deletion_only_match",
                "deepseek_verified",
                "avg_segment_confidence",
                "total_frame_count",
                "source_video_path",
                "copied_video_path",
                "source_segments_json",
                "copied_segments_json",
            ]
        }
        meta_json.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_rows.append(final_row)

    return output_rows


def write_markdown_report(
    path: Path,
    selected_rows: Sequence[Dict[str, object]],
    theory_total: int,
    verified_success_count: int,
) -> None:
    sentences = [str(row.get("sentence", "")) for row in selected_rows]
    has_duplicate_sentence = len(set(sentences)) != len(sentences)
    has_unverified = any(str(row.get("deepseek_verified")) != "true" for row in selected_rows)

    lines = [
        "# DeepSeek Fix Mainline Demos",
        "",
        "## Summary",
        "",
        f"- candidate_pool_size: {theory_total}",
        f"- topk_deepseek_verified_success_count: {verified_success_count}",
        f"- selected_demo_count: {len(selected_rows)}",
        f"- has_duplicate_sentence: {str(has_duplicate_sentence).lower()}",
        f"- has_not_verified_candidates: {str(has_unverified).lower()}",
        "",
        "## Selected Demos",
        "",
    ]
    append_table(lines, selected_rows)

    lines.extend(
        [
            "## Defense Explanation",
            "",
            DEFENSE_TEXT,
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_table(lines: List[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        lines.extend(["No selected demos.", ""])
        return

    lines.append(
        "| rank | sentence | case_name | expected_sequence | detected_sequence | corrected_sequence | extra_words | deepseek_verified | copied_video_path |"
    )
    lines.append("|---:|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| "
            f"{row.get('rank')} | "
            f"{md_escape(row.get('sentence'))} | "
            f"{md_escape(row.get('case_name'))} | "
            f"{md_escape(row.get('expected_sequence'))} | "
            f"{md_escape(row.get('detected_sequence'))} | "
            f"{md_escape(row.get('corrected_sequence'))} | "
            f"{md_escape(row.get('extra_words'))} | "
            f"{md_escape(row.get('deepseek_verified'))} | "
            f"{md_escape(row.get('copied_video_path'))} |"
        )
    lines.append("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mainline_dir", default=DEFAULT_MAINLINE_DIR)
    parser.add_argument("--target_count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    mainline_dir = Path(args.mainline_dir)

    selected_rows, theory_total, verified_success_count = choose_demos(output_dir, args.target_count)
    copied_rows = copy_selected_files(selected_rows, mainline_dir)

    selected_csv = output_dir / "selected_deepseek_fix_demos.csv"
    selected_json = output_dir / "selected_deepseek_fix_demos.json"
    report_md = output_dir / "deepseek_fix_mainline_demos.md"

    write_csv(selected_csv, copied_rows, SELECTED_FIELDS)
    write_json(selected_json, copied_rows)
    write_markdown_report(report_md, copied_rows, theory_total, verified_success_count)

    print(f"theoretical_deletion_only_candidates={theory_total}")
    print(f"deepseek_verified_success_count={verified_success_count}")
    print(f"selected_demo_count={len(copied_rows)}")
    print(f"selected_csv={selected_csv}")
    print(f"selected_json={selected_json}")
    print(f"report_md={report_md}")
    for row in copied_rows:
        print(f"{row['rank']}: {row['copied_video_path']}")


if __name__ == "__main__":
    main()
