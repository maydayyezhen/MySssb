# -*- coding: utf-8 -*-
"""Select theoretical deletion-only correction demos from an existing search CSV.

This script is intentionally read-only with respect to model/search results. It
does not run MediaPipe, TensorFlow, or video inference again.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_REPORT_CSV = "D:/datasets/WLASL-mini-v2-20/demo_search_20/demo_candidate_report.csv"
DEFAULT_OUTPUT_DIR = "D:/datasets/WLASL-mini-v2-20/demo_search_20"

OUTPUT_FIELDS = [
    "rank",
    "sentence",
    "case_name",
    "expected_sequence",
    "detected_sequence",
    "extra_words",
    "missing_words",
    "extra_count",
    "expected_count",
    "detected_count",
    "exact_match",
    "deletion_only_match",
    "avg_segment_confidence",
    "total_frame_count",
    "video_path",
    "segments_json",
    "status",
]


def parse_bool(value: object) -> bool:
    """Parse common CSV boolean values."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def split_words(text: object) -> List[str]:
    """Convert a blank-separated text field to a label list."""
    return [item.strip() for item in str(text or "").split() if item.strip()]


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or default))
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


def is_deletion_only_fix_candidate(row: Dict[str, str]) -> bool:
    return (
        row.get("status") == "ok"
        and not parse_bool(row.get("exact_match"))
        and parse_bool(row.get("deletion_only_match"))
        and not split_words(row.get("missing_words"))
        and bool(split_words(row.get("extra_words")))
    )


def score_candidate(row: Dict[str, object]) -> Tuple[object, ...]:
    """Lower tuple is better for deletion-only correction demos."""
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


def rank_with_sentence_diversity(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
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


def normalize_candidate(row: Dict[str, str]) -> Dict[str, object]:
    expected = split_words(row.get("expected_sequence"))
    detected = split_words(row.get("detected_sequence"))
    extra = split_words(row.get("extra_words"))
    missing = split_words(row.get("missing_words"))

    return {
        "rank": 0,
        "sentence": row.get("sentence", ""),
        "case_name": row.get("case_name", ""),
        "expected_sequence": " ".join(expected),
        "detected_sequence": " ".join(detected),
        "extra_words": " ".join(extra),
        "missing_words": " ".join(missing),
        "extra_count": len(extra),
        "expected_count": len(expected),
        "detected_count": len(detected),
        "exact_match": int(parse_bool(row.get("exact_match"))),
        "deletion_only_match": int(parse_bool(row.get("deletion_only_match"))),
        "avg_segment_confidence": row.get("avg_segment_confidence", ""),
        "total_frame_count": row.get("total_frame_count", ""),
        "video_path": row.get("video_path", ""),
        "segments_json": row.get("segments_json", ""),
        "status": row.get("status", ""),
    }


def write_markdown_report(
    path: Path,
    total_count: int,
    ranked_rows: Sequence[Dict[str, object]],
) -> None:
    top_rows = list(ranked_rows[:10])
    lines = [
        "# Deletion-Only Fix Demo Candidates",
        "",
        "## Summary",
        "",
        f"- total_cases: {total_count}",
        f"- deletion_only_fix_candidates: {len(ranked_rows)}",
        "",
        "## Recommended Theoretical Candidates Top 10",
        "",
    ]

    append_table(lines, top_rows)

    lines.extend(
        [
            "## Detailed Candidates",
            "",
        ]
    )
    append_table(lines, ranked_rows)

    lines.extend(
        [
            "## Reminder",
            "",
            "These are theoretical deletion-only candidates selected from existing search results. "
            "They still need verification through the existing backend DeepSeek semantic correction chain.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_table(lines: List[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        lines.extend(["No candidates found.", ""])
        return

    lines.append(
        "| rank | sentence | case_name | expected_sequence | detected_sequence | extra_words | avg_conf | frames | video_path | segments_json |"
    )
    lines.append("|---:|---|---|---|---|---|---:|---:|---|---|")
    for row in rows:
        lines.append(
            "| "
            f"{row.get('rank')} | "
            f"{md_escape(row.get('sentence'))} | "
            f"{md_escape(row.get('case_name'))} | "
            f"{md_escape(row.get('expected_sequence'))} | "
            f"{md_escape(row.get('detected_sequence'))} | "
            f"{md_escape(row.get('extra_words'))} | "
            f"{as_float(row.get('avg_segment_confidence')):.6f} | "
            f"{as_int(row.get('total_frame_count'))} | "
            f"{md_escape(row.get('video_path'))} | "
            f"{md_escape(row.get('segments_json'))} |"
        )
    lines.append("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_csv", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_csv = Path(args.report_csv)
    output_dir = Path(args.output_dir)

    rows = read_csv(report_csv)
    candidates = [
        normalize_candidate(row)
        for row in rows
        if is_deletion_only_fix_candidate(row)
    ]
    ranked_rows = rank_with_sentence_diversity(candidates)

    if args.top_k > 0:
        ranked_rows = ranked_rows[:args.top_k]

    for index, row in enumerate(ranked_rows, start=1):
        row["rank"] = index

    candidates_csv = output_dir / "deletion_only_fix_candidates.csv"
    candidates_json = output_dir / "deletion_only_fix_candidates.json"
    report_md = output_dir / "deletion_only_fix_demo.md"

    write_csv(candidates_csv, ranked_rows, OUTPUT_FIELDS)
    write_json(candidates_json, ranked_rows)
    write_markdown_report(report_md, len(rows), ranked_rows)

    print(f"total_cases={len(rows)}")
    print(f"deletion_only_fix_candidates={len(ranked_rows)}")
    print(f"candidates_csv={candidates_csv}")
    print(f"candidates_json={candidates_json}")
    print(f"report_md={report_md}")


if __name__ == "__main__":
    main()
