# -*- coding: utf-8 -*-
"""Build backend semantic-correction candidates from the active v2-25 suite.

The v2-25 demo set already has per-case semantic evaluation artifacts. This
script converts those artifacts into the same CSV shape consumed by
``verify_deepseek_deletion_fix_candidates.py`` so the Java DeepSeek endpoint can
be tested without rerunning video inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_SUMMARY_JSON = (
    "D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/semantic_sentence_summary.json"
)
DEFAULT_OUTPUT_CSV = (
    "D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/topk_backend_eval_candidates.csv"
)

OUTPUT_FIELDS = [
    "rank",
    "sentence",
    "case_name",
    "expected_sequence",
    "detected_sequence",
    "exact_match",
    "deletion_only_match",
    "missing_words",
    "extra_words",
    "segment_count",
    "expected_count",
    "avg_segment_confidence",
    "total_frame_count",
    "video_path",
    "segments_json",
    "status",
    "error_message",
]


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def split_labels(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split() if item.strip()]


def is_subsequence(expected: Sequence[str], detected: Sequence[str]) -> bool:
    expected_index = 0
    for label in detected:
        if expected_index < len(expected) and label == expected[expected_index]:
            expected_index += 1
    return expected_index == len(expected)


def sequence_diff(expected: Sequence[str], detected: Sequence[str]) -> Tuple[List[str], List[str]]:
    expected_index = 0
    matched_expected = [False for _ in expected]
    matched_detected = [False for _ in detected]

    for detected_index, label in enumerate(detected):
        if expected_index < len(expected) and label == expected[expected_index]:
            matched_expected[expected_index] = True
            matched_detected[detected_index] = True
            expected_index += 1

    missing_words = [
        label for index, label in enumerate(expected) if not matched_expected[index]
    ]
    extra_words = [
        label for index, label in enumerate(detected) if not matched_detected[index]
    ]
    return missing_words, extra_words


def locate_segments_json(summary_path: Path, case: Dict[str, object]) -> Path:
    name = str(case.get("name", "")).strip()
    video_path = Path(str(case.get("videoPath", "")))
    if video_path.name:
        candidate = summary_path.parent / name / f"{video_path.stem}_segments.json"
        if candidate.exists():
            return candidate

    case_dir = summary_path.parent / name
    matches = sorted(case_dir.glob("*_segments.json"))
    if matches:
        return matches[0]
    return case_dir / f"{name}_trimmed_segments.json"


def avg_segment_confidence(case: Dict[str, object]) -> float:
    segment_topk = case.get("segmentTopK", [])
    if not isinstance(segment_topk, list) or not segment_topk:
        return 0.0
    values = []
    for segment in segment_topk:
        if isinstance(segment, dict):
            try:
                values.append(float(segment.get("avgConfidence") or 0.0))
            except (TypeError, ValueError):
                pass
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def total_frame_count(segments_json: Path) -> int:
    if not segments_json.exists():
        return 0
    try:
        payload = read_json(segments_json)
    except (OSError, json.JSONDecodeError):
        return 0
    video_info = payload.get("video_info", {})
    if not isinstance(video_info, dict):
        return 0
    try:
        return int(video_info.get("total_frame_count") or 0)
    except (TypeError, ValueError):
        return 0


def build_rows(summary_path: Path) -> List[Dict[str, object]]:
    summary = read_json(summary_path)
    results = summary.get("results", [])
    if not isinstance(results, list):
        raise ValueError("summary.results must be a list")

    rows: List[Dict[str, object]] = []
    for rank, case in enumerate(results, start=1):
        if not isinstance(case, dict):
            continue

        expected = split_labels(case.get("expectedSequence"))
        detected = split_labels(case.get("rawSequence"))
        missing, extra = sequence_diff(expected, detected)
        exact_match = expected == detected
        segments_json = locate_segments_json(summary_path, case)

        rows.append(
            {
                "rank": rank,
                "sentence": str(case.get("sentence", "")).replace(",", " "),
                "case_name": str(case.get("name", "")).strip(),
                "expected_sequence": " ".join(expected),
                "detected_sequence": " ".join(detected),
                "exact_match": int(exact_match),
                "deletion_only_match": int(is_subsequence(expected, detected)),
                "missing_words": " ".join(missing),
                "extra_words": " ".join(extra),
                "segment_count": len(detected),
                "expected_count": len(expected),
                "avg_segment_confidence": avg_segment_confidence(case),
                "total_frame_count": total_frame_count(segments_json),
                "video_path": case.get("videoPath", ""),
                "segments_json": str(segments_json),
                "status": "ok" if segments_json.exists() else "missing_segments_json",
                "error_message": "" if segments_json.exists() else str(segments_json),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    output_path = Path(args.output_csv)

    rows = build_rows(summary_path)
    write_csv(output_path, rows)

    exact_count = sum(1 for row in rows if int(row["exact_match"]) == 1)
    deletion_count = sum(1 for row in rows if int(row["deletion_only_match"]) == 1)
    print(f"cases={len(rows)}")
    print(f"raw_exact={exact_count}")
    print(f"deletion_only_theory={deletion_count}")
    print(f"output_csv={output_path}")


if __name__ == "__main__":
    main()
