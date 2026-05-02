# -*- coding: utf-8 -*-
"""Verify deletion-only candidates through the existing backend API.

The script calls the already implemented Java endpoint. It does not implement a
new DeepSeek client and does not read or store any API key.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_CANDIDATES_CSV = "D:/datasets/WLASL-mini-v2-20/demo_search_20/deletion_only_fix_candidates.csv"
DEFAULT_OUTPUT_DIR = "D:/datasets/WLASL-mini-v2-20/demo_search_20"
DEFAULT_API_URL = "http://localhost:8080/app/sign-video/semantic-correct"

OUTPUT_FIELDS = [
    "rank",
    "sentence",
    "case_name",
    "expected_sequence",
    "detected_sequence",
    "extra_words",
    "missing_words",
    "corrected_sequence",
    "exact_match",
    "deletion_only_match",
    "deepseek_verified",
    "api_status",
    "http_status",
    "error_message",
    "avg_segment_confidence",
    "total_frame_count",
    "video_path",
    "segments_json",
    "raw_response",
]


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


def load_segments_json(path_text: str) -> Dict[str, object]:
    path = Path(path_text)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_segment_topk(row: Dict[str, str]) -> List[Dict[str, object]]:
    payload = load_segments_json(row.get("segments_json", ""))
    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        return []

    segment_topk: List[Dict[str, object]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue

        label = str(segment.get("label", "")).strip()
        segment_topk.append(
            {
                "segmentIndex": index,
                "startFrame": as_int(segment.get("start_frame")),
                "endFrame": as_int(segment.get("end_frame")),
                "rawLabel": label,
                "rawLabelZh": label,
                "avgConfidence": as_float(segment.get("avg_confidence")),
                "maxConfidence": as_float(segment.get("max_confidence")),
                "topK": [],
            }
        )

    return segment_topk


def build_request_payload(row: Dict[str, str]) -> Dict[str, object]:
    detected_sequence = split_words(row.get("detected_sequence"))
    expected_sequence = split_words(row.get("expected_sequence"))

    return {
        "rawSequence": detected_sequence,
        "rawTextZh": " ".join(detected_sequence),
        "segmentTopK": build_segment_topk(row),
        "expectedSequence": expected_sequence,
        "detectedSequence": detected_sequence,
    }


def post_json(
    api_url: str,
    payload: Dict[str, object],
    timeout: float,
    auth_token: str = "",
) -> Tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    if auth_token.strip():
        headers["Authorization"] = "Bearer " + auth_token.strip()

    request = urllib.request.Request(
        api_url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return int(response.status), text
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        return int(error.code), text


def verify_one(
    row: Dict[str, str],
    api_url: str,
    timeout: float,
    auth_token: str = "",
) -> Dict[str, object]:
    expected_sequence = split_words(row.get("expected_sequence"))
    detected_sequence = split_words(row.get("detected_sequence"))
    result: Dict[str, object] = {
        "rank": row.get("rank", ""),
        "sentence": row.get("sentence", ""),
        "case_name": row.get("case_name", ""),
        "expected_sequence": " ".join(expected_sequence),
        "detected_sequence": " ".join(detected_sequence),
        "extra_words": row.get("extra_words", ""),
        "missing_words": row.get("missing_words", ""),
        "corrected_sequence": "",
        "exact_match": row.get("exact_match", "0"),
        "deletion_only_match": row.get("deletion_only_match", "1"),
        "deepseek_verified": "false",
        "api_status": "not_called",
        "http_status": "",
        "error_message": "",
        "avg_segment_confidence": row.get("avg_segment_confidence", ""),
        "total_frame_count": row.get("total_frame_count", ""),
        "video_path": row.get("video_path", ""),
        "segments_json": row.get("segments_json", ""),
        "raw_response": "",
    }

    try:
        http_status, response_text = post_json(
            api_url,
            build_request_payload(row),
            timeout,
            auth_token,
        )
        result["http_status"] = http_status
        result["raw_response"] = response_text

        if http_status < 200 or http_status >= 300:
            result["api_status"] = "http_error"
            result["error_message"] = response_text[:500]
            return result

        response_json = json.loads(response_text)
        corrected_sequence = response_json.get("correctedSequence", [])
        if not isinstance(corrected_sequence, list):
            corrected_sequence = []

        corrected_sequence = [str(item).strip() for item in corrected_sequence if str(item).strip()]
        result["corrected_sequence"] = " ".join(corrected_sequence)
        result["api_status"] = "ok"
        result["deepseek_verified"] = str(corrected_sequence == expected_sequence).lower()
        result["raw_response"] = json.dumps(response_json, ensure_ascii=False, separators=(",", ":"))
        return result
    except urllib.error.URLError as error:
        result["api_status"] = "request_failed"
        result["error_message"] = str(error)
        return result
    except TimeoutError as error:
        result["api_status"] = "request_failed"
        result["error_message"] = str(error)
        return result
    except Exception as error:  # Keep verification batch running.
        result["api_status"] = "parse_or_request_failed"
        result["error_message"] = str(error)
        return result


def write_markdown_report(path: Path, api_url: str, rows: Sequence[Dict[str, object]]) -> None:
    verified_count = sum(1 for row in rows if parse_bool(row.get("deepseek_verified")))
    lines = [
        "# DeepSeek Verified Deletion-Only Fix Candidates",
        "",
        "## Summary",
        "",
        f"- api_url: `{api_url}`",
        f"- candidates_checked: {len(rows)}",
        f"- deepseek_verified_count: {verified_count}",
        "",
        "## Results",
        "",
    ]
    append_table(lines, rows)

    if verified_count == 0:
        lines.extend(
            [
                "## Note",
                "",
                "No candidate was verified as corrected through the live backend call in this run. "
                "Use the generated manual request examples for Postman or backend-side debugging.",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_table(lines: List[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        lines.extend(["No results.", ""])
        return

    lines.append(
        "| rank | sentence | case_name | expected_sequence | detected_sequence | extra_words | corrected_sequence | verified | api_status | http_status |"
    )
    lines.append("|---:|---|---|---|---|---|---|---|---|---:|")
    for row in rows:
        lines.append(
            "| "
            f"{row.get('rank')} | "
            f"{md_escape(row.get('sentence'))} | "
            f"{md_escape(row.get('case_name'))} | "
            f"{md_escape(row.get('expected_sequence'))} | "
            f"{md_escape(row.get('detected_sequence'))} | "
            f"{md_escape(row.get('extra_words'))} | "
            f"{md_escape(row.get('corrected_sequence'))} | "
            f"{md_escape(row.get('deepseek_verified'))} | "
            f"{md_escape(row.get('api_status'))} | "
            f"{md_escape(row.get('http_status'))} |"
        )
    lines.append("")


def write_manual_examples(path: Path, rows: Sequence[Dict[str, str]], api_url: str) -> None:
    lines = [
        "# DeepSeek Manual Verification Request Examples",
        "",
        f"Existing backend endpoint found: `{api_url}`",
        "",
        "The live verification run did not produce a verified candidate. "
        "The following examples can be sent manually through the existing backend endpoint.",
        "",
    ]

    for index, row in enumerate(rows, start=1):
        expected = split_words(row.get("expected_sequence"))
        detected = split_words(row.get("detected_sequence"))
        request_payload = {
            "rawSequence": detected,
            "rawTextZh": " ".join(detected),
            "segmentTopK": build_segment_topk(row),
            "expectedSequence": expected,
            "detectedSequence": detected,
        }
        lines.extend(
            [
                f"## Candidate {index}",
                "",
                f"- case_name: {row.get('case_name', '')}",
                f"- expected_sequence: {' '.join(expected)}",
                f"- detected_sequence: {' '.join(detected)}",
                f"- extra_words: {row.get('extra_words', '')}",
                "",
                "### Request JSON",
                "",
                "```json",
                json.dumps(request_payload, ensure_ascii=False, indent=2),
                "```",
                "",
                "### Expected correctedSequence",
                "",
                "```json",
                json.dumps(expected, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--auth_token", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = read_csv(Path(args.candidates_csv))
    if args.top_k > 0:
        rows = rows[:args.top_k]

    verified_rows = [
        verify_one(row, args.api_url, args.timeout, args.auth_token)
        for row in rows
    ]

    output_csv = output_dir / "deepseek_verified_deletion_fix_candidates.csv"
    output_json = output_dir / "deepseek_verified_deletion_fix_candidates.json"
    output_md = output_dir / "deepseek_verified_deletion_fix_demo.md"
    manual_md = output_dir / "deepseek_verify_manual_request_examples.md"

    write_csv(output_csv, verified_rows, OUTPUT_FIELDS)
    write_json(output_json, verified_rows)
    write_markdown_report(output_md, args.api_url, verified_rows)

    verified_count = sum(1 for row in verified_rows if parse_bool(row.get("deepseek_verified")))
    if verified_count == 0:
        write_manual_examples(manual_md, rows, args.api_url)

    print(f"checked={len(verified_rows)}")
    print(f"deepseek_verified={verified_count}")
    print(f"output_csv={output_csv}")
    print(f"output_json={output_json}")
    print(f"output_md={output_md}")
    if verified_count == 0:
        print(f"manual_examples={manual_md}")


if __name__ == "__main__":
    main()
