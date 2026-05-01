# -*- coding: utf-8 -*-
"""
分析 v2-25 语义句子是否具备 TopK + LLM 重排修正潜力。

判定规则：
expectedSequence 是否可以按顺序匹配到 segmentTopK 中。
允许跳过多余 segment，但不允许凭空造词。
"""

import json
import csv
from pathlib import Path


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[完成] 已写出 CSV：{path}")


def get_topk_labels(segment):
    return [item["label"] for item in segment.get("topK", [])]


def match_expected_by_topk(expected, segments):
    """
    判断 expected 是否能作为 segmentTopK 的有序子序列被匹配。
    """
    pos = 0
    matched = []
    skipped = []

    for index, segment in enumerate(segments, start=1):
        labels = get_topk_labels(segment)

        if pos < len(expected) and expected[pos] in labels:
            matched.append({
                "expected": expected[pos],
                "segmentIndex": index,
                "rawLabel": segment.get("rawLabel", ""),
                "topK": "/".join(labels),
            })
            pos += 1
        else:
            skipped.append({
                "segmentIndex": index,
                "rawLabel": segment.get("rawLabel", ""),
                "topK": "/".join(labels),
            })

    return pos == len(expected), matched, skipped


def main():
    summary_path = Path("D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/semantic_sentence_summary.json")
    output_dir = summary_path.parent

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []

    for item in payload["results"]:
        expected = item["expectedSequence"]
        raw = item["rawSequence"]
        segments = item["segmentTopK"]

        fixable, matched, skipped = match_expected_by_topk(expected, segments)

        rows.append({
            "name": item["name"],
            "expected": " ".join(expected),
            "raw": " ".join(raw),
            "exact_match": int(item["exactMatch"]),
            "topk_fixable": int(fixable),
            "segment_count": len(segments),
            "matched_path": " ; ".join(
                f"{m['expected']}@seg{m['segmentIndex']}[{m['rawLabel']}|{m['topK']}]"
                for m in matched
            ),
            "skipped_segments": " ; ".join(
                f"seg{s['segmentIndex']}[{s['rawLabel']}|{s['topK']}]"
                for s in skipped
            ),
        })

    exact_count = sum(row["exact_match"] for row in rows)
    fixable_count = sum(row["topk_fixable"] for row in rows)

    write_csv(
        output_dir / "semantic_topk_fixability_report.csv",
        rows,
        [
            "name",
            "expected",
            "raw",
            "exact_match",
            "topk_fixable",
            "segment_count",
            "matched_path",
            "skipped_segments",
        ],
    )

    print("\n========== TopK 可修正性统计 ==========")
    print(f"case_count：{len(rows)}")
    print(f"raw_exact：{exact_count}/{len(rows)} = {exact_count / len(rows):.4f}")
    print(f"topk_fixable：{fixable_count}/{len(rows)} = {fixable_count / len(rows):.4f}")

    print("\n========== 推荐展示样例 ==========")
    for row in rows:
        if row["topk_fixable"] and not row["exact_match"]:
            print(f"[可展示] {row['name']}: {row['raw']} -> {row['expected']}")

    print("\n========== 不适合语义修正的样例 ==========")
    for row in rows:
        if not row["topk_fixable"]:
            print(f"[不适合] {row['name']}: {row['raw']} -> {row['expected']}")


if __name__ == "__main__":
    main()
