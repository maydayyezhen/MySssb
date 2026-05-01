# -*- coding: utf-8 -*-
"""
调用 DeepSeek 快速非思考模型，对 WLASL v2-25 的 clean cases 做 deletion-only 语义后处理。

输入：
- D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/llm_semantic_rerank_cases_clean.json

输出：
- deepseek_fast_rerank_results.json
- deepseek_fast_rerank_results.csv

设计原则：
1. 一次一个 case，避免模型批量串题。
2. 使用 deletion-only prompt，只允许删除明显多余段，不允许替换、增加、重排。
3. 默认使用 deepseek-v4-flash + thinking disabled。
"""

import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List


SYSTEM_PROMPT = """You are a conservative deletion-only post-processor for a sign-language recognition system.

The recognizer provides:
- rawSequence: the original predicted gloss sequence.
- segments: ordered recognition segments.
- each segment has rawLabel and topK visual candidates.

Your task:
Clean the rawSequence only by removing obviously extra inserted segments.

Important:
This is NOT a maximum-probability reranking task.
Do NOT choose the highest-probability candidate automatically.
The rawLabel is the default trusted output.

Strict rules:
1. Default behavior: keep every segment's rawLabel.
2. You may ONLY remove segments. Do not replace labels.
3. Do not add new segments.
4. Do not reorder segments.
5. Do not invent words.
6. Do not translate into natural English. Keep gloss words only.
7. Remove a segment only if it is clearly an extra insertion, duplicate, transition noise, or makes the sequence obviously unnatural.
8. If uncertain, keep the rawSequence unchanged.
9. Avoid adjacent duplicated words unless repetition is clearly meaningful.
10. Return JSON only.

Good correction examples:
- ["teacher", "learn", "help", "you"] -> remove "learn" -> ["teacher", "help", "you"]
- ["you", "want", "work", "help"] -> remove "work" -> ["you", "want", "help"]
- ["please", "meet", "teacher", "learn"] -> remove "learn" -> ["please", "meet", "teacher"]

Bad corrections:
- Do NOT change ["you", "want", "work", "help"] into ["you", "work", "work", "help"].
- Do NOT choose a candidate just because it has higher probability.
- Do NOT add missing words.
- Do NOT output a natural English sentence.

Return format:
{
  "correctedSequence": ["word1", "word2"],
  "correctionApplied": true,
  "selectedSegments": [
    {
      "segmentIndex": 1,
      "rawLabel": "word1",
      "selectedLabel": "word1",
      "action": "keep"
    },
    {
      "segmentIndex": 2,
      "rawLabel": "extra",
      "selectedLabel": null,
      "action": "remove"
    }
  ],
  "removedSegments": [
    {
      "segmentIndex": 2,
      "rawLabel": "extra",
      "reason": "extra insertion due to overlapping segmentation"
    }
  ],
  "reason": "brief explanation"
}
"""


def load_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[完成] 已写出 JSON：{path}")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """写出 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[完成] 已写出 CSV：{path}")


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])

        raise


def build_user_prompt(case_id: str, llm_input: Dict[str, Any]) -> str:
    """构建单个 case 的用户提示词，不包含 expectedSequence。"""
    return (
        f"CASE_ID: {case_id}\n\n"
        f"INPUT:\n"
        f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}"
    )


def call_deepseek(
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    """调用 DeepSeek Chat Completions API。"""
    url = "https://api.deepseek.com/chat/completions"

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "stream": False,
        "thinking": {
            "type": "disabled",
        },
    }

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        default="D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/llm_semantic_rerank_cases_clean.json",
        help="clean cases JSON 路径",
    )
    parser.add_argument(
        "--summary_json",
        default="D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/semantic_sentence_summary.json",
        help="可选：用于离线对照 expectedSequence",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/deepseek_fast",
        help="输出目录",
    )
    parser.add_argument(
        "--model",
        default="deepseek-v4-flash",
        help="DeepSeek 模型名；如果报模型不可用，可改成 deepseek-chat",
    )
    parser.add_argument(
        "--case_id",
        default="",
        help="只跑某一个 case，例如 case_004；为空则跑全部",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="最多跑多少条；0 表示全部",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="快速模式建议 0",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="单次请求超时时间，秒",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="每个 case 之间暂停秒数，避免触发限流",
    )

    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("没有读取到环境变量 DEEPSEEK_API_KEY，请先在 PowerShell 设置：$env:DEEPSEEK_API_KEY='sk-...'")

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_json(input_json)

    if args.case_id.strip():
        cases = [
            item for item in cases
            if item.get("caseId") == args.case_id.strip()
        ]

    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    expected_map: Dict[str, List[str]] = {}

    summary_path = Path(args.summary_json)
    if summary_path.exists():
        summary = load_json(summary_path)
        for index, item in enumerate(summary.get("results", []), start=1):
            expected_map[f"case_{index:03d}"] = item.get("expectedSequence", [])

    results = []
    rows = []

    print("========== DeepSeek 快速非思考语义后处理 ==========")
    print(f"[信息] model：{args.model}")
    print(f"[信息] case_count：{len(cases)}")
    print(f"[信息] output_dir：{output_dir}")

    for idx, item in enumerate(cases, start=1):
        case_id = item["caseId"]
        llm_input = item["llmInput"]
        raw_sequence = llm_input.get("rawSequence", [])

        print(f"\n[{idx}/{len(cases)}] 调用：{case_id}")
        print(f"rawSequence：{' '.join(raw_sequence)}")

        user_prompt = build_user_prompt(case_id, llm_input)

        start = time.perf_counter()

        try:
            response_payload = call_deepseek(
                api_key=api_key,
                model=args.model,
                user_prompt=user_prompt,
                temperature=args.temperature,
                timeout=args.timeout,
            )

            elapsed = time.perf_counter() - start

            content = response_payload["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)

            expected = expected_map.get(case_id, [])
            corrected = parsed.get("correctedSequence", [])

            expected_match = bool(expected) and corrected == expected

            result = {
                "caseId": case_id,
                "ok": True,
                "elapsedSeconds": round(elapsed, 3),
                "rawSequence": raw_sequence,
                "expectedSequence": expected,
                "expectedMatch": expected_match,
                "model": args.model,
                "response": parsed,
                "rawContent": content,
                "usage": response_payload.get("usage", {}),
            }

            print(f"[完成] {case_id} 耗时 {elapsed:.3f}s")
            print(f"corrected：{' '.join(corrected)}")
            if expected:
                print(f"expected ：{' '.join(expected)}")
                print(f"match    ：{expected_match}")

        except urllib.error.HTTPError as e:
            elapsed = time.perf_counter() - start
            error_body = e.read().decode("utf-8", errors="replace")

            result = {
                "caseId": case_id,
                "ok": False,
                "elapsedSeconds": round(elapsed, 3),
                "rawSequence": raw_sequence,
                "error": f"HTTPError {e.code}: {error_body}",
            }

            print(f"[错误] {case_id} HTTP {e.code}: {error_body}")

        except Exception as e:
            elapsed = time.perf_counter() - start

            result = {
                "caseId": case_id,
                "ok": False,
                "elapsedSeconds": round(elapsed, 3),
                "rawSequence": raw_sequence,
                "error": str(e),
            }

            print(f"[错误] {case_id}: {e}")

        results.append(result)

        response = result.get("response", {})

        rows.append({
            "caseId": case_id,
            "ok": int(bool(result.get("ok"))),
            "elapsedSeconds": result.get("elapsedSeconds", ""),
            "rawSequence": " ".join(result.get("rawSequence", [])),
            "correctedSequence": " ".join(response.get("correctedSequence", [])) if isinstance(response, dict) else "",
            "correctionApplied": response.get("correctionApplied", "") if isinstance(response, dict) else "",
            "expectedSequence": " ".join(result.get("expectedSequence", [])),
            "expectedMatch": int(bool(result.get("expectedMatch", False))),
            "reason": response.get("reason", "") if isinstance(response, dict) else result.get("error", ""),
        })

        time.sleep(args.sleep)

    ok_count = sum(1 for item in results if item.get("ok"))
    match_count = sum(1 for item in results if item.get("expectedMatch"))

    summary_result = {
        "model": args.model,
        "caseCount": len(results),
        "okCount": ok_count,
        "expectedMatchCount": match_count,
        "expectedMatchRate": round(match_count / max(1, len(results)), 6),
        "results": results,
    }

    save_json(output_dir / "deepseek_fast_rerank_results.json", summary_result)

    write_csv(
        output_dir / "deepseek_fast_rerank_results.csv",
        rows,
        [
            "caseId",
            "ok",
            "elapsedSeconds",
            "rawSequence",
            "correctedSequence",
            "correctionApplied",
            "expectedSequence",
            "expectedMatch",
            "reason",
        ],
    )

    print("\n========== 完成 ==========")
    print(f"成功请求：{ok_count}/{len(results)}")
    print(f"匹配 expected：{match_count}/{len(results)}")
    print(f"匹配率：{summary_result['expectedMatchRate']:.4f}")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
