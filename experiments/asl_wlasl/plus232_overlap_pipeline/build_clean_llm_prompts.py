# -*- coding: utf-8 -*-
"""
生成不泄露答案的 LLM 语义重排提示词。

输入：
- semantic_sentence_summary.json

输出：
- llm_semantic_rerank_prompts_clean.txt
- llm_semantic_rerank_cases_clean.json

注意：
不会输出 case name、expectedSequence、sentence、videoPath 等答案泄露字段。
"""

import json
from pathlib import Path


def make_clean_prompt(case_id, llm_input):
    return (
        f"CASE_ID: {case_id}\n"
        f"You are a semantic reranker for a sign-language recognition system.\n"
        f"The recognizer outputs one raw gloss sequence and TopK candidates for each segment.\n"
        f"Your job is to choose the most natural corrected gloss sequence.\n\n"
        f"Strict rules:\n"
        f"1. Only choose labels that appear in each segment's topK list.\n"
        f"2. Do not invent any word outside the provided candidates.\n"
        f"3. Keep the original segment order.\n"
        f"4. You may remove an obviously extra segment only when it makes the sentence unnatural, duplicated, or transitional.\n"
        f"5. Do not add new segments.\n"
        f"6. Do not translate into natural English. Keep gloss words only.\n"
        f"7. Return JSON only.\n\n"
        f"Return format:\n"
        f'{{\n'
        f'  "correctedSequence": ["word1", "word2"],\n'
        f'  "correctionApplied": true,\n'
        f'  "removedSegments": [\n'
        f'    {{\n'
        f'      "segmentIndex": 2,\n'
        f'      "rawLabel": "example",\n'
        f'      "reason": "extra insertion"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "reason": "brief explanation"\n'
        f'}}\n\n'
        f"INPUT:\n"
        f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}\n"
    )


def main():
    summary_path = Path("D:/datasets/WLASL-mini-v2-25/demo_eval_semantic/semantic_sentence_summary.json")
    output_dir = summary_path.parent

    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    clean_cases = []
    prompt_chunks = []

    for index, item in enumerate(payload["results"], start=1):
        case_id = f"case_{index:03d}"

        # 只保留 llmInput，不保留 name / sentence / expectedSequence / videoPath
        llm_input = item["llmInput"]

        clean_case = {
            "caseId": case_id,
            "llmInput": llm_input
        }

        clean_cases.append(clean_case)
        prompt_chunks.append(make_clean_prompt(case_id, llm_input))

    clean_json_path = output_dir / "llm_semantic_rerank_cases_clean.json"
    clean_prompt_path = output_dir / "llm_semantic_rerank_prompts_clean.txt"

    clean_json_path.write_text(
        json.dumps(clean_cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    clean_prompt_path.write_text(
        ("\n\n" + "=" * 80 + "\n\n").join(prompt_chunks),
        encoding="utf-8",
    )

    print(f"[完成] 已写出：{clean_json_path}")
    print(f"[完成] 已写出：{clean_prompt_path}")


if __name__ == "__main__":
    main()
