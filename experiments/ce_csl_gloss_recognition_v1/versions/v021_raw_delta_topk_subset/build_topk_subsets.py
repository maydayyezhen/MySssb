"""
V21 top-k subset 构建脚本

作用：
1. 从 train_ctc_ready.jsonl 统计 gloss 频率。
2. 分别构建 top100 / top200 / top300 / top500 / top700 / top1000 子集。
3. 对 train/dev/test 使用同一套 top-k 词表过滤样本。
4. 只保留 glossIds 全部在 top-k 词表内的样本。
5. 重新映射 token id：
   0 = CTC blank
   1..K = top-k gloss
6. 原始数据、原始 feature 文件、原始 ctc_ready 文件全部不修改。

输出目录：
D:\\CE-CSL\\CE-CSL\\processed\\subsets\\v021_raw_delta_topk_subset
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


# =========================================================
# 1. 路径配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

OUTPUT_ROOT = PROCESSED_DIR / "subsets" / "v021_raw_delta_topk_subset"

BLANK_ID = 0

TOP_K_LIST = [
    100,
    200,
    300,
    500,
    700,
    1000,
]


# =========================================================
# 2. 文件工具
# =========================================================

def read_jsonl(path: Path) -> List[Dict]:
    """
    读取 jsonl 文件。

    Args:
        path: jsonl 文件路径。

    Returns:
        字典列表。
    """
    rows: List[Dict] = []

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 文件。

    Args:
        path: 输出路径。
        rows: 字典列表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict) -> None:
    """
    写出 json 文件。

    Args:
        path: 输出路径。
        data: 字典数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# =========================================================
# 3. gloss 字段兼容工具
# =========================================================

def get_sample_id(row: Dict, index: int, split: str) -> str:
    """
    获取样本 ID。

    Args:
        row: 样本行。
        index: 当前序号。
        split: 数据集 split。

    Returns:
        样本 ID。
    """
    return str(
        row.get("sampleId")
        or row.get("sample_id")
        or row.get("id")
        or f"{split}-{index + 1:05d}"
    )


def get_chinese(row: Dict) -> str:
    """
    获取中文句子。
    """
    return str(
        row.get("chinese")
        or row.get("text")
        or row.get("sentence")
        or ""
    )


def get_gloss_tokens(row: Dict) -> List[str]:
    """
    获取 gloss 文本列表。

    说明：
    不同中间文件可能字段名不完全一样，所以这里做兼容。
    """
    gloss_value = (
        row.get("gloss")
        or row.get("glosses")
        or row.get("glossList")
        or row.get("gloss_list")
        or []
    )

    if isinstance(gloss_value, list):
        return [str(item) for item in gloss_value]

    if isinstance(gloss_value, str):
        if "/" in gloss_value:
            return [part.strip() for part in gloss_value.split("/") if part.strip()]

        return [part.strip() for part in gloss_value.split() if part.strip()]

    return []


def build_old_id_to_gloss(rows: List[Dict]) -> Dict[int, str]:
    """
    从训练集里尽量构造 old gloss id 到 gloss 文本的映射。

    Args:
        rows: train ctc_ready 行。

    Returns:
        old_id -> gloss 文本。
    """
    old_id_to_gloss: Dict[int, str] = {}

    for row in rows:
        gloss_ids = row.get("glossIds", [])
        gloss_tokens = get_gloss_tokens(row)

        if len(gloss_ids) != len(gloss_tokens):
            continue

        for old_id, gloss_token in zip(gloss_ids, gloss_tokens):
            old_id_to_gloss[int(old_id)] = str(gloss_token)

    return old_id_to_gloss


# =========================================================
# 4. 词表与子集构建
# =========================================================

def count_train_tokens(train_rows: List[Dict]) -> Counter:
    """
    统计 train 中每个 gloss id 的出现次数。

    Args:
        train_rows: train ctc_ready 行。

    Returns:
        token 频次计数器。
    """
    counter: Counter = Counter()

    for row in train_rows:
        for old_id in row.get("glossIds", []):
            counter[int(old_id)] += 1

    return counter


def build_vocab_for_top_k(
    token_counter: Counter,
    old_id_to_gloss: Dict[int, str],
    top_k: int,
) -> Dict:
    """
    构建 top-k closed vocab。

    Args:
        token_counter: train token 频次。
        old_id_to_gloss: old id 到 gloss 文本的映射。
        top_k: 保留前 K 个高频 token。

    Returns:
        vocab 字典。
    """
    most_common = token_counter.most_common(top_k)

    kept_old_ids = [int(old_id) for old_id, _count in most_common]

    old_to_new = {
        str(old_id): new_id
        for new_id, old_id in enumerate(kept_old_ids, start=1)
    }

    new_to_old = {
        str(new_id): old_id
        for old_id, new_id in old_to_new.items()
    }

    new_to_gloss = {
        "0": "<blank>",
    }

    for old_id, new_id in old_to_new.items():
        new_to_gloss[str(new_id)] = old_id_to_gloss.get(int(old_id), f"<old_id:{old_id}>")

    vocab = {
        "name": f"top_k_{top_k}",
        "blankId": BLANK_ID,
        "topK": top_k,
        "vocabSize": top_k + 1,
        "keptOldIds": kept_old_ids,
        "oldToNew": old_to_new,
        "newToOld": new_to_old,
        "newToGloss": new_to_gloss,
        "tokenFrequency": {
            str(old_id): int(count)
            for old_id, count in most_common
        },
    }

    return vocab


def filter_and_remap_rows(
    rows: List[Dict],
    split: str,
    vocab: Dict,
) -> Dict:
    """
    按 top-k 词表过滤并重新映射样本。

    Args:
        rows: 原始 ctc_ready 行。
        split: train / dev / test。
        vocab: top-k vocab。

    Returns:
        包含 rows 和统计信息的字典。
    """
    old_to_new = vocab["oldToNew"]

    kept_rows: List[Dict] = []
    removed_rows: List[Dict] = []

    total_tokens = 0
    kept_tokens = 0

    for index, row in enumerate(rows):
        gloss_ids = [int(old_id) for old_id in row.get("glossIds", [])]

        total_tokens += len(gloss_ids)

        is_all_kept = all(str(old_id) in old_to_new for old_id in gloss_ids)

        if not is_all_kept:
            removed_rows.append(
                {
                    "sampleId": get_sample_id(row, index, split),
                    "chinese": get_chinese(row),
                    "glossIds": gloss_ids,
                    "gloss": get_gloss_tokens(row),
                    "unknownOldIds": [
                        old_id
                        for old_id in gloss_ids
                        if str(old_id) not in old_to_new
                    ],
                }
            )
            continue

        remapped_gloss_ids = [
            int(old_to_new[str(old_id)])
            for old_id in gloss_ids
        ]

        kept_tokens += len(gloss_ids)

        new_row = dict(row)
        new_row["originalGlossIds"] = gloss_ids
        new_row["glossIds"] = remapped_gloss_ids
        new_row["subsetName"] = vocab["name"]
        new_row["subsetVocabSize"] = vocab["vocabSize"]

        kept_rows.append(new_row)

    original_sample_count = len(rows)
    kept_sample_count = len(kept_rows)
    removed_sample_count = len(removed_rows)

    summary = {
        "split": split,
        "originalSampleCount": original_sample_count,
        "keptSampleCount": kept_sample_count,
        "removedSampleCount": removed_sample_count,
        "keptSampleRate": kept_sample_count / original_sample_count if original_sample_count else 0.0,
        "totalTokens": total_tokens,
        "keptTokens": kept_tokens,
        "keptTokenRate": kept_tokens / total_tokens if total_tokens else 0.0,
    }

    return {
        "rows": kept_rows,
        "removedRows": removed_rows,
        "summary": summary,
    }


def build_one_subset(
    top_k: int,
    train_rows: List[Dict],
    dev_rows: List[Dict],
    test_rows: List[Dict],
    token_counter: Counter,
    old_id_to_gloss: Dict[int, str],
) -> Dict:
    """
    构建一个 top-k 子集。

    Args:
        top_k: top-k 数量。
        train_rows: train 行。
        dev_rows: dev 行。
        test_rows: test 行。
        token_counter: train token 频次。
        old_id_to_gloss: old id 到 gloss 文本映射。

    Returns:
        summary。
    """
    subset_name = f"top_k_{top_k}"
    subset_dir = OUTPUT_ROOT / subset_name

    vocab = build_vocab_for_top_k(
        token_counter=token_counter,
        old_id_to_gloss=old_id_to_gloss,
        top_k=top_k,
    )

    split_to_rows = {
        "train": train_rows,
        "dev": dev_rows,
        "test": test_rows,
    }

    split_summaries = {}

    for split, rows in split_to_rows.items():
        result = filter_and_remap_rows(
            rows=rows,
            split=split,
            vocab=vocab,
        )

        write_jsonl(
            subset_dir / f"{split}_subset.jsonl",
            result["rows"],
        )

        write_jsonl(
            subset_dir / f"{split}_removed.jsonl",
            result["removedRows"],
        )

        split_summaries[split] = result["summary"]

    summary = {
        "subsetName": subset_name,
        "topK": top_k,
        "vocabSize": vocab["vocabSize"],
        "splitSummaries": split_summaries,
    }

    write_json(subset_dir / "vocab.json", vocab)
    write_json(subset_dir / "summary.json", summary)

    return summary


# =========================================================
# 5. 主入口
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== V21 top-k subset 构建开始 =====")
    print("DATASET_ROOT:", DATASET_ROOT)
    print("CTC_READY_DIR:", CTC_READY_DIR)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)

    train_rows = read_jsonl(CTC_READY_DIR / "train_ctc_ready.jsonl")
    dev_rows = read_jsonl(CTC_READY_DIR / "dev_ctc_ready.jsonl")
    test_rows = read_jsonl(CTC_READY_DIR / "test_ctc_ready.jsonl")

    print("train rows:", len(train_rows))
    print("dev rows:", len(dev_rows))
    print("test rows:", len(test_rows))

    token_counter = count_train_tokens(train_rows)
    old_id_to_gloss = build_old_id_to_gloss(train_rows)

    print("train token type:", len(token_counter))
    print("old_id_to_gloss size:", len(old_id_to_gloss))

    all_summaries: List[Dict] = []

    for top_k in TOP_K_LIST:
        print("\n" + "=" * 80)
        print("构建 subset:", f"top_k_{top_k}")
        print("=" * 80)

        summary = build_one_subset(
            top_k=top_k,
            train_rows=train_rows,
            dev_rows=dev_rows,
            test_rows=test_rows,
            token_counter=token_counter,
            old_id_to_gloss=old_id_to_gloss,
        )

        all_summaries.append(summary)

        for split in ["train", "dev", "test"]:
            split_summary = summary["splitSummaries"][split]

            print(
                split,
                "kept=",
                split_summary["keptSampleCount"],
                "/",
                split_summary["originalSampleCount"],
                "sampleRate=",
                round(split_summary["keptSampleRate"], 4),
                "tokenRate=",
                round(split_summary["keptTokenRate"], 4),
            )

    write_json(
        OUTPUT_ROOT / "all_subset_summary.json",
        {
            "topKList": TOP_K_LIST,
            "summaries": all_summaries,
        },
    )

    print("\n===== 汇总 =====")
    for summary in all_summaries:
        train_summary = summary["splitSummaries"]["train"]
        dev_summary = summary["splitSummaries"]["dev"]
        test_summary = summary["splitSummaries"]["test"]

        print(
            summary["subsetName"],
            "vocab=",
            summary["vocabSize"],
            "train=",
            train_summary["keptSampleCount"],
            "dev=",
            dev_summary["keptSampleCount"],
            "test=",
            test_summary["keptSampleCount"],
            "devRate=",
            round(dev_summary["keptSampleRate"], 4),
        )

    print("\n已写出:", OUTPUT_ROOT)
    print("===== V21 top-k subset 构建结束 =====")


if __name__ == "__main__":
    main()