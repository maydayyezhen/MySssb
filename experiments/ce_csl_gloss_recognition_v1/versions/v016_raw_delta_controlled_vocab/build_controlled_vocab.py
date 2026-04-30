"""
v016 controlled vocab 构建与覆盖率诊断

目的：
1. 统计 train/dev/test 的 gloss 频次。
2. 基于 train 频次构建受控词表。
3. 支持 min_count / top_k 两种策略。
4. 输出 controlled_vocab 映射和覆盖率报告。
5. 本脚本不训练模型，只做词表诊断。

核心判断：
如果受控词表覆盖率高、词表规模明显变小，后续训练 TER 大幅下降，
说明当前 full vocab 3840 长尾是主要瓶颈。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


# =========================================================
# 1. 路径配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

OUTPUT_DIR = (
    PROCESSED_DIR
    / "controlled_vocab"
    / "v016_raw_delta_controlled_vocab"
)

BLANK_ID = 0
UNK_ID = 1
BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"


# =========================================================
# 2. 实验配置
# =========================================================

CONTROLLED_VOCAB_CONFIGS = [
    {
        "name": "min_count_5",
        "mode": "min_count",
        "min_count": 5,
        "top_k": None,
    },
    {
        "name": "min_count_10",
        "mode": "min_count",
        "min_count": 10,
        "top_k": None,
    },
    {
        "name": "min_count_20",
        "mode": "min_count",
        "min_count": 20,
        "top_k": None,
    },
    {
        "name": "top_k_500",
        "mode": "top_k",
        "min_count": None,
        "top_k": 500,
    },
    {
        "name": "top_k_1000",
        "mode": "top_k",
        "min_count": None,
        "top_k": 1000,
    },
]


# =========================================================
# 3. 文件工具
# =========================================================

def read_jsonl(path: Path) -> List[Dict]:
    """
    读取 jsonl 文件。
    """
    rows: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def write_json(path: Path, data: Dict | List) -> None:
    """
    写出 json 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_split_rows(split: str) -> List[Dict]:
    """
    读取某个 split 的 ctc_ready 行。
    """
    path = CTC_READY_DIR / f"{split}_ctc_ready.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    return read_jsonl(path)


# =========================================================
# 4. 统计工具
# =========================================================

def count_train_tokens(train_rows: List[Dict]) -> Tuple[Counter, Dict[int, str]]:
    """
    统计 train 中每个原始 gloss_id 的出现次数。
    """
    token_counter: Counter = Counter()
    id_to_gloss: Dict[int, str] = {}

    for row in train_rows:
        gloss_list = row.get("gloss", [])
        gloss_ids = row.get("glossIds", [])

        for gloss, gloss_id in zip(gloss_list, gloss_ids):
            gloss_id = int(gloss_id)

            token_counter[gloss_id] += 1
            id_to_gloss[gloss_id] = str(gloss)

    return token_counter, id_to_gloss


def build_controlled_vocab(
    config: Dict,
    train_counter: Counter,
    id_to_gloss: Dict[int, str],
) -> Dict:
    """
    根据配置构建 controlled vocab。

    新 ID 规则：
    0 = <blank>
    1 = <unk>
    2... = 保留 token
    """
    mode = config["mode"]

    if mode == "min_count":
        min_count = int(config["min_count"])

        kept_old_ids = [
            old_id
            for old_id, count in train_counter.items()
            if count >= min_count
        ]

        kept_old_ids = sorted(
            kept_old_ids,
            key=lambda old_id: (-train_counter[old_id], id_to_gloss.get(old_id, "")),
        )

    elif mode == "top_k":
        top_k = int(config["top_k"])

        kept_old_ids = [
            old_id
            for old_id, _ in train_counter.most_common(top_k)
        ]

    else:
        raise ValueError(f"未知 mode：{mode}")

    old_to_new: Dict[str, int] = {}
    new_to_old: Dict[str, int | None] = {
        str(BLANK_ID): None,
        str(UNK_ID): None,
    }
    new_to_gloss: Dict[str, str] = {
        str(BLANK_ID): BLANK_TOKEN,
        str(UNK_ID): UNK_TOKEN,
    }

    for new_id, old_id in enumerate(kept_old_ids, start=2):
        old_to_new[str(old_id)] = new_id
        new_to_old[str(new_id)] = int(old_id)
        new_to_gloss[str(new_id)] = id_to_gloss.get(old_id, f"<old:{old_id}>")

    return {
        "config": config,
        "blankId": BLANK_ID,
        "unkId": UNK_ID,
        "blankToken": BLANK_TOKEN,
        "unkToken": UNK_TOKEN,
        "controlledVocabSize": len(kept_old_ids) + 2,
        "keptTokenCount": len(kept_old_ids),
        "oldToNew": old_to_new,
        "newToOld": new_to_old,
        "newToGloss": new_to_gloss,
        "keptTokens": [
            {
                "oldId": int(old_id),
                "newId": int(old_to_new[str(old_id)]),
                "gloss": id_to_gloss.get(old_id, ""),
                "trainCount": int(train_counter[old_id]),
            }
            for old_id in kept_old_ids
        ],
    }


def analyze_split_coverage(
    split: str,
    rows: List[Dict],
    controlled_vocab: Dict,
) -> Dict:
    """
    分析某个 split 在 controlled vocab 下的覆盖率。
    """
    old_to_new = controlled_vocab["oldToNew"]

    total_tokens = 0
    kept_tokens = 0
    unk_tokens = 0

    total_samples = len(rows)
    samples_with_unk = 0
    all_kept_samples = 0

    target_lengths = []
    unk_counts_per_sample = []

    for row in rows:
        gloss_ids = [int(x) for x in row.get("glossIds", [])]

        sample_total = len(gloss_ids)
        sample_unk = 0

        for old_id in gloss_ids:
            total_tokens += 1

            if str(old_id) in old_to_new:
                kept_tokens += 1
            else:
                unk_tokens += 1
                sample_unk += 1

        target_lengths.append(sample_total)
        unk_counts_per_sample.append(sample_unk)

        if sample_unk > 0:
            samples_with_unk += 1
        else:
            all_kept_samples += 1

    coverage = kept_tokens / total_tokens if total_tokens else 0.0
    unk_rate = unk_tokens / total_tokens if total_tokens else 0.0
    sample_unk_rate = samples_with_unk / total_samples if total_samples else 0.0
    all_kept_sample_rate = all_kept_samples / total_samples if total_samples else 0.0

    return {
        "split": split,
        "sampleCount": total_samples,
        "totalTokens": total_tokens,
        "keptTokens": kept_tokens,
        "unkTokens": unk_tokens,
        "tokenCoverage": coverage,
        "tokenUnkRate": unk_rate,
        "samplesWithUnk": samples_with_unk,
        "samplesWithUnkRate": sample_unk_rate,
        "allKeptSamples": all_kept_samples,
        "allKeptSampleRate": all_kept_sample_rate,
        "avgTargetLength": sum(target_lengths) / len(target_lengths) if target_lengths else 0.0,
        "avgUnkPerSample": sum(unk_counts_per_sample) / len(unk_counts_per_sample) if unk_counts_per_sample else 0.0,
    }


# =========================================================
# 5. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== v016 controlled vocab 构建开始 =====")
    print("DATASET_ROOT:", DATASET_ROOT)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    train_rows = load_split_rows("train")
    dev_rows = load_split_rows("dev")
    test_rows = load_split_rows("test")

    train_counter, id_to_gloss = count_train_tokens(train_rows)

    print("train samples:", len(train_rows))
    print("dev samples:", len(dev_rows))
    print("test samples:", len(test_rows))
    print("train token types:", len(train_counter))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_reports = []

    for config in CONTROLLED_VOCAB_CONFIGS:
        name = config["name"]

        print("\n" + "=" * 80)
        print("构建 controlled vocab:", name)
        print("=" * 80)

        controlled_vocab = build_controlled_vocab(
            config=config,
            train_counter=train_counter,
            id_to_gloss=id_to_gloss,
        )

        report = {
            "name": name,
            "config": config,
            "controlledVocabSize": controlled_vocab["controlledVocabSize"],
            "keptTokenCount": controlled_vocab["keptTokenCount"],
            "train": analyze_split_coverage("train", train_rows, controlled_vocab),
            "dev": analyze_split_coverage("dev", dev_rows, controlled_vocab),
            "test": analyze_split_coverage("test", test_rows, controlled_vocab),
        }

        config_dir = OUTPUT_DIR / name
        config_dir.mkdir(parents=True, exist_ok=True)

        write_json(config_dir / "controlled_vocab.json", controlled_vocab)
        write_json(config_dir / "coverage_report.json", report)

        all_reports.append(report)

        print("controlledVocabSize:", report["controlledVocabSize"])
        print("train tokenCoverage:", round(report["train"]["tokenCoverage"], 4), "unkRate:", round(report["train"]["tokenUnkRate"], 4))
        print("dev   tokenCoverage:", round(report["dev"]["tokenCoverage"], 4), "unkRate:", round(report["dev"]["tokenUnkRate"], 4))
        print("test  tokenCoverage:", round(report["test"]["tokenCoverage"], 4), "unkRate:", round(report["test"]["tokenUnkRate"], 4))
        print("dev samplesWithUnkRate:", round(report["dev"]["samplesWithUnkRate"], 4))
        print("dev allKeptSampleRate:", round(report["dev"]["allKeptSampleRate"], 4))

    write_json(OUTPUT_DIR / "all_coverage_reports.json", all_reports)

    print("\n===== 汇总 =====")
    for report in all_reports:
        print(
            report["name"],
            "vocab=",
            report["controlledVocabSize"],
            "trainCov=",
            round(report["train"]["tokenCoverage"], 4),
            "devCov=",
            round(report["dev"]["tokenCoverage"], 4),
            "devUnkRate=",
            round(report["dev"]["tokenUnkRate"], 4),
            "devAllKeptSampleRate=",
            round(report["dev"]["allKeptSampleRate"], 4),
        )

    print("\n已写出:", OUTPUT_DIR)
    print("===== v016 controlled vocab 构建结束 =====")


if __name__ == "__main__":
    main()