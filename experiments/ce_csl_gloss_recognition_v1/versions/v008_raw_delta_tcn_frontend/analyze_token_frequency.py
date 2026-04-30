"""
CE-CSL token 频率与预测错误诊断脚本

作用：
1. 统计 train/dev/test 中每个 gloss token 的出现次数。
2. 读取 full_bilstm_ctc 的预测诊断结果。
3. 按 train 频次分组统计 dev token 错误率。
4. 分析错误是否集中在低频 token / OOV token 上。
5. 输出低频高错 token、高频预测 token、OOV token 等诊断报告。

前置条件：
- 已经运行过 inspect_predictions.py。
- 已经生成：
  D:\\CE-CSL\\CE-CSL\\processed\\logs\\full_bilstm_ctc_raw_delta_tcn_frontend\\prediction_diagnosis\\best_dev_ter_all_examples.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# =========================================================
# 1. 基础路径配置
# =========================================================

# CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# processed 目录
PROCESSED_DIR = DATASET_ROOT / "processed"

# CTC ready 样本清单目录
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

# full_bilstm_ctc 预测诊断目录
PREDICTION_DIAGNOSIS_DIR = (
    PROCESSED_DIR / "logs" / "full_bilstm_ctc_raw_delta_tcn_frontend" / "prediction_diagnosis"
)

OUTPUT_DIR = PROCESSED_DIR / "logs" / "full_bilstm_ctc_raw_delta_tcn_frontend" / "token_frequency_diagnosis"

# 当前主要分析的预测文件
PREDICTION_FILE = PREDICTION_DIAGNOSIS_DIR / "best_dev_ter_all_examples.jsonl"


# =========================================================
# 2. 文件工具
# =========================================================

def read_jsonl(path: Path) -> List[Dict]:
    """
    读取 jsonl 文件。

    Args:
        path: jsonl 文件路径。

    Returns:
        每一行解析后的字典列表。
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
    写出 JSON 文件。

    Args:
        path: 输出路径。
        data: 待写出的数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 文件。

    Args:
        path: 输出路径。
        rows: 待写出的字典列表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# 3. 频次统计工具
# =========================================================

def count_tokens_for_split(split: str) -> Counter:
    """
    统计某个 split 中 gloss token 出现次数。

    Args:
        split: train/dev/test。

    Returns:
        gloss token 到出现次数的 Counter。
    """
    ready_path = CTC_READY_DIR / f"{split}_ctc_ready.jsonl"

    if not ready_path.exists():
        raise FileNotFoundError(f"找不到 CTC ready 文件：{ready_path}")

    rows = read_jsonl(ready_path)

    counter: Counter = Counter()

    for row in rows:
        gloss_list = row.get("gloss", [])

        counter.update(gloss_list)

    return counter


def get_frequency_bucket(train_count: int) -> str:
    """
    根据 token 在 train 中的出现次数进行分桶。

    Args:
        train_count: token 在 train 中的出现次数。

    Returns:
        频次桶名称。
    """
    if train_count == 0:
        return "00_oov"

    if train_count == 1:
        return "01_once"

    if train_count <= 3:
        return "02_count_2_3"

    if train_count <= 10:
        return "03_count_4_10"

    if train_count <= 50:
        return "04_count_11_50"

    return "05_count_51_plus"


def safe_divide(numerator: float, denominator: float) -> float:
    """
    安全除法。

    Args:
        numerator: 分子。
        denominator: 分母。

    Returns:
        denominator 为 0 时返回 0，否则返回 numerator / denominator。
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


# =========================================================
# 4. 序列对齐工具
# =========================================================

def align_prediction_to_reference(
    prediction: List[str],
    reference: List[str],
) -> List[Dict]:
    """
    对预测 gloss 序列和真实 gloss 序列做编辑距离对齐。

    对齐操作包括：
    - correct: 预测 token 和真实 token 相同
    - substitute: 预测 token 替换真实 token
    - delete: 真实 token 被漏掉
    - insert: 预测中多出来的 token

    Args:
        prediction: 预测 gloss 序列。
        reference: 真实 gloss 序列。

    Returns:
        对齐操作列表。
    """
    pred_len = len(prediction)
    ref_len = len(reference)

    dp = [[0] * (ref_len + 1) for _ in range(pred_len + 1)]

    for i in range(pred_len + 1):
        dp[i][0] = i

    for j in range(ref_len + 1):
        dp[0][j] = j

    for i in range(1, pred_len + 1):
        for j in range(1, ref_len + 1):
            if prediction[i - 1] == reference[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # insert
                dp[i][j - 1] + 1,        # delete
                dp[i - 1][j - 1] + cost, # correct / substitute
            )

    operations: List[Dict] = []

    i = pred_len
    j = ref_len

    while i > 0 or j > 0:
        # correct / substitute
        if i > 0 and j > 0:
            cost = 0 if prediction[i - 1] == reference[j - 1] else 1

            if dp[i][j] == dp[i - 1][j - 1] + cost:
                operations.append(
                    {
                        "op": "correct" if cost == 0 else "substitute",
                        "prediction": prediction[i - 1],
                        "reference": reference[j - 1],
                    }
                )
                i -= 1
                j -= 1
                continue

        # delete：真实 token 被漏掉
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(
                {
                    "op": "delete",
                    "prediction": None,
                    "reference": reference[j - 1],
                }
            )
            j -= 1
            continue

        # insert：预测中多出来 token
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(
                {
                    "op": "insert",
                    "prediction": prediction[i - 1],
                    "reference": None,
                }
            )
            i -= 1
            continue

        raise RuntimeError("编辑距离回溯失败")

    operations.reverse()

    return operations


# =========================================================
# 5. 诊断主逻辑
# =========================================================

def analyze_prediction_errors(
    prediction_rows: List[Dict],
    train_counter: Counter,
    dev_counter: Counter,
) -> Dict:
    """
    分析预测错误与 train 频次之间的关系。

    Args:
        prediction_rows: 预测样例列表。
        train_counter: train token 频次。
        dev_counter: dev token 频次。

    Returns:
        诊断结果字典。
    """
    bucket_stats: Dict[str, Dict] = defaultdict(
        lambda: {
            "referenceTokenCount": 0,
            "correctCount": 0,
            "substituteCount": 0,
            "deleteCount": 0,
            "errorCount": 0,
            "tokens": Counter(),
        }
    )

    token_stats: Dict[str, Dict] = defaultdict(
        lambda: {
            "trainCount": 0,
            "devCount": 0,
            "referenceCountInEval": 0,
            "correctCount": 0,
            "substituteCount": 0,
            "deleteCount": 0,
            "errorCount": 0,
        }
    )

    insertion_counter: Counter = Counter()

    total_reference_tokens = 0
    total_correct_tokens = 0
    total_error_tokens = 0
    total_substitute = 0
    total_delete = 0
    total_insert = 0

    for row in prediction_rows:
        reference_gloss = row.get("referenceGloss", [])
        prediction_gloss = row.get("predictionGloss", [])

        operations = align_prediction_to_reference(
            prediction=prediction_gloss,
            reference=reference_gloss,
        )

        for operation in operations:
            op = operation["op"]
            reference_token = operation["reference"]
            prediction_token = operation["prediction"]

            if op == "insert":
                insertion_counter[prediction_token] += 1
                total_insert += 1
                continue

            if reference_token is None:
                continue

            train_count = int(train_counter.get(reference_token, 0))
            bucket = get_frequency_bucket(train_count)

            bucket_stats[bucket]["referenceTokenCount"] += 1
            bucket_stats[bucket]["tokens"][reference_token] += 1

            token_stats[reference_token]["trainCount"] = train_count
            token_stats[reference_token]["devCount"] = int(dev_counter.get(reference_token, 0))
            token_stats[reference_token]["referenceCountInEval"] += 1

            total_reference_tokens += 1

            if op == "correct":
                bucket_stats[bucket]["correctCount"] += 1
                token_stats[reference_token]["correctCount"] += 1
                total_correct_tokens += 1
            elif op == "substitute":
                bucket_stats[bucket]["substituteCount"] += 1
                bucket_stats[bucket]["errorCount"] += 1
                token_stats[reference_token]["substituteCount"] += 1
                token_stats[reference_token]["errorCount"] += 1
                total_substitute += 1
                total_error_tokens += 1
            elif op == "delete":
                bucket_stats[bucket]["deleteCount"] += 1
                bucket_stats[bucket]["errorCount"] += 1
                token_stats[reference_token]["deleteCount"] += 1
                token_stats[reference_token]["errorCount"] += 1
                total_delete += 1
                total_error_tokens += 1

    bucket_summary: List[Dict] = []

    for bucket in sorted(bucket_stats.keys()):
        stats = bucket_stats[bucket]
        reference_count = stats["referenceTokenCount"]

        bucket_summary.append(
            {
                "bucket": bucket,
                "referenceTokenCount": reference_count,
                "correctCount": stats["correctCount"],
                "substituteCount": stats["substituteCount"],
                "deleteCount": stats["deleteCount"],
                "errorCount": stats["errorCount"],
                "tokenAccuracy": safe_divide(stats["correctCount"], reference_count),
                "tokenErrorRate": safe_divide(stats["errorCount"], reference_count),
                "deleteRate": safe_divide(stats["deleteCount"], reference_count),
                "substituteRate": safe_divide(stats["substituteCount"], reference_count),
                "topReferenceTokens": [
                    {"token": token, "count": count}
                    for token, count in stats["tokens"].most_common(20)
                ],
            }
        )

    token_rows: List[Dict] = []

    for token, stats in token_stats.items():
        reference_count = stats["referenceCountInEval"]

        token_rows.append(
            {
                "token": token,
                "trainCount": stats["trainCount"],
                "devCount": stats["devCount"],
                "referenceCountInEval": reference_count,
                "correctCount": stats["correctCount"],
                "substituteCount": stats["substituteCount"],
                "deleteCount": stats["deleteCount"],
                "errorCount": stats["errorCount"],
                "tokenAccuracy": safe_divide(stats["correctCount"], reference_count),
                "tokenErrorRate": safe_divide(stats["errorCount"], reference_count),
                "frequencyBucket": get_frequency_bucket(stats["trainCount"]),
            }
        )

    # 低频但在 dev 中出现的错误 token
    low_frequency_error_tokens = sorted(
        [
            row for row in token_rows
            if row["trainCount"] <= 3 and row["referenceCountInEval"] > 0
        ],
        key=lambda item: (
            -item["errorCount"],
            item["trainCount"],
            item["token"],
        ),
    )

    # 高频但仍然错得多的 token
    high_frequency_error_tokens = sorted(
        [
            row for row in token_rows
            if row["trainCount"] >= 10 and row["errorCount"] > 0
        ],
        key=lambda item: (
            -item["errorCount"],
            -item["referenceCountInEval"],
            item["token"],
        ),
    )

    # dev OOV token
    dev_oov_tokens = sorted(
        [
            {
                "token": token,
                "devCount": count,
                "trainCount": int(train_counter.get(token, 0)),
            }
            for token, count in dev_counter.items()
            if int(train_counter.get(token, 0)) == 0
        ],
        key=lambda item: (-item["devCount"], item["token"]),
    )

    insertion_summary = [
        {"token": token, "insertCount": count}
        for token, count in insertion_counter.most_common(50)
    ]

    summary = {
        "totalReferenceTokens": total_reference_tokens,
        "totalCorrectTokens": total_correct_tokens,
        "totalErrorTokens": total_error_tokens,
        "totalSubstitute": total_substitute,
        "totalDelete": total_delete,
        "totalInsert": total_insert,
        "tokenAccuracy": safe_divide(total_correct_tokens, total_reference_tokens),
        "tokenErrorRate": safe_divide(total_error_tokens, total_reference_tokens),
        "substituteRate": safe_divide(total_substitute, total_reference_tokens),
        "deleteRate": safe_divide(total_delete, total_reference_tokens),
        "insertPerReferenceToken": safe_divide(total_insert, total_reference_tokens),
        "bucketSummary": bucket_summary,
        "devOovTokens": dev_oov_tokens,
        "devOovTokenTotal": sum(item["devCount"] for item in dev_oov_tokens),
        "devOovTypeCount": len(dev_oov_tokens),
        "lowFrequencyErrorTokensTop50": low_frequency_error_tokens[:50],
        "highFrequencyErrorTokensTop50": high_frequency_error_tokens[:50],
        "insertionTokensTop50": insertion_summary,
    }

    return {
        "summary": summary,
        "tokenRows": token_rows,
    }


# =========================================================
# 6. 打印工具
# =========================================================

def print_summary(result: Dict) -> None:
    """
    打印诊断摘要。

    Args:
        result: analyze_prediction_errors 的返回结果。
    """
    summary = result["summary"]

    print("\n===== 总体错误统计 =====")
    print("totalReferenceTokens:", summary["totalReferenceTokens"])
    print("totalCorrectTokens:", summary["totalCorrectTokens"])
    print("totalErrorTokens:", summary["totalErrorTokens"])
    print("totalSubstitute:", summary["totalSubstitute"])
    print("totalDelete:", summary["totalDelete"])
    print("totalInsert:", summary["totalInsert"])
    print("tokenAccuracy:", round(summary["tokenAccuracy"], 4))
    print("tokenErrorRate:", round(summary["tokenErrorRate"], 4))
    print("substituteRate:", round(summary["substituteRate"], 4))
    print("deleteRate:", round(summary["deleteRate"], 4))
    print("insertPerReferenceToken:", round(summary["insertPerReferenceToken"], 4))

    print("\n===== 按 train 频次分桶 =====")
    for row in summary["bucketSummary"]:
        print(
            row["bucket"],
            "refTokens=",
            row["referenceTokenCount"],
            "acc=",
            round(row["tokenAccuracy"], 4),
            "err=",
            round(row["tokenErrorRate"], 4),
            "delete=",
            round(row["deleteRate"], 4),
            "sub=",
            round(row["substituteRate"], 4),
        )

    print("\n===== dev OOV =====")
    print("devOovTypeCount:", summary["devOovTypeCount"])
    print("devOovTokenTotal:", summary["devOovTokenTotal"])
    for row in summary["devOovTokens"][:20]:
        print(row["token"], "devCount=", row["devCount"])

    print("\n===== 低频错误 token Top 20 =====")
    for row in summary["lowFrequencyErrorTokensTop50"][:20]:
        print(
            row["token"],
            "train=",
            row["trainCount"],
            "dev=",
            row["devCount"],
            "evalRef=",
            row["referenceCountInEval"],
            "err=",
            row["errorCount"],
            "acc=",
            round(row["tokenAccuracy"], 4),
        )

    print("\n===== 高频仍错误 token Top 20 =====")
    for row in summary["highFrequencyErrorTokensTop50"][:20]:
        print(
            row["token"],
            "train=",
            row["trainCount"],
            "dev=",
            row["devCount"],
            "evalRef=",
            row["referenceCountInEval"],
            "err=",
            row["errorCount"],
            "acc=",
            round(row["tokenAccuracy"], 4),
        )

    print("\n===== 插入 token Top 20 =====")
    for row in summary["insertionTokensTop50"][:20]:
        print(row["token"], "insert=", row["insertCount"])


# =========================================================
# 7. 主入口
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== CE-CSL token 频率与错误诊断开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("预测文件:", PREDICTION_FILE)
    print("输出目录:", OUTPUT_DIR)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"找不到预测诊断文件：{PREDICTION_FILE}\n"
            f"请先运行 inspect_predictions.py"
        )

    train_counter = count_tokens_for_split("train")
    dev_counter = count_tokens_for_split("dev")
    test_counter = count_tokens_for_split("test")

    prediction_rows = read_jsonl(PREDICTION_FILE)

    print("train token type:", len(train_counter))
    print("dev token type:", len(dev_counter))
    print("test token type:", len(test_counter))
    print("prediction rows:", len(prediction_rows))

    result = analyze_prediction_errors(
        prediction_rows=prediction_rows,
        train_counter=train_counter,
        dev_counter=dev_counter,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_json(
        OUTPUT_DIR / "frequency_error_summary.json",
        result["summary"],
    )

    write_jsonl(
        OUTPUT_DIR / "token_error_rows.jsonl",
        result["tokenRows"],
    )

    print_summary(result)

    print("\n已写出:")
    print("summary:", OUTPUT_DIR / "frequency_error_summary.json")
    print("token rows:", OUTPUT_DIR / "token_error_rows.jsonl")

    print("===== CE-CSL token 频率与错误诊断结束 =====")


if __name__ == "__main__":
    main()
