"""
v016 top_k_500 controlled vocab 预测诊断

作用：
1. 加载 controlled top_k_500 的 best_dev_ter.pt。
2. 在 dev 上计算 controlled vocab TER。
3. 统计 <unk> 在真实标签和预测中的比例。
4. 分开统计：
   - 不含 <unk> 的样本 TER
   - 含 <unk> 的样本 TER
   - kept token 的识别准确率
   - unk token 的识别准确率
5. 判断 controlled vocab 的提升是否主要来自 <unk> 简化。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


# =========================================================
# 1. 项目路径配置
# =========================================================

CURRENT_FILE = Path(__file__).resolve()


def find_experiment_root(start_file: Path) -> Path:
    """
    从当前文件向上查找实验根目录。
    """
    for parent in start_file.parents:
        if (parent / "src" / "ce_csl").exists():
            return parent

    raise RuntimeError(f"无法从 {start_file} 定位实验根目录")


EXPERIMENT_ROOT = find_experiment_root(CURRENT_FILE)
SRC_DIR = EXPERIMENT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 让当前版本目录里的 controlled_dataset.py 可导入
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))


from ce_csl.ctc_decode import ctc_greedy_decode, edit_distance  # noqa: E402
from ce_csl.dataset import ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402
from controlled_dataset import ControlledVocabCeCslGlossDataset  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"

CONTROLLED_VOCAB_NAME = "top_k_500"

CONTROLLED_VOCAB_PATH = (
    PROCESSED_DIR
    / "controlled_vocab"
    / "v016_raw_delta_controlled_vocab"
    / CONTROLLED_VOCAB_NAME
    / "controlled_vocab.json"
)

CHECKPOINT_PATH = (
    PROCESSED_DIR
    / "checkpoints"
    / "full_bilstm_ctc_raw_delta_controlled_top_k_500"
    / "best_dev_ter.pt"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "logs"
    / "full_bilstm_ctc_raw_delta_controlled_top_k_500"
    / "prediction_diagnosis"
)

SPLIT = "dev"
MAX_ITEMS = None

BLANK_ID = 0
UNK_ID = 1
FEATURE_DIM = 166
FEATURE_MODE = "raw_delta"

BATCH_SIZE = 16
NUM_WORKERS = 0

PRINT_EXAMPLES = 20


# =========================================================
# 3. 文件工具
# =========================================================

def write_json(path: Path, data: Dict | List) -> None:
    """
    写出 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_controlled_vocab() -> Dict:
    """
    读取 controlled vocab。
    """
    if not CONTROLLED_VOCAB_PATH.exists():
        raise FileNotFoundError(f"找不到 controlled vocab：{CONTROLLED_VOCAB_PATH}")

    with CONTROLLED_VOCAB_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_device() -> torch.device:
    """
    获取设备。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def safe_divide(numerator: float, denominator: float) -> float:
    """
    安全除法。
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


# =========================================================
# 4. 对齐工具
# =========================================================

def align_prediction_to_reference(
    prediction: List[int],
    reference: List[int],
) -> List[Dict]:
    """
    对预测 ID 序列和真实 ID 序列做编辑距离对齐。
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
            cost = 0 if prediction[i - 1] == reference[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    operations: List[Dict] = []

    i = pred_len
    j = ref_len

    while i > 0 or j > 0:
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
# 5. 模型加载
# =========================================================

def load_model(device: torch.device) -> tuple[BiLstmCtcModel, Dict]:
    """
    加载模型 checkpoint。
    """
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    model = BiLstmCtcModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# =========================================================
# 6. 评估
# =========================================================

def evaluate() -> Dict:
    """
    执行诊断评估。
    """
    device = get_device()
    controlled_vocab = load_controlled_vocab()

    new_to_gloss = {
        int(new_id): gloss
        for new_id, gloss in controlled_vocab["newToGloss"].items()
    }

    model, checkpoint = load_model(device)

    dataset = ControlledVocabCeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split=SPLIT,
        controlled_vocab_path=CONTROLLED_VOCAB_PATH,
        max_items=MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
        feature_mode=FEATURE_MODE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ce_csl_collate_fn,
    )

    total_edit_distance = 0
    total_target_tokens = 0
    total_prediction_tokens = 0
    empty_prediction_count = 0

    ref_unk_count = 0
    pred_unk_count = 0

    group_stats = defaultdict(
        lambda: {
            "sampleCount": 0,
            "editDistance": 0,
            "targetTokens": 0,
            "predictionTokens": 0,
        }
    )

    token_type_stats = {
        "kept": {
            "referenceCount": 0,
            "correctCount": 0,
            "substituteCount": 0,
            "deleteCount": 0,
            "errorCount": 0,
        },
        "unk": {
            "referenceCount": 0,
            "correctCount": 0,
            "substituteCount": 0,
            "deleteCount": 0,
            "errorCount": 0,
        },
    }

    predicted_token_counter: Counter = Counter()
    reference_token_counter: Counter = Counter()

    all_examples: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            chinese_list = batch["chinese_list"]

            log_probs_btc = model(features)

            decoded_id_list = ctc_greedy_decode(
                log_probs_btc=log_probs_btc,
                input_lengths=input_lengths,
                blank_id=BLANK_ID,
            )

            targets_cpu = batch["targets"].detach().cpu().tolist()
            target_lengths_cpu = batch["target_lengths"].detach().cpu().tolist()

            target_offset = 0

            for index, decoded_ids in enumerate(decoded_id_list):
                target_length = int(target_lengths_cpu[index])

                reference_ids = targets_cpu[target_offset: target_offset + target_length]
                target_offset += target_length

                distance = edit_distance(
                    source=[str(token_id) for token_id in decoded_ids],
                    target=[str(token_id) for token_id in reference_ids],
                )

                total_edit_distance += distance
                total_target_tokens += len(reference_ids)
                total_prediction_tokens += len(decoded_ids)

                if len(decoded_ids) == 0:
                    empty_prediction_count += 1

                sample_ref_unk_count = sum(1 for token_id in reference_ids if token_id == UNK_ID)
                sample_pred_unk_count = sum(1 for token_id in decoded_ids if token_id == UNK_ID)

                ref_unk_count += sample_ref_unk_count
                pred_unk_count += sample_pred_unk_count

                group_name = "samples_with_unk" if sample_ref_unk_count > 0 else "samples_without_unk"

                group_stats[group_name]["sampleCount"] += 1
                group_stats[group_name]["editDistance"] += distance
                group_stats[group_name]["targetTokens"] += len(reference_ids)
                group_stats[group_name]["predictionTokens"] += len(decoded_ids)

                operations = align_prediction_to_reference(
                    prediction=decoded_ids,
                    reference=reference_ids,
                )

                for operation in operations:
                    reference_token = operation["reference"]
                    prediction_token = operation["prediction"]
                    op = operation["op"]

                    if prediction_token is not None:
                        predicted_token_counter[prediction_token] += 1

                    if reference_token is not None:
                        reference_token_counter[reference_token] += 1

                        token_type = "unk" if reference_token == UNK_ID else "kept"
                        token_type_stats[token_type]["referenceCount"] += 1

                        if op == "correct":
                            token_type_stats[token_type]["correctCount"] += 1
                        elif op == "substitute":
                            token_type_stats[token_type]["substituteCount"] += 1
                            token_type_stats[token_type]["errorCount"] += 1
                        elif op == "delete":
                            token_type_stats[token_type]["deleteCount"] += 1
                            token_type_stats[token_type]["errorCount"] += 1

                reference_gloss = [
                    new_to_gloss.get(token_id, f"<id:{token_id}>")
                    for token_id in reference_ids
                ]

                prediction_gloss = [
                    new_to_gloss.get(token_id, f"<id:{token_id}>")
                    for token_id in decoded_ids
                ]

                all_examples.append(
                    {
                        "sampleId": sample_ids[index],
                        "chinese": chinese_list[index],
                        "referenceIds": reference_ids,
                        "predictionIds": decoded_ids,
                        "referenceGloss": reference_gloss,
                        "predictionGloss": prediction_gloss,
                        "referenceLength": len(reference_ids),
                        "predictionLength": len(decoded_ids),
                        "referenceUnkCount": sample_ref_unk_count,
                        "predictionUnkCount": sample_pred_unk_count,
                        "editDistance": distance,
                        "TER": safe_divide(distance, len(reference_ids)),
                    }
                )

    sample_count = len(dataset)

    group_summary = {}

    for group_name, stats in group_stats.items():
        group_summary[group_name] = {
            "sampleCount": stats["sampleCount"],
            "editDistance": stats["editDistance"],
            "targetTokens": stats["targetTokens"],
            "predictionTokens": stats["predictionTokens"],
            "TER": safe_divide(stats["editDistance"], stats["targetTokens"]),
            "avgPredictionLength": safe_divide(stats["predictionTokens"], stats["sampleCount"]),
        }

    token_type_summary = {}

    for token_type, stats in token_type_stats.items():
        ref_count = stats["referenceCount"]

        token_type_summary[token_type] = {
            **stats,
            "accuracy": safe_divide(stats["correctCount"], ref_count),
            "errorRate": safe_divide(stats["errorCount"], ref_count),
            "deleteRate": safe_divide(stats["deleteCount"], ref_count),
            "substituteRate": safe_divide(stats["substituteCount"], ref_count),
        }

    top_predicted_tokens = [
        {
            "id": int(token_id),
            "gloss": new_to_gloss.get(int(token_id), f"<id:{token_id}>"),
            "count": int(count),
        }
        for token_id, count in predicted_token_counter.most_common(30)
    ]

    top_reference_tokens = [
        {
            "id": int(token_id),
            "gloss": new_to_gloss.get(int(token_id), f"<id:{token_id}>"),
            "count": int(count),
        }
        for token_id, count in reference_token_counter.most_common(30)
    ]

    worst_examples = sorted(
        all_examples,
        key=lambda row: (row["TER"], row["editDistance"]),
        reverse=True,
    )[:PRINT_EXAMPLES]

    best_examples = sorted(
        all_examples,
        key=lambda row: (row["TER"], row["editDistance"]),
    )[:PRINT_EXAMPLES]

    summary = {
        "checkpointPath": str(CHECKPOINT_PATH),
        "checkpointEpoch": checkpoint.get("epoch"),
        "checkpointDevLoss": checkpoint.get("dev_loss"),
        "checkpointDevTER": checkpoint.get("dev_ter"),
        "controlledVocabName": CONTROLLED_VOCAB_NAME,
        "controlledVocabPath": str(CONTROLLED_VOCAB_PATH),
        "controlledVocabSize": int(controlled_vocab["controlledVocabSize"]),
        "split": SPLIT,
        "sampleCount": sample_count,
        "totalEditDistance": total_edit_distance,
        "totalTargetTokens": total_target_tokens,
        "overallTER": safe_divide(total_edit_distance, total_target_tokens),
        "avgReferenceLength": safe_divide(total_target_tokens, sample_count),
        "avgPredictionLength": safe_divide(total_prediction_tokens, sample_count),
        "emptyPredictionCount": empty_prediction_count,
        "emptyPredictionRatio": safe_divide(empty_prediction_count, sample_count),
        "referenceUnkCount": ref_unk_count,
        "referenceUnkRate": safe_divide(ref_unk_count, total_target_tokens),
        "predictionUnkCount": pred_unk_count,
        "predictionUnkRateOverPredTokens": safe_divide(pred_unk_count, total_prediction_tokens),
        "groupSummary": group_summary,
        "tokenTypeSummary": token_type_summary,
        "topPredictedTokens": top_predicted_tokens,
        "topReferenceTokens": top_reference_tokens,
    }

    return {
        "summary": summary,
        "allExamples": all_examples,
        "worstExamples": worst_examples,
        "bestExamples": best_examples,
    }


# =========================================================
# 7. 打印
# =========================================================

def print_result(result: Dict) -> None:
    """
    打印诊断结果。
    """
    summary = result["summary"]

    print("\n===== controlled top_k_500 预测诊断 =====")
    print("checkpoint epoch:", summary["checkpointEpoch"])
    print("checkpoint dev_loss:", summary["checkpointDevLoss"])
    print("checkpoint dev_TER:", summary["checkpointDevTER"])
    print("controlledVocabSize:", summary["controlledVocabSize"])
    print("sampleCount:", summary["sampleCount"])
    print("overallTER:", round(summary["overallTER"], 4))
    print("avgReferenceLength:", round(summary["avgReferenceLength"], 4))
    print("avgPredictionLength:", round(summary["avgPredictionLength"], 4))
    print("emptyPredictionRatio:", round(summary["emptyPredictionRatio"], 4))
    print("referenceUnkRate:", round(summary["referenceUnkRate"], 4))
    print("predictionUnkRateOverPredTokens:", round(summary["predictionUnkRateOverPredTokens"], 4))

    print("\n===== 分组 TER =====")
    for group_name, row in summary["groupSummary"].items():
        print(
            group_name,
            "samples=",
            row["sampleCount"],
            "TER=",
            round(row["TER"], 4),
            "avgPredLen=",
            round(row["avgPredictionLength"], 4),
        )

    print("\n===== kept vs unk token 统计 =====")
    for token_type, row in summary["tokenTypeSummary"].items():
        print(
            token_type,
            "ref=",
            row["referenceCount"],
            "acc=",
            round(row["accuracy"], 4),
            "err=",
            round(row["errorRate"], 4),
            "delete=",
            round(row["deleteRate"], 4),
            "sub=",
            round(row["substituteRate"], 4),
        )

    print("\n===== 预测高频 token Top 15 =====")
    for row in summary["topPredictedTokens"][:15]:
        print(row["gloss"], row["count"])

    print("\n===== 真实高频 token Top 15 =====")
    for row in summary["topReferenceTokens"][:15]:
        print(row["gloss"], row["count"])

    print("\n===== 最好样例 Top 5 =====")
    for row in result["bestExamples"][:5]:
        print("-" * 80)
        print("sampleId:", row["sampleId"])
        print("中文:", row["chinese"])
        print("真实:", " / ".join(row["referenceGloss"]))
        print("预测:", " / ".join(row["predictionGloss"]))
        print("TER:", round(row["TER"], 4))

    print("\n===== 最差样例 Top 5 =====")
    for row in result["worstExamples"][:5]:
        print("-" * 80)
        print("sampleId:", row["sampleId"])
        print("中文:", row["chinese"])
        print("真实:", " / ".join(row["referenceGloss"]))
        print("预测:", " / ".join(row["predictionGloss"]))
        print("TER:", round(row["TER"], 4))


def main() -> None:
    """
    主入口。
    """
    print("===== v016 top_k_500 controlled prediction inspect 开始 =====")
    print("checkpoint:", CHECKPOINT_PATH)
    print("controlled vocab:", CONTROLLED_VOCAB_PATH)
    print("output:", OUTPUT_DIR)

    result = evaluate()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_json(OUTPUT_DIR / "summary.json", result["summary"])
    write_jsonl(OUTPUT_DIR / "all_examples.jsonl", result["allExamples"])
    write_jsonl(OUTPUT_DIR / "worst_examples.jsonl", result["worstExamples"])
    write_jsonl(OUTPUT_DIR / "best_examples.jsonl", result["bestExamples"])

    print_result(result)

    print("\n已写出:", OUTPUT_DIR)
    print("===== v016 top_k_500 controlled prediction inspect 结束 =====")


if __name__ == "__main__":
    main()