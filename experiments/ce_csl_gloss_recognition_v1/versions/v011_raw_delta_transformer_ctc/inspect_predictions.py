"""
检查 full_bilstm_ctc 全量模型预测结果

作用：
1. 加载 best_dev_ter.pt 和 best_dev_loss.pt。
2. 在 dev 全量 515 条上做 CTC greedy decode。
3. 输出真实 gloss / 预测 gloss / TER。
4. 统计预测长度、空预测比例、高频预测 token、按真实 gloss 长度分组的 TER。
5. 帮助判断当前 baseline 的主要错误类型。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


# =========================================================
# 1. 项目路径配置
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

# experiments/ce_csl_gloss_recognition_v1
EXPERIMENT_ROOT = CURRENT_FILE.parents[2]

# experiments/ce_csl_gloss_recognition_v1/src
SRC_DIR = EXPERIMENT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ce_csl.ctc_decode import ctc_greedy_decode, edit_distance, ids_to_gloss  # noqa: E402
from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import TransformerCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints" / "full_transformer_ctc_raw_delta"
LOG_DIR = PROCESSED_DIR / "logs" / "full_transformer_ctc_raw_delta" / "prediction_diagnosis"

CHECKPOINTS = [
    CHECKPOINT_DIR / "best_dev_ter.pt",
    CHECKPOINT_DIR / "best_dev_loss.pt",
]

SPLIT = "dev"
MAX_ITEMS = None

BLANK_ID = 0
FEATURE_DIM = 166
FEATURE_MODE = "raw_delta"

BATCH_SIZE = 16
NUM_WORKERS = 0

PRINT_EXAMPLES = 20


# =========================================================
# 3. 工具函数
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


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict) -> None:
    """
    写出 json 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_id_to_gloss() -> Dict[int, str]:
    """
    从 ctc_ready 文件构建 id -> gloss 映射。
    """
    id_to_gloss: Dict[int, str] = {
        0: "<blank>",
        1: "<unk>",
    }

    for split in ["train", "dev", "test"]:
        ready_path = CTC_READY_DIR / f"{split}_ctc_ready.jsonl"

        if not ready_path.exists():
            raise FileNotFoundError(f"找不到 CTC ready 文件：{ready_path}")

        rows = read_jsonl(ready_path)

        for row in rows:
            gloss_list = row.get("gloss", [])
            gloss_ids = row.get("glossIds", [])

            for gloss, gloss_id in zip(gloss_list, gloss_ids):
                id_to_gloss[int(gloss_id)] = str(gloss)

    return id_to_gloss


def get_device() -> torch.device:
    """
    获取设备。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_length_bucket(length: int) -> str:
    """
    按真实 gloss 长度分组。
    """
    if length <= 3:
        return "01_len_1_3"

    if length <= 5:
        return "02_len_4_5"

    if length <= 8:
        return "03_len_6_8"

    if length <= 12:
        return "04_len_9_12"

    return "05_len_13_plus"


def safe_divide(numerator: float, denominator: float) -> float:
    """
    安全除法。
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[TransformerCtcModel, Dict]:
    """
    加载 checkpoint 并构建模型。
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    model = TransformerCtcModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# =========================================================
# 4. 评估逻辑
# =========================================================

def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset: CeCslGlossDataset,
    dataloader: DataLoader,
    device: torch.device,
    id_to_gloss: Dict[int, str],
) -> Dict:
    """
    评估一个 checkpoint。
    """
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    all_examples: List[Dict] = []

    total_edit_distance = 0
    total_target_tokens = 0

    total_pred_tokens = 0
    empty_prediction_count = 0

    predicted_token_counter: Counter[str] = Counter()
    reference_token_counter: Counter[str] = Counter()

    bucket_edit_distance: Dict[str, int] = defaultdict(int)
    bucket_target_tokens: Dict[str, int] = defaultdict(int)
    bucket_sample_count: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            reference_gloss_list = batch["gloss_list"]
            chinese_list = batch["chinese_list"]

            log_probs_btc = model(features, input_lengths)

            decoded_id_list = ctc_greedy_decode(
                log_probs_btc=log_probs_btc,
                input_lengths=input_lengths,
                blank_id=BLANK_ID,
            )

            for sample_id, decoded_ids, reference_gloss, chinese in zip(
                sample_ids,
                decoded_id_list,
                reference_gloss_list,
                chinese_list,
            ):
                prediction_gloss = ids_to_gloss(decoded_ids, id_to_gloss)

                distance = edit_distance(
                    source=prediction_gloss,
                    target=reference_gloss,
                )

                reference_length = len(reference_gloss)
                prediction_length = len(prediction_gloss)

                ter = safe_divide(distance, reference_length)

                total_edit_distance += distance
                total_target_tokens += reference_length

                total_pred_tokens += prediction_length

                if prediction_length == 0:
                    empty_prediction_count += 1

                predicted_token_counter.update(prediction_gloss)
                reference_token_counter.update(reference_gloss)

                bucket = get_length_bucket(reference_length)

                bucket_edit_distance[bucket] += distance
                bucket_target_tokens[bucket] += reference_length
                bucket_sample_count[bucket] += 1

                all_examples.append(
                    {
                        "sampleId": sample_id,
                        "chinese": chinese,
                        "referenceGloss": reference_gloss,
                        "predictionGloss": prediction_gloss,
                        "referenceLength": reference_length,
                        "predictionLength": prediction_length,
                        "editDistance": distance,
                        "TER": ter,
                    }
                )

    sample_count = len(dataset)

    overall_ter = safe_divide(total_edit_distance, total_target_tokens)
    avg_reference_length = safe_divide(total_target_tokens, sample_count)
    avg_prediction_length = safe_divide(total_pred_tokens, sample_count)
    empty_prediction_ratio = safe_divide(empty_prediction_count, sample_count)

    bucket_summary = []

    for bucket in sorted(bucket_sample_count.keys()):
        bucket_summary.append(
            {
                "bucket": bucket,
                "sampleCount": bucket_sample_count[bucket],
                "targetTokens": bucket_target_tokens[bucket],
                "editDistance": bucket_edit_distance[bucket],
                "TER": safe_divide(
                    bucket_edit_distance[bucket],
                    bucket_target_tokens[bucket],
                ),
            }
        )

    top_predicted_tokens = [
        {"token": token, "count": count}
        for token, count in predicted_token_counter.most_common(30)
    ]

    top_reference_tokens = [
        {"token": token, "count": count}
        for token, count in reference_token_counter.most_common(30)
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
        "checkpoint": checkpoint_path.name,
        "checkpointPath": str(checkpoint_path),
        "checkpointEpoch": checkpoint.get("epoch"),
        "checkpointTrainLoss": checkpoint.get("train_loss"),
        "checkpointDevLoss": checkpoint.get("dev_loss"),
        "checkpointDevTER": checkpoint.get("dev_ter"),
        "split": SPLIT,
        "sampleCount": sample_count,
        "totalEditDistance": total_edit_distance,
        "totalTargetTokens": total_target_tokens,
        "overallTER": overall_ter,
        "avgReferenceLength": avg_reference_length,
        "avgPredictionLength": avg_prediction_length,
        "emptyPredictionCount": empty_prediction_count,
        "emptyPredictionRatio": empty_prediction_ratio,
        "lengthBucketSummary": bucket_summary,
        "topPredictedTokens": top_predicted_tokens,
        "topReferenceTokens": top_reference_tokens,
    }

    return {
        "summary": summary,
        "allExamples": all_examples,
        "worstExamples": worst_examples,
        "bestExamples": best_examples,
    }


def print_summary(summary: Dict) -> None:
    """
    打印摘要。
    """
    print("\n" + "=" * 80)
    print("checkpoint:", summary["checkpoint"])
    print("checkpoint epoch:", summary["checkpointEpoch"])
    print("checkpoint dev_loss:", summary["checkpointDevLoss"])
    print("checkpoint dev_TER:", summary["checkpointDevTER"])
    print("split:", summary["split"])
    print("sampleCount:", summary["sampleCount"])
    print("overallTER:", round(summary["overallTER"], 4))
    print("avgReferenceLength:", round(summary["avgReferenceLength"], 4))
    print("avgPredictionLength:", round(summary["avgPredictionLength"], 4))
    print("emptyPredictionRatio:", round(summary["emptyPredictionRatio"], 4))

    print("\n按真实 gloss 长度分组:")
    for row in summary["lengthBucketSummary"]:
        print(
            row["bucket"],
            "samples=",
            row["sampleCount"],
            "TER=",
            round(row["TER"], 4),
        )

    print("\n预测高频 token Top 15:")
    for row in summary["topPredictedTokens"][:15]:
        print(row["token"], row["count"])

    print("\n真实高频 token Top 15:")
    for row in summary["topReferenceTokens"][:15]:
        print(row["token"], row["count"])


def print_examples(title: str, examples: List[Dict]) -> None:
    """
    打印样例。
    """
    print("\n" + "#" * 80)
    print(title)
    print("#" * 80)

    for row in examples:
        print("-" * 80)
        print("sampleId:", row["sampleId"])
        print("中文:", row["chinese"])
        print("真实:", " / ".join(row["referenceGloss"]))
        print("预测:", " / ".join(row["predictionGloss"]))
        print(
            "len:",
            row["referenceLength"],
            "->",
            row["predictionLength"],
            "editDistance:",
            row["editDistance"],
            "TER:",
            round(row["TER"], 4),
        )


# =========================================================
# 5. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== full_bilstm_ctc 预测诊断开始 =====")

    device = get_device()
    id_to_gloss = build_id_to_gloss()

    print("device:", device)
    print("split:", SPLIT)
    print("feature_mode:", FEATURE_MODE)
    print("数据集目录:", DATASET_ROOT)
    print("输出目录:", LOG_DIR)

    dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split=SPLIT,
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

    print("dataset size:", len(dataset))
    print("batch count:", len(dataloader))

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict] = []

    for checkpoint_path in CHECKPOINTS:
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            dataloader=dataloader,
            device=device,
            id_to_gloss=id_to_gloss,
        )

        summary = result["summary"]
        all_summaries.append(summary)

        checkpoint_name = checkpoint_path.stem

        write_json(
            LOG_DIR / f"{checkpoint_name}_summary.json",
            summary,
        )

        write_jsonl(
            LOG_DIR / f"{checkpoint_name}_all_examples.jsonl",
            result["allExamples"],
        )

        write_jsonl(
            LOG_DIR / f"{checkpoint_name}_worst_examples.jsonl",
            result["worstExamples"],
        )

        write_jsonl(
            LOG_DIR / f"{checkpoint_name}_best_examples.jsonl",
            result["bestExamples"],
        )

        print_summary(summary)
        print_examples(
            title=f"{checkpoint_path.name} 最差样例",
            examples=result["worstExamples"][:8],
        )
        print_examples(
            title=f"{checkpoint_path.name} 最好样例",
            examples=result["bestExamples"][:8],
        )

    write_json(
        LOG_DIR / "diagnosis_summary.json",
        {
            "summaries": all_summaries,
        },
    )

    print("\n===== 汇总对比 =====")
    for summary in all_summaries:
        print(
            summary["checkpoint"],
            "epoch=",
            summary["checkpointEpoch"],
            "overallTER=",
            round(summary["overallTER"], 4),
            "avgPredLen=",
            round(summary["avgPredictionLength"], 4),
            "avgRefLen=",
            round(summary["avgReferenceLength"], 4),
            "emptyRatio=",
            round(summary["emptyPredictionRatio"], 4),
        )

    print("\n输出目录:", LOG_DIR)
    print("===== full_bilstm_ctc 预测诊断结束 =====")


if __name__ == "__main__":
    main()
