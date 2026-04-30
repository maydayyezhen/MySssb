"""
V18 decode calibration 评估脚本

作用：
1. 加载 v018 best_dev_ter.pt。
2. 在 dev split 上评估不同 CTC 解码配置。
3. 尝试通过 blank_penalty 和 token_insert_bonus 缓解输出偏短问题。
4. 不训练模型，只做解码层面的参数搜索。

说明：
- blank_penalty: 对 blank log probability 做惩罚，鼓励模型吐出更多 token。
- token_insert_bonus: beam search 排序时给更长 token 序列一点奖励。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

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


from ce_csl.ctc_decode import ctc_greedy_decode, edit_distance  # noqa: E402
from ce_csl.ctc_beam_decode import ctc_prefix_beam_search_batch  # noqa: E402
from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"

CHECKPOINT_PATH = (
    PROCESSED_DIR
    / "checkpoints"
    / "full_bilstm_ctc_raw_delta_finetune_v18_lr1e4"
    / "best_dev_ter.pt"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "logs"
    / "full_bilstm_ctc_raw_delta_finetune_v18_lr1e4"
    / "decode_calibration"
)

SPLIT = "dev"
MAX_ITEMS = None

BLANK_ID = 0
FEATURE_DIM = 166
FEATURE_MODE = "raw_delta"

BATCH_SIZE = 16
NUM_WORKERS = 0


# =========================================================
# 3. 解码配置
# =========================================================

DECODE_CONFIGS = [
    # greedy baseline
    {
        "name": "greedy_blank0",
        "decode_type": "greedy",
        "blank_penalty": 0.0,
        "beam_size": None,
        "top_k_per_frame": None,
        "token_insert_bonus": 0.0,
    },

    # greedy + blank penalty
    {
        "name": "greedy_blank02",
        "decode_type": "greedy",
        "blank_penalty": 0.2,
        "beam_size": None,
        "top_k_per_frame": None,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "greedy_blank04",
        "decode_type": "greedy",
        "blank_penalty": 0.4,
        "beam_size": None,
        "top_k_per_frame": None,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "greedy_blank06",
        "decode_type": "greedy",
        "blank_penalty": 0.6,
        "beam_size": None,
        "top_k_per_frame": None,
        "token_insert_bonus": 0.0,
    },

    # beam search baseline
    {
        "name": "beam3_top20_blank0_bonus0",
        "decode_type": "beam",
        "blank_penalty": 0.0,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.0,
    },

    # beam + blank penalty
    {
        "name": "beam3_top20_blank02_bonus0",
        "decode_type": "beam",
        "blank_penalty": 0.2,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "beam3_top20_blank04_bonus0",
        "decode_type": "beam",
        "blank_penalty": 0.4,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "beam3_top20_blank06_bonus0",
        "decode_type": "beam",
        "blank_penalty": 0.6,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.0,
    },

    # beam + blank penalty + token insert bonus
    {
        "name": "beam3_top20_blank04_bonus01",
        "decode_type": "beam",
        "blank_penalty": 0.4,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.1,
    },
    {
        "name": "beam3_top20_blank04_bonus02",
        "decode_type": "beam",
        "blank_penalty": 0.4,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.2,
    },
    {
        "name": "beam3_top20_blank06_bonus01",
        "decode_type": "beam",
        "blank_penalty": 0.6,
        "beam_size": 3,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.1,
    },

    # 稍大 beam，最后再试
    {
        "name": "beam5_top20_blank04_bonus01",
        "decode_type": "beam",
        "blank_penalty": 0.4,
        "beam_size": 5,
        "top_k_per_frame": 20,
        "token_insert_bonus": 0.1,
    },
]


# =========================================================
# 4. 工具函数
# =========================================================

def get_device() -> torch.device:
    """
    获取当前设备。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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


def safe_divide(numerator: float, denominator: float) -> float:
    """
    安全除法。
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def apply_blank_penalty(
    log_probs_btc: torch.Tensor,
    blank_penalty: float,
    blank_id: int,
) -> torch.Tensor:
    """
    对 blank log probability 做惩罚。

    注意：
    这里不重新 softmax 归一化，因为解码只关心相对排序。
    减小 blank 的 log 概率后，模型会更愿意输出非 blank token。
    """
    if blank_penalty <= 0:
        return log_probs_btc

    adjusted = log_probs_btc.clone()
    adjusted[:, :, blank_id] = adjusted[:, :, blank_id] - blank_penalty

    return adjusted


def load_model(device: torch.device) -> tuple[BiLstmCtcModel, Dict]:
    """
    加载 V18 checkpoint。
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
# 5. 单个配置评估
# =========================================================

def evaluate_one_config(
    model: BiLstmCtcModel,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict,
) -> Dict:
    """
    评估单个解码配置。
    """
    start_time = time.time()

    total_edit_distance = 0
    total_target_tokens = 0
    total_prediction_tokens = 0
    empty_prediction_count = 0

    all_examples: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            chinese_list = batch["chinese_list"]
            reference_gloss_list = batch["gloss_list"]

            log_probs_btc = model(features)

            log_probs_btc = apply_blank_penalty(
                log_probs_btc=log_probs_btc,
                blank_penalty=float(config["blank_penalty"]),
                blank_id=BLANK_ID,
            )

            if config["decode_type"] == "greedy":
                decoded_id_list = ctc_greedy_decode(
                    log_probs_btc=log_probs_btc,
                    input_lengths=input_lengths,
                    blank_id=BLANK_ID,
                )

            elif config["decode_type"] == "beam":
                decoded_id_list = ctc_prefix_beam_search_batch(
                    log_probs_btv=log_probs_btc,
                    input_lengths=input_lengths,
                    blank_id=BLANK_ID,
                    beam_size=int(config["beam_size"]),
                    top_k_per_frame=int(config["top_k_per_frame"]),
                    token_insert_bonus=float(config["token_insert_bonus"]),
                )

            else:
                raise ValueError(f"未知 decode_type：{config['decode_type']}")

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

                all_examples.append(
                    {
                        "sampleId": sample_ids[index],
                        "chinese": chinese_list[index],
                        "referenceLength": len(reference_ids),
                        "predictionLength": len(decoded_ids),
                        "editDistance": distance,
                        "TER": safe_divide(distance, len(reference_ids)),
                        "referenceGloss": reference_gloss_list[index],
                    }
                )

    elapsed_seconds = time.time() - start_time
    sample_count = len(all_examples)

    summary = {
        "name": config["name"],
        "decodeType": config["decode_type"],
        "blankPenalty": config["blank_penalty"],
        "beamSize": config["beam_size"],
        "topKPerFrame": config["top_k_per_frame"],
        "tokenInsertBonus": config["token_insert_bonus"],
        "sampleCount": sample_count,
        "totalEditDistance": total_edit_distance,
        "totalTargetTokens": total_target_tokens,
        "overallTER": safe_divide(total_edit_distance, total_target_tokens),
        "avgReferenceLength": safe_divide(total_target_tokens, sample_count),
        "avgPredictionLength": safe_divide(total_prediction_tokens, sample_count),
        "emptyPredictionCount": empty_prediction_count,
        "emptyPredictionRatio": safe_divide(empty_prediction_count, sample_count),
        "elapsedSeconds": elapsed_seconds,
    }

    worst_examples = sorted(
        all_examples,
        key=lambda row: (row["TER"], row["editDistance"]),
        reverse=True,
    )[:20]

    best_examples = sorted(
        all_examples,
        key=lambda row: (row["TER"], row["editDistance"]),
    )[:20]

    return {
        "summary": summary,
        "worstExamples": worst_examples,
        "bestExamples": best_examples,
    }


# =========================================================
# 6. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== V18 decode calibration 开始 =====")
    print("checkpoint:", CHECKPOINT_PATH)
    print("output:", OUTPUT_DIR)

    device = get_device()
    model, checkpoint = load_model(device)

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

    print("device:", device)
    print("split:", SPLIT)
    print("feature_mode:", FEATURE_MODE)
    print("dataset size:", len(dataset))
    print("batch count:", len(dataloader))
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint dev_TER:", checkpoint.get("dev_ter"))
    print("checkpoint dev_loss:", checkpoint.get("dev_loss"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []

    for config in DECODE_CONFIGS:
        print("\n" + "=" * 80)
        print("评估解码配置:", config["name"])
        print("=" * 80)

        result = evaluate_one_config(
            model=model,
            dataloader=dataloader,
            device=device,
            config=config,
        )

        summary = result["summary"]
        summaries.append(summary)

        write_json(
            OUTPUT_DIR / f"{config['name']}_summary.json",
            summary,
        )

        write_jsonl(
            OUTPUT_DIR / f"{config['name']}_worst_examples.jsonl",
            result["worstExamples"],
        )

        write_jsonl(
            OUTPUT_DIR / f"{config['name']}_best_examples.jsonl",
            result["bestExamples"],
        )

        print("overallTER:", round(summary["overallTER"], 4))
        print("avgReferenceLength:", round(summary["avgReferenceLength"], 4))
        print("avgPredictionLength:", round(summary["avgPredictionLength"], 4))
        print("emptyPredictionRatio:", round(summary["emptyPredictionRatio"], 4))
        print("elapsedSeconds:", round(summary["elapsedSeconds"], 2))

    sorted_summaries = sorted(
        summaries,
        key=lambda row: row["overallTER"],
    )

    write_json(
        OUTPUT_DIR / "decode_calibration_summary.json",
        {
            "checkpointPath": str(CHECKPOINT_PATH),
            "checkpointEpoch": checkpoint.get("epoch"),
            "checkpointDevTER": checkpoint.get("dev_ter"),
            "checkpointDevLoss": checkpoint.get("dev_loss"),
            "summaries": summaries,
            "sortedSummaries": sorted_summaries,
        },
    )

    print("\n===== 排名汇总 =====")
    for index, summary in enumerate(sorted_summaries, start=1):
        print(
            f"{index:02d}.",
            summary["name"],
            "TER=",
            round(summary["overallTER"], 4),
            "avgPredLen=",
            round(summary["avgPredictionLength"], 4),
            "emptyRatio=",
            round(summary["emptyPredictionRatio"], 4),
            "sec=",
            round(summary["elapsedSeconds"], 2),
        )

    print("\n输出目录:", OUTPUT_DIR)
    print("===== V18 decode calibration 结束 =====")


if __name__ == "__main__":
    main()