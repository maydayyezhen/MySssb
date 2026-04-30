"""
评估 CTC greedy decode 与 beam search decode

作用：
1. 加载 full_bilstm_ctc/best_dev_ter.pt。
2. 在 dev 全量 515 条上评估 greedy decode。
3. 在 dev 全量 515 条上评估多个 beam search 配置。
4. 比较 overall TER、平均预测长度、空预测比例。

说明：
- 不重新训练。
- 只比较不同解码策略。
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


from ce_csl.ctc_beam_decode import ctc_prefix_beam_search_batch  # noqa: E402
from ce_csl.ctc_decode import ctc_greedy_decode, edit_distance  # noqa: E402
from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"

CHECKPOINT_PATH = PROCESSED_DIR / "checkpoints" / "full_bilstm_ctc" / "best_dev_ter.pt"
OUTPUT_DIR = PROCESSED_DIR / "logs" / "full_bilstm_ctc" / "beam_search_eval"

SPLIT = "dev"
MAX_ITEMS = None

BLANK_ID = 0
FEATURE_DIM = 166

BATCH_SIZE = 16
NUM_WORKERS = 0

# 先别配太大，Python beam search 会比 greedy 慢很多
DECODE_CONFIGS = [
    {
        "name": "greedy",
        "type": "greedy",
    },
    {
        "name": "beam3_top30_bonus0",
        "type": "beam",
        "beam_size": 3,
        "top_k_per_frame": 30,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "beam5_top30_bonus0",
        "type": "beam",
        "beam_size": 5,
        "top_k_per_frame": 30,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "beam5_top50_bonus0",
        "type": "beam",
        "beam_size": 5,
        "top_k_per_frame": 50,
        "token_insert_bonus": 0.0,
    },
    {
        "name": "beam5_top50_bonus02",
        "type": "beam",
        "beam_size": 5,
        "top_k_per_frame": 50,
        "token_insert_bonus": 0.2,
    },
    {
        "name": "beam10_top50_bonus02",
        "type": "beam",
        "beam_size": 10,
        "top_k_per_frame": 50,
        "token_insert_bonus": 0.2,
    },
]


# =========================================================
# 3. 工具函数
# =========================================================

def get_device() -> torch.device:
    """
    获取设备。
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


def load_model(device: torch.device) -> BiLstmCtcModel:
    """
    加载 best_dev_ter checkpoint。

    Args:
        device: 设备。

    Returns:
        模型。
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

    print("checkpoint:", CHECKPOINT_PATH)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint dev_TER:", checkpoint.get("dev_ter"))
    print("checkpoint dev_loss:", checkpoint.get("dev_loss"))

    return model


def evaluate_decode_config(
    model: BiLstmCtcModel,
    dataloader: DataLoader,
    device: torch.device,
    decode_config: Dict,
) -> Dict:
    """
    评估一个解码配置。

    Args:
        model: 模型。
        dataloader: dev DataLoader。
        device: 设备。
        decode_config: 解码配置。

    Returns:
        评估摘要。
    """
    start_time = time.time()

    total_edit_distance = 0
    total_target_tokens = 0
    total_prediction_tokens = 0
    empty_prediction_count = 0
    sample_count = 0

    examples: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            gloss_list = batch["gloss_list"]
            chinese_list = batch["chinese_list"]

            log_probs_btv = model(features)

            if decode_config["type"] == "greedy":
                decoded_id_list = ctc_greedy_decode(
                    log_probs_btc=log_probs_btv,
                    input_lengths=input_lengths,
                    blank_id=BLANK_ID,
                )
            elif decode_config["type"] == "beam":
                decoded_id_list = ctc_prefix_beam_search_batch(
                    log_probs_btv=log_probs_btv,
                    input_lengths=input_lengths,
                    blank_id=BLANK_ID,
                    beam_size=int(decode_config["beam_size"]),
                    top_k_per_frame=int(decode_config["top_k_per_frame"]),
                    token_insert_bonus=float(decode_config["token_insert_bonus"]),
                )
            else:
                raise ValueError(f"未知 decode type：{decode_config['type']}")

            for sample_id, decoded_ids, reference_gloss, chinese in zip(
                sample_ids,
                decoded_id_list,
                gloss_list,
                chinese_list,
            ):
                reference_ids_as_text = [str(token) for token in reference_gloss]
                prediction_ids_as_text = [str(token_id) for token_id in decoded_ids]

                # 注意：
                # 这里用 id 和 gloss 文本不混合比较。
                # 为了评估 TER，应该比较 token id。
                # 但 batch 里目前没有原始 target ids，所以用 gloss 文本对 greedy 诊断不够严谨。
                # 因此这里重新使用 reference_gloss 文本只用于样例展示；
                # 正确 TER 需要 id_to_gloss 或 targets 切片。
                # 为避免重构 DataLoader，这里直接基于 gloss 文本和预测 id 文本会不匹配。
                # 所以下面会改为使用 targets 切片逻辑。
                _ = reference_ids_as_text
                _ = prediction_ids_as_text

            # 正确地从拼接 targets 中切回 reference ids
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

                sample_count += 1

                if len(examples) < 20:
                    examples.append(
                        {
                            "sampleId": sample_ids[index],
                            "chinese": chinese_list[index],
                            "referenceGloss": gloss_list[index],
                            "referenceIds": reference_ids,
                            "predictionIds": decoded_ids,
                            "editDistance": distance,
                            "targetLength": len(reference_ids),
                            "predictionLength": len(decoded_ids),
                            "TER": distance / len(reference_ids) if reference_ids else 0.0,
                        }
                    )

    elapsed_seconds = time.time() - start_time

    overall_ter = total_edit_distance / total_target_tokens if total_target_tokens > 0 else 0.0
    avg_reference_length = total_target_tokens / sample_count if sample_count > 0 else 0.0
    avg_prediction_length = total_prediction_tokens / sample_count if sample_count > 0 else 0.0
    empty_ratio = empty_prediction_count / sample_count if sample_count > 0 else 0.0

    return {
        "name": decode_config["name"],
        "config": decode_config,
        "sampleCount": sample_count,
        "totalEditDistance": total_edit_distance,
        "totalTargetTokens": total_target_tokens,
        "overallTER": overall_ter,
        "avgReferenceLength": avg_reference_length,
        "avgPredictionLength": avg_prediction_length,
        "emptyPredictionCount": empty_prediction_count,
        "emptyPredictionRatio": empty_ratio,
        "elapsedSeconds": elapsed_seconds,
        "examples": examples,
    }


# =========================================================
# 4. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== CTC beam search 评估开始 =====")

    device = get_device()

    print("device:", device)
    print("dataset:", DATASET_ROOT)
    print("split:", SPLIT)
    print("output:", OUTPUT_DIR)

    model = load_model(device)

    dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split=SPLIT,
        max_items=MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []

    for decode_config in DECODE_CONFIGS:
        print("\n" + "=" * 80)
        print("评估解码配置:", decode_config["name"])
        print("=" * 80)

        result = evaluate_decode_config(
            model=model,
            dataloader=dataloader,
            device=device,
            decode_config=decode_config,
        )

        summary = {
            key: value
            for key, value in result.items()
            if key != "examples"
        }

        summaries.append(summary)

        write_json(
            OUTPUT_DIR / f"{decode_config['name']}_summary.json",
            summary,
        )

        write_jsonl(
            OUTPUT_DIR / f"{decode_config['name']}_examples.jsonl",
            result["examples"],
        )

        print("overallTER:", round(result["overallTER"], 4))
        print("avgReferenceLength:", round(result["avgReferenceLength"], 4))
        print("avgPredictionLength:", round(result["avgPredictionLength"], 4))
        print("emptyPredictionRatio:", round(result["emptyPredictionRatio"], 4))
        print("elapsedSeconds:", round(result["elapsedSeconds"], 2))

    write_json(
        OUTPUT_DIR / "beam_search_summary.json",
        {
            "checkpoint": str(CHECKPOINT_PATH),
            "summaries": summaries,
        },
    )

    print("\n===== 汇总 =====")
    for summary in summaries:
        print(
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
    print("===== CTC beam search 评估结束 =====")


if __name__ == "__main__":
    main()