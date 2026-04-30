"""
检查 subset_500 模型在 train/dev 上的预测效果

作用：
1. 加载 best_subset_500.pt 和 last_subset_500.pt。
2. 分别评估 train 前 500 条、dev 前 200 条。
3. 输出 TER，判断模型是训练不够，还是泛化/数据覆盖不足。
"""

from __future__ import annotations

import json
import sys
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


from ce_csl.ctc_decode import ctc_greedy_decode, edit_distance, ids_to_gloss  # noqa: E402
from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints" / "subset_500"

CHECKPOINTS = [
    CHECKPOINT_DIR / "best_subset_500.pt",
    CHECKPOINT_DIR / "last_subset_500.pt",
]

BLANK_ID = 0
FEATURE_DIM = 166

TRAIN_MAX_ITEMS = 500
DEV_MAX_ITEMS = 200

BATCH_SIZE = 16
NUM_WORKERS = 0

PRINT_EXAMPLES = 8


# =========================================================
# 3. 工具函数
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

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def build_id_to_gloss() -> Dict[int, str]:
    """
    从 ctc_ready 文件构建 id -> gloss 映射。

    Returns:
        id 到 gloss 的映射。
    """
    id_to_gloss: Dict[int, str] = {
        0: "<blank>",
        1: "<unk>",
    }

    for split in ["train", "dev", "test"]:
        ready_path = CTC_READY_DIR / f"{split}_ctc_ready.jsonl"

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

    Returns:
        CUDA 可用则返回 cuda，否则返回 cpu。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate_checkpoint_on_split(
    checkpoint_path: Path,
    split: str,
    max_items: int,
    device: torch.device,
    id_to_gloss: Dict[int, str],
) -> Dict:
    """
    评估某个 checkpoint 在某个 split 上的预测效果。

    Args:
        checkpoint_path: checkpoint 路径。
        split: train/dev/test。
        max_items: 最大评估样本数。
        device: 设备。
        id_to_gloss: id 到 gloss 的映射。

    Returns:
        评估结果。
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    model = BiLstmCtcModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split=split,
        max_items=max_items,
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

    total_edit_distance = 0
    total_target_tokens = 0
    examples: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            gloss_list = batch["gloss_list"]
            chinese_list = batch["chinese_list"]

            log_probs_btc = model(features)

            decoded_id_list = ctc_greedy_decode(
                log_probs_btc=log_probs_btc,
                input_lengths=input_lengths,
                blank_id=BLANK_ID,
            )

            for sample_id, decoded_ids, reference_gloss, chinese in zip(
                sample_ids,
                decoded_id_list,
                gloss_list,
                chinese_list,
            ):
                prediction_gloss = ids_to_gloss(decoded_ids, id_to_gloss)

                distance = edit_distance(
                    source=prediction_gloss,
                    target=reference_gloss,
                )

                total_edit_distance += distance
                total_target_tokens += len(reference_gloss)

                if len(examples) < PRINT_EXAMPLES:
                    examples.append(
                        {
                            "sampleId": sample_id,
                            "chinese": chinese,
                            "reference": reference_gloss,
                            "prediction": prediction_gloss,
                            "editDistance": distance,
                            "targetLength": len(reference_gloss),
                            "TER": distance / len(reference_gloss) if reference_gloss else 0.0,
                        }
                    )

    ter = total_edit_distance / total_target_tokens if total_target_tokens > 0 else 0.0

    return {
        "checkpoint": checkpoint_path.name,
        "checkpointEpoch": checkpoint.get("epoch"),
        "split": split,
        "maxItems": max_items,
        "sampleCount": len(dataset),
        "totalEditDistance": total_edit_distance,
        "totalTargetTokens": total_target_tokens,
        "TER": ter,
        "examples": examples,
    }


def print_result(result: Dict) -> None:
    """
    打印评估结果。

    Args:
        result: 评估结果。
    """
    print("\n" + "=" * 80)
    print("checkpoint:", result["checkpoint"])
    print("checkpoint epoch:", result["checkpointEpoch"])
    print("split:", result["split"])
    print("sampleCount:", result["sampleCount"])
    print("totalEditDistance:", result["totalEditDistance"])
    print("totalTargetTokens:", result["totalTargetTokens"])
    print("TER:", round(result["TER"], 4))

    print("\n示例:")
    for example in result["examples"]:
        print("-" * 80)
        print("sampleId:", example["sampleId"])
        print("中文:", example["chinese"])
        print("真实:", " / ".join(example["reference"]))
        print("预测:", " / ".join(example["prediction"]))
        print("TER:", round(example["TER"], 4))


# =========================================================
# 4. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== 检查 subset_500 预测效果开始 =====")

    device = get_device()
    id_to_gloss = build_id_to_gloss()

    print("device:", device)
    print("checkpoints:")
    for checkpoint_path in CHECKPOINTS:
        print(" ", checkpoint_path)

    all_results: List[Dict] = []

    for checkpoint_path in CHECKPOINTS:
        for split, max_items in [
            ("train", TRAIN_MAX_ITEMS),
            ("dev", DEV_MAX_ITEMS),
        ]:
            result = evaluate_checkpoint_on_split(
                checkpoint_path=checkpoint_path,
                split=split,
                max_items=max_items,
                device=device,
                id_to_gloss=id_to_gloss,
            )

            all_results.append(result)
            print_result(result)

    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)

    for result in all_results:
        print(
            f"{result['checkpoint']} | "
            f"epoch={result['checkpointEpoch']} | "
            f"{result['split']} | "
            f"TER={result['TER']:.4f} | "
            f"samples={result['sampleCount']}"
        )

    print("===== 检查 subset_500 预测效果结束 =====")


if __name__ == "__main__":
    main()