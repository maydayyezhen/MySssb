"""
检查 overfit 20 模型的预测结果

作用：
1. 加载 overfit_20/best_overfit_20.pt。
2. 读取 train 前 20 条样本。
3. 对每条样本做 CTC greedy decode。
4. 打印真实 gloss、预测 gloss、错误率。
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


from ce_csl.ctc_decode import ctc_greedy_decode, ids_to_gloss, token_error_rate  # noqa: E402
from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

CHECKPOINT_PATH = PROCESSED_DIR / "checkpoints" / "overfit_20" / "best_overfit_20.pt"

BLANK_ID = 0
FEATURE_DIM = 166
MAX_ITEMS = 20
BATCH_SIZE = 4


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
    从 ctc_ready 文件中构建 id -> gloss 映射。

    Returns:
        id 到 gloss 字符串的映射。
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
    获取推理设备。

    Returns:
        CUDA 可用时使用 CUDA，否则使用 CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =========================================================
# 4. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== 检查 overfit 20 预测结果开始 =====")
    print("checkpoint:", CHECKPOINT_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{CHECKPOINT_PATH}")

    device = get_device()

    print("device:", device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    print("model_config:", json.dumps(model_config, ensure_ascii=False, indent=2))

    model = BiLstmCtcModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    id_to_gloss = build_id_to_gloss()

    dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split="train",
        max_items=MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=ce_csl_collate_fn,
    )

    total_distance = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            reference_gloss_list = batch["gloss_list"]
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
                reference_gloss_list,
                chinese_list,
            ):
                prediction_gloss = ids_to_gloss(decoded_ids, id_to_gloss)

                error_rate = token_error_rate(
                    prediction=prediction_gloss,
                    reference=reference_gloss,
                )

                distance = int(round(error_rate * len(reference_gloss)))
                total_distance += distance
                total_tokens += len(reference_gloss)

                print("\n" + "-" * 80)
                print("sample_id:", sample_id)
                print("中文:", chinese)
                print("真实:", " / ".join(reference_gloss))
                print("预测:", " / ".join(prediction_gloss))
                print("TER:", round(error_rate, 4))

    overall_ter = total_distance / total_tokens if total_tokens > 0 else 0.0

    print("\n===== 汇总 =====")
    print("total_distance:", total_distance)
    print("total_tokens:", total_tokens)
    print("overall_TER:", round(overall_ter, 4))
    print("===== 检查 overfit 20 预测结果结束 =====")


if __name__ == "__main__":
    main()