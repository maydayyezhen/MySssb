"""
CE-CSL DataLoader 检查脚本

作用：
1. 加载 CeCslGlossDataset。
2. 使用 DataLoader 取一个 batch。
3. 检查 features / targets / input_lengths / target_lengths 的形状。
4. 验证 batch 是否符合后续 CTC 训练要求。

本脚本不训练模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# =========================================================
# 1. 项目路径配置
# =========================================================

# 当前文件：
# experiments/ce_csl_gloss_recognition_v1/scripts/04_training_checks/inspect_dataloader.py
CURRENT_FILE = Path(__file__).resolve()

# 实验根目录：
# experiments/ce_csl_gloss_recognition_v1
EXPERIMENT_ROOT = CURRENT_FILE.parents[2]

# src 目录：
# experiments/ce_csl_gloss_recognition_v1/src
SRC_DIR = EXPERIMENT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

BATCH_SIZE = 4
MAX_ITEMS = 8
FEATURE_DIM = 166


# =========================================================
# 3. 检查函数
# =========================================================

def inspect_one_split(split: str) -> None:
    """
    检查某个 split 的 DataLoader 输出。

    Args:
        split: train / dev / test。
    """
    print("\n" + "#" * 80)
    print(f"检查 split: {split}")
    print("#" * 80)

    dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split=split,
        max_items=MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=0,
    )

    print("dataset size:", len(dataset))

    first_item = dataset[0]

    print("\n===== 单条样本检查 =====")
    print("sample_id:", first_item["sample_id"])
    print("feature shape:", tuple(first_item["feature"].shape))
    print("target shape:", tuple(first_item["target"].shape))
    print("input_length:", first_item["input_length"])
    print("target_length:", first_item["target_length"])
    print("gloss:", " / ".join(first_item["gloss"]))
    print("chinese:", first_item["chinese"])

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=ce_csl_collate_fn,
    )

    batch = next(iter(dataloader))

    features = batch["features"]
    targets = batch["targets"]
    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]
    sample_ids = batch["sample_ids"]

    print("\n===== Batch 检查 =====")
    print("sample_ids:", sample_ids)
    print("features shape:", tuple(features.shape))
    print("targets shape:", tuple(targets.shape))
    print("input_lengths:", input_lengths.tolist())
    print("target_lengths:", target_lengths.tolist())
    print("targets 前 30 个:", targets[:30].tolist())

    # 基础断言
    assert features.ndim == 3, "features 应为 B × T × 166"
    assert features.shape[0] == BATCH_SIZE, "batch size 不符合预期"
    assert features.shape[2] == FEATURE_DIM, "feature_dim 不符合预期"

    assert targets.ndim == 1, "CTC targets 应拼接成一维"
    assert input_lengths.ndim == 1, "input_lengths 应为一维"
    assert target_lengths.ndim == 1, "target_lengths 应为一维"

    assert input_lengths.shape[0] == BATCH_SIZE, "input_lengths 数量应等于 batch size"
    assert target_lengths.shape[0] == BATCH_SIZE, "target_lengths 数量应等于 batch size"

    assert int(target_lengths.sum().item()) == int(targets.shape[0]), (
        "target_lengths 之和必须等于 targets 总长度"
    )

    assert torch.all(input_lengths <= features.shape[1]), (
        "input_lengths 不应超过 padding 后的 max_T"
    )

    assert torch.all(target_lengths > 0), "target_lengths 必须大于 0"
    assert torch.all(input_lengths >= target_lengths), (
        "input_lengths 必须大于等于 target_lengths"
    )

    assert not torch.any(targets == 0), "targets 中不应包含 CTC blank id 0"

    print("\n检查结果: OK")


def main() -> None:
    """
    主入口。
    """
    print("===== CE-CSL DataLoader 检查开始 =====")
    print("实验根目录:", EXPERIMENT_ROOT)
    print("src 目录:", SRC_DIR)
    print("数据集目录:", DATASET_ROOT)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("MAX_ITEMS:", MAX_ITEMS)

    for split in ["train", "dev", "test"]:
        inspect_one_split(split)

    print("\n===== CE-CSL DataLoader 检查结束 =====")


if __name__ == "__main__":
    main()