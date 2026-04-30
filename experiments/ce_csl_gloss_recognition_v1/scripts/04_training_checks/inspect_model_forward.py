"""
CE-CSL BiLSTM-CTC 模型前向检查脚本

作用：
1. 加载 DataLoader 的一个 batch。
2. 自动从 CTC ready 文件中估算 vocab_size。
3. 创建 BiLstmCtcModel。
4. 检查模型 forward 输出形状。
5. 检查输出能否接入 torch.nn.CTCLoss。

本脚本不训练模型，只做前向链路检查。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
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


from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

FEATURE_DIM = 166
BLANK_ID = 0
BATCH_SIZE = 4
MAX_ITEMS = 8


# =========================================================
# 3. 工具函数
# =========================================================

def read_jsonl(path: Path) -> list[dict]:
    """
    读取 jsonl 文件。

    Args:
        path: jsonl 文件路径。

    Returns:
        字典列表。
    """
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def infer_vocab_size_from_ready_files() -> int:
    """
    从 train/dev/test 的 ctc_ready 文件中估算 vocab_size。

    由于 <blank> = 0，且 glossIds 是整数 id，
    vocab_size 至少应为 max(glossIds) + 1。

    Returns:
        词表大小。
    """
    max_id = 0

    for split in ["train", "dev", "test"]:
        ready_path = CTC_READY_DIR / f"{split}_ctc_ready.jsonl"

        if not ready_path.exists():
            raise FileNotFoundError(f"找不到 CTC ready 文件：{ready_path}")

        rows = read_jsonl(ready_path)

        for row in rows:
            gloss_ids = row["glossIds"]

            if gloss_ids:
                max_id = max(max_id, max(gloss_ids))

    vocab_size = max_id + 1

    return vocab_size


# =========================================================
# 4. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== CE-CSL BiLSTM-CTC 模型前向检查开始 =====")
    print("实验根目录:", EXPERIMENT_ROOT)
    print("src 目录:", SRC_DIR)
    print("数据集目录:", DATASET_ROOT)

    vocab_size = infer_vocab_size_from_ready_files()

    print("推断 vocab_size:", vocab_size)

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

    batch = next(iter(dataloader))

    features = batch["features"]
    targets = batch["targets"]
    input_lengths = batch["input_lengths"]
    target_lengths = batch["target_lengths"]
    sample_ids = batch["sample_ids"]

    print("\n===== Batch 输入 =====")
    print("sample_ids:", sample_ids)
    print("features shape:", tuple(features.shape))
    print("targets shape:", tuple(targets.shape))
    print("input_lengths:", input_lengths.tolist())
    print("target_lengths:", target_lengths.tolist())

    model = BiLstmCtcModel(
        input_dim=FEATURE_DIM,
        projection_dim=256,
        hidden_size=256,
        num_layers=2,
        vocab_size=vocab_size,
        input_dropout=0.2,
        lstm_dropout=0.3,
        output_dropout=0.3,
    )

    model.train()

    log_probs_btc = model(features)

    print("\n===== 模型输出 =====")
    print("log_probs_btc shape:", tuple(log_probs_btc.shape))

    expected_b = features.shape[0]
    expected_t = features.shape[1]

    assert log_probs_btc.ndim == 3, "模型输出应为 B × T × vocab_size"
    assert log_probs_btc.shape[0] == expected_b, "模型输出 B 不正确"
    assert log_probs_btc.shape[1] == expected_t, "模型输出 T 不正确"
    assert log_probs_btc.shape[2] == vocab_size, "模型输出 vocab_size 不正确"

    # PyTorch CTCLoss 需要 T × B × vocab_size
    log_probs_tbc = log_probs_btc.transpose(0, 1)

    print("log_probs_tbc shape:", tuple(log_probs_tbc.shape))

    ctc_loss_fn = nn.CTCLoss(
        blank=BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )

    loss = ctc_loss_fn(
        log_probs=log_probs_tbc,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )

    print("\n===== CTC Loss 检查 =====")
    print("loss:", float(loss.item()))

    assert torch.isfinite(loss), "CTC loss 不是有限值"

    print("\n检查结果: OK")
    print("===== CE-CSL BiLSTM-CTC 模型前向检查结束 =====")


if __name__ == "__main__":
    main()