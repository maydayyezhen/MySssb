"""
CE-CSL BiLSTM-CTC 20 条样本 overfit 测试脚本

作用：
1. 只读取 train 前 20 条样本。
2. 使用 BiLSTM-CTC 模型进行小样本训练。
3. 观察 CTC loss 是否能明显下降。
4. 验证 Dataset / DataLoader / Model / CTCLoss / optimizer / backward 是否完整打通。

注意：
- 本脚本不是正式训练。
- 本脚本故意关闭 dropout，目的是让模型尽量记住这 20 条样本。
- 如果 20 条样本都无法 overfit，则不应该直接进入全量训练。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

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

# checkpoint 输出目录
CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints" / "overfit_20"

# CTC blank 类别
BLANK_ID = 0

# 特征维度
FEATURE_DIM = 166

# 只训练前 20 条
MAX_ITEMS = 20

# batch size
BATCH_SIZE = 4

# 训练轮数
EPOCHS = 300

# 学习率
LEARNING_RATE = 1e-3

# 梯度裁剪阈值，防止梯度爆炸
GRAD_CLIP_NORM = 5.0

# 打印间隔
PRINT_EVERY_EPOCH = 1


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
    从 train/dev/test 的 ctc_ready 文件中推断 vocab_size。

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

    return max_id + 1


def get_device() -> torch.device:
    """
    获取训练设备。

    Returns:
        优先使用 CUDA，否则使用 CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict,
) -> None:
    """
    保存 checkpoint。

    Args:
        path: checkpoint 路径。
        model: 模型。
        optimizer: 优化器。
        epoch: 当前 epoch。
        loss: 当前 loss。
        config: 模型与训练配置。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


# =========================================================
# 4. 单轮训练
# =========================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    训练一个 epoch。

    Args:
        model: BiLSTM-CTC 模型。
        dataloader: 训练 DataLoader。
        ctc_loss_fn: CTC Loss。
        optimizer: 优化器。
        device: 训练设备。

    Returns:
        当前 epoch 的平均 loss。
    """
    model.train()

    total_loss = 0.0
    batch_count = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # B × T × vocab_size
        log_probs_btc = model(features)

        # CTCLoss 需要 T × B × vocab_size
        log_probs_tbc = log_probs_btc.transpose(0, 1)

        loss = ctc_loss_fn(
            log_probs=log_probs_tbc,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
        )

        loss.backward()

        # 梯度裁剪，避免 LSTM 梯度爆炸
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
        )

        optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("当前 epoch 没有任何 batch")

    return total_loss / batch_count


# =========================================================
# 5. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    print("===== CE-CSL BiLSTM-CTC overfit 20 开始 =====")
    print("实验根目录:", EXPERIMENT_ROOT)
    print("src 目录:", SRC_DIR)
    print("数据集目录:", DATASET_ROOT)
    print("checkpoint 目录:", CHECKPOINT_DIR)

    device = get_device()

    print("device:", device)

    vocab_size = infer_vocab_size_from_ready_files()

    print("vocab_size:", vocab_size)

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
        shuffle=True,
        num_workers=0,
        collate_fn=ce_csl_collate_fn,
    )

    print("dataset size:", len(dataset))
    print("batch size:", BATCH_SIZE)
    print("batch count:", len(dataloader))

    model_config = {
        "input_dim": FEATURE_DIM,
        "projection_dim": 256,
        "hidden_size": 256,
        "num_layers": 2,
        "vocab_size": vocab_size,
        "input_dropout": 0.0,
        "lstm_dropout": 0.0,
        "output_dropout": 0.0,
    }

    model = BiLstmCtcModel(**model_config).to(device)

    ctc_loss_fn = nn.CTCLoss(
        blank=BLANK_ID,
        reduction="mean",
        zero_infinity=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    train_config = {
        "max_items": MAX_ITEMS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "device": str(device),
        "blank_id": BLANK_ID,
    }

    full_config = {
        "model": model_config,
        "train": train_config,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            ctc_loss_fn=ctc_loss_fn,
            optimizer=optimizer,
            device=device,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            save_checkpoint(
                path=CHECKPOINT_DIR / "best_overfit_20.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                config=full_config,
            )

        save_checkpoint(
            path=CHECKPOINT_DIR / "last_overfit_20.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_loss,
            config=full_config,
        )

        if epoch % PRINT_EVERY_EPOCH == 0:
            print(
                f"epoch={epoch:03d}/{EPOCHS} "
                f"loss={avg_loss:.4f} "
                f"best_loss={best_loss:.4f}"
            )

    print("\n===== 训练结束 =====")
    print("best_loss:", round(best_loss, 4))
    print("best checkpoint:", CHECKPOINT_DIR / "best_overfit_20.pt")
    print("last checkpoint:", CHECKPOINT_DIR / "last_overfit_20.pt")
    print("===== CE-CSL BiLSTM-CTC overfit 20 结束 =====")


if __name__ == "__main__":
    main()