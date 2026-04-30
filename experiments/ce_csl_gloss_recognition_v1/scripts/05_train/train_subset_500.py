"""
CE-CSL BiLSTM-CTC 500 条样本小训练脚本

作用：
1. 使用 train 前 500 条样本进行小规模训练。
2. 使用 dev 前 200 条样本进行验证。
3. 每轮输出 train_loss / dev_loss / dev_TER。
4. 保存 dev_TER 最好的 checkpoint。
5. 用于验证模型是否具备基本泛化能力。

注意：
- 本脚本不是最终全量训练。
- overfit_20 已经证明训练链路能跑通。
- 这一步开始恢复 dropout 和 weight_decay。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
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


from ce_csl.ctc_decode import ctc_greedy_decode, edit_distance  # noqa: E402
from ce_csl.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl.model import BiLstmCtcModel  # noqa: E402


# =========================================================
# 2. 基础配置
# =========================================================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints" / "subset_500"
LOG_DIR = PROCESSED_DIR / "logs" / "subset_500"

BLANK_ID = 0
FEATURE_DIM = 166

TRAIN_MAX_ITEMS = 500
DEV_MAX_ITEMS = 200

BATCH_SIZE = 16
EPOCHS = 120

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
GRAD_CLIP_NORM = 5.0

NUM_WORKERS = 0
SEED = 42


# =========================================================
# 3. 工具函数
# =========================================================

def set_seed(seed: int) -> None:
    """
    固定随机种子，方便复现实验。

    Args:
        seed: 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def infer_vocab_size_from_ready_files() -> int:
    """
    从 train/dev/test 的 ctc_ready 文件中推断 vocab_size。

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
        CUDA 可用时返回 cuda，否则返回 cpu。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    dev_ter: float,
    config: Dict,
) -> None:
    """
    保存 checkpoint。

    Args:
        path: checkpoint 文件路径。
        model: 模型。
        optimizer: 优化器。
        epoch: 当前 epoch。
        train_loss: 当前训练 loss。
        dev_loss: 当前验证 loss。
        dev_ter: 当前验证 TER。
        config: 实验配置。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_ter": dev_ter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 文件。

    Args:
        path: 输出路径。
        rows: 字典列表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# 4. 训练与验证
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
        model: 模型。
        dataloader: 训练 DataLoader。
        ctc_loss_fn: CTC Loss。
        optimizer: 优化器。
        device: 训练设备。

    Returns:
        平均训练 loss。
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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
        )

        optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError("训练集没有 batch")

    return total_loss / batch_count


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    device: torch.device,
) -> Dict:
    """
    在验证集上评估。

    Args:
        model: 模型。
        dataloader: 验证 DataLoader。
        ctc_loss_fn: CTC Loss。
        device: 评估设备。

    Returns:
        包含 dev_loss、dev_TER 和预测样例的字典。
    """
    model.eval()

    total_loss = 0.0
    batch_count = 0

    total_edit_distance = 0
    total_target_tokens = 0

    prediction_examples: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            gloss_list = batch["gloss_list"]
            chinese_list = batch["chinese_list"]

            # B × T × vocab_size
            log_probs_btc = model(features)

            # T × B × vocab_size
            log_probs_tbc = log_probs_btc.transpose(0, 1)

            loss = ctc_loss_fn(
                log_probs=log_probs_tbc,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )

            total_loss += float(loss.item())
            batch_count += 1

            decoded_id_list = ctc_greedy_decode(
                log_probs_btc=log_probs_btc,
                input_lengths=input_lengths,
                blank_id=BLANK_ID,
            )

            # targets 是拼接后的一维，所以这里按 target_lengths 切回每条样本
            target_offset = 0
            targets_cpu = targets.detach().cpu().tolist()

            for index, decoded_ids in enumerate(decoded_id_list):
                target_length = int(target_lengths[index].item())

                reference_ids = targets_cpu[target_offset: target_offset + target_length]
                target_offset += target_length

                distance = edit_distance(
                    source=[str(x) for x in decoded_ids],
                    target=[str(x) for x in reference_ids],
                )

                total_edit_distance += distance
                total_target_tokens += len(reference_ids)

                if len(prediction_examples) < 10:
                    prediction_examples.append(
                        {
                            "sampleId": sample_ids[index],
                            "chinese": chinese_list[index],
                            "referenceGloss": gloss_list[index],
                            "referenceIds": reference_ids,
                            "predictionIds": decoded_ids,
                            "editDistance": distance,
                            "targetLength": len(reference_ids),
                        }
                    )

    if batch_count == 0:
        raise RuntimeError("验证集没有 batch")

    avg_loss = total_loss / batch_count
    ter = total_edit_distance / total_target_tokens if total_target_tokens > 0 else 0.0

    return {
        "loss": avg_loss,
        "ter": ter,
        "totalEditDistance": total_edit_distance,
        "totalTargetTokens": total_target_tokens,
        "predictionExamples": prediction_examples,
    }


# =========================================================
# 5. 主流程
# =========================================================

def main() -> None:
    """
    主入口。
    """
    set_seed(SEED)

    print("===== CE-CSL BiLSTM-CTC subset 500 训练开始 =====")
    print("实验根目录:", EXPERIMENT_ROOT)
    print("src 目录:", SRC_DIR)
    print("数据集目录:", DATASET_ROOT)
    print("checkpoint 目录:", CHECKPOINT_DIR)
    print("log 目录:", LOG_DIR)

    device = get_device()
    vocab_size = infer_vocab_size_from_ready_files()

    print("device:", device)
    print("vocab_size:", vocab_size)
    print("TRAIN_MAX_ITEMS:", TRAIN_MAX_ITEMS)
    print("DEV_MAX_ITEMS:", DEV_MAX_ITEMS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("EPOCHS:", EPOCHS)

    train_dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split="train",
        max_items=TRAIN_MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
    )

    dev_dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split="dev",
        max_items=DEV_MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=ce_csl_collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ce_csl_collate_fn,
    )

    print("train dataset size:", len(train_dataset))
    print("dev dataset size:", len(dev_dataset))
    print("train batch count:", len(train_loader))
    print("dev batch count:", len(dev_loader))

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
        weight_decay=WEIGHT_DECAY,
    )

    train_config = {
        "train_max_items": TRAIN_MAX_ITEMS,
        "dev_max_items": DEV_MAX_ITEMS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "device": str(device),
        "blank_id": BLANK_ID,
        "seed": SEED,
    }

    full_config = {
        "model": model_config,
        "train": train_config,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_dev_ter = float("inf")
    best_dev_loss = float("inf")
    history_rows: List[Dict] = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            ctc_loss_fn=ctc_loss_fn,
            optimizer=optimizer,
            device=device,
        )

        dev_result = evaluate(
            model=model,
            dataloader=dev_loader,
            ctc_loss_fn=ctc_loss_fn,
            device=device,
        )

        dev_loss = float(dev_result["loss"])
        dev_ter = float(dev_result["ter"])

        row = {
            "epoch": epoch,
            "trainLoss": train_loss,
            "devLoss": dev_loss,
            "devTER": dev_ter,
            "devEditDistance": dev_result["totalEditDistance"],
            "devTargetTokens": dev_result["totalTargetTokens"],
        }

        history_rows.append(row)

        is_best = dev_ter < best_dev_ter

        if is_best:
            best_dev_ter = dev_ter
            best_dev_loss = dev_loss

            save_checkpoint(
                path=CHECKPOINT_DIR / "best_subset_500.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                dev_loss=dev_loss,
                dev_ter=dev_ter,
                config=full_config,
            )

            write_jsonl(
                LOG_DIR / "best_prediction_examples.jsonl",
                dev_result["predictionExamples"],
            )

        save_checkpoint(
            path=CHECKPOINT_DIR / "last_subset_500.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            dev_loss=dev_loss,
            dev_ter=dev_ter,
            config=full_config,
        )

        write_jsonl(LOG_DIR / "history.jsonl", history_rows)

        print(
            f"epoch={epoch:03d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dev_loss:.4f} "
            f"dev_TER={dev_ter:.4f} "
            f"best_dev_TER={best_dev_ter:.4f}"
        )

    print("\n===== 训练结束 =====")
    print("best_dev_TER:", round(best_dev_ter, 4))
    print("best_dev_loss:", round(best_dev_loss, 4))
    print("best checkpoint:", CHECKPOINT_DIR / "best_subset_500.pt")
    print("last checkpoint:", CHECKPOINT_DIR / "last_subset_500.pt")
    print("history:", LOG_DIR / "history.jsonl")
    print("best examples:", LOG_DIR / "best_prediction_examples.jsonl")
    print("===== CE-CSL BiLSTM-CTC subset 500 训练结束 =====")


if __name__ == "__main__":
    main()