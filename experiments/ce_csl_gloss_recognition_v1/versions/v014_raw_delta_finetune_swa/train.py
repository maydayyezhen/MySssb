"""
CE-CSL BiLSTM-CTC 全量训练脚本

作用：
1. 使用 train 全量 4973 条样本训练。
2. 使用 dev 全量 515 条样本验证。
3. 每轮输出 train_loss / dev_loss / dev_TER。
4. 保存 dev_TER 最好的模型。
5. 同时保存 dev_loss 最好的模型。
6. 保存最后一轮模型。
7. 写出训练历史日志和最佳预测示例。

说明：
- 本脚本用于正式全量训练。
- 前面已经通过 overfit_20 和 subset_500 验证训练链路可用。
- 当前使用 BiLSTM-CTC baseline。
"""

from __future__ import annotations

import json
import random
import sys
import time
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

def find_experiment_root(start_file: Path) -> Path:
    """
    从当前文件向上查找实验根目录。
    """
    for parent in start_file.parents:
        if (parent / "src" / "ce_csl").exists():
            return parent

    raise RuntimeError(f"无法从 {start_file} 定位实验根目录")


# experiments/ce_csl_gloss_recognition_v1
EXPERIMENT_ROOT = find_experiment_root(CURRENT_FILE)

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

SOURCE_CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints" / "full_bilstm_ctc_raw_delta"
SOURCE_CHECKPOINT_PATH = SOURCE_CHECKPOINT_DIR / "best_dev_ter.pt"

CHECKPOINT_DIR = PROCESSED_DIR / "checkpoints" / "full_bilstm_ctc_raw_delta_finetune_swa"
LOG_DIR = PROCESSED_DIR / "logs" / "full_bilstm_ctc_raw_delta_finetune_swa"

BLANK_ID = 0
FEATURE_DIM = 166
FEATURE_MODE = "raw_delta"
MODEL_INPUT_DIM = 332

# None 表示全量
TRAIN_MAX_ITEMS = None
DEV_MAX_ITEMS = None

BATCH_SIZE = 16
EPOCHS = 40

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
SWA_START_EPOCH = 3
EARLY_STOP_PATIENCE = 12

# Windows 下先用 0 最稳；如果后面想提速，可以尝试改成 2
NUM_WORKERS = 0

SEED = 42

# 每几轮验证一次；全量 dev 不大，先每轮验证
EVAL_EVERY = 1

# 最多保存多少条预测示例
MAX_PREDICTION_EXAMPLES = 20


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


def write_json(path: Path, data: Dict) -> None:
    """
    写出 json 文件。

    Args:
        path: 输出路径。
        data: 字典数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    dev_loss: float | None,
    dev_ter: float | None,
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
        dev_loss: 当前验证 loss，可为空。
        dev_ter: 当前验证 TER，可为空。
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


def clone_model_state_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Clone a model state_dict to CPU for SWA-style averaging.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def update_average_state(
    average_state: Dict[str, torch.Tensor],
    model: nn.Module,
    average_count: int,
) -> None:
    """
    Update an equal-weight running average of model weights.
    """
    current_state = model.state_dict()
    next_count = average_count + 1

    for key, average_value in average_state.items():
        current_value = current_state[key].detach().cpu()

        if torch.is_floating_point(average_value):
            average_value.add_((current_value - average_value) / next_count)
        else:
            average_value.copy_(current_value)


def load_average_state(
    model: nn.Module,
    average_state: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """
    Load a CPU averaged state_dict into a model on the target device.
    """
    model.load_state_dict(
        {
            key: value.to(device)
            for key, value in average_state.items()
        }
    )


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

    for batch_index, batch in enumerate(dataloader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

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

        # 梯度裁剪，防止 LSTM 梯度爆炸
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
        验证结果。
    """
    model.eval()

    total_loss = 0.0
    batch_count = 0

    total_edit_distance = 0
    total_target_tokens = 0

    prediction_examples: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            input_lengths = batch["input_lengths"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

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

            # targets 是拼接后的一维，需要按 target_lengths 切回每条样本
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

                if len(prediction_examples) < MAX_PREDICTION_EXAMPLES:
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

    print("===== CE-CSL BiLSTM-CTC 全量训练开始 =====")
    print("实验根目录:", EXPERIMENT_ROOT)
    print("src 目录:", SRC_DIR)
    print("数据集目录:", DATASET_ROOT)
    print("source checkpoint:", SOURCE_CHECKPOINT_PATH)
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
    print("LEARNING_RATE:", LEARNING_RATE)
    print("SWA_START_EPOCH:", SWA_START_EPOCH)
    print("EARLY_STOP_PATIENCE:", EARLY_STOP_PATIENCE)
    print("EVAL_EVERY:", EVAL_EVERY)

    pin_memory = device.type == "cuda"

    train_dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split="train",
        max_items=TRAIN_MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
        feature_mode=FEATURE_MODE,
    )

    dev_dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split="dev",
        max_items=DEV_MAX_ITEMS,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
        feature_mode=FEATURE_MODE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=ce_csl_collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=ce_csl_collate_fn,
    )

    print("train dataset size:", len(train_dataset))
    print("dev dataset size:", len(dev_dataset))
    print("train batch count:", len(train_loader))
    print("dev batch count:", len(dev_loader))

    model_config = {
        "input_dim": MODEL_INPUT_DIM,
        "projection_dim": 256,
        "hidden_size": 256,
        "num_layers": 2,
        "vocab_size": vocab_size,
        "input_dropout": 0.2,
        "lstm_dropout": 0.3,
        "output_dropout": 0.3,
    }

    model = BiLstmCtcModel(**model_config).to(device)

    if not SOURCE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"找不到 source checkpoint: {SOURCE_CHECKPOINT_PATH}")

    source_checkpoint = torch.load(
        SOURCE_CHECKPOINT_PATH,
        map_location=device,
    )
    source_model_config = source_checkpoint["config"]["model"]

    if source_model_config != model_config:
        raise RuntimeError(
            "source checkpoint model config does not match v012 model config: "
            f"source={source_model_config}, current={model_config}"
        )

    model.load_state_dict(source_checkpoint["model_state_dict"])
    swa_model = BiLstmCtcModel(**model_config).to(device)
    swa_model.load_state_dict(model.state_dict())
    swa_model.eval()

    source_metrics = {
        "checkpoint_path": str(SOURCE_CHECKPOINT_PATH),
        "epoch": int(source_checkpoint["epoch"]),
        "train_loss": float(source_checkpoint["train_loss"]),
        "dev_loss": float(source_checkpoint["dev_loss"]),
        "dev_ter": float(source_checkpoint["dev_ter"]),
    }

    print("loaded source epoch:", source_metrics["epoch"])
    print("loaded source dev_TER:", source_metrics["dev_ter"])

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
        "feature_mode": FEATURE_MODE,
        "model_input_dim": MODEL_INPUT_DIM,
        "seed": SEED,
        "eval_every": EVAL_EVERY,
        "num_workers": NUM_WORKERS,
        "swa_start_epoch": SWA_START_EPOCH,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "source_checkpoint_path": str(SOURCE_CHECKPOINT_PATH),
        "source_checkpoint_epoch": source_metrics["epoch"],
        "source_checkpoint_dev_ter": source_metrics["dev_ter"],
        "source_checkpoint_dev_loss": source_metrics["dev_loss"],
    }

    full_config = {
        "model": model_config,
        "train": train_config,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    write_json(LOG_DIR / "config.json", full_config)
    write_json(LOG_DIR / "source_checkpoint.json", source_metrics)

    best_dev_ter = source_metrics["dev_ter"]
    best_dev_loss = float("inf")
    best_dev_ter_epoch = 0
    best_dev_ter_source = "source"

    save_checkpoint(
        path=CHECKPOINT_DIR / "best_dev_ter.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        train_loss=source_metrics["train_loss"],
        dev_loss=source_metrics["dev_loss"],
        dev_ter=source_metrics["dev_ter"],
        config=full_config,
    )

    history_rows: List[Dict] = [
        {
            "epoch": 0,
            "trainLoss": source_metrics["train_loss"],
            "devLoss": source_metrics["dev_loss"],
            "devTER": source_metrics["dev_ter"],
            "devEditDistance": None,
            "devTargetTokens": None,
            "bestDevTER": best_dev_ter,
            "bestDevLoss": None,
            "bestDevTERSource": best_dev_ter_source,
            "swaDevLoss": None,
            "swaDevTER": None,
            "swaAverageCount": 0,
            "epochSeconds": 0.0,
            "elapsedMinutes": 0.0,
            "sourceCheckpoint": True,
        }
    ]
    write_jsonl(LOG_DIR / "history.jsonl", history_rows)

    start_time = time.time()
    swa_average_state: Dict[str, torch.Tensor] | None = None
    swa_average_count = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            ctc_loss_fn=ctc_loss_fn,
            optimizer=optimizer,
            device=device,
        )

        dev_loss = None
        dev_ter = None
        dev_total_edit_distance = None
        dev_total_target_tokens = None
        swa_dev_loss = None
        swa_dev_ter = None
        swa_total_edit_distance = None
        swa_total_target_tokens = None

        should_eval = epoch % EVAL_EVERY == 0 or epoch == EPOCHS

        if should_eval:
            dev_result = evaluate(
                model=model,
                dataloader=dev_loader,
                ctc_loss_fn=ctc_loss_fn,
                device=device,
            )

            dev_loss = float(dev_result["loss"])
            dev_ter = float(dev_result["ter"])
            dev_total_edit_distance = int(dev_result["totalEditDistance"])
            dev_total_target_tokens = int(dev_result["totalTargetTokens"])

            if dev_ter < best_dev_ter:
                best_dev_ter = dev_ter
                best_dev_ter_epoch = epoch
                best_dev_ter_source = "model"

                save_checkpoint(
                    path=CHECKPOINT_DIR / "best_dev_ter.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    dev_loss=dev_loss,
                    dev_ter=dev_ter,
                    config=full_config,
                )

                write_jsonl(
                    LOG_DIR / "best_dev_ter_prediction_examples.jsonl",
                    dev_result["predictionExamples"],
                )

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss

                save_checkpoint(
                    path=CHECKPOINT_DIR / "best_dev_loss.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    dev_loss=dev_loss,
                    dev_ter=dev_ter,
                    config=full_config,
                )

                write_jsonl(
                    LOG_DIR / "best_dev_loss_prediction_examples.jsonl",
                    dev_result["predictionExamples"],
                )

            if epoch >= SWA_START_EPOCH:
                if swa_average_state is None:
                    swa_average_state = clone_model_state_to_cpu(model)
                    swa_average_count = 1
                else:
                    update_average_state(
                        average_state=swa_average_state,
                        model=model,
                        average_count=swa_average_count,
                    )
                    swa_average_count += 1

                load_average_state(
                    model=swa_model,
                    average_state=swa_average_state,
                    device=device,
                )

                swa_result = evaluate(
                    model=swa_model,
                    dataloader=dev_loader,
                    ctc_loss_fn=ctc_loss_fn,
                    device=device,
                )

                swa_dev_loss = float(swa_result["loss"])
                swa_dev_ter = float(swa_result["ter"])
                swa_total_edit_distance = int(swa_result["totalEditDistance"])
                swa_total_target_tokens = int(swa_result["totalTargetTokens"])

                if swa_dev_ter < best_dev_ter:
                    best_dev_ter = swa_dev_ter
                    best_dev_ter_epoch = epoch
                    best_dev_ter_source = "swa"

                    save_checkpoint(
                        path=CHECKPOINT_DIR / "best_dev_ter.pt",
                        model=swa_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        train_loss=train_loss,
                        dev_loss=swa_dev_loss,
                        dev_ter=swa_dev_ter,
                        config=full_config,
                    )

                    write_jsonl(
                        LOG_DIR / "best_dev_ter_prediction_examples.jsonl",
                        swa_result["predictionExamples"],
                    )

                if swa_dev_loss < best_dev_loss:
                    best_dev_loss = swa_dev_loss

                    save_checkpoint(
                        path=CHECKPOINT_DIR / "best_dev_loss.pt",
                        model=swa_model,
                        optimizer=optimizer,
                        epoch=epoch,
                        train_loss=train_loss,
                        dev_loss=swa_dev_loss,
                        dev_ter=swa_dev_ter,
                        config=full_config,
                    )

                    write_jsonl(
                        LOG_DIR / "best_dev_loss_prediction_examples.jsonl",
                        swa_result["predictionExamples"],
                    )

        save_checkpoint(
            path=CHECKPOINT_DIR / "last_full.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            dev_loss=dev_loss,
            dev_ter=dev_ter,
            config=full_config,
        )

        epoch_seconds = time.time() - epoch_start_time
        elapsed_minutes = (time.time() - start_time) / 60.0

        row = {
            "epoch": epoch,
            "trainLoss": train_loss,
            "devLoss": dev_loss,
            "devTER": dev_ter,
            "devEditDistance": dev_total_edit_distance,
            "devTargetTokens": dev_total_target_tokens,
            "swaDevLoss": swa_dev_loss,
            "swaDevTER": swa_dev_ter,
            "swaEditDistance": swa_total_edit_distance,
            "swaTargetTokens": swa_total_target_tokens,
            "swaAverageCount": swa_average_count,
            "bestDevTER": best_dev_ter if best_dev_ter != float("inf") else None,
            "bestDevLoss": best_dev_loss if best_dev_loss != float("inf") else None,
            "bestDevTERSource": best_dev_ter_source,
            "epochSeconds": epoch_seconds,
            "elapsedMinutes": elapsed_minutes,
        }

        history_rows.append(row)
        write_jsonl(LOG_DIR / "history.jsonl", history_rows)

        if should_eval:
            print(
                f"epoch={epoch:03d}/{EPOCHS} "
                f"train_loss={train_loss:.4f} "
                f"dev_loss={dev_loss:.4f} "
                f"dev_TER={dev_ter:.4f} "
                f"swa_TER={swa_dev_ter if swa_dev_ter is not None else 'N/A'} "
                f"best_dev_TER={best_dev_ter:.4f} "
                f"best_source={best_dev_ter_source} "
                f"best_dev_loss={best_dev_loss:.4f} "
                f"epoch_sec={epoch_seconds:.1f} "
                f"elapsed_min={elapsed_minutes:.1f}"
            )
        else:
            print(
                f"epoch={epoch:03d}/{EPOCHS} "
                f"train_loss={train_loss:.4f} "
                f"dev_loss=SKIP "
                f"dev_TER=SKIP "
                f"best_dev_TER={best_dev_ter if best_dev_ter != float('inf') else 'N/A'} "
                f"epoch_sec={epoch_seconds:.1f} "
                f"elapsed_min={elapsed_minutes:.1f}"
            )

        if should_eval and epoch - best_dev_ter_epoch >= EARLY_STOP_PATIENCE:
            print(
                f"early_stop: no dev_TER improvement for {EARLY_STOP_PATIENCE} evals "
                f"(best_epoch={best_dev_ter_epoch}, best_source={best_dev_ter_source}, "
                f"best_dev_TER={best_dev_ter:.4f})"
            )
            break

    print("\n===== 全量训练结束 =====")
    print("best_dev_TER:", None if best_dev_ter == float("inf") else round(best_dev_ter, 4))
    print("best_dev_TER_source:", best_dev_ter_source)
    print("best_dev_loss:", None if best_dev_loss == float("inf") else round(best_dev_loss, 4))
    print("best TER checkpoint:", CHECKPOINT_DIR / "best_dev_ter.pt")
    print("best loss checkpoint:", CHECKPOINT_DIR / "best_dev_loss.pt")
    print("last checkpoint:", CHECKPOINT_DIR / "last_full.pt")
    print("history:", LOG_DIR / "history.jsonl")
    print("===== CE-CSL BiLSTM-CTC 全量训练结束 =====")


if __name__ == "__main__":
    main()
