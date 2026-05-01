"""
检查 Conformer-CTC checkpoint 在 train/dev/test 指定 split 上的预测效果。

用法：
python experiments\ce_csl_conformer_ctc\inspect_checkpoint_split.py --split train --max-items 20
python experiments\ce_csl_conformer_ctc\inspect_checkpoint_split.py --split dev --max-items 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader


CURRENT_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = CURRENT_FILE.parent
SRC_DIR = EXPERIMENT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ce_csl_conformer_ctc.ctc_decode import ctc_greedy_decode, edit_distance  # noqa: E402
from ce_csl_conformer_ctc.dataset import CeCslGlossDataset, ce_csl_collate_fn  # noqa: E402
from ce_csl_conformer_ctc.conformer_model import ConformerCtcModel  # noqa: E402


DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

DEFAULT_CHECKPOINT_PATH = (
    PROCESSED_DIR
    / "checkpoints"
    / "experiment_conformer_ctc_raw_delta"
    / "best_dev_ter.pt"
)

BLANK_ID = 0
FEATURE_DIM = 166
FEATURE_MODE = "raw_delta"


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


def build_id_to_gloss() -> Dict[int, str]:
    """
    从 ctc_ready 文件中构造 id -> gloss 映射。
    """
    id_to_gloss: Dict[int, str] = {}

    for split in ["train", "dev", "test"]:
        ready_path = CTC_READY_DIR / f"{split}_ctc_ready.jsonl"

        if not ready_path.exists():
            continue

        rows = read_jsonl(ready_path)

        for row in rows:
            gloss_ids = row.get("glossIds", [])
            gloss_list = row.get("gloss", [])

            for token_id, gloss in zip(gloss_ids, gloss_list):
                id_to_gloss[int(token_id)] = str(gloss)

    return id_to_gloss


def ids_to_gloss(token_ids: List[int], id_to_gloss: Dict[int, str]) -> List[str]:
    """
    将 id 序列转成 gloss 序列。
    """
    return [id_to_gloss.get(int(token_id), f"<id:{token_id}>") for token_id in token_ids]


def main() -> None:
    """
    主入口。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="要检查的数据划分。",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="最多检查多少条样本。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="推理 batch size。",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="checkpoint 路径。",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("===== checkpoint split inspect =====")
    print("split:", args.split)
    print("max_items:", args.max_items)
    print("checkpoint:", checkpoint_path)
    print("device:", device)

    id_to_gloss = build_id_to_gloss()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model_config = checkpoint["config"]["model"]

    model = ConformerCtcModel(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CeCslGlossDataset(
        dataset_root=DATASET_ROOT,
        split=args.split,
        max_items=args.max_items,
        feature_dim=FEATURE_DIM,
        blank_id=BLANK_ID,
        feature_mode=FEATURE_MODE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ce_csl_collate_fn,
    )

    total_edit_distance = 0
    total_target_tokens = 0
    prediction_counter: Counter[str] = Counter()

    printed = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            sample_ids = batch["sample_ids"]
            gloss_list = batch["gloss_list"]
            chinese_list = batch["chinese_list"]

            log_probs_btc = model(
                features=features,
                input_lengths=input_lengths,
            )

            decoded_id_list = ctc_greedy_decode(
                log_probs_btc=log_probs_btc,
                input_lengths=input_lengths,
                blank_id=BLANK_ID,
            )

            targets_cpu = targets.detach().cpu().tolist()
            target_offset = 0

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

                prediction_gloss = ids_to_gloss(decoded_ids, id_to_gloss)
                prediction_counter.update(prediction_gloss)

                if printed < args.max_items:
                    print()
                    print(f"[{printed + 1}] {sample_ids[index]}")
                    print("中文:", chinese_list[index])
                    print("参考:", " / ".join(gloss_list[index]))
                    print("预测:", " / ".join(prediction_gloss))
                    print("编辑距离:", distance, "/", len(reference_ids))

                    printed += 1

    ter = total_edit_distance / total_target_tokens if total_target_tokens > 0 else 0.0

    print()
    print("===== summary =====")
    print("split:", args.split)
    print("items:", len(dataset))
    print("total_edit_distance:", total_edit_distance)
    print("total_target_tokens:", total_target_tokens)
    print("TER:", round(ter, 4))

    print()
    print("===== prediction gloss freq =====")
    for gloss, count in prediction_counter.most_common(30):
        print(f"{gloss}\t{count}")


if __name__ == "__main__":
    main()
