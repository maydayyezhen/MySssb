"""
v016 controlled vocab Dataset 包装器

作用：
1. 复用原始 CeCslGlossDataset 读取 raw_delta 特征。
2. 把原始 glossIds 映射到 controlled vocab id。
3. 不修改原始 ctc_ready 文件。
4. 不影响 v002 raw_delta baseline。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from ce_csl.dataset import CeCslGlossDataset


class ControlledVocabCeCslGlossDataset(Dataset):
    """
    CE-CSL controlled vocab 数据集包装器。

    这个类不重新读取特征逻辑，而是包装原始 CeCslGlossDataset：
    - feature 仍然由原始 Dataset 根据 feature_mode 构造
    - target 从原始 glossIds 映射为 controlled vocab ids
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        controlled_vocab_path: str | Path,
        max_items: int | None = None,
        feature_dim: int = 166,
        blank_id: int = 0,
        feature_mode: str = "raw_delta",
    ) -> None:
        """
        初始化 controlled vocab 数据集。

        Args:
            dataset_root: CE-CSL 数据集根目录。
            split: train / dev / test。
            controlled_vocab_path: controlled_vocab.json 路径。
            max_items: 最多读取样本数。
            feature_dim: 原始特征维度。
            blank_id: CTC blank id。
            feature_mode: 特征模式，当前建议 raw_delta。
        """
        super().__init__()

        self.controlled_vocab_path = Path(controlled_vocab_path)

        if not self.controlled_vocab_path.exists():
            raise FileNotFoundError(f"找不到 controlled vocab 文件：{self.controlled_vocab_path}")

        with self.controlled_vocab_path.open("r", encoding="utf-8") as file:
            self.controlled_vocab = json.load(file)

        self.old_to_new: Dict[str, int] = {
            str(old_id): int(new_id)
            for old_id, new_id in self.controlled_vocab["oldToNew"].items()
        }

        self.blank_id = int(self.controlled_vocab["blankId"])
        self.unk_id = int(self.controlled_vocab["unkId"])
        self.controlled_vocab_size = int(self.controlled_vocab["controlledVocabSize"])

        if self.blank_id != blank_id:
            raise ValueError(
                f"controlled vocab blank_id={self.blank_id} 与传入 blank_id={blank_id} 不一致"
            )

        self.base_dataset = CeCslGlossDataset(
            dataset_root=dataset_root,
            split=split,
            max_items=max_items,
            feature_dim=feature_dim,
            blank_id=blank_id,
            feature_mode=feature_mode,
        )

    def __len__(self) -> int:
        """
        返回样本数量。
        """
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict:
        """
        读取并映射单条样本。
        """
        item = self.base_dataset[index]

        original_target_ids = item["target"].tolist()

        controlled_target_ids = [
            self.old_to_new.get(str(old_id), self.unk_id)
            for old_id in original_target_ids
        ]

        item["original_target"] = item["target"]
        item["target"] = torch.tensor(controlled_target_ids, dtype=torch.long)
        item["target_length"] = int(len(controlled_target_ids))
        item["controlled_vocab_path"] = str(self.controlled_vocab_path)
        item["controlled_vocab_size"] = self.controlled_vocab_size

        return item