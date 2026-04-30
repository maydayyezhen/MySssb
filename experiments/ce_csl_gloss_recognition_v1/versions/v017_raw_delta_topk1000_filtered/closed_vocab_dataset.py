"""
v017 closed vocab filtered Dataset

作用：
1. 复用原始 CeCslGlossDataset 读取 raw_delta 特征。
2. 使用 v016 top_k_1000 controlled_vocab.json 中的 top_k_1000 映射。
3. 过滤掉包含低频 token 的样本。
4. 不使用 <unk> 作为训练标签。
5. 构造真正 closed vocabulary：
   0 = <blank>
   1..1000 = top_k_1000 gloss
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from ce_csl.dataset import CeCslGlossDataset


class ClosedVocabCeCslGlossDataset(Dataset):
    """
    CE-CSL closed vocab 数据集。

    与 v016 controlled vocab 的区别：
    - v016 会把低频词映射成 <unk>
    - v017 会直接过滤掉含低频词的样本
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
        初始化 closed vocab 数据集。

        Args:
            dataset_root: CE-CSL 数据集根目录。
            split: train / dev / test。
            controlled_vocab_path: v016 生成的 controlled_vocab.json。
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

        self.blank_id = blank_id

        if int(self.controlled_vocab["blankId"]) != blank_id:
            raise ValueError(
                f"controlled vocab blankId={self.controlled_vocab['blankId']} "
                f"与 blank_id={blank_id} 不一致"
            )

        # v016 controlled vocab:
        # 0 = <blank>
        # 1 = <unk>
        # 2..1001 = top_k_1000 gloss
        #
        # v017 closed vocab:
        # 0 = <blank>
        # 1..1000 = top_k_1000 gloss
        self.old_to_closed_new: Dict[str, int] = {
            str(old_id): int(controlled_new_id) - 1
            for old_id, controlled_new_id in self.controlled_vocab["oldToNew"].items()
        }

        self.closed_vocab_size = len(self.old_to_closed_new) + 1

        self.base_dataset = CeCslGlossDataset(
            dataset_root=dataset_root,
            split=split,
            max_items=max_items,
            feature_dim=feature_dim,
            blank_id=blank_id,
            feature_mode=feature_mode,
        )

        original_sample_count = len(self.base_dataset.samples)

        filtered_samples = []

        for sample in self.base_dataset.samples:
            gloss_ids = sample.get("glossIds", [])

            if all(str(old_id) in self.old_to_closed_new for old_id in gloss_ids):
                filtered_samples.append(sample)

        self.base_dataset.samples = filtered_samples

        self.original_sample_count = original_sample_count
        self.filtered_sample_count = len(filtered_samples)
        self.removed_sample_count = original_sample_count - self.filtered_sample_count

        if self.filtered_sample_count == 0:
            raise RuntimeError(
                f"{split} split 过滤后没有样本，请检查 controlled vocab 或过滤策略"
            )

    def __len__(self) -> int:
        """
        返回过滤后的样本数。
        """
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict:
        """
        读取单条样本，并把原始 gloss id 映射成 closed vocab id。
        """
        item = self.base_dataset[index]

        original_target_ids = item["target"].tolist()

        closed_target_ids = [
            self.old_to_closed_new[str(old_id)]
            for old_id in original_target_ids
        ]

        item["original_target"] = item["target"]
        item["target"] = torch.tensor(closed_target_ids, dtype=torch.long)
        item["target_length"] = int(len(closed_target_ids))
        item["closed_vocab_size"] = self.closed_vocab_size
        item["controlled_vocab_path"] = str(self.controlled_vocab_path)

        return item