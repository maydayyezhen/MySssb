"""
V21 top-k subset Dataset

作用：
1. 读取 processed/subsets/v021_raw_delta_topk_subset/top_k_xxx/{split}_subset.jsonl。
2. 特征文件仍然从 processed/features/{split}/{sampleId}.npy 读取。
3. 不复制、不移动原始 .npy 特征。
4. glossIds 已经在 subset jsonl 里被重新映射为：
   0 = CTC blank
   1..K = top-k gloss
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ce_csl.dataset import CeCslGlossDataset


class TopKSubsetCeCslGlossDataset(CeCslGlossDataset):
    """
    CE-CSL top-k subset 数据集。

    复用 CeCslGlossDataset 的特征读取和 raw_delta 构造逻辑，
    只把样本清单从原始 ctc_ready 切换到 subset jsonl。
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        subset_dir: str | Path,
        max_items: Optional[int] = None,
        feature_dim: int = 166,
        blank_id: int = 0,
        feature_mode: str = "raw_delta",
    ) -> None:
        """
        初始化 top-k subset 数据集。

        Args:
            dataset_root: CE-CSL 数据集根目录。
            split: train / dev / test。
            subset_dir: top_k_xxx 子集目录。
            max_items: 最多读取样本数。
            feature_dim: 原始特征维度。
            blank_id: CTC blank id。
            feature_mode: 特征模式，当前使用 raw_delta。
        """
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            max_items=None,
            feature_dim=feature_dim,
            blank_id=blank_id,
            feature_mode=feature_mode,
        )

        self.subset_dir = Path(subset_dir)
        self.subset_ready_path = self.subset_dir / f"{split}_subset.jsonl"

        if not self.subset_ready_path.exists():
            raise FileNotFoundError(f"找不到 subset 文件：{self.subset_ready_path}")

        subset_samples = self._read_jsonl(self.subset_ready_path)

        if max_items is not None:
            subset_samples = subset_samples[:max_items]

        if not subset_samples:
            raise RuntimeError(f"{split} subset 数据集为空：{self.subset_ready_path}")

        self.samples = subset_samples
        self.original_ready_path = self.ready_path
        self.ready_path = self.subset_ready_path
        self.subset_sample_count = len(self.samples)

    @staticmethod
    def read_vocab(path: str | Path) -> Dict:
        """
        读取 subset vocab.json。
        """
        vocab_path = Path(path)

        if not vocab_path.exists():
            raise FileNotFoundError(f"找不到 vocab 文件：{vocab_path}")

        with vocab_path.open("r", encoding="utf-8") as file:
            return json.load(file)