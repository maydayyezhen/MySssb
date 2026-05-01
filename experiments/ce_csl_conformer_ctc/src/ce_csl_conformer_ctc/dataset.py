"""
CE-CSL gloss 识别 Dataset 与 DataLoader 工具

作用：
1. 读取 processed/ctc_ready/{split}_ctc_ready.jsonl。
2. 根据 sampleId 读取 processed/features/{split}/{sampleId}.npy。
3. 返回单条样本的 feature / target / length 信息。
4. 提供 collate_fn，把不同长度的样本 padding 成 batch。

注意：
- 本文件不负责训练。
- 本文件不重新提取特征。
- 本文件只负责把已经生成好的 .npy 特征整理成 PyTorch 能训练的格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CeCslGlossDataset(Dataset):
    """
    CE-CSL gloss 序列识别数据集。

    每条样本对应：
    - 一个 T × 166 的 .npy 特征文件
    - 一个 glossIds 标签序列
    """

    def __init__(
            self,
            dataset_root: str | Path,
            split: str,
            max_items: Optional[int] = None,
            feature_dim: int = 166,
            blank_id: int = 0,
            feature_mode: str = "raw",
    ) -> None:
        """
        初始化数据集。

        Args:
            dataset_root: CE-CSL 数据集根目录，例如 D:\\CE-CSL\\CE-CSL。
            split: 数据划分，train / dev / test。
            max_items: 最多读取多少条样本；None 表示读取全部。
            feature_dim: 单帧特征维度，当前为 166。
            blank_id: CTC blank id，真实标签中不应包含该 id。
        """
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.processed_dir = self.dataset_root / "processed"
        self.feature_dir = self.processed_dir / "features" / split
        self.ready_path = self.processed_dir / "ctc_ready" / f"{split}_ctc_ready.jsonl"

        self.split = split
        self.feature_dim = feature_dim
        self.blank_id = blank_id
        self.feature_mode = feature_mode

        if feature_mode not in {"raw", "raw_delta", "raw_delta_delta", "raw_delta_presence"}:
            raise ValueError(
                "feature_mode 只支持 raw / raw_delta / raw_delta_delta / raw_delta_presence，"
                f"实际为：{feature_mode}"
            )

        if split not in {"train", "dev", "test"}:
            raise ValueError(f"split 必须是 train/dev/test，实际为：{split}")

        if not self.ready_path.exists():
            raise FileNotFoundError(f"找不到 CTC ready 文件：{self.ready_path}")

        self.samples = self._read_jsonl(self.ready_path)

        if max_items is not None:
            self.samples = self.samples[:max_items]

        if not self.samples:
            raise RuntimeError(f"{split} 数据集为空：{self.ready_path}")

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict]:
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

    def __len__(self) -> int:
        """
        返回样本数量。
        """
        return len(self.samples)

    def _build_feature_by_mode(self, feature_np: np.ndarray) -> np.ndarray:
        """
        根据 feature_mode 构造最终输入特征。

        raw:
            T × 166

        raw_delta:
            原始特征 T × 166
            一阶差分 T × 166
            拼接后 T × 332

        raw_delta_presence:
            原始特征 T × 166
            一阶差分 T × 166
            left/right/arm presence mask T × 3
            拼接后 T × 335

        raw_delta_delta:
            原始特征 T × 166
            一阶差分 T × 166
            二阶差分 T × 166
            拼接后 T × 498

        Args:
            feature_np: 原始 T × 166 特征。

        Returns:
            根据 feature_mode 处理后的特征。
        """
        feature_np = feature_np.astype(np.float32)

        if self.feature_mode == "raw":
            return feature_np

        delta_np = np.zeros_like(feature_np, dtype=np.float32)

        if feature_np.shape[0] > 1:
            delta_np[1:] = feature_np[1:] - feature_np[:-1]

        if self.feature_mode == "raw_delta":
            return np.concatenate(
                [feature_np, delta_np],
                axis=1,
            )

        if self.feature_mode == "raw_delta_presence":
            left_present = ~np.all(np.isclose(feature_np[:, 0:78], 0.0), axis=1)
            right_present = ~np.all(np.isclose(feature_np[:, 78:156], 0.0), axis=1)
            arm_present = ~np.all(np.isclose(feature_np[:, 156:166], 0.0), axis=1)

            presence_np = np.stack(
                [
                    left_present.astype(np.float32),
                    right_present.astype(np.float32),
                    arm_present.astype(np.float32),
                ],
                axis=1,
            )

            return np.concatenate(
                [feature_np, delta_np, presence_np],
                axis=1,
            )

        if self.feature_mode == "raw_delta_delta":
            delta_delta_np = np.zeros_like(delta_np, dtype=np.float32)

            if delta_np.shape[0] > 1:
                delta_delta_np[1:] = delta_np[1:] - delta_np[:-1]

            return np.concatenate(
                [feature_np, delta_np, delta_delta_np],
                axis=1,
            )

        raise ValueError(f"未知 feature_mode：{self.feature_mode}")

    def __getitem__(self, index: int) -> Dict:
        """
        读取单条样本。

        Args:
            index: 样本下标。

        Returns:
            单条样本字典。
        """
        sample = self.samples[index]

        sample_id = sample["sampleId"]
        split = sample["split"]

        feature_path = self.feature_dir / f"{sample_id}.npy"

        if not feature_path.exists():
            raise FileNotFoundError(f"找不到特征文件：{feature_path}")

        feature_np = np.load(feature_path)

        if feature_np.ndim != 2:
            raise ValueError(f"{sample_id} 特征不是二维矩阵，shape={feature_np.shape}")

        if feature_np.shape[1] != self.feature_dim:
            raise ValueError(
                f"{sample_id} 特征维度错误，期望 {self.feature_dim}，实际 {feature_np.shape[1]}"
            )

        gloss_ids = sample["glossIds"]

        if not isinstance(gloss_ids, list) or len(gloss_ids) == 0:
            raise ValueError(f"{sample_id} glossIds 非法：{gloss_ids}")

        if self.blank_id in gloss_ids:
            raise ValueError(f"{sample_id} 的 glossIds 中包含 blank_id={self.blank_id}")

        final_feature_np = self._build_feature_by_mode(feature_np)
        feature = torch.from_numpy(final_feature_np).float()
        target = torch.tensor(gloss_ids, dtype=torch.long)

        input_length = int(feature.shape[0])
        target_length = int(target.shape[0])

        return {
            "sample_id": sample_id,
            "split": split,
            "feature": feature,
            "target": target,
            "input_length": input_length,
            "target_length": target_length,
            "gloss": sample.get("gloss", []),
            "chinese": sample.get("chinese", ""),
            "feature_path": str(feature_path),
        }


def ce_csl_collate_fn(batch: List[Dict]) -> Dict:
    """
    CE-CSL batch 整理函数。

    作用：
    1. 将不同长度的 T × 166 特征 padding 成 B × max_T × 166。
    2. 将不同长度的 target 拼接成一维 targets。
    3. 记录每条样本的 input_lengths 和 target_lengths。

    Args:
        batch: Dataset 返回的样本列表。

    Returns:
        一个 batch 字典。
    """
    features = [item["feature"] for item in batch]
    targets = [item["target"] for item in batch]

    input_lengths = torch.tensor(
        [item["input_length"] for item in batch],
        dtype=torch.long,
    )

    target_lengths = torch.tensor(
        [item["target_length"] for item in batch],
        dtype=torch.long,
    )

    # padding 后形状：B × max_T × 166
    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0,
    )

    # CTC Loss 常用格式：把所有 target 拼成一维
    concat_targets = torch.cat(targets, dim=0)

    sample_ids = [item["sample_id"] for item in batch]
    gloss_list = [item["gloss"] for item in batch]
    chinese_list = [item["chinese"] for item in batch]

    return {
        "features": padded_features,
        "targets": concat_targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "sample_ids": sample_ids,
        "gloss_list": gloss_list,
        "chinese_list": chinese_list,
    }
