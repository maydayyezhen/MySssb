"""
CE-CSL Feature V1 少量样本特征检查脚本

作用：
1. 读取 features_sample 中生成的 .npy 文件。
2. 检查 shape 是否为 T × 166。
3. 检查是否存在 NaN / Inf。
4. 检查数值范围。
5. 检查左手、右手、手臂部分是否出现大量全 0。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# 少量样本特征目录
FEATURE_DIR = DATASET_ROOT / "processed" / "features_sample"


def block_zero_ratio(features: np.ndarray, start: int, end: int) -> float:
    """
    计算某个特征块中，每一帧是否全 0 的比例。

    Args:
        features: T × 166 特征矩阵。
        start: 起始维度，包含。
        end: 结束维度，不包含。

    Returns:
        全 0 帧比例。
    """
    block = features[:, start:end]

    zero_rows = np.all(np.isclose(block, 0.0), axis=1)

    return float(np.mean(zero_rows))


def inspect_one_feature(path: Path) -> None:
    """
    检查单个 .npy 特征文件。

    Args:
        path: .npy 文件路径。
    """
    features = np.load(path)

    print("=" * 80)
    print("文件:", path.relative_to(FEATURE_DIR))
    print("shape:", features.shape)
    print("dtype:", features.dtype)

    if features.ndim != 2:
        print("异常: 特征不是二维矩阵")
        return

    if features.shape[1] != 166:
        print("异常: 第二维不是 166")
        return

    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()

    print("存在 NaN:", has_nan)
    print("存在 Inf:", has_inf)

    print("最小值:", float(np.min(features)))
    print("最大值:", float(np.max(features)))
    print("均值:", float(np.mean(features)))
    print("标准差:", float(np.std(features)))

    # 按 FEATURE_SPEC.md 中的维度切块
    left_zero_ratio = block_zero_ratio(features, 0, 78)
    right_zero_ratio = block_zero_ratio(features, 78, 156)
    arm_zero_ratio = block_zero_ratio(features, 156, 166)

    print("左手 78 维全 0 帧比例:", round(left_zero_ratio, 3))
    print("右手 78 维全 0 帧比例:", round(right_zero_ratio, 3))
    print("手臂 10 维全 0 帧比例:", round(arm_zero_ratio, 3))

    # 检查是否存在整帧全 0
    whole_zero_ratio = float(np.mean(np.all(np.isclose(features, 0.0), axis=1)))
    print("整帧全 0 比例:", round(whole_zero_ratio, 3))


def main() -> None:
    """
    主入口。
    """
    print("===== CE-CSL Feature V1 少量样本特征检查开始 =====")
    print("特征目录:", FEATURE_DIR)

    npy_files = sorted(FEATURE_DIR.rglob("*.npy"))

    print("npy 文件数:", len(npy_files))

    if not npy_files:
        print("没有找到 .npy 文件")
        return

    for path in npy_files:
        inspect_one_feature(path)

    print("\n===== CE-CSL Feature V1 少量样本特征检查结束 =====")


if __name__ == "__main__":
    main()