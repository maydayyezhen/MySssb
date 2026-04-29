"""
CE-CSL Feature V1 全量特征检查脚本

作用：
1. 检查 processed/features 下 train/dev/test 的 .npy 数量。
2. 检查每个特征文件是否为 T × 166。
3. 检查 NaN / Inf。
4. 统计 T 长度分布。
5. 统计左手、右手、手臂部分全 0 帧比例。
6. 发现异常样本并输出报告。

本脚本只读取 .npy，不重新提取特征，不训练模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


# =========================
# 1. 路径配置
# =========================

DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")
PROCESSED_DIR = DATASET_ROOT / "processed"
FEATURE_DIR = PROCESSED_DIR / "features"

REPORT_DIR = PROCESSED_DIR / "feature_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COUNTS = {
    "train": 4973,
    "dev": 515,
    "test": 500,
}

FEATURE_DIM = 166


# =========================
# 2. 工具函数
# =========================

def read_jsonl(path: Path) -> List[Dict]:
    """
    读取 jsonl 文件。
    """
    samples: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def block_zero_ratio(features: np.ndarray, start: int, end: int) -> float:
    """
    计算某个特征块中，每一帧该块是否全 0 的比例。
    """
    block = features[:, start:end]
    zero_rows = np.all(np.isclose(block, 0.0), axis=1)
    return float(np.mean(zero_rows))


def safe_percentile(values: List[float], q: float) -> float:
    """
    安全计算分位数。
    """
    if not values:
        return 0.0

    return float(np.percentile(np.array(values, dtype=np.float32), q))


def inspect_one_feature(path: Path, sample: Dict) -> Dict:
    """
    检查单个 .npy 特征文件。
    """
    result = {
        "sampleId": sample["sampleId"],
        "split": sample["split"],
        "videoPath": sample["videoPath"],
        "featurePath": str(path.relative_to(FEATURE_DIR)).replace("\\", "/"),
        "glossLength": sample.get("glossLength"),
        "shape": None,
        "validShape": False,
        "hasNaN": False,
        "hasInf": False,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "maxAbs": None,
        "leftZeroRatio": None,
        "rightZeroRatio": None,
        "armZeroRatio": None,
        "wholeZeroRatio": None,
        "status": "ok",
        "message": "",
    }

    if not path.exists():
        result["status"] = "missing"
        result["message"] = "feature file not found"
        return result

    try:
        features = np.load(path)
    except Exception as exc:
        result["status"] = "load_error"
        result["message"] = f"{type(exc).__name__}: {exc}"
        return result

    result["shape"] = list(features.shape)

    if features.ndim != 2:
        result["status"] = "bad_shape"
        result["message"] = f"feature ndim is {features.ndim}, expected 2"
        return result

    if features.shape[1] != FEATURE_DIM:
        result["status"] = "bad_dim"
        result["message"] = f"feature dim is {features.shape[1]}, expected {FEATURE_DIM}"
        return result

    result["validShape"] = True

    result["hasNaN"] = bool(np.isnan(features).any())
    result["hasInf"] = bool(np.isinf(features).any())

    if result["hasNaN"] or result["hasInf"]:
        result["status"] = "nan_or_inf"
        result["message"] = "feature contains NaN or Inf"
        return result

    result["min"] = float(np.min(features))
    result["max"] = float(np.max(features))
    result["mean"] = float(np.mean(features))
    result["std"] = float(np.std(features))
    result["maxAbs"] = float(np.max(np.abs(features)))

    # 按 FEATURE_SPEC.md 切块
    result["leftZeroRatio"] = block_zero_ratio(features, 0, 78)
    result["rightZeroRatio"] = block_zero_ratio(features, 78, 156)
    result["armZeroRatio"] = block_zero_ratio(features, 156, 166)
    result["wholeZeroRatio"] = float(np.mean(np.all(np.isclose(features, 0.0), axis=1)))

    # 轻度异常标记，不直接判死刑
    warnings = []

    if result["wholeZeroRatio"] > 0:
        warnings.append("whole_zero_frame_exists")

    if result["leftZeroRatio"] > 0.8:
        warnings.append("left_hand_mostly_missing")

    if result["rightZeroRatio"] > 0.8:
        warnings.append("right_hand_mostly_missing")

    if result["armZeroRatio"] > 0.2:
        warnings.append("arm_pose_missing_too_much")

    if result["maxAbs"] > 20:
        warnings.append("max_abs_too_large")

    if warnings:
        result["status"] = "warning"
        result["message"] = ",".join(warnings)

    return result


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 报告。
    """
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_distribution(name: str, values: List[float]) -> None:
    """
    打印数值分布。
    """
    if not values:
        print(f"{name}: 无数据")
        return

    arr = np.array(values, dtype=np.float32)

    print(f"{name}:")
    print("  min:", round(float(np.min(arr)), 4))
    print("  p25:", round(float(np.percentile(arr, 25)), 4))
    print("  p50:", round(float(np.percentile(arr, 50)), 4))
    print("  p75:", round(float(np.percentile(arr, 75)), 4))
    print("  p95:", round(float(np.percentile(arr, 95)), 4))
    print("  max:", round(float(np.max(arr)), 4))
    print("  mean:", round(float(np.mean(arr)), 4))


# =========================
# 3. 主流程
# =========================

def main() -> None:
    """
    主入口。
    """
    print("===== CE-CSL Feature V1 全量特征检查开始 =====")
    print("特征目录:", FEATURE_DIR)
    print("报告目录:", REPORT_DIR)

    all_results: List[Dict] = []

    split_summary = {}

    for split in ["train", "dev", "test"]:
        print("\n" + "#" * 80)
        print(f"检查 split: {split}")
        print("#" * 80)

        manifest_path = PROCESSED_DIR / f"{split}.jsonl"

        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 manifest 文件: {manifest_path}")

        samples = read_jsonl(manifest_path)

        feature_split_dir = FEATURE_DIR / split
        npy_files = sorted(feature_split_dir.glob("*.npy")) if feature_split_dir.exists() else []

        print("manifest 样本数:", len(samples))
        print("npy 文件数:", len(npy_files))
        print("预期数量:", EXPECTED_COUNTS[split])

        missing_count = 0
        ok_count = 0
        warning_count = 0
        error_count = 0

        split_results: List[Dict] = []

        for index, sample in enumerate(samples, start=1):
            feature_path = feature_split_dir / f"{sample['sampleId']}.npy"

            result = inspect_one_feature(feature_path, sample)
            split_results.append(result)
            all_results.append(result)

            status = result["status"]

            if status == "ok":
                ok_count += 1
            elif status == "warning":
                warning_count += 1
            elif status == "missing":
                missing_count += 1
            else:
                error_count += 1

            if index % 500 == 0:
                print(f"已检查 {index}/{len(samples)}")

        split_summary[split] = {
            "manifestCount": len(samples),
            "npyCount": len(npy_files),
            "expectedCount": EXPECTED_COUNTS[split],
            "okCount": ok_count,
            "warningCount": warning_count,
            "missingCount": missing_count,
            "errorCount": error_count,
        }

        print(f"\n{split} 检查完成:")
        print("  OK:", ok_count)
        print("  WARNING:", warning_count)
        print("  MISSING:", missing_count)
        print("  ERROR:", error_count)

        write_jsonl(REPORT_DIR / f"{split}_feature_report.jsonl", split_results)

    # 全局统计
    print("\n" + "=" * 80)
    print("全局统计")
    print("=" * 80)

    valid_results = [
        row for row in all_results
        if row["validShape"] and not row["hasNaN"] and not row["hasInf"]
    ]

    t_lengths = [row["shape"][0] for row in valid_results]
    gloss_lengths = [row["glossLength"] for row in valid_results if row["glossLength"] is not None]

    min_values = [row["min"] for row in valid_results]
    max_values = [row["max"] for row in valid_results]
    mean_values = [row["mean"] for row in valid_results]
    std_values = [row["std"] for row in valid_results]
    max_abs_values = [row["maxAbs"] for row in valid_results]

    left_zero_ratios = [row["leftZeroRatio"] for row in valid_results]
    right_zero_ratios = [row["rightZeroRatio"] for row in valid_results]
    arm_zero_ratios = [row["armZeroRatio"] for row in valid_results]
    whole_zero_ratios = [row["wholeZeroRatio"] for row in valid_results]

    print("总样本数:", len(all_results))
    print("有效 shape 且无 NaN/Inf:", len(valid_results))

    print("\n===== split summary =====")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))

    print("\n===== T 长度分布 =====")
    print_distribution("T", t_lengths)

    print("\n===== gloss 长度分布 =====")
    print_distribution("glossLength", gloss_lengths)

    print("\n===== 数值范围分布 =====")
    print_distribution("min", min_values)
    print_distribution("max", max_values)
    print_distribution("mean", mean_values)
    print_distribution("std", std_values)
    print_distribution("maxAbs", max_abs_values)

    print("\n===== 全 0 比例分布 =====")
    print_distribution("leftZeroRatio", left_zero_ratios)
    print_distribution("rightZeroRatio", right_zero_ratios)
    print_distribution("armZeroRatio", arm_zero_ratios)
    print_distribution("wholeZeroRatio", whole_zero_ratios)

    abnormal_results = [
        row for row in all_results
        if row["status"] != "ok"
    ]

    print("\n===== 异常 / 警告样本 =====")
    print("数量:", len(abnormal_results))

    for row in abnormal_results[:30]:
        print(
            row["sampleId"],
            row["split"],
            row["status"],
            row["shape"],
            row["message"],
        )

    write_jsonl(REPORT_DIR / "all_feature_report.jsonl", all_results)
    write_jsonl(REPORT_DIR / "abnormal_feature_report.jsonl", abnormal_results)

    summary_path = REPORT_DIR / "feature_summary.json"

    summary = {
        "splitSummary": split_summary,
        "totalCount": len(all_results),
        "validCount": len(valid_results),
        "abnormalCount": len(abnormal_results),
        "tLength": {
            "min": min(t_lengths) if t_lengths else None,
            "max": max(t_lengths) if t_lengths else None,
            "mean": float(np.mean(t_lengths)) if t_lengths else None,
            "p50": safe_percentile(t_lengths, 50),
            "p95": safe_percentile(t_lengths, 95),
        },
        "glossLength": {
            "min": min(gloss_lengths) if gloss_lengths else None,
            "max": max(gloss_lengths) if gloss_lengths else None,
            "mean": float(np.mean(gloss_lengths)) if gloss_lengths else None,
            "p50": safe_percentile(gloss_lengths, 50),
            "p95": safe_percentile(gloss_lengths, 95),
        },
        "maxAbs": {
            "max": max(max_abs_values) if max_abs_values else None,
            "mean": float(np.mean(max_abs_values)) if max_abs_values else None,
            "p95": safe_percentile(max_abs_values, 95),
        },
        "zeroRatio": {
            "leftMean": float(np.mean(left_zero_ratios)) if left_zero_ratios else None,
            "rightMean": float(np.mean(right_zero_ratios)) if right_zero_ratios else None,
            "armMean": float(np.mean(arm_zero_ratios)) if arm_zero_ratios else None,
            "wholeMean": float(np.mean(whole_zero_ratios)) if whole_zero_ratios else None,
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n报告已写出:")
    print("  all:", REPORT_DIR / "all_feature_report.jsonl")
    print("  abnormal:", REPORT_DIR / "abnormal_feature_report.jsonl")
    print("  summary:", summary_path)

    print("\n===== CE-CSL Feature V1 全量特征检查结束 =====")


if __name__ == "__main__":
    main()