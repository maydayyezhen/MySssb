"""
CE-CSL CTC 训练前检查脚本

作用：
1. 检查 train/dev/test 的 manifest 是否能对应到 features 目录中的 .npy 特征文件。
2. 检查每个特征文件是否为 T × 166。
3. 检查 glossIds 是否存在、是否为空、是否包含 CTC blank id。
4. 检查输入长度 T 是否满足 CTC 对齐所需的最小长度。
5. 输出可训练清单和异常清单，供后续 Dataset / DataLoader 使用。

注意：
- 本脚本不训练模型。
- 本脚本只做训练前体检。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =========================================================
# 1. 路径与基础配置
# =========================================================

# CE-CSL 数据集根目录
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# processed 目录
PROCESSED_DIR = DATASET_ROOT / "processed"

# 特征目录
FEATURE_DIR = PROCESSED_DIR / "features"

# CTC 检查报告输出目录
CTC_REPORT_DIR = PROCESSED_DIR / "ctc_reports"

# CTC 可训练清单输出目录
CTC_READY_DIR = PROCESSED_DIR / "ctc_ready"

# 单帧特征维度
FEATURE_DIM = 166

# CTC blank 类别编号
BLANK_ID = 0

# 当前检查的数据划分
SPLITS = ["train", "dev", "test"]


# =========================================================
# 2. 基础文件工具
# =========================================================

def read_jsonl(path: Path) -> List[Dict]:
    """
    读取 jsonl 文件。

    Args:
        path: jsonl 文件路径。

    Returns:
        每一行解析后的字典列表。
    """
    rows: List[Dict] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """
    写出 jsonl 文件。

    Args:
        path: 输出文件路径。
        rows: 待写出的字典列表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict | List) -> None:
    """
    写出 JSON 文件。

    Args:
        path: 输出文件路径。
        data: 待写出的 JSON 数据。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# =========================================================
# 3. CTC 长度检查工具
# =========================================================

def count_consecutive_repeats(token_ids: List[int]) -> int:
    """
    统计标签序列中相邻重复 token 的数量。

    CTC 中如果目标标签存在相邻重复，例如：
        A A B

    通常需要在两个 A 中间插入 blank 才能区分：
        A blank A B

    因此 CTC 的最小输入长度不仅是 target_length，
    更严格地说是：
        target_length + consecutive_repeat_count

    Args:
        token_ids: glossIds 列表。

    Returns:
        相邻重复 token 的数量。
    """
    repeat_count = 0

    for index in range(1, len(token_ids)):
        if token_ids[index] == token_ids[index - 1]:
            repeat_count += 1

    return repeat_count


def get_ctc_min_input_length(token_ids: List[int]) -> int:
    """
    计算 CTC 理论上需要的最小输入长度。

    Args:
        token_ids: glossIds 列表。

    Returns:
        CTC 最小输入长度。
    """
    return len(token_ids) + count_consecutive_repeats(token_ids)


# =========================================================
# 4. 单条样本检查
# =========================================================

def inspect_one_sample(sample: Dict) -> Tuple[bool, Dict]:
    """
    检查单条样本是否适合进入 CTC 训练。

    Args:
        sample: manifest 中的一条样本。

    Returns:
        二元组：
        - 是否有效
        - 检查结果行
    """
    split = sample.get("split", "")
    sample_id = sample.get("sampleId", "")

    feature_path = FEATURE_DIR / split / f"{sample_id}.npy"

    gloss_ids = sample.get("glossIds")
    gloss = sample.get("gloss")
    raw_gloss = sample.get("rawGloss")
    gloss_length = sample.get("glossLength")

    result = {
        "sampleId": sample_id,
        "split": split,
        "videoPath": sample.get("videoPath"),
        "featurePath": str(feature_path.relative_to(DATASET_ROOT)).replace("\\", "/"),
        "chinese": sample.get("chinese"),
        "rawGloss": raw_gloss,
        "gloss": gloss,
        "glossIds": gloss_ids,
        "glossLength": gloss_length,
        "inputLength": None,
        "featureShape": None,
        "ctcMinInputLength": None,
        "consecutiveRepeatCount": None,
        "status": "ok",
        "message": "",
    }

    # 1. 检查特征文件是否存在
    if not feature_path.exists():
        result["status"] = "missing_feature"
        result["message"] = "feature file not found"
        return False, result

    # 2. 检查标签是否存在
    if not isinstance(gloss_ids, list):
        result["status"] = "bad_gloss_ids"
        result["message"] = "glossIds is not a list"
        return False, result

    if len(gloss_ids) == 0:
        result["status"] = "empty_target"
        result["message"] = "glossIds is empty"
        return False, result

    # 3. 检查 glossLength 是否匹配
    if gloss_length != len(gloss_ids):
        result["status"] = "bad_target_length"
        result["message"] = f"glossLength={gloss_length}, len(glossIds)={len(gloss_ids)}"
        return False, result

    # 4. 检查 target 中是否混入 blank
    if BLANK_ID in gloss_ids:
        result["status"] = "target_contains_blank"
        result["message"] = f"target contains blank id {BLANK_ID}"
        return False, result

    # 5. 检查 glossIds 是否全是整数
    if not all(isinstance(token_id, int) for token_id in gloss_ids):
        result["status"] = "bad_gloss_id_type"
        result["message"] = "some glossIds are not int"
        return False, result

    # 6. 读取特征并检查 shape
    try:
        features = np.load(feature_path)
    except Exception as exc:
        result["status"] = "feature_load_error"
        result["message"] = f"{type(exc).__name__}: {exc}"
        return False, result

    result["featureShape"] = list(features.shape)

    if features.ndim != 2:
        result["status"] = "bad_feature_ndim"
        result["message"] = f"features.ndim={features.ndim}, expected 2"
        return False, result

    if features.shape[1] != FEATURE_DIM:
        result["status"] = "bad_feature_dim"
        result["message"] = f"feature_dim={features.shape[1]}, expected {FEATURE_DIM}"
        return False, result

    if np.isnan(features).any():
        result["status"] = "feature_contains_nan"
        result["message"] = "features contains NaN"
        return False, result

    if np.isinf(features).any():
        result["status"] = "feature_contains_inf"
        result["message"] = "features contains Inf"
        return False, result

    input_length = int(features.shape[0])
    target_length = int(len(gloss_ids))

    repeat_count = count_consecutive_repeats(gloss_ids)
    ctc_min_input_length = get_ctc_min_input_length(gloss_ids)

    result["inputLength"] = input_length
    result["consecutiveRepeatCount"] = repeat_count
    result["ctcMinInputLength"] = ctc_min_input_length

    # 7. 最基本长度检查
    if input_length < target_length:
        result["status"] = "input_shorter_than_target"
        result["message"] = f"T={input_length}, targetLength={target_length}"
        return False, result

    # 8. 更严格的 CTC 最小长度检查
    if input_length < ctc_min_input_length:
        result["status"] = "input_shorter_than_ctc_min"
        result["message"] = f"T={input_length}, ctcMinInputLength={ctc_min_input_length}"
        return False, result

    return True, result


# =========================================================
# 5. split 检查
# =========================================================

def inspect_split(split: str) -> Dict:
    """
    检查一个数据划分。

    Args:
        split: train / dev / test。

    Returns:
        当前 split 的检查摘要。
    """
    manifest_path = PROCESSED_DIR / f"{split}.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest 文件：{manifest_path}")

    samples = read_jsonl(manifest_path)

    valid_rows: List[Dict] = []
    invalid_rows: List[Dict] = []

    input_lengths: List[int] = []
    target_lengths: List[int] = []
    ctc_min_lengths: List[int] = []
    repeat_counts: List[int] = []

    for index, sample in enumerate(samples, start=1):
        is_valid, row = inspect_one_sample(sample)

        if is_valid:
            valid_rows.append(row)
            input_lengths.append(int(row["inputLength"]))
            target_lengths.append(int(row["glossLength"]))
            ctc_min_lengths.append(int(row["ctcMinInputLength"]))
            repeat_counts.append(int(row["consecutiveRepeatCount"]))
        else:
            invalid_rows.append(row)

        if index % 500 == 0:
            print(f"{split}: 已检查 {index}/{len(samples)}")

    # 输出当前 split 的可训练清单和异常清单
    write_jsonl(CTC_READY_DIR / f"{split}_ctc_ready.jsonl", valid_rows)
    write_jsonl(CTC_REPORT_DIR / f"{split}_ctc_invalid.jsonl", invalid_rows)
    write_jsonl(CTC_REPORT_DIR / f"{split}_ctc_all_checked.jsonl", valid_rows + invalid_rows)

    summary = {
        "split": split,
        "total": len(samples),
        "valid": len(valid_rows),
        "invalid": len(invalid_rows),
        "invalidExamples": invalid_rows[:20],
        "inputLength": summarize_numbers(input_lengths),
        "targetLength": summarize_numbers(target_lengths),
        "ctcMinInputLength": summarize_numbers(ctc_min_lengths),
        "consecutiveRepeatCount": summarize_numbers(repeat_counts),
    }

    return summary


def summarize_numbers(values: List[int]) -> Dict:
    """
    生成简单数值统计。

    Args:
        values: 整数列表。

    Returns:
        统计信息。
    """
    if not values:
        return {
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    arr = np.array(values, dtype=np.float32)

    return {
        "min": int(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


# =========================================================
# 6. 主入口
# =========================================================

def main() -> None:
    """
    主入口。
    """
    CTC_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CTC_READY_DIR.mkdir(parents=True, exist_ok=True)

    print("===== CE-CSL CTC 训练前检查开始 =====")
    print("数据集目录:", DATASET_ROOT)
    print("特征目录:", FEATURE_DIR)
    print("CTC ready 输出目录:", CTC_READY_DIR)
    print("CTC report 输出目录:", CTC_REPORT_DIR)
    print("FEATURE_DIM:", FEATURE_DIM)
    print("BLANK_ID:", BLANK_ID)

    summaries: List[Dict] = []

    for split in SPLITS:
        print("\n" + "#" * 80)
        print("检查 split:", split)
        print("#" * 80)

        summary = inspect_split(split)
        summaries.append(summary)

        print(f"\n{split} 检查完成:")
        print("  total:", summary["total"])
        print("  valid:", summary["valid"])
        print("  invalid:", summary["invalid"])
        print("  inputLength:", summary["inputLength"])
        print("  targetLength:", summary["targetLength"])
        print("  ctcMinInputLength:", summary["ctcMinInputLength"])
        print("  consecutiveRepeatCount:", summary["consecutiveRepeatCount"])

        if summary["invalidExamples"]:
            print("\n  invalid 示例:")
            for row in summary["invalidExamples"]:
                print(
                    " ",
                    row["sampleId"],
                    row["status"],
                    row["message"],
                    "gloss=",
                    "/".join(row["gloss"] or []),
                )

    summary_path = CTC_REPORT_DIR / "ctc_ready_summary.json"
    write_json(summary_path, summaries)

    print("\n===== 总结 =====")
    for summary in summaries:
        print(
            summary["split"],
            "total=", summary["total"],
            "valid=", summary["valid"],
            "invalid=", summary["invalid"],
        )

    print("\n已写出:")
    print("  ready train:", CTC_READY_DIR / "train_ctc_ready.jsonl")
    print("  ready dev:", CTC_READY_DIR / "dev_ctc_ready.jsonl")
    print("  ready test:", CTC_READY_DIR / "test_ctc_ready.jsonl")
    print("  summary:", summary_path)

    print("\n===== CE-CSL CTC 训练前检查结束 =====")


if __name__ == "__main__":
    main()