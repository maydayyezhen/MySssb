"""
CE-CSL 数据集 Manifest 构建脚本

作用：
1. 读取 CE-CSL 的 train.csv / dev.csv / test.csv。
2. 根据 Number + Translator + split 定位对应视频文件。
3. 将 Gloss 字段按 "/" 切分成 gloss 序列。
4. 去掉标点符号。
5. 构建 gloss 词表。
6. 生成 train.jsonl / dev.jsonl / test.jsonl / all.jsonl。
7. 生成 gloss_to_id.json / id_to_gloss.json。

使用方式：
    1. 修改 DATASET_ROOT 为你的 CE-CSL 数据集根目录。
    2. 在项目根目录运行：
       python experiments/ce_csl_gloss_recognition_v1/build_manifest.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Set


# =========================================================
# 1. 路径配置
# =========================================================

# TODO：改成你的 CE-CSL 数据集根目录
# 这个目录下面应该有 label/ 和 video/
DATASET_ROOT = Path(r"D:\CE-CSL\CE-CSL")

# 输出目录，会自动创建
OUTPUT_DIR = DATASET_ROOT / "processed"


# =========================================================
# 2. Gloss 清洗配置
# =========================================================

# 第一版 CTC 不训练标点，后面交给 AI 翻译层补自然标点
PUNCTUATION_SET: Set[str] = {
    "。", "？", "！", "，", "、", "；", "：",
    ".", "?", "!", ",", ";", ":",
    "“", "”", "\"", "'", "‘", "’",
    "（", "）", "(", ")",
}


def clean_gloss(raw_gloss: str) -> List[str]:
    """
    将 CSV 里的 Gloss 字段转换成 gloss token 列表。

    示例：
        原始：五/百/万/衣服/买/多少/可以/？
        输出：["五", "百", "万", "衣服", "买", "多少", "可以"]

    Args:
        raw_gloss: CSV 中的原始 Gloss 字符串。

    Returns:
        清洗后的 gloss token 列表。
    """
    tokens = str(raw_gloss).split("/")

    result: List[str] = []

    for token in tokens:
        token = token.strip()

        if not token:
            continue

        if token in PUNCTUATION_SET:
            continue

        result.append(token)

    return result


def read_csv_split(split: str) -> List[Dict]:
    """
    读取某一个 split 的 CSV 文件。

    Args:
        split: train / dev / test。

    Returns:
        当前 split 下的样本列表。
    """
    csv_path = DATASET_ROOT / "label" / f"{split}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    samples: List[Dict] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            sample_id = row["Number"].strip()
            translator = row["Translator"].strip()
            chinese = row["Chinese Sentences"].strip()
            raw_gloss = row["Gloss"].strip()

            video_path = DATASET_ROOT / "video" / split / translator / f"{sample_id}.mp4"

            if not video_path.exists():
                raise FileNotFoundError(f"找不到视频文件：{video_path}")

            gloss_tokens = clean_gloss(raw_gloss)

            sample = {
                "sampleId": sample_id,
                "split": split,
                "translator": translator,
                "videoPath": str(video_path.relative_to(DATASET_ROOT)).replace("\\", "/"),
                "chinese": chinese,
                "rawGloss": raw_gloss,
                "gloss": gloss_tokens,
                "glossLength": len(gloss_tokens),
            }

            samples.append(sample)

    return samples


def build_vocab_from_train(train_samples: List[Dict]) -> Dict[str, int]:
    """
    只用训练集构建 gloss 词表。

    这样更规范：dev/test 只用于验证和测试，不参与训练词表构建。

    Args:
        train_samples: 训练集样本。

    Returns:
        gloss_to_id 映射。
    """
    gloss_set: Set[str] = set()

    for sample in train_samples:
        for token in sample["gloss"]:
            gloss_set.add(token)

    sorted_tokens = sorted(gloss_set)

    # CTC 的 blank 固定为 0
    gloss_to_id: Dict[str, int] = {
        "<blank>": 0,
        "<unk>": 1,
    }

    for index, token in enumerate(sorted_tokens, start=2):
        gloss_to_id[token] = index

    return gloss_to_id


def attach_gloss_ids(samples: List[Dict], gloss_to_id: Dict[str, int]) -> None:
    """
    给每条样本添加 glossIds。

    如果 dev/test 中出现训练集没见过的词，映射为 <unk>。

    Args:
        samples: 样本列表。
        gloss_to_id: gloss token 到数字 id 的映射。
    """
    unk_id = gloss_to_id["<unk>"]

    for sample in samples:
        sample["glossIds"] = [
            gloss_to_id.get(token, unk_id)
            for token in sample["gloss"]
        ]


def collect_oov(samples: List[Dict], gloss_to_id: Dict[str, int]) -> Set[str]:
    """
    收集样本中不在训练词表里的 token。

    Args:
        samples: 样本列表。
        gloss_to_id: 训练词表。

    Returns:
        OOV token 集合。
    """
    oov: Set[str] = set()

    for sample in samples:
        for token in sample["gloss"]:
            if token not in gloss_to_id:
                oov.add(token)

    return oov


def write_jsonl(path: Path, samples: List[Dict]) -> None:
    """
    写出 jsonl 文件。

    Args:
        path: 输出路径。
        samples: 样本列表。
    """
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_json(path: Path, data) -> None:
    """
    写出 JSON 文件。

    Args:
        path: 输出路径。
        data: 待写入对象。
    """
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    """
    主流程。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== CE-CSL Manifest 构建开始 =====")
    print("数据集目录：", DATASET_ROOT)
    print("输出目录：", OUTPUT_DIR)

    train_samples = read_csv_split("train")
    dev_samples = read_csv_split("dev")
    test_samples = read_csv_split("test")

    all_samples = train_samples + dev_samples + test_samples

    gloss_to_id = build_vocab_from_train(train_samples)
    id_to_gloss = {
        str(idx): token
        for token, idx in gloss_to_id.items()
    }

    dev_oov = collect_oov(dev_samples, gloss_to_id)
    test_oov = collect_oov(test_samples, gloss_to_id)

    attach_gloss_ids(train_samples, gloss_to_id)
    attach_gloss_ids(dev_samples, gloss_to_id)
    attach_gloss_ids(test_samples, gloss_to_id)
    attach_gloss_ids(all_samples, gloss_to_id)

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_samples)
    write_jsonl(OUTPUT_DIR / "dev.jsonl", dev_samples)
    write_jsonl(OUTPUT_DIR / "test.jsonl", test_samples)
    write_jsonl(OUTPUT_DIR / "all.jsonl", all_samples)

    write_json(OUTPUT_DIR / "gloss_to_id.json", gloss_to_id)
    write_json(OUTPUT_DIR / "id_to_gloss.json", id_to_gloss)

    gloss_lengths = [sample["glossLength"] for sample in all_samples]

    print("\n===== 构建完成 =====")
    print("train 样本数：", len(train_samples))
    print("dev 样本数：", len(dev_samples))
    print("test 样本数：", len(test_samples))
    print("all 样本数：", len(all_samples))
    print("词表大小，含 <blank>/<unk>：", len(gloss_to_id))
    print("最短 gloss 长度：", min(gloss_lengths))
    print("最长 gloss 长度：", max(gloss_lengths))
    print("平均 gloss 长度：", round(sum(gloss_lengths) / len(gloss_lengths), 2))

    print("\n===== OOV 检查 =====")
    print("dev 中训练集未出现 token 数：", len(dev_oov))
    print("test 中训练集未出现 token 数：", len(test_oov))

    if dev_oov:
        print("dev OOV 示例：", sorted(dev_oov)[:20])

    if test_oov:
        print("test OOV 示例：", sorted(test_oov)[:20])

    print("\n===== 样本预览 =====")
    for sample in all_samples[:3]:
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()