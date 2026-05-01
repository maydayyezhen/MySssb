# -*- coding: utf-8 -*-
"""
从 How2Sign 的句子 CSV 中筛选包含目标 ASL 单词的句子。

用途：
1. 先看 How2Sign 句子视频里有哪些适合演示的简单句
2. 再反过来决定 WLASL 要下载哪些单词视频
"""

import re
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TARGETS = [
    "help",
    "go",
    "want",
    "school",
    "today",
    "meet",
    "work",
    "teacher",
    "learn",
    "you",
    "friend",
    "tomorrow",
    "sorry",
    "please",
]


def split_targets(text: str):
    """
    解析逗号分隔的目标词。
    """
    return [
        item.strip().lower()
        for item in text.replace("，", ",").split(",")
        if item.strip()
    ]


def extract_words(sentence: str):
    """
    从英文句子中提取小写单词。
    """
    return set(re.findall(r"[a-z']+", sentence.lower()))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        default="D:/datasets/How2Sign_HF_meta/how2sign_realigned_val.csv",
        help="How2Sign realigned CSV 路径",
    )

    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="逗号分隔的目标词",
    )

    parser.add_argument(
        "--min_hits",
        type=int,
        default=2,
        help="句子中至少命中多少个目标词",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=80,
        help="最多输出多少条结果",
    )

    parser.add_argument(
        "--output_csv",
        default="D:/datasets/How2Sign_HF_meta/how2sign_val_target_sentence_candidates.csv",
        help="筛选结果 CSV 输出路径",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_csv = Path(args.output_csv)

    targets = split_targets(args.targets)

    df = pd.read_csv(csv_path, sep=None, engine="python")

    rows = []

    for _, row in df.iterrows():
        sentence = str(row["SENTENCE"])
        words = extract_words(sentence)

        hits = [target for target in targets if target in words]

        if len(hits) >= args.min_hits:
            rows.append({
                "hit_count": len(hits),
                "hits": ",".join(hits),
                "sentence_name": row["SENTENCE_NAME"],
                "video_name": row["VIDEO_NAME"],
                "start_realigned": row["START_REALIGNED"],
                "end_realigned": row["END_REALIGNED"],
                "sentence": sentence,
            })

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        print("[结果] 没有找到符合条件的句子")
        return

    result_df = result_df.sort_values(
        by=["hit_count", "start_realigned"],
        ascending=[False, True],
    )

    result_df.head(args.top_k).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("========== How2Sign 句子筛选完成 ==========")
    print(f"CSV：{csv_path}")
    print(f"目标词：{', '.join(targets)}")
    print(f"min_hits：{args.min_hits}")
    print(f"命中句子数：{len(result_df)}")
    print(f"已输出前 {args.top_k} 条到：{output_csv}")

    print("\n========== 前若干候选 ==========")

    for _, item in result_df.head(args.top_k).iterrows():
        print()
        print(f"HITS = {item['hits']}")
        print(f"SENTENCE_NAME = {item['sentence_name']}")
        print(f"VIDEO_NAME = {item['video_name']}")
        print(f"TIME = {item['start_realigned']} -> {item['end_realigned']}")
        print(f"TEXT = {item['sentence']}")


if __name__ == "__main__":
    main()