# -*- coding: utf-8 -*-
"""
从 NationalCSL-DP 中抽取 HearBridge 小词表训练集。

作用：
1. 不全量解压 Participant_xx.zip
2. 只抽取指定词 ID 的 front 视角图片帧
3. 将图片整理成适合后续特征提取/训练的小数据集目录
4. 生成 samples.csv，记录每个样本的标签、参与者、帧目录和帧数

推荐输出结构：
D:/datasets/HearBridge-NationalCSL-mini/
  raw_frames/
    我__1928/
      Participant_01/
        00001.jpg
        00002.jpg
    你__1925/
      Participant_01/
        00001.jpg
  samples.csv
"""

import argparse
import csv
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List


# HearBridge 第一批小词表资源。
# resource_id：对应 gloss.csv 中的 #ID
# source_word：数据集原始中文词名
# label：训练时使用的归一化标签
# translation：英文说明，仅用于排查
TARGET_RESOURCES: List[Dict[str, str]] = [
    {"resource_id": "1928", "source_word": "我", "label": "我", "translation": "I"},
    {"resource_id": "1925", "source_word": "你", "label": "你", "translation": "You"},
    {"resource_id": "0488", "source_word": "我们", "label": "我们", "translation": "We"},
    {"resource_id": "6571", "source_word": "你们", "label": "你们", "translation": "You plural"},
    {"resource_id": "1251", "source_word": "学校", "label": "学校", "translation": "School"},
    {"resource_id": "1284", "source_word": "医院", "label": "医院", "translation": "Hospital"},
    {"resource_id": "1597", "source_word": "老师", "label": "老师", "translation": "Teacher"},
    {"resource_id": "1770", "source_word": "今天", "label": "今天", "translation": "Today"},
    {"resource_id": "4542", "source_word": "明天", "label": "明天", "translation": "Tomorrow"},
    {"resource_id": "4046", "source_word": "再见", "label": "再见", "translation": "Goodbye"},
    {"resource_id": "5008", "source_word": "朋友", "label": "朋友", "translation": "Friend"},
    {"resource_id": "5094", "source_word": "帮助", "label": "帮助", "translation": "Help"},
    {"resource_id": "5610", "source_word": "需要1-1", "label": "需要", "translation": "Need"},
    {"resource_id": "5311", "source_word": "对不起1-2", "label": "对不起", "translation": "Sorry"},
    {"resource_id": "3701", "source_word": "学习1-2", "label": "学习", "translation": "Study variant 1"},
    {"resource_id": "4462", "source_word": "学习1-1", "label": "学习", "translation": "Study variant 2"},
]


def build_label_dir_name(label: str, resource_id: str) -> str:
    """
    构造标签目录名。

    目录里带 resource_id 是为了避免多个同名词变体覆盖。
    例如：
    学习__3701
    学习__4462
    """
    return f"{label}__{resource_id}"


def get_participant_zip_path(dataset_root: Path, participant_index: int) -> Path:
    """
    获取指定参与者的 Pics 压缩包路径。

    :param dataset_root: NationalCSL-DP 根目录
    :param participant_index: 参与者编号，1 到 10
    :return: Participant_xx.zip 路径
    """
    zip_name = f"Participant_{participant_index:02d}.zip"
    return dataset_root / "Pics" / zip_name


def extract_resource_from_zip(
    zip_path: Path,
    participant_name: str,
    view: str,
    resource: Dict[str, str],
    output_root: Path,
) -> Dict[str, str]:
    """
    从单个 Participant 压缩包中抽取一个词资源。

    :param zip_path: Participant_xx.zip 路径
    :param participant_name: 参与者目录名，例如 Participant_01
    :param view: 视角名称，推荐先用 front
    :param resource: 目标词资源配置
    :param output_root: 小数据集输出根目录
    :return: 样本元信息
    """
    resource_id = resource["resource_id"]
    label = resource["label"]
    source_word = resource["source_word"]
    translation = resource["translation"]

    # zip 内部路径前缀，例如：
    # Participant_01/front/1928/
    zip_prefix = f"{participant_name}/{view}/{resource_id}/"

    # 输出目录，例如：
    # raw_frames/我__1928/Participant_01/
    label_dir_name = build_label_dir_name(label=label, resource_id=resource_id)
    frame_output_dir = output_root / "raw_frames" / label_dir_name / participant_name

    extracted_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        # 找到该词对应的所有 jpg 帧
        frame_names = [
            name
            for name in zf.namelist()
            if name.startswith(zip_prefix) and name.lower().endswith(".jpg")
        ]

        frame_names.sort()

        if not frame_names:
            return {
                "resource_id": resource_id,
                "label": label,
                "source_word": source_word,
                "translation": translation,
                "participant": participant_name,
                "view": view,
                "frame_dir": str(frame_output_dir),
                "frame_count": "0",
                "status": "missing",
            }

        frame_output_dir.mkdir(parents=True, exist_ok=True)

        for frame_name in frame_names:
            # 只保留帧文件名，不保留 zip 内部的多级目录
            target_path = frame_output_dir / Path(frame_name).name

            with zf.open(frame_name, "r") as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted_count += 1

    return {
        "resource_id": resource_id,
        "label": label,
        "source_word": source_word,
        "translation": translation,
        "participant": participant_name,
        "view": view,
        "frame_dir": str(frame_output_dir),
        "frame_count": str(extracted_count),
        "status": "ok",
    }


def write_samples_csv(output_root: Path, rows: List[Dict[str, str]]) -> None:
    """
    写出 samples.csv 样本索引。

    后续特征提取脚本可以直接读取这个文件。
    """
    csv_path = output_root / "samples.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "resource_id",
        "label",
        "source_word",
        "translation",
        "participant",
        "view",
        "frame_dir",
        "frame_count",
        "status",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[完成] 已写出样本索引：{csv_path}")


def extract_mini_dataset(
    dataset_root: Path,
    output_root: Path,
    view: str,
    participant_count: int,
    overwrite: bool,
) -> None:
    """
    抽取小词表数据集。

    :param dataset_root: NationalCSL-DP 根目录
    :param output_root: 输出的小训练集根目录
    :param view: 视角，建议先用 front
    :param participant_count: 使用多少个参与者，默认 10
    :param overwrite: 是否清空旧输出目录
    """
    if overwrite and output_root.exists():
        print(f"[警告] 清空旧输出目录：{output_root}")
        shutil.rmtree(output_root)

    rows: List[Dict[str, str]] = []

    print("========== 开始抽取 NationalCSL-DP 小词表 ==========")
    print(f"[信息] 数据集根目录：{dataset_root}")
    print(f"[信息] 输出目录：{output_root}")
    print(f"[信息] 使用视角：{view}")
    print(f"[信息] 参与者数量：{participant_count}")
    print(f"[信息] 目标资源数：{len(TARGET_RESOURCES)}")

    for participant_index in range(1, participant_count + 1):
        participant_name = f"Participant_{participant_index:02d}"
        zip_path = get_participant_zip_path(dataset_root, participant_index)

        if not zip_path.exists():
            print(f"[跳过] 找不到压缩包：{zip_path}")
            continue

        print(f"\n========== 处理 {participant_name} ==========")
        print(f"[信息] zip：{zip_path}")

        for resource in TARGET_RESOURCES:
            row = extract_resource_from_zip(
                zip_path=zip_path,
                participant_name=participant_name,
                view=view,
                resource=resource,
                output_root=output_root,
            )
            rows.append(row)

            print(
                f"[{row['status']}] "
                f"{row['resource_id']} {row['source_word']} -> {row['label']} "
                f"{participant_name} 帧数={row['frame_count']}"
            )

    write_samples_csv(output_root=output_root, rows=rows)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    missing_count = sum(1 for row in rows if row["status"] != "ok")

    print("\n========== 抽取总结 ==========")
    print(f"[信息] 成功样本数：{ok_count}")
    print(f"[信息] 缺失样本数：{missing_count}")
    print(f"[信息] 总记录数：{len(rows)}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="NationalCSL-DP 根目录，例如 D:/datasets/NationalCSL-DP",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="输出小训练集目录，例如 D:/datasets/HearBridge-NationalCSL-mini",
    )
    parser.add_argument(
        "--view",
        default="front",
        choices=["front", "side"],
        help="抽取视角，第一版建议使用 front",
    )
    parser.add_argument(
        "--participant_count",
        type=int,
        default=10,
        help="使用前多少个参与者，默认 10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，是否先清空",
    )

    args = parser.parse_args()

    extract_mini_dataset(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        view=args.view,
        participant_count=args.participant_count,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()