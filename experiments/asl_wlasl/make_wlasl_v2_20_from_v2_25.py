# -*- coding: utf-8 -*-
"""
从 WLASL-mini-v2-25 派生 WLASL-mini-v2-20。

用途：
1. 根据 keep_labels 过滤 samples.csv；
2. 尽量复制 samples.csv 中引用的视频文件；
3. 从已有 features_20f_plus 中筛选 20 词样本；
4. 重新映射 y 标签编号；
5. 输出新的 labels.json。

注意：
- 不改变模型结构；
- 不重新跑 MediaPipe；
- 适合作为 25词 vs 20词 的严格 A/B 对照。
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def read_text_labels(text: str) -> list[str]:
    """读取逗号分隔的标签列表。"""
    return [item.strip() for item in text.split(",") if item.strip()]


def load_json(path: Path):
    """读取 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_labels(labels_path: Path) -> list[str]:
    """读取 labels.json，兼容 list 或 {'labels': [...]} 两种格式。"""
    payload = load_json(labels_path)

    if isinstance(payload, dict) and "labels" in payload:
        return list(payload["labels"])

    if isinstance(payload, list):
        return list(payload)

    raise ValueError(f"无法识别 labels.json 格式：{labels_path}")


def find_y_file(feature_dir: Path, class_count: int) -> Path:
    """寻找 y 标签文件。"""
    preferred = [
        "y.npy",
        "labels.npy",
        "label_indices.npy",
        "target.npy",
        "targets.npy",
    ]

    for name in preferred:
        path = feature_dir / name
        if path.exists():
            return path

    for path in feature_dir.glob("*.npy"):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 1:
            continue
        if not np.issubdtype(arr.dtype, np.integer):
            continue
        if len(arr) == 0:
            continue
        if int(arr.min()) >= 0 and int(arr.max()) < class_count:
            return path

    raise FileNotFoundError(
        f"没有在 {feature_dir} 中找到 y 标签 npy 文件。"
    )


def filter_samples_csv(src_root: Path, dst_root: Path, keep_labels: list[str]) -> None:
    """过滤 samples.csv，并尽量复制对应视频。"""
    src_csv = src_root / "samples.csv"
    dst_csv = dst_root / "samples.csv"

    if not src_csv.exists():
        print(f"[跳过] 未找到 samples.csv：{src_csv}")
        return

    dst_root.mkdir(parents=True, exist_ok=True)

    with src_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    label_col = None
    for candidate in ["label", "gloss", "word", "class", "class_name"]:
        if candidate in fieldnames:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"samples.csv 找不到标签列，现有列：{fieldnames}")

    keep_set = set(keep_labels)
    filtered_rows = [row for row in rows if row.get(label_col) in keep_set]

    path_cols = [
        col for col in fieldnames
        if any(key in col.lower() for key in ["path", "file", "video"])
    ]

    src_root_str = str(src_root)
    dst_root_str = str(dst_root)
    src_root_posix = src_root.as_posix()
    dst_root_posix = dst_root.as_posix()

    copied = 0

    for row in filtered_rows:
        for col in path_cols:
            value = row.get(col) or ""
            if not value:
                continue

            raw_path = Path(value)
            src_path = raw_path if raw_path.is_absolute() else (src_root / raw_path)

            if not src_path.exists():
                row[col] = (
                    value
                    .replace(src_root_str, dst_root_str)
                    .replace(src_root_posix, dst_root_posix)
                )
                continue

            try:
                rel = src_path.relative_to(src_root)
            except ValueError:
                rel = Path("videos") / src_path.name

            dst_path = dst_root / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

            row[col] = dst_path.as_posix()

    with dst_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"[完成] samples.csv：{dst_csv}")
    print(f"[信息] 原始样本数：{len(rows)}")
    print(f"[信息] 20词样本数：{len(filtered_rows)}")
    print(f"[信息] 复制视频数：{copied}")


def maybe_filter_json_payload(payload, mask, old_to_new_index):
    """尝试过滤 JSON payload。"""
    selected_old_indices = set(old_to_new_index.keys())

    if isinstance(payload, list) and len(payload) == len(mask):
        return [item for item, keep in zip(payload, mask) if bool(keep)]

    if isinstance(payload, dict):
        # 处理 split 索引形式，例如 {"train": [0, 1], "test": [2]}
        converted = {}
        changed = False

        for key, value in payload.items():
            if (
                isinstance(value, list)
                and all(isinstance(x, int) for x in value)
            ):
                converted[key] = [
                    old_to_new_index[x]
                    for x in value
                    if x in selected_old_indices
                ]
                changed = True
            else:
                converted[key] = value

        if changed:
            return converted

    return payload


def filter_feature_dir(src_feature_dir: Path, dst_feature_dir: Path, keep_labels: list[str]) -> None:
    """从 25 词 features_20f_plus 筛出 20 词 features_20f_plus。"""
    labels_path = src_feature_dir / "labels.json"
    old_labels = load_labels(labels_path)

    keep_set = set(keep_labels)
    missing = [label for label in keep_labels if label not in old_labels]
    if missing:
        raise ValueError(f"keep_labels 中有标签不在原 labels.json 中：{missing}")

    y_path = find_y_file(src_feature_dir, len(old_labels))
    y = np.load(y_path, allow_pickle=True)

    old_label_for_y = np.array([old_labels[int(idx)] for idx in y])
    mask = np.array([label in keep_set for label in old_label_for_y], dtype=bool)

    new_label_to_id = {label: idx for idx, label in enumerate(keep_labels)}
    new_y = np.array(
        [new_label_to_id[old_labels[int(idx)]] for idx in y[mask]],
        dtype=y.dtype,
    )

    old_to_new_index = {}
    new_index = 0
    for old_index, keep in enumerate(mask):
        if bool(keep):
            old_to_new_index[old_index] = new_index
            new_index += 1

    if dst_feature_dir.exists():
        shutil.rmtree(dst_feature_dir)

    dst_feature_dir.mkdir(parents=True, exist_ok=True)

    # 复制/过滤 npy
    for path in src_feature_dir.glob("*.npy"):
        arr = np.load(path, allow_pickle=True)

        if path.name == y_path.name:
            out = new_y
            action = "remap-y"
        elif arr.shape and arr.shape[0] == len(y):
            out = arr[mask]
            action = "filter"
        else:
            out = arr
            action = "copy"

        np.save(dst_feature_dir / path.name, out)
        print(f"[{action}] {path.name}: {arr.shape} -> {out.shape}")

    # 复制/重写 json
    for path in src_feature_dir.glob("*.json"):
        if path.name == "labels.json":
            save_json(dst_feature_dir / path.name, {"labels": keep_labels})
            print(f"[rewrite] labels.json -> {len(keep_labels)} labels")
            continue

        payload = load_json(path)
        new_payload = maybe_filter_json_payload(payload, mask, old_to_new_index)
        save_json(dst_feature_dir / path.name, new_payload)
        print(f"[json] {path.name}")

    # 复制/过滤 csv
    for path in src_feature_dir.glob("*.csv"):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if len(rows) == len(mask):
            out_rows = [row for row, keep in zip(rows, mask) if bool(keep)]
            action = "filter"
        else:
            out_rows = rows
            action = "copy"

        with (dst_feature_dir / path.name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"[{action}-csv] {path.name}: {len(rows)} -> {len(out_rows)}")

    save_json(
        dst_feature_dir / "subset_info.json",
        {
            "source_feature_dir": src_feature_dir.as_posix(),
            "class_count": len(keep_labels),
            "labels": keep_labels,
            "source_sample_count": int(len(y)),
            "subset_sample_count": int(mask.sum()),
            "removed_labels": [label for label in old_labels if label not in keep_set],
        },
    )

    print(f"[完成] 20词特征目录：{dst_feature_dir}")
    print(f"[信息] 原始样本数：{len(y)}")
    print(f"[信息] 20词样本数：{int(mask.sum())}")
    print(f"[信息] 标签：{keep_labels}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--dst_root", required=True)
    parser.add_argument("--keep_labels", required=True)
    parser.add_argument("--feature_name", default="features_20f_plus")
    parser.add_argument("--only_dataset", action="store_true")
    parser.add_argument("--only_features", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    keep_labels = read_text_labels(args.keep_labels)

    if not src_root.exists():
        raise FileNotFoundError(f"源数据目录不存在：{src_root}")

    if not args.only_features:
        filter_samples_csv(src_root, dst_root, keep_labels)

        save_json(
            dst_root / "labels_20.json",
            {
                "labels": keep_labels,
                "removed_labels": [],
            },
        )

    if not args.only_dataset:
        filter_feature_dir(
            src_feature_dir=src_root / args.feature_name,
            dst_feature_dir=dst_root / args.feature_name,
            keep_labels=keep_labels,
        )


if __name__ == "__main__":
    main()
