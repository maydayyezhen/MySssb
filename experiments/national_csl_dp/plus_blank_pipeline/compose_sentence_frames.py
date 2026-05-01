# -*- coding: utf-8 -*-
"""
将 NationalCSL-DP 已抽取的 raw_frames 词语帧拼接成一个连续句子帧目录。

用途：
不需要自己录手语。
直接用已有词语帧目录，例如：

raw_frames/
  朋友__5008/Participant_10/*.jpg
  帮助__5094/Participant_10/*.jpg
  我们__0488/Participant_10/*.jpg

拼成：

stream_frames/
  000001.jpg
  000002.jpg
  ...

后续 infer_stream_plus.py 可以直接读取这个连续帧目录做真实帧推理。
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def list_images(frame_dir: Path) -> List[Path]:
    """
    获取目录下图片帧。
    """
    if not frame_dir.exists():
        return []

    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def parse_sentence(sentence: str) -> List[str]:
    """
    解析逗号分隔词序列。
    """
    labels = [
        item.strip()
        for item in sentence.replace("，", ",").split(",")
    ]
    labels = [item for item in labels if item]

    if not labels:
        raise ValueError("sentence 不能为空，例如：朋友,帮助,我们")

    return labels


def find_label_dir(raw_frames_root: Path, label: str) -> Path:
    """
    根据中文 label 查找 raw_frames 下对应目录。

    例如：
    label=朋友
    匹配：
    朋友__5008

    如果同一个 label 有多个资源，例如 学习__3701 / 学习__4462，
    默认取排序后的第一个。
    """
    candidates = sorted([
        path for path in raw_frames_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{label}__")
    ])

    if not candidates:
        raise RuntimeError(f"找不到 label 目录：{label}，root={raw_frames_root}")

    return candidates[0]


def copy_frame(src: Path, dst: Path) -> None:
    """
    复制一帧图片。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def compose_sentence_frames(
    raw_frames_root: Path,
    output_dir: Path,
    sentence: str,
    participant: str,
    overwrite: bool,
    tail_frames: int,
    tail_mode: str,
) -> None:
    """
    拼接词语帧目录为连续句子帧目录。
    """
    labels = parse_sentence(sentence)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []

    frame_no = 1
    last_frame_path: Optional[Path] = None

    print("========== 开始拼接连续句子帧 ==========")
    print(f"[信息] raw_frames_root：{raw_frames_root}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] participant：{participant}")
    print(f"[信息] sentence：{' '.join(labels)}")

    for label in labels:
        label_dir = find_label_dir(raw_frames_root, label)
        participant_dir = label_dir / participant

        image_paths = list_images(participant_dir)

        if not image_paths:
            raise RuntimeError(f"没有找到图片帧：{participant_dir}")

        start_frame = frame_no

        print(f"[词] {label} <- {participant_dir} 帧数={len(image_paths)}")

        for image_path in image_paths:
            dst = output_dir / f"{frame_no:06d}{image_path.suffix.lower()}"
            copy_frame(image_path, dst)

            manifest_rows.append({
                "frame_no": frame_no,
                "filename": dst.name,
                "type": "word",
                "label": label,
                "source": str(image_path),
            })

            last_frame_path = image_path
            frame_no += 1

        end_frame = frame_no - 1

        print(f"     输出帧范围：{start_frame} ~ {end_frame}")

    if tail_frames > 0 and tail_mode != "none":
        if last_frame_path is None:
            raise RuntimeError("没有 last_frame_path，无法生成 tail")

        print(f"[tail] tail_frames={tail_frames}, tail_mode={tail_mode}")

        for _ in range(tail_frames):
            if tail_mode == "repeat_last":
                src = last_frame_path
            else:
                raise ValueError(f"暂不支持 tail_mode：{tail_mode}")

            dst = output_dir / f"{frame_no:06d}{src.suffix.lower()}"
            copy_frame(src, dst)

            manifest_rows.append({
                "frame_no": frame_no,
                "filename": dst.name,
                "type": "tail",
                "label": "blank",
                "source": str(src),
            })

            frame_no += 1

    manifest_csv = output_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_no", "filename", "type", "label", "source"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_json = output_dir / "manifest.json"
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sentence": labels,
                "participant": participant,
                "frame_count": frame_no - 1,
                "tail_frames": tail_frames,
                "tail_mode": tail_mode,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== 拼接完成 ==========")
    print(f"[完成] 输出目录：{output_dir}")
    print(f"[完成] 总帧数：{frame_no - 1}")
    print(f"[完成] manifest：{manifest_csv}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_frames_root",
        required=True,
        help="raw_frames 根目录，例如 D:/datasets/HearBridge-NationalCSL-mini/raw_frames",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="连续句子帧输出目录",
    )
    parser.add_argument(
        "--sentence",
        required=True,
        help="逗号分隔词序列，例如：朋友,帮助,我们",
    )
    parser.add_argument(
        "--participant",
        default="Participant_10",
        help="参与者编号",
    )
    parser.add_argument(
        "--tail_frames",
        type=int,
        default=12,
        help="句尾重复最后一帧的数量",
    )
    parser.add_argument(
        "--tail_mode",
        default="repeat_last",
        choices=["repeat_last", "none"],
        help="句尾补帧方式",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出目录",
    )

    args = parser.parse_args()

    compose_sentence_frames(
        raw_frames_root=Path(args.raw_frames_root),
        output_dir=Path(args.output_dir),
        sentence=args.sentence,
        participant=args.participant,
        overwrite=args.overwrite,
        tail_frames=args.tail_frames,
        tail_mode=args.tail_mode,
    )


if __name__ == "__main__":
    main()