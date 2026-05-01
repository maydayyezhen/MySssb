# -*- coding: utf-8 -*-
"""
NationalCSL-DP 小数据集 MediaPipe 可见性检查脚本。

作用：
1. 读取 HearBridge-NationalCSL-mini/samples.csv
2. 对每个样本的图片帧运行 MediaPipe Pose + Hands
3. 统计 pose_present、hand_present、both_hands_present 等指标
4. 生成 CSV 统计文件，方便判断是否可以直接使用旧特征方案
5. 生成带关键点覆盖的预览图，方便肉眼检查

推荐运行：
python experiments/national_csl_dp/check_mediapipe_visibility.py ^
  --samples_csv "D:/datasets/HearBridge-NationalCSL-mini/samples.csv" ^
  --output_dir "D:/datasets/HearBridge-NationalCSL-mini/mediapipe_check" ^
  --max_samples 40 ^
  --frames_per_sample 6
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe 绘图工具
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
MP_POSE = mp.solutions.pose
MP_HANDS = mp.solutions.hands


def read_samples(samples_csv: Path) -> List[Dict[str, str]]:
    """
    读取 samples.csv。
    """
    with samples_csv.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_image_frames(frame_dir: Path) -> List[Path]:
    """
    获取样本目录中的所有图片帧。
    """
    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def sample_indices(total: int, count: int) -> List[int]:
    """
    从 total 个帧中均匀抽取 count 个索引。
    """
    if total <= 0:
        return []

    if total <= count:
        return list(range(total))

    return [
        round(i * (total - 1) / (count - 1))
        for i in range(count)
    ]

def read_image_bgr_unicode(image_path: Path) -> np.ndarray | None:
    """
    读取可能包含中文路径的图片。

    说明：
    1. Windows 下 cv2.imread(str(path)) 对中文路径可能返回 None
    2. 这里使用 np.fromfile + cv2.imdecode 绕开路径编码问题
    3. 返回值仍然是 OpenCV 常用的 BGR 图片
    """
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)

        if image_bytes.size == 0:
            return None

        image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        return image_bgr

    except Exception as e:
        print(f"[错误] 读取图片异常：{image_path}，原因：{e}")
        return None

def detect_frame(
    image_bgr: np.ndarray,
    pose_model,
    hands_model,
) -> Tuple[Dict[str, object], np.ndarray]:
    """
    对单帧图片运行 Pose + Hands 检测，并返回统计结果和绘制后的图片。
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe 推荐将输入标记为不可写，提升性能
    image_rgb.flags.writeable = False
    pose_result = pose_model.process(image_rgb)
    hands_result = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    overlay_bgr = image_bgr.copy()

    # 统计 Pose
    pose_present = pose_result.pose_landmarks is not None

    if pose_present:
        MP_DRAWING.draw_landmarks(
            overlay_bgr,
            pose_result.pose_landmarks,
            MP_POSE.POSE_CONNECTIONS,
            landmark_drawing_spec=MP_DRAWING_STYLES.get_default_pose_landmarks_style(),
        )

    # 统计 Hands
    hand_count = 0
    left_count = 0
    right_count = 0

    if hands_result.multi_hand_landmarks:
        hand_count = len(hands_result.multi_hand_landmarks)

        for hand_landmarks in hands_result.multi_hand_landmarks:
            MP_DRAWING.draw_landmarks(
                overlay_bgr,
                hand_landmarks,
                MP_HANDS.HAND_CONNECTIONS,
                MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
                MP_DRAWING_STYLES.get_default_hand_connections_style(),
            )

    if hands_result.multi_handedness:
        for handedness in hands_result.multi_handedness:
            label = handedness.classification[0].label
            if label == "Left":
                left_count += 1
            elif label == "Right":
                right_count += 1

    stats = {
        "pose_present": pose_present,
        "hand_count": hand_count,
        "any_hand_present": hand_count >= 1,
        "both_hands_present": hand_count >= 2,
        "left_hand_detected": left_count >= 1,
        "right_hand_detected": right_count >= 1,
    }

    return stats, overlay_bgr


def build_overlay_contact_sheet(
    overlay_images: List[Tuple[str, np.ndarray]],
    output_path: Path,
    thumb_width: int = 240,
    thumb_height: int = 180,
    cols: int = 3,
) -> None:
    """
    将若干张带骨架的预览图拼成一张 contact sheet。
    """
    if not overlay_images:
        print("[警告] 没有可用于生成预览图的图片")
        return

    rows = int(np.ceil(len(overlay_images) / cols))
    title_height = 32
    gap = 8

    sheet_width = cols * thumb_width + (cols + 1) * gap
    sheet_height = rows * (thumb_height + title_height) + (rows + 1) * gap

    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)

    for index, (title, image_bgr) in enumerate(overlay_images):
        row = index // cols
        col = index % cols

        x = gap + col * (thumb_width + gap)
        y = gap + row * (thumb_height + title_height + gap)

        # 写标题
        cv2.putText(
            sheet,
            title[:36],
            (x, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # 保持比例缩放图片
        h, w = image_bgr.shape[:2]
        scale = min(thumb_width / w, thumb_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((thumb_height, thumb_width, 3), 255, dtype=np.uint8)
        px = (thumb_width - new_w) // 2
        py = (thumb_height - new_h) // 2
        canvas[py:py + new_h, px:px + new_w] = resized

        image_y = y + title_height
        sheet[image_y:image_y + thumb_height, x:x + thumb_width] = canvas

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_bgr_unicode(output_path, sheet)
    print(f"[完成] 已生成 MediaPipe 预览图：{output_path}")

def save_image_bgr_unicode(output_path: Path, image_bgr: np.ndarray) -> None:
    """
    保存可能包含中文路径的图片。

    说明：
    1. Windows 下 cv2.imwrite(str(path)) 对中文路径也可能不稳定
    2. 这里使用 cv2.imencode + tofile 保存
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        success, encoded_image = cv2.imencode(".jpg", image_bgr)
    elif suffix == ".png":
        success, encoded_image = cv2.imencode(".png", image_bgr)
    else:
        success, encoded_image = cv2.imencode(".jpg", image_bgr)

    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")

    encoded_image.tofile(str(output_path))

def summarize_sample(
    row: Dict[str, str],
    pose_model,
    hands_model,
    frames_per_sample: int,
    overlay_images: List[Tuple[str, np.ndarray]],
    max_overlay_images: int,
) -> Dict[str, object]:
    """
    检查单个样本的 MediaPipe 检测情况。
    """
    label = row["label"]
    resource_id = row["resource_id"]
    participant = row["participant"]
    frame_dir = Path(row["frame_dir"])

    image_paths = list_image_frames(frame_dir)
    indices = sample_indices(len(image_paths), frames_per_sample)

    pose_count = 0
    any_hand_count = 0
    both_hand_count = 0
    left_hand_count = 0
    right_hand_count = 0
    total_hand_count = 0
    checked_count = 0

    for local_i, frame_index in enumerate(indices):
        image_path = image_paths[frame_index]
        image_bgr = read_image_bgr_unicode(image_path)

        if image_bgr is None:
            print(f"[警告] 图片读取失败：{image_path}")
            continue

        stats, overlay_bgr = detect_frame(
            image_bgr=image_bgr,
            pose_model=pose_model,
            hands_model=hands_model,
        )

        checked_count += 1
        pose_count += int(stats["pose_present"])
        any_hand_count += int(stats["any_hand_present"])
        both_hand_count += int(stats["both_hands_present"])
        left_hand_count += int(stats["left_hand_detected"])
        right_hand_count += int(stats["right_hand_detected"])
        total_hand_count += int(stats["hand_count"])

        if len(overlay_images) < max_overlay_images:
            title = f"{label}_{resource_id}_{participant}_f{frame_index + 1}"
            overlay_images.append((title, overlay_bgr))

    def ratio(count: int) -> float:
        if checked_count == 0:
            return 0.0
        return count / checked_count

    return {
        "resource_id": resource_id,
        "label": label,
        "source_word": row["source_word"],
        "participant": participant,
        "view": row["view"],
        "raw_frame_count": row["frame_count"],
        "checked_frame_count": checked_count,
        "pose_ratio": f"{ratio(pose_count):.4f}",
        "any_hand_ratio": f"{ratio(any_hand_count):.4f}",
        "both_hand_ratio": f"{ratio(both_hand_count):.4f}",
        "left_hand_ratio": f"{ratio(left_hand_count):.4f}",
        "right_hand_ratio": f"{ratio(right_hand_count):.4f}",
        "avg_hand_count": f"{(total_hand_count / checked_count) if checked_count else 0.0:.4f}",
        "frame_dir": str(frame_dir),
    }


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出 CSV。
    """
    if not rows:
        print("[警告] 没有统计结果可写出")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出样本检测统计：{output_path}")


def write_overall_json(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出总体统计 JSON。
    """
    if not rows:
        return

    def avg_float(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return float(sum(values) / len(values)) if values else 0.0

    overall = {
        "sample_count": len(rows),
        "avg_pose_ratio": round(avg_float("pose_ratio"), 4),
        "avg_any_hand_ratio": round(avg_float("any_hand_ratio"), 4),
        "avg_both_hand_ratio": round(avg_float("both_hand_ratio"), 4),
        "avg_left_hand_ratio": round(avg_float("left_hand_ratio"), 4),
        "avg_right_hand_ratio": round(avg_float("right_hand_ratio"), 4),
        "avg_hand_count": round(avg_float("avg_hand_count"), 4),
        "decision_hint": {
            "good": "pose_ratio >= 0.90 且 any_hand_ratio >= 0.70，基本可以直接使用 Pose + Hands 特征",
            "warning": "如果 any_hand_ratio 明显偏低，需要检查图片尺度、手部遮挡或降低检测阈值",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出总体统计：{output_path}")
    print("\n========== 总体统计 ==========")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples_csv",
        required=True,
        help="小数据集 samples.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="检测结果输出目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=40,
        help="最多检查多少个样本，先不要全量跑，确认没问题再加大",
    )
    parser.add_argument(
        "--frames_per_sample",
        type=int,
        default=6,
        help="每个样本均匀抽取多少帧检查",
    )
    parser.add_argument(
        "--max_overlay_images",
        type=int,
        default=36,
        help="最多生成多少张预览图",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe 检测阈值",
    )

    args = parser.parse_args()

    samples_csv = Path(args.samples_csv)
    output_dir = Path(args.output_dir)

    rows = read_samples(samples_csv)
    rows = [row for row in rows if row.get("status") == "ok"]
    rows = rows[:args.max_samples]

    overlay_images: List[Tuple[str, np.ndarray]] = []
    result_rows: List[Dict[str, object]] = []

    print("========== 开始检查 MediaPipe 可见性 ==========")
    print(f"[信息] 样本索引：{samples_csv}")
    print(f"[信息] 输出目录：{output_dir}")
    print(f"[信息] 检查样本数：{len(rows)}")
    print(f"[信息] 每个样本抽帧数：{args.frames_per_sample}")

    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
    ) as pose_model, MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as hands_model:

        for index, row in enumerate(rows, start=1):
            summary = summarize_sample(
                row=row,
                pose_model=pose_model,
                hands_model=hands_model,
                frames_per_sample=args.frames_per_sample,
                overlay_images=overlay_images,
                max_overlay_images=args.max_overlay_images,
            )
            result_rows.append(summary)

            print(
                f"[{index}/{len(rows)}] "
                f"{summary['label']} {summary['participant']} "
                f"pose={summary['pose_ratio']} "
                f"any_hand={summary['any_hand_ratio']} "
                f"both_hand={summary['both_hand_ratio']} "
                f"avg_hand={summary['avg_hand_count']}"
            )

    write_csv(output_dir / "mediapipe_visibility_by_sample.csv", result_rows)
    write_overall_json(output_dir / "mediapipe_visibility_overall.json", result_rows)
    build_overlay_contact_sheet(
        overlay_images=overlay_images,
        output_path=output_dir / "mediapipe_overlay_preview.jpg",
    )


if __name__ == "__main__":
    main()