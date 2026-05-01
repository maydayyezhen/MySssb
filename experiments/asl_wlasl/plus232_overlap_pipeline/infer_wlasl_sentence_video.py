# -*- coding: utf-8 -*-
"""
对 WLASL 拼接句子视频进行滑窗识别。

输入：
- 一个 mp4 句子视频，例如 friend_meet_today_trimmed.mp4
- WLASL 训练好的 20f plus 模型
- labels.json

输出：
- dense_predictions.csv：每个滑窗的 Top1/Top2/Top3 预测
- segments.json：后处理后的词段结果
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_wlasl_features_20f_plus import (
    read_all_frames,
    process_frame,
    extract_frame_feature,
)

from build_wlasl_features_20f_base166 import (
    extract_frame_feature_base166,
)

from build_wlasl_features_20f_static194 import (
    extract_frame_feature_static194,
)


def load_json(path: Path) -> Dict:
    """读取 JSON。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    """写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    """写出 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{path}")


def parse_expected(expected: str) -> List[str]:
    """解析期望 gloss 序列。"""
    return [
        item.strip().lower()
        for item in expected.replace("，", ",").split(",")
        if item.strip()
    ]


def get_video_info(video_path: Path) -> Dict[str, object]:
    """读取视频基础信息。"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "opened": False,
            "frame_width": 0,
            "frame_height": 0,
            "fps": 0.0,
            "total_frame_count": 0,
        }

    info = {
        "opened": True,
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "total_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    cap.release()
    return info


def build_result_cache(video_path: Path):
    """
    读取视频并缓存每帧 MediaPipe 结果。

    注意：
    后续每个滑窗都会从缓存结果中重新构造 20 帧特征；
    每个滑窗内部 dynamic_delta 的第一帧会重置为 0，
    这样与训练样本的构造方式保持一致。
    """
    frames = read_all_frames(video_path)

    if not frames:
        raise RuntimeError(f"视频为空或无法读取：{video_path}")

    results = []
    hand_flags = []
    pose_flags = []

    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for index, frame in enumerate(frames):
            result = process_frame(holistic, frame)
            results.append(result)

            has_pose = result.pose_landmarks is not None
            has_left = result.left_hand_landmarks is not None
            has_right = result.right_hand_landmarks is not None

            pose_flags.append(bool(has_pose))
            hand_flags.append(bool(has_left or has_right))

            if (index + 1) % 20 == 0:
                print(f"[MediaPipe] 已处理帧：{index + 1}/{len(frames)}")

    return frames, results, pose_flags, hand_flags


def build_window_feature(
    results: List[object],
    start: int,
    window_size: int,
    feature_dim: int,
) -> np.ndarray:
    """
    根据缓存的 MediaPipe 结果构建单个滑窗特征。

    feature_dim:
    - 232：使用 plus 特征
    - 166：使用 base166 特征
    """
    features = []
    previous_dynamic_source = None

    for offset in range(window_size):
        frame_index = start + offset

        if frame_index >= len(results):
            frame_index = len(results) - 1

        result = results[frame_index]

        if feature_dim == 232:
            feature, previous_dynamic_source = extract_frame_feature(
                result,
                previous_dynamic_source,
            )
        elif feature_dim == 194:
            feature = extract_frame_feature_static194(result)
        elif feature_dim == 166:
            feature = extract_frame_feature_base166(result)
        else:
            raise RuntimeError(f"不支持的 feature_dim：{feature_dim}")

        features.append(feature)

    return np.array(features, dtype=np.float32)


def build_windows(
    results: List[object],
    window_size: int,
    stride: int,
    feature_dim: int,
) -> tuple[np.ndarray, List[Dict[str, int]]]:
    """
    构建所有滑窗。
    """
    if len(results) <= window_size:
        starts = [0]
    else:
        starts = list(range(0, len(results) - window_size + 1, stride))

        last_start = len(results) - window_size
        if starts[-1] != last_start:
            starts.append(last_start)

    X_list = []
    window_meta = []

    for window_index, start in enumerate(starts):
        X = build_window_feature(
            results=results,
            start=start,
            window_size=window_size,
            feature_dim=feature_dim,
        )

        X_list.append(X)

        end = min(start + window_size - 1, len(results) - 1)
        center = (start + end) // 2

        window_meta.append({
            "window_index": window_index,
            "start_frame": start,
            "end_frame": end,
            "center_frame": center,
        })

    return np.stack(X_list, axis=0).astype(np.float32), window_meta


def make_dense_rows(
    probs: np.ndarray,
    window_meta: List[Dict[str, int]],
    labels: List[str],
    confidence_threshold: float,
    margin_threshold: float,
) -> List[Dict[str, object]]:
    """
    生成逐滑窗预测结果。
    """
    rows = []

    for i, prob in enumerate(probs):
        sorted_ids = np.argsort(prob)[::-1]

        top1_id = int(sorted_ids[0])
        top2_id = int(sorted_ids[1]) if len(sorted_ids) > 1 else top1_id
        top3_ids = sorted_ids[:min(3, len(sorted_ids))]

        top1_prob = float(prob[top1_id])
        top2_prob = float(prob[top2_id])
        margin = top1_prob - top2_prob

        accepted = (
            top1_prob >= confidence_threshold
            and margin >= margin_threshold
        )

        meta = window_meta[i]

        row = {
            "window_index": meta["window_index"],
            "start_frame": meta["start_frame"],
            "end_frame": meta["end_frame"],
            "center_frame": meta["center_frame"],
            "top1": labels[top1_id],
            "top1_prob": round(top1_prob, 6),
            "top2": labels[top2_id],
            "top2_prob": round(top2_prob, 6),
            "margin": round(margin, 6),
            "accepted": int(accepted),
        }

        for rank, label_id in enumerate(top3_ids, start=1):
            row[f"top{rank}_label"] = labels[int(label_id)]
            row[f"top{rank}_prob"] = round(float(prob[int(label_id)]), 6)

        rows.append(row)

    return rows


def build_raw_segments(
    dense_rows: List[Dict[str, object]],
    same_label_merge_gap: int,
) -> List[Dict[str, object]]:
    """
    将连续滑窗预测合并成原始词段。

    规则：
    - 只使用 accepted=1 的窗口
    - 相邻窗口 label 相同，且时间间隔不大，就合并
    """
    accepted_rows = [
        row for row in dense_rows
        if int(row["accepted"]) == 1
    ]

    if not accepted_rows:
        return []

    segments = []
    current = None

    for row in accepted_rows:
        label = str(row["top1"])
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        center = int(row["center_frame"])
        confidence = float(row["top1_prob"])

        if current is None:
            current = {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "center_frame": center,
                "window_count": 1,
                "confidence_sum": confidence,
                "max_confidence": confidence,
                "max_center_frame": center,
            }
            continue

        same_label = label == current["label"]
        close_enough = start <= int(current["end_frame"]) + same_label_merge_gap

        if same_label and close_enough:
            current["end_frame"] = max(int(current["end_frame"]), end)
            current["window_count"] = int(current["window_count"]) + 1
            current["confidence_sum"] = float(current["confidence_sum"]) + confidence

            if confidence > float(current["max_confidence"]):
                current["max_confidence"] = confidence
                current["max_center_frame"] = center

            current["center_frame"] = (
                int(current["start_frame"]) + int(current["end_frame"])
            ) // 2
        else:
            segments.append(finalize_segment(current))
            current = {
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "center_frame": center,
                "window_count": 1,
                "confidence_sum": confidence,
                "max_confidence": confidence,
                "max_center_frame": center,
            }

    if current is not None:
        segments.append(finalize_segment(current))

    return segments


def finalize_segment(segment: Dict[str, object]) -> Dict[str, object]:
    """补充词段平均置信度。"""
    window_count = int(segment["window_count"])

    segment["avg_confidence"] = round(
        float(segment["confidence_sum"]) / max(1, window_count),
        6,
    )
    segment["max_confidence"] = round(float(segment["max_confidence"]), 6)

    segment.pop("confidence_sum", None)

    return segment


def filter_segments(
    segments: List[Dict[str, object]],
    min_segment_windows: int,
    min_segment_avg_confidence: float,
    min_segment_max_confidence: float,
    blank_label: str = "blank",
) -> List[Dict[str, object]]:
    """
    过滤太短、置信度太低或 blank 的词段。

    规则：
    1. blank 不进入最终 gloss 序列
    2. 太短的词段丢弃
    3. 平均置信度和峰值置信度都低的词段丢弃
    """
    result = []

    for segment in segments:
        label = str(segment["label"])

        if label == blank_label:
            continue

        if int(segment["window_count"]) < min_segment_windows:
            continue

        avg_conf = float(segment["avg_confidence"])
        max_conf = float(segment["max_confidence"])

        if avg_conf < min_segment_avg_confidence and max_conf < min_segment_max_confidence:
            continue

        result.append(segment)

    return result


def get_segment_length(segment: Dict[str, object]) -> int:
    """
    计算词段长度。
    """
    return max(
        1,
        int(segment["end_frame"]) - int(segment["start_frame"]) + 1,
    )


def get_overlap_length(
    a: Dict[str, object],
    b: Dict[str, object],
) -> int:
    """
    计算两个词段的重叠帧数。
    """
    start = max(int(a["start_frame"]), int(b["start_frame"]))
    end = min(int(a["end_frame"]), int(b["end_frame"]))

    if end < start:
        return 0

    return end - start + 1


def get_segment_score(segment: Dict[str, object]) -> float:
    """
    词段评分。

    不只看 max_confidence，因为短暂误报也可能 max 很高。
    这里更偏向：
    - 平均置信度高
    - 持续窗口数多

    score = avg_confidence * log(1 + window_count)
    """
    avg_confidence = float(segment.get("avg_confidence", 0.0))
    window_count = max(1, int(segment.get("window_count", 1)))

    return avg_confidence * float(np.log1p(window_count))


def merge_two_same_label_segments(
    a: Dict[str, object],
    b: Dict[str, object],
) -> Dict[str, object]:
    """
    合并两个同标签词段。
    """
    a_count = max(1, int(a.get("window_count", 1)))
    b_count = max(1, int(b.get("window_count", 1)))

    total_count = a_count + b_count

    a_avg = float(a.get("avg_confidence", 0.0))
    b_avg = float(b.get("avg_confidence", 0.0))

    merged = dict(a)

    merged["start_frame"] = min(int(a["start_frame"]), int(b["start_frame"]))
    merged["end_frame"] = max(int(a["end_frame"]), int(b["end_frame"]))
    merged["center_frame"] = (
        int(merged["start_frame"]) + int(merged["end_frame"])
    ) // 2
    merged["window_count"] = total_count
    merged["avg_confidence"] = round(
        (a_avg * a_count + b_avg * b_count) / total_count,
        6,
    )
    merged["max_confidence"] = round(
        max(
            float(a.get("max_confidence", 0.0)),
            float(b.get("max_confidence", 0.0)),
        ),
        6,
    )

    return merged


def merge_close_same_label_segments(
    segments: List[Dict[str, object]],
    merge_gap: int,
) -> List[Dict[str, object]]:
    """
    合并时间相邻或重叠的同标签词段。

    解决：
    today learn -> today learn learn
    这类同一个词被切成两个连续词段的问题。
    """
    if not segments:
        return []

    ordered = sorted(
        segments,
        key=lambda item: (int(item["start_frame"]), int(item["end_frame"])),
    )

    merged: List[Dict[str, object]] = []

    for segment in ordered:
        if not merged:
            merged.append(dict(segment))
            continue

        last = merged[-1]

        same_label = str(last["label"]) == str(segment["label"])
        close_enough = int(segment["start_frame"]) <= int(last["end_frame"]) + merge_gap

        if same_label and close_enough:
            merged[-1] = merge_two_same_label_segments(last, segment)
        else:
            merged.append(dict(segment))

    return merged


def is_heavily_overlapped(
    candidate: Dict[str, object],
    existing: Dict[str, object],
    overlap_ratio_threshold: float,
    containment_ratio_threshold: float,
) -> bool:
    """
    判断 candidate 是否应被已有词段抑制。

    规则：
    1. 如果两个词段高度重叠，认为它们竞争同一段时间。
    2. 如果 candidate 大部分被 existing 覆盖，也认为是边界误报。
    """
    overlap = get_overlap_length(candidate, existing)

    if overlap <= 0:
        return False

    candidate_len = get_segment_length(candidate)
    existing_len = get_segment_length(existing)

    overlap_over_candidate = overlap / candidate_len
    overlap_over_existing = overlap / existing_len
    overlap_over_min = overlap / min(candidate_len, existing_len)

    if overlap_over_min >= overlap_ratio_threshold:
        return True

    if overlap_over_candidate >= containment_ratio_threshold:
        return True

    if overlap_over_existing >= containment_ratio_threshold:
        return True

    return False


def nms_segments(
    segments: List[Dict[str, object]],
    suppress_radius: int,
) -> List[Dict[str, object]]:
    """
    区间重叠版 NMS。

    相比旧版只看 center_frame 距离，这版额外处理：
    1. 同标签相邻词段合并
    2. 不同标签但区间高度重叠时，只保留 score 更高者
    3. 短暂插入误报如果大部分被真实词段覆盖，会被删除

    适合处理：
    friend meet today -> friend school meet go want today
    这类边界插入误报。
    """
    if not segments:
        return []

    # 第一步：先合并相邻同标签词段，避免 learn learn 这类重复。
    merged_segments = merge_close_same_label_segments(
        segments=segments,
        merge_gap=max(4, suppress_radius),
    )

    # 第二步：按稳定分数排序，而不是只按 max_confidence。
    candidates = sorted(
        merged_segments,
        key=get_segment_score,
        reverse=True,
    )

    selected: List[Dict[str, object]] = []

    overlap_ratio_threshold = 0.55
    containment_ratio_threshold = 0.70

    for candidate in candidates:
        candidate_center = int(candidate["center_frame"])

        should_suppress = False

        for existing in selected:
            existing_center = int(existing["center_frame"])

            # 旧规则：中心太近，仍然视为竞争。
            center_too_close = abs(candidate_center - existing_center) <= suppress_radius

            # 新规则：区间明显重叠，也视为竞争。
            overlap_too_much = is_heavily_overlapped(
                candidate=candidate,
                existing=existing,
                overlap_ratio_threshold=overlap_ratio_threshold,
                containment_ratio_threshold=containment_ratio_threshold,
            )

            if center_too_close or overlap_too_much:
                should_suppress = True
                break

        if not should_suppress:
            selected.append(candidate)

    # 第三步：按时间顺序输出。
    selected = sorted(
        selected,
        key=lambda item: int(item["start_frame"]),
    )

    # 第四步：NMS 后再合并一次同标签相邻段。
    selected = merge_close_same_label_segments(
        segments=selected,
        merge_gap=max(4, suppress_radius),
    )

    return sorted(
        selected,
        key=lambda item: int(item["start_frame"]),
    )


def compare_sequence(expected: List[str], detected: List[str]) -> Dict[str, object]:
    """
    简单比较期望序列和检测序列。
    """
    return {
        "expected_sequence": expected,
        "detected_sequence": detected,
        "exact_match": expected == detected if expected else None,
        "expected_text": " ".join(expected),
        "detected_text": " ".join(detected),
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", required=True)
    parser.add_argument(
        "--feature_dir",
        default="D:/datasets/WLASL-mini/features_20f_plus",
    )
    parser.add_argument(
        "--model_dir",
        default="D:/datasets/WLASL-mini/models_20f_plus",
    )
    parser.add_argument(
        "--output_dir",
        default="D:/datasets/WLASL-mini/demo_infer",
    )
    parser.add_argument("--expected", default="")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.45)
    parser.add_argument("--margin_threshold", type=float, default=0.05)
    parser.add_argument("--min_segment_windows", type=int, default=2)
    parser.add_argument("--min_segment_avg_confidence", type=float, default=0.45)
    parser.add_argument("--min_segment_max_confidence", type=float, default=0.55)
    parser.add_argument("--same_label_merge_gap", type=int, default=8)
    parser.add_argument("--nms_suppress_radius", type=int, default=6)

    args = parser.parse_args()

    video_path = Path(args.video_path)
    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_payload = load_json(feature_dir / "labels.json")
    labels = labels_payload["labels"]

    model_path = model_dir / "best_wlasl_20f_plus_classifier.keras"

    if not model_path.exists():
        model_path = model_dir / "wlasl_20f_plus_classifier.keras"

    print("========== WLASL 句子视频推理 ==========")
    print(f"[信息] video_path：{video_path}")
    print(f"[信息] model_path：{model_path}")
    print(f"[信息] labels：{labels}")
    print(f"[信息] output_dir：{output_dir}")
    print(f"[信息] window_size：{args.window_size}")
    print(f"[信息] stride：{args.stride}")

    model = tf.keras.models.load_model(model_path)

    model_input_shape = model.input_shape
    feature_dim = int(model_input_shape[-1])

    print(f"[信息] model input shape：{model_input_shape}")
    print(f"[信息] feature_dim：{feature_dim}")

    video_info = get_video_info(video_path)
    frames, results, pose_flags, hand_flags = build_result_cache(video_path)

    X, window_meta = build_windows(
        results=results,
        window_size=args.window_size,
        stride=args.stride,
        feature_dim=feature_dim,
    )

    print(f"[信息] 视频帧数：{len(frames)}")
    print(f"[信息] X windows shape：{X.shape}")
    print(f"[信息] pose_ratio：{sum(pose_flags) / len(pose_flags):.4f}")
    print(f"[信息] any_hand_ratio：{sum(hand_flags) / len(hand_flags):.4f}")

    probs = model.predict(X, verbose=0)

    dense_rows = make_dense_rows(
        probs=probs,
        window_meta=window_meta,
        labels=labels,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )

    raw_segments = build_raw_segments(
        dense_rows=dense_rows,
        same_label_merge_gap=args.same_label_merge_gap,
    )

    filtered_segments = filter_segments(
        segments=raw_segments,
        min_segment_windows=args.min_segment_windows,
        min_segment_avg_confidence=args.min_segment_avg_confidence,
        min_segment_max_confidence=args.min_segment_max_confidence,
        blank_label="blank",
    )

    final_segments = nms_segments(
        segments=filtered_segments,
        suppress_radius=args.nms_suppress_radius,
    )

    detected_sequence = [
        str(segment["label"])
        for segment in final_segments
    ]

    expected_sequence = parse_expected(args.expected)

    comparison = compare_sequence(
        expected=expected_sequence,
        detected=detected_sequence,
    )

    dense_csv_path = output_dir / f"{video_path.stem}_dense_predictions.csv"
    segments_json_path = output_dir / f"{video_path.stem}_segments.json"

    write_csv(
        dense_csv_path,
        dense_rows,
        [
            "window_index",
            "start_frame",
            "end_frame",
            "center_frame",
            "top1",
            "top1_prob",
            "top2",
            "top2_prob",
            "margin",
            "accepted",
            "top1_label",
            "top1_prob",
            "top2_label",
            "top2_prob",
            "top3_label",
            "top3_prob",
        ],
    )

    save_json(
        segments_json_path,
        {
            "video_path": str(video_path),
            "model_path": str(model_path),
            "video_info": video_info,
            "window_size": args.window_size,
            "stride": args.stride,
            "thresholds": {
                "confidence_threshold": args.confidence_threshold,
                "margin_threshold": args.margin_threshold,
                "min_segment_windows": args.min_segment_windows,
                "min_segment_avg_confidence": args.min_segment_avg_confidence,
                "min_segment_max_confidence": args.min_segment_max_confidence,
                "same_label_merge_gap": args.same_label_merge_gap,
                "nms_suppress_radius": args.nms_suppress_radius,
            },
            "pose_ratio": round(sum(pose_flags) / len(pose_flags), 6),
            "any_hand_ratio": round(sum(hand_flags) / len(hand_flags), 6),
            "raw_segments": raw_segments,
            "filtered_segments": filtered_segments,
            "segments": final_segments,
            **comparison,
        },
    )

    print("\n========== 推理完成 ==========")
    print(f"期望序列：{comparison['expected_text']}")
    print(f"检测序列：{comparison['detected_text']}")
    print(f"是否完全匹配：{comparison['exact_match']}")
    print(f"逐窗口预测：{dense_csv_path}")
    print(f"词段结果：{segments_json_path}")

    print("\n========== 最终词段 ==========")

    if not final_segments:
        print("无")
    else:
        for segment in final_segments:
            print(
                f"{segment['label']} "
                f"[{segment['start_frame']}, {segment['end_frame']}] "
                f"windows={segment['window_count']} "
                f"avg={segment['avg_confidence']} "
                f"max={segment['max_confidence']}"
            )


if __name__ == "__main__":
    main()
