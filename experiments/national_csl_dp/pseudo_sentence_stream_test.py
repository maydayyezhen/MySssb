# -*- coding: utf-8 -*-
"""
NationalCSL-DP 伪连续句子滚动窗口测试脚本。

作用：
1. 从 features_20f / features_20f_blank 的 X.npy 中取若干孤立词特征序列
2. 按指定句子顺序拼接成伪连续特征流
3. 使用固定窗口做滚动预测，模拟“吃满帧后每帧输出一次”
4. 对 dense prediction stream 做 uncertain / blank 过滤
5. 对词段做高置信筛选和 NMS，压制过渡误报
6. 支持句尾 tail_frames，用于模拟用户做完最后一个词后的短暂停顿
7. 输出逐窗口预测 CSV 和分段结果 JSON

推荐当前主线：
features_20f_blank + models_20f_blank
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tensorflow import keras


def load_json(path: Path) -> Dict:
    """
    读取 JSON 文件。
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    """
    保存 JSON 文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写出 JSON：{path}")


def read_sample_index(sample_index_csv: Path) -> List[Dict[str, str]]:
    """
    读取 sample_index.csv。

    只保留：
    1. status=ok
    2. label_id 非空

    注意：
    返回顺序必须与 X.npy / y.npy 中的样本顺序一致。
    """
    with sample_index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    valid_rows = [
        row for row in rows
        if row.get("status") == "ok" and str(row.get("label_id", "")).strip() != ""
    ]

    return valid_rows


def write_csv(output_path: Path, rows: List[Dict[str, object]]) -> None:
    """
    写出 CSV 文件。
    """
    if not rows:
        print("[警告] 没有 CSV 数据可写出")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] 已写出 CSV：{output_path}")


def apply_normalizer(X: np.ndarray, normalizer_path: Path) -> np.ndarray:
    """
    应用训练阶段保存的标准化参数。
    """
    data = np.load(normalizer_path)
    mean = data["mean"]
    std = data["std"]

    std = np.where(std < 1e-6, 1.0, std)

    return ((X - mean) / std).astype(np.float32)


def parse_sentence(sentence: str) -> List[str]:
    """
    解析命令行传入的句子标签。

    示例：
    --sentence "我们,需要,帮助"
    """
    labels = [
        item.strip()
        for item in sentence.replace("，", ",").split(",")
    ]
    labels = [item for item in labels if item]

    if not labels:
        raise ValueError("sentence 为空，请传入类似：我们,需要,帮助")

    return labels


def find_sample_index(
    rows: List[Dict[str, str]],
    label: str,
    participant: str,
    used_resource_ids: Optional[set] = None,
) -> int:
    """
    根据 label + participant 查找样本索引。

    对于“学习”这种多个 resource_id 合并成同一标签的情况：
    1. 优先避免重复使用同一个 resource_id
    2. 如果都用过，则退回第一个匹配项
    """
    matched_indices = []

    for index, row in enumerate(rows):
        if row["label"] == label and row["participant"] == participant:
            matched_indices.append(index)

    if not matched_indices:
        raise RuntimeError(f"找不到样本：label={label}, participant={participant}")

    if used_resource_ids is not None:
        for index in matched_indices:
            resource_id = rows[index]["resource_id"]
            if resource_id not in used_resource_ids:
                used_resource_ids.add(resource_id)
                return index

    return matched_indices[0]


def build_pseudo_stream(
    X: np.ndarray,
    rows: List[Dict[str, str]],
    sentence_labels: List[str],
    participant: str,
    gap_frames: int,
    gap_mode: str,
    tail_frames: int,
    tail_mode: str,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    构造伪连续特征流。

    输入：
    - 每个词样本本身是 shape=(T, D)，当前通常是 (20, 166)
    - 词之间可以插入 gap_frames 个间隔帧
    - 句尾可以插入 tail_frames 个尾部帧，用于模拟最后一个动作后的短暂停顿

    gap_mode:
    - zero：插入全 0 特征帧
    - repeat_last：重复上一个词的最后一帧
    - none：不插入间隔

    tail_mode:
    - zero：句尾插入全 0 特征帧
    - repeat_last：句尾重复最后一个词最后一帧
    - none：不插入尾部帧
    """
    stream_parts = []
    source_segments = []

    used_resource_ids = set()
    current_frame = 0

    feature_dim = X.shape[-1]

    for word_index, label in enumerate(sentence_labels):
        sample_index = find_sample_index(
            rows=rows,
            label=label,
            participant=participant,
            used_resource_ids=used_resource_ids,
        )

        word_features = X[sample_index]
        row = rows[sample_index]

        start_frame = current_frame
        end_frame = current_frame + len(word_features) - 1

        stream_parts.append(word_features)

        source_segments.append({
            "type": "word",
            "label": label,
            "resource_id": row["resource_id"],
            "source_word": row["source_word"],
            "participant": row["participant"],
            "sample_index": sample_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": int(len(word_features)),
        })

        current_frame += len(word_features)

        if word_index < len(sentence_labels) - 1 and gap_frames > 0 and gap_mode != "none":
            if gap_mode == "zero":
                gap = np.zeros((gap_frames, feature_dim), dtype=np.float32)
            elif gap_mode == "repeat_last":
                gap = np.repeat(word_features[-1:, :], gap_frames, axis=0)
            else:
                raise ValueError(f"不支持的 gap_mode：{gap_mode}")

            gap_start = current_frame
            gap_end = current_frame + gap_frames - 1

            stream_parts.append(gap)

            source_segments.append({
                "type": "gap",
                "label": "blank",
                "resource_id": "",
                "source_word": "",
                "participant": participant,
                "sample_index": "",
                "start_frame": gap_start,
                "end_frame": gap_end,
                "frame_count": int(gap_frames),
                "gap_mode": gap_mode,
            })

            current_frame += gap_frames

    if tail_frames > 0 and tail_mode != "none":
        if not stream_parts:
            raise RuntimeError("stream_parts 为空，无法生成 tail")

        if tail_mode == "zero":
            tail = np.zeros((tail_frames, feature_dim), dtype=np.float32)
        elif tail_mode == "repeat_last":
            tail = np.repeat(stream_parts[-1][-1:, :], tail_frames, axis=0)
        else:
            raise ValueError(f"不支持的 tail_mode：{tail_mode}")

        tail_start = current_frame
        tail_end = current_frame + tail_frames - 1

        stream_parts.append(tail)

        source_segments.append({
            "type": "tail",
            "label": "blank",
            "resource_id": "",
            "source_word": "",
            "participant": participant,
            "sample_index": "",
            "start_frame": tail_start,
            "end_frame": tail_end,
            "frame_count": int(tail_frames),
            "tail_mode": tail_mode,
        })

        current_frame += tail_frames

    stream = np.concatenate(stream_parts, axis=0).astype(np.float32)

    return stream, source_segments


def rolling_predict(
    stream: np.ndarray,
    model,
    normalizer_path: Path,
    id_to_label: Dict[int, str],
    window_size: int,
    top_k: int,
) -> List[Dict[str, object]]:
    """
    对伪连续特征流做滚动窗口预测。

    每个输出表示：
    以 window_end 为结尾的最近 window_size 帧窗口的预测结果。
    """
    if len(stream) < window_size:
        raise RuntimeError(
            f"特征流长度不足：stream_len={len(stream)}, window_size={window_size}"
        )

    windows = []
    meta = []

    for end_frame in range(window_size - 1, len(stream)):
        start_frame = end_frame - window_size + 1
        window = stream[start_frame:end_frame + 1]

        windows.append(window)
        meta.append((start_frame, end_frame))

    X_windows = np.stack(windows, axis=0).astype(np.float32)
    X_windows = apply_normalizer(X_windows, normalizer_path)

    prob = model.predict(X_windows, verbose=0)

    topk_ids = np.argsort(prob, axis=1)[:, -top_k:][:, ::-1]
    topk_probs = np.take_along_axis(prob, topk_ids, axis=1)

    prediction_rows = []

    for i, (start_frame, end_frame) in enumerate(meta):
        top_ids = [int(x) for x in topk_ids[i].tolist()]
        top_probs = [float(x) for x in topk_probs[i].tolist()]
        top_labels = [id_to_label[class_id] for class_id in top_ids]

        prediction_rows.append({
            "step": i + 1,
            "window_start": start_frame,
            "window_end": end_frame,

            "top1_label": top_labels[0],
            "top1_id": top_ids[0],
            "top1_prob": f"{top_probs[0]:.6f}",

            "top2_label": top_labels[1] if len(top_labels) > 1 else "",
            "top2_id": top_ids[1] if len(top_ids) > 1 else "",
            "top2_prob": f"{top_probs[1]:.6f}" if len(top_probs) > 1 else "",

            "top3_label": top_labels[2] if len(top_labels) > 2 else "",
            "top3_id": top_ids[2] if len(top_ids) > 2 else "",
            "top3_prob": f"{top_probs[2]:.6f}" if len(top_probs) > 2 else "",
        })

    return prediction_rows


def normalize_prediction_row(
    row: Dict[str, object],
    strong_confidence_threshold: float,
    margin_threshold: float,
    blank_label: str = "blank",
) -> Tuple[str, float, str]:
    """
    将单个滚动窗口预测结果归一化为 label/confidence/reason。

    规则：
    1. top1 是 blank：直接视为 blank
    2. top1 置信度低于阈值：视为 blank
    3. top1 和 top2 差距太小：视为 blank
    4. 其他情况才接收 top1
    """
    top1_label = str(row["top1_label"])
    top1_prob = float(row["top1_prob"])

    top2_prob_text = str(row.get("top2_prob", "")).strip()
    top2_prob = float(top2_prob_text) if top2_prob_text else 0.0

    if top1_label == blank_label:
        return blank_label, top1_prob, "top1_blank"

    if top1_prob < strong_confidence_threshold:
        return blank_label, top1_prob, "low_confidence"

    if top1_prob - top2_prob < margin_threshold:
        return blank_label, top1_prob, "small_margin"

    return top1_label, top1_prob, "accepted"


class PredictionSegmenter:
    """
    将逐窗口预测流转换为词段。

    说明：
    这里的 label 已经经过 normalize_prediction_row 处理。
    低置信、小 margin、blank 都会被转为 blank。
    """

    def __init__(
        self,
        stable_frames: int,
        blank_end_frames: int,
        same_label_merge_gap: int,
    ):
        """
        初始化词段提取器。

        :param stable_frames: 同一标签连续出现多少次才开启词段
        :param blank_end_frames: 连续多少个 blank 后结束当前词段
        :param same_label_merge_gap: 相同标签间隔多少帧以内进行合并
        """
        self.stable_frames = stable_frames
        self.blank_end_frames = blank_end_frames
        self.same_label_merge_gap = same_label_merge_gap

        self.candidate_label = None
        self.candidate_count = 0

        self.blank_count = 0
        self.active_segment = None
        self.finished_segments = []

    def _finalize_active(self, end_frame: int) -> None:
        """
        结束当前活动词段。
        """
        if self.active_segment is None:
            return

        self.active_segment["end_frame"] = end_frame

        confidences = self.active_segment.get("confidences", [])

        self.active_segment["avg_confidence"] = round(
            float(np.mean(confidences)),
            6,
        ) if confidences else 0.0

        self.active_segment["max_confidence"] = round(
            float(np.max(confidences)),
            6,
        ) if confidences else 0.0

        self.active_segment["duration"] = (
            int(self.active_segment["end_frame"])
            - int(self.active_segment["start_frame"])
            + 1
        )

        self.finished_segments.append(self.active_segment)
        self.active_segment = None

    def update(self, window_end: int, label: str, confidence: float) -> None:
        """
        输入一个滚动窗口预测结果。
        """
        if label == "blank":
            self.blank_count += 1
            self.candidate_label = None
            self.candidate_count = 0

            if self.active_segment is not None and self.blank_count >= self.blank_end_frames:
                self._finalize_active(window_end - self.blank_count + 1)

            return

        self.blank_count = 0

        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1

        if self.candidate_count < self.stable_frames:
            return

        stable_start = window_end - self.stable_frames + 1

        if self.active_segment is None:
            self.active_segment = {
                "label": label,
                "start_frame": stable_start,
                "end_frame": window_end,
                "confidences": [confidence],
            }
            return

        if self.active_segment["label"] == label:
            self.active_segment["end_frame"] = window_end
            self.active_segment["confidences"].append(confidence)
            return

        self._finalize_active(stable_start - 1)

        self.active_segment = {
            "label": label,
            "start_frame": stable_start,
            "end_frame": window_end,
            "confidences": [confidence],
        }

    def finish(self, final_frame: int) -> List[Dict[str, object]]:
        """
        输入结束后关闭当前活动词段，并合并短间隔重复标签。
        """
        self._finalize_active(final_frame)

        return merge_same_label_segments(
            self.finished_segments,
            same_label_merge_gap=self.same_label_merge_gap,
        )


def merge_same_label_segments(
    segments: List[Dict[str, object]],
    same_label_merge_gap: int,
) -> List[Dict[str, object]]:
    """
    合并短间隔内重复出现的同一标签。
    """
    if not segments:
        return []

    merged = [segments[0]]

    for segment in segments[1:]:
        last = merged[-1]

        gap = int(segment["start_frame"]) - int(last["end_frame"]) - 1

        if segment["label"] == last["label"] and gap <= same_label_merge_gap:
            last["end_frame"] = segment["end_frame"]
            last["duration"] = int(last["end_frame"]) - int(last["start_frame"]) + 1

            old_conf = list(last.get("confidences", []))
            new_conf = list(segment.get("confidences", []))
            all_conf = old_conf + new_conf

            last["confidences"] = all_conf
            last["avg_confidence"] = round(float(np.mean(all_conf)), 6) if all_conf else 0.0
            last["max_confidence"] = round(float(np.max(all_conf)), 6) if all_conf else 0.0
        else:
            merged.append(segment)

    for segment in merged:
        segment.pop("confidences", None)

    return merged


def get_segment_center(segment: Dict[str, object]) -> float:
    """
    获取词段中心帧。
    """
    return (int(segment["start_frame"]) + int(segment["end_frame"])) / 2.0


def filter_and_nms_segments(
    segments: List[Dict[str, object]],
    min_segment_avg_confidence: float,
    min_segment_max_confidence: float,
    suppress_radius: int,
    blank_label: str = "blank",
    min_segment_duration: int = 3,
    short_segment_max_confidence: float = 0.90,
) -> List[Dict[str, object]]:
    """
    对词段做最终过滤和峰值 NMS。

    规则：
    1. 去掉 blank
    2. 去掉平均置信度和峰值置信度都不够的词段
    3. 去掉短促低置信误报词段：
       - duration < min_segment_duration
       - 且 max_confidence < short_segment_max_confidence
    4. 对时间上太近的多个词段，只保留 max_confidence 更高的那个

    典型作用：
    - 过滤类似 “老师 [20,21] max=0.833989” 这种短促误报
    - 保留 “我们 [19,24] max=0.993615” 这种短但高置信的真实词段
    """
    candidates = []

    for segment in segments:
        label = str(segment["label"])

        # 1. blank 不进入最终 gloss 序列
        if label == blank_label:
            continue

        avg_confidence = float(segment.get("avg_confidence", 0.0))
        max_confidence = float(segment.get("max_confidence", 0.0))

        start_frame = int(segment.get("start_frame", 0))
        end_frame = int(segment.get("end_frame", start_frame))
        duration = int(segment.get("duration", end_frame - start_frame + 1))

        # 2. 置信度整体太弱的词段丢弃
        if avg_confidence < min_segment_avg_confidence and max_confidence < min_segment_max_confidence:
            continue

        # 3. 短促且峰值不够高的词段，通常是误报
        if duration < min_segment_duration and max_confidence < short_segment_max_confidence:
            continue

        copied = dict(segment)
        copied["duration"] = duration
        copied["center_frame"] = get_segment_center(segment)
        candidates.append(copied)

    # 4. 先按峰值置信度从高到低选择
    candidates.sort(key=lambda item: float(item["max_confidence"]), reverse=True)

    selected = []

    for candidate in candidates:
        center = float(candidate["center_frame"])

        too_close = False

        for existing in selected:
            existing_center = float(existing["center_frame"])

            if abs(center - existing_center) <= suppress_radius:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

    # 5. 最后按时间顺序输出
    selected.sort(key=lambda item: int(item["start_frame"]))

    for item in selected:
        item.pop("center_frame", None)

    return selected


def segment_predictions(
    prediction_rows: List[Dict[str, object]],
    confidence_threshold: float,
    stable_frames: int,
    blank_end_frames: int,
    same_label_merge_gap: int,
    margin_threshold: float,
) -> List[Dict[str, object]]:
    """
    从滚动窗口预测结果中提取原始词段。

    流程：
    1. 对每个窗口做 uncertain / blank 归一化
    2. 把高置信、margin 足够的标签送入 segmenter
    3. 得到 raw segments
    """
    segmenter = PredictionSegmenter(
        stable_frames=stable_frames,
        blank_end_frames=blank_end_frames,
        same_label_merge_gap=same_label_merge_gap,
    )

    final_frame = 0

    for row in prediction_rows:
        window_end = int(row["window_end"])

        label, confidence, reason = normalize_prediction_row(
            row=row,
            strong_confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            blank_label="blank",
        )

        segmenter.update(
            window_end=window_end,
            label=label,
            confidence=confidence,
        )

        final_frame = window_end

    return segmenter.finish(final_frame=final_frame)


def run_pseudo_sentence_test(
    feature_dir: Path,
    model_dir: Path,
    output_dir: Path,
    sentence: str,
    participant: str,
    model_name: str,
    window_size: int,
    top_k: int,
    gap_frames: int,
    gap_mode: str,
    tail_frames: int,
    tail_mode: str,
    confidence_threshold: float,
    stable_frames: int,
    blank_end_frames: int,
    same_label_merge_gap: int,
    margin_threshold: float,
    min_segment_avg_confidence: float,
    min_segment_max_confidence: float,
    nms_suppress_radius: int,
) -> None:
    """
    执行伪连续句子测试。
    """
    X = np.load(feature_dir / "X.npy").astype(np.float32)

    label_map = load_json(feature_dir / "label_map.json")
    id_to_label = {int(v): k for k, v in label_map.items()}

    rows = read_sample_index(feature_dir / "sample_index.csv")

    if len(rows) != len(X):
        raise RuntimeError(f"rows 数量与 X 不一致：rows={len(rows)}, X={len(X)}")

    sentence_labels = parse_sentence(sentence)

    stream, source_segments = build_pseudo_stream(
        X=X,
        rows=rows,
        sentence_labels=sentence_labels,
        participant=participant,
        gap_frames=gap_frames,
        gap_mode=gap_mode,
        tail_frames=tail_frames,
        tail_mode=tail_mode,
    )

    model_path = model_dir / model_name
    normalizer_path = model_dir / "normalizer.npz"

    model = keras.models.load_model(model_path)

    prediction_rows = rolling_predict(
        stream=stream,
        model=model,
        normalizer_path=normalizer_path,
        id_to_label=id_to_label,
        window_size=window_size,
        top_k=top_k,
    )

    raw_detected_segments = segment_predictions(
        prediction_rows=prediction_rows,
        confidence_threshold=confidence_threshold,
        stable_frames=stable_frames,
        blank_end_frames=blank_end_frames,
        same_label_merge_gap=same_label_merge_gap,
        margin_threshold=margin_threshold,
    )

    detected_segments = filter_and_nms_segments(
        segments=raw_detected_segments,
        min_segment_avg_confidence=min_segment_avg_confidence,
        min_segment_max_confidence=min_segment_max_confidence,
        suppress_radius=nms_suppress_radius,
        blank_label="blank",
    )

    detected_sequence = [
        segment["label"]
        for segment in detected_segments
        if segment["label"] != "blank"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    safe_sentence_name = "_".join(sentence_labels)
    base_name = f"{participant}_{safe_sentence_name}"

    dense_csv_path = output_dir / f"{base_name}_dense_predictions.csv"
    result_json_path = output_dir / f"{base_name}_segments.json"

    write_csv(dense_csv_path, prediction_rows)

    result_payload = {
        "sentence": sentence_labels,
        "participant": participant,
        "stream_shape": list(stream.shape),
        "source_segments": source_segments,
        "settings": {
            "model_path": str(model_path),
            "window_size": window_size,
            "top_k": top_k,
            "gap_frames": gap_frames,
            "gap_mode": gap_mode,
            "tail_frames": tail_frames,
            "tail_mode": tail_mode,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
            "stable_frames": stable_frames,
            "blank_end_frames": blank_end_frames,
            "same_label_merge_gap": same_label_merge_gap,
            "min_segment_avg_confidence": min_segment_avg_confidence,
            "min_segment_max_confidence": min_segment_max_confidence,
            "nms_suppress_radius": nms_suppress_radius,
        },
        "raw_detected_segments": raw_detected_segments,
        "detected_segments": detected_segments,
        "detected_sequence": detected_sequence,
        "expected_sequence": sentence_labels,
        "exact_match": detected_sequence == sentence_labels,
    }

    save_json(result_json_path, result_payload)

    print("\n========== 伪连续句子测试完成 ==========")
    print(f"输入句子：{' '.join(sentence_labels)}")
    print(f"检测序列：{' '.join(detected_sequence) if detected_sequence else '(空)'}")
    print(f"是否完全匹配：{detected_sequence == sentence_labels}")
    print(f"逐窗口预测：{dense_csv_path}")
    print(f"分段结果：{result_json_path}")

    print("\n========== 原始词段 raw_detected_segments ==========")
    if not raw_detected_segments:
        print("(空)")
    else:
        for segment in raw_detected_segments:
            print(
                f"{segment['label']} "
                f"[{segment['start_frame']}, {segment['end_frame']}] "
                f"avg={segment.get('avg_confidence')} "
                f"max={segment.get('max_confidence')}"
            )

    print("\n========== 最终词段 detected_segments ==========")
    if not detected_segments:
        print("(空)")
    else:
        for segment in detected_segments:
            print(
                f"{segment['label']} "
                f"[{segment['start_frame']}, {segment['end_frame']}] "
                f"avg={segment.get('avg_confidence')} "
                f"max={segment.get('max_confidence')}"
            )


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_dir", required=True, help="features_20f 或 features_20f_blank 目录")
    parser.add_argument("--model_dir", required=True, help="models_20f 或 models_20f_blank 目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")

    parser.add_argument("--sentence", required=True, help="逗号分隔词序列，例如：我们,需要,帮助")
    parser.add_argument("--participant", default="Participant_10", help="使用哪个参与者样本")
    parser.add_argument("--model_name", default="best_national_csl_20f_classifier.keras")

    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=3)

    parser.add_argument("--gap_frames", type=int, default=0)
    parser.add_argument("--gap_mode", default="none", choices=["zero", "repeat_last", "none"])

    parser.add_argument("--tail_frames", type=int, default=0)
    parser.add_argument("--tail_mode", default="repeat_last", choices=["zero", "repeat_last", "none"])

    parser.add_argument("--confidence_threshold", type=float, default=0.80)
    parser.add_argument("--margin_threshold", type=float, default=0.15)

    parser.add_argument("--stable_frames", type=int, default=1)
    parser.add_argument("--blank_end_frames", type=int, default=3)
    parser.add_argument("--same_label_merge_gap", type=int, default=8)

    parser.add_argument("--min_segment_avg_confidence", type=float, default=0.75)
    parser.add_argument("--min_segment_max_confidence", type=float, default=0.85)
    parser.add_argument("--nms_suppress_radius", type=int, default=10)

    args = parser.parse_args()

    run_pseudo_sentence_test(
        feature_dir=Path(args.feature_dir),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        sentence=args.sentence,
        participant=args.participant,
        model_name=args.model_name,
        window_size=args.window_size,
        top_k=args.top_k,
        gap_frames=args.gap_frames,
        gap_mode=args.gap_mode,
        tail_frames=args.tail_frames,
        tail_mode=args.tail_mode,
        confidence_threshold=args.confidence_threshold,
        stable_frames=args.stable_frames,
        blank_end_frames=args.blank_end_frames,
        same_label_merge_gap=args.same_label_merge_gap,
        margin_threshold=args.margin_threshold,
        min_segment_avg_confidence=args.min_segment_avg_confidence,
        min_segment_max_confidence=args.min_segment_max_confidence,
        nms_suppress_radius=args.nms_suppress_radius,
    )


if __name__ == "__main__":
    main()