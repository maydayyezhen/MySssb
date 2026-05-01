# -*- coding: utf-8 -*-
"""
预览 MediaPipe Hands 左右手标注结果。

作用：
1. 从指定样本目录读取若干帧图片
2. 运行 MediaPipe Pose + Hands
3. 在图上标出：
   - Hands 输出的 Left / Right
   - Pose 的 LEFT_WRIST / RIGHT_WRIST
4. 用来判断 NationalCSL-DP 是否需要交换 handedness
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


MP_HANDS = mp.solutions.hands
MP_POSE = mp.solutions.pose
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles


def read_image_bgr_unicode(image_path: Path) -> Optional[np.ndarray]:
    """
    读取可能包含中文路径的图片。
    """
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        if image_bytes.size == 0:
            return None
        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[错误] 图片读取失败：{image_path}，原因：{e}")
        return None


def save_image_bgr_unicode(output_path: Path, image_bgr: np.ndarray) -> None:
    """
    保存可能包含中文路径的图片。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")

    encoded.tofile(str(output_path))


def list_image_frames(frame_dir: Path) -> List[Path]:
    """
    获取样本目录中的图片帧。
    """
    return sorted([
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])


def sample_evenly(items: List[Path], max_count: int) -> List[Path]:
    """
    均匀抽取若干帧。
    """
    if len(items) <= max_count:
        return items

    if max_count <= 1:
        return [items[0]]

    indices = [
        round(i * (len(items) - 1) / (max_count - 1))
        for i in range(max_count)
    ]

    return [items[index] for index in indices]


def put_label(
    image_bgr: np.ndarray,
    text: str,
    point_xy: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    """
    在图上写英文标签。
    """
    x, y = point_xy

    cv2.putText(
        image_bgr,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_handedness_on_frame(
    image_bgr: np.ndarray,
    pose_model,
    hands_model,
) -> np.ndarray:
    """
    在单帧图上绘制 Pose / Hands 左右手标注。
    """
    output = image_bgr.copy()
    h, w = output.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    pose_result = pose_model.process(image_rgb)
    hands_result = hands_model.process(image_rgb)
    image_rgb.flags.writeable = True

    # 画 Pose，并标出 Pose 的左右手腕
    if pose_result.pose_landmarks:
        MP_DRAWING.draw_landmarks(
            output,
            pose_result.pose_landmarks,
            MP_POSE.POSE_CONNECTIONS,
            landmark_drawing_spec=MP_DRAWING_STYLES.get_default_pose_landmarks_style(),
        )

        left_wrist = pose_result.pose_landmarks.landmark[
            MP_POSE.PoseLandmark.LEFT_WRIST.value
        ]
        right_wrist = pose_result.pose_landmarks.landmark[
            MP_POSE.PoseLandmark.RIGHT_WRIST.value
        ]

        left_xy = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_xy = (int(right_wrist.x * w), int(right_wrist.y * h))

        put_label(output, "Pose LEFT_WRIST", (left_xy[0] + 8, left_xy[1] - 8), (255, 0, 0))
        put_label(output, "Pose RIGHT_WRIST", (right_xy[0] + 8, right_xy[1] - 8), (0, 0, 255))

    # 画 Hands，并标出 Hands 的 handedness
    if hands_result.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
            MP_DRAWING.draw_landmarks(
                output,
                hand_landmarks,
                MP_HANDS.HAND_CONNECTIONS,
                MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
                MP_DRAWING_STYLES.get_default_hand_connections_style(),
            )

            wrist = hand_landmarks.landmark[0]
            wrist_xy = (int(wrist.x * w), int(wrist.y * h))

            hand_label = "Unknown"
            hand_score = 0.0

            if hands_result.multi_handedness and hand_index < len(hands_result.multi_handedness):
                classification = hands_result.multi_handedness[hand_index].classification[0]
                hand_label = classification.label
                hand_score = classification.score

            put_label(
                output,
                f"Hands {hand_label} {hand_score:.2f}",
                (wrist_xy[0] + 8, wrist_xy[1] + 24),
                (0, 140, 255),
            )

    return output


def build_sheet(
    frame_paths: List[Path],
    output_path: Path,
    thumb_width: int,
    thumb_height: int,
) -> None:
    """
    生成标注拼图。
    """
    cols = 3
    gap = 10
    title_height = 30

    rows = int(np.ceil(len(frame_paths) / cols))
    sheet_width = cols * thumb_width + (cols + 1) * gap
    sheet_height = rows * (thumb_height + title_height + gap) + gap

    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)

    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose_model, MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands_model:

        for index, frame_path in enumerate(frame_paths):
            image_bgr = read_image_bgr_unicode(frame_path)

            if image_bgr is None:
                continue

            overlay = draw_handedness_on_frame(
                image_bgr=image_bgr,
                pose_model=pose_model,
                hands_model=hands_model,
            )

            h, w = overlay.shape[:2]
            scale = min(thumb_width / w, thumb_height / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)

            canvas = np.full((thumb_height, thumb_width, 3), 255, dtype=np.uint8)
            px = (thumb_width - new_w) // 2
            py = (thumb_height - new_h) // 2
            canvas[py:py + new_h, px:px + new_w] = resized

            row = index // cols
            col = index % cols

            x = gap + col * (thumb_width + gap)
            y = gap + row * (thumb_height + title_height + gap)

            cv2.putText(
                sheet,
                frame_path.name,
                (x, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            sheet[y + title_height:y + title_height + thumb_height, x:x + thumb_width] = canvas

    save_image_bgr_unicode(output_path, sheet)
    print(f"[完成] 已生成左右手标注图：{output_path}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_dir",
        required=True,
        help="某个样本的帧目录，例如 D:/datasets/HearBridge-NationalCSL-mini/raw_frames/你__1925/Participant_10",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出图片路径",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=9,
        help="最多抽取多少帧",
    )
    parser.add_argument(
        "--thumb_width",
        type=int,
        default=360,
    )
    parser.add_argument(
        "--thumb_height",
        type=int,
        default=420,
    )

    args = parser.parse_args()

    frame_paths = list_image_frames(Path(args.frame_dir))
    selected = sample_evenly(frame_paths, args.max_frames)

    if not selected:
        print(f"[错误] 没有找到图片帧：{args.frame_dir}")
        return

    build_sheet(
        frame_paths=selected,
        output_path=Path(args.output_path),
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
    )


if __name__ == "__main__":
    main()