import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from utils.hand_features import (
    build_frame_feature,
    extract_palm_center,
    extract_palm_scale,
)


def load_label_map(label_map_path: str):
    """加载标签映射，并构造反向映射。"""
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    reverse_label_map = {v: k for k, v in label_map.items()}
    return label_map, reverse_label_map


def main():
    # ========= 可改参数 =========
    model_path = "artifacts/gesture_cnn.keras"
    label_map_path = "artifacts/label_map.json"
    camera_index = 0
    window_size = 30
    confidence_threshold = 0.7
    max_missing_frames = 10
    # ==========================

    # 1. 加载模型和标签映射
    model = tf.keras.models.load_model(model_path)
    _, reverse_label_map = load_label_map(label_map_path)

    # 2. 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 3. 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        hands.close()
        raise ValueError(f"无法打开摄像头：{camera_index}")

    # 4. 滑动窗口缓存
    feature_window = deque(maxlen=window_size)   # 每帧 78 维
    center_window = deque(maxlen=window_size)    # 每帧掌心中心 (2,)
    scale_window = deque(maxlen=window_size)     # 每帧掌部尺度
    missing_frames = 0

    predicted_label = "Waiting..."
    predicted_conf = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("读取摄像头画面失败。")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        has_valid_hand = False

        # 画骨架
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        # 只有同时拿到 image/world landmarks 才算有效帧
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            has_valid_hand = True
            missing_frames = 0

            first_hand_landmarks = results.multi_hand_landmarks[0]
            first_hand_world_landmarks = results.multi_hand_world_landmarks[0]

            # 单帧 78 维主特征
            frame_feature = build_frame_feature(first_hand_world_landmarks)
            palm_center = extract_palm_center(first_hand_landmarks)
            palm_scale = extract_palm_scale(first_hand_landmarks)

            feature_window.append(frame_feature)
            center_window.append(palm_center)
            scale_window.append(palm_scale)

            # 窗口满 30 帧后开始预测
            if len(feature_window) == window_size:
                base_sample = np.array(feature_window, dtype=np.float32)   # (30, 78)
                centers = np.array(center_window, dtype=np.float32)        # (30, 2)

                start_motion_scale = max(float(scale_window[0]), 1e-6)
                motion = (centers - centers[0]) / start_motion_scale       # (30, 2)

                sample = np.concatenate([base_sample, motion], axis=1).astype(np.float32)  # (30, 80)
                sample = np.expand_dims(sample, axis=0)  # (1, 30, 80)

                probs = model.predict(sample, verbose=0)[0]
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])

                if conf >= confidence_threshold:
                    predicted_label = reverse_label_map[pred_id]
                    predicted_conf = conf
                else:
                    predicted_label = "Uncertain"
                    predicted_conf = conf

        else:
            missing_frames += 1

            # 连续太久没检测到手，就清空窗口，防止旧窗口一直残留
            if missing_frames >= max_missing_frames:
                feature_window.clear()
                center_window.clear()
                scale_window.clear()
                predicted_label = "Waiting..."
                predicted_conf = 0.0

        # 画状态文字
        cv2.putText(
            frame,
            f"Prediction: {predicted_label}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Confidence: {predicted_conf:.3f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Valid Frames In Window: {len(feature_window)}/{window_size}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        if not has_valid_hand:
            cv2.putText(
                frame,
                "No valid hand detected",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )

        cv2.putText(
            frame,
            "Q: Quit",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Real-time Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()