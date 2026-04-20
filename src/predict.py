import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from utils.hand_features import build_frame_feature, extract_palm_center, extract_palm_scale



def load_label_map(label_map_path: str):
    """加载标签映射，并构造反向映射。"""
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    reverse_label_map = {v: k for k, v in label_map.items()}
    return label_map, reverse_label_map


class GesturePredictSession:
    """手势实时预测会话。负责维护模型、MediaPipe 和滑动窗口状态。"""

    def __init__(
        self,
        model_path: str = "artifacts/gesture_cnn.keras",
        label_map_path: str = "artifacts/label_map.json",
        window_size: int = 30,
        confidence_threshold: float = 0.7,
        max_missing_frames: int = 10,
    ):
        # ========= 配置 =========
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames

        # ========= 模型与标签 =========
        self.model = tf.keras.models.load_model(model_path)
        _, self.reverse_label_map = load_label_map(label_map_path)

        # ========= MediaPipe =========
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ========= 滑动窗口缓存 =========
        self.feature_window = deque(maxlen=self.window_size)   # 每帧 78 维
        self.center_window = deque(maxlen=self.window_size)    # 每帧掌心中心 (2,)
        self.scale_window = deque(maxlen=self.window_size)     # 每帧掌部尺度

        # ========= 运行时状态 =========
        self.missing_frames = 0
        self.predicted_label = "Waiting..."
        self.predicted_conf = 0.0

    def close(self):
        """释放 MediaPipe 资源。"""
        self.hands.close()

    def reset(self):
        """清空窗口和当前识别状态。"""
        self.feature_window.clear()
        self.center_window.clear()
        self.scale_window.clear()
        self.missing_frames = 0
        self.predicted_label = "Waiting..."
        self.predicted_conf = 0.0

    def _draw_landmarks(self, frame, results):
        """在画面上绘制手部骨架。"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
                )

    def process_frame(self, frame, draw_landmarks: bool = False):
        """
        处理单帧图像。
        :param frame: OpenCV BGR 图像
        :param draw_landmarks: 是否在 frame 上绘制骨架
        :return: 当前帧对应的识别结果字典
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if draw_landmarks:
            self._draw_landmarks(frame, results)

        has_valid_hand = False
        status = "no_hand"

        # 只有同时拿到 image/world landmarks 才算有效帧
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            has_valid_hand = True
            self.missing_frames = 0

            first_hand_landmarks = results.multi_hand_landmarks[0]
            first_hand_world_landmarks = results.multi_hand_world_landmarks[0]

            # 单帧 78 维主特征
            frame_feature = build_frame_feature(first_hand_world_landmarks)
            palm_center = extract_palm_center(first_hand_landmarks)
            palm_scale = extract_palm_scale(first_hand_landmarks)

            self.feature_window.append(frame_feature)
            self.center_window.append(palm_center)
            self.scale_window.append(palm_scale)

            # 窗口满 30 帧后开始预测
            if len(self.feature_window) == self.window_size:
                base_sample = np.array(self.feature_window, dtype=np.float32)   # (30, 78)
                centers = np.array(self.center_window, dtype=np.float32)        # (30, 2)

                start_motion_scale = max(float(self.scale_window[0]), 1e-6)
                motion = (centers - centers[0]) / start_motion_scale            # (30, 2)

                sample = np.concatenate([base_sample, motion], axis=1).astype(np.float32)  # (30, 80)
                sample = np.expand_dims(sample, axis=0)  # (1, 30, 80)

                probs = self.model.predict(sample, verbose=0)[0]
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])

                if conf >= self.confidence_threshold:
                    self.predicted_label = self.reverse_label_map[pred_id]
                    self.predicted_conf = conf
                else:
                    self.predicted_label = "Uncertain"
                    self.predicted_conf = conf

                status = "predicted"
            else:
                status = "warming_up"

        else:
            self.missing_frames += 1

            # 连续太久没检测到手，就清空窗口，防止旧窗口一直残留
            if self.missing_frames >= self.max_missing_frames:
                self.reset()
            else:
                status = "missing_hand"

        return {
            "status": status,
            "label": self.predicted_label,
            "confidence": self.predicted_conf,
            "has_valid_hand": has_valid_hand,
            "valid_frames": len(self.feature_window),
            "window_size": self.window_size,
        }