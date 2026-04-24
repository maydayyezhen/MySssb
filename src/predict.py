from pathlib import Path
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    from src.utils.hand_features import extract_two_hand_frame_parts, build_two_hand_sample
except ImportError:
    from utils.hand_features import extract_two_hand_frame_parts, build_two_hand_sample


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "gesture_cnn_twohand.keras"
DEFAULT_LABEL_MAP_PATH = ARTIFACTS_DIR / "label_map_twohand.json"



def load_label_map(label_map_path: str):
    """加载标签映射，并构造反向映射。"""
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    reverse_label_map = {v: k for k, v in label_map.items()}
    return label_map, reverse_label_map


class GesturePredictSession:
    def __init__(
        self,
        model_path=None,
        label_map_path=None,
        window_size: int = 30,
        confidence_threshold: float = 0.7,
        max_missing_frames: int = 10,
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames

        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        label_map_path = Path(label_map_path) if label_map_path else DEFAULT_LABEL_MAP_PATH

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        if not label_map_path.exists():
            raise FileNotFoundError(f"标签映射文件不存在：{label_map_path}")

        self.model = tf.keras.models.load_model(str(model_path))
        _, self.reverse_label_map = load_label_map(str(label_map_path))

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
        self.frame_parts_window = deque(maxlen=self.window_size)

        # ========= 运行时状态 =========
        self.missing_frames = 0
        self.predicted_label = "Waiting..."
        self.predicted_conf = 0.0

    def close(self):
        """释放 MediaPipe 资源。"""
        self.hands.close()

    def reset(self):
        """清空窗口和当前识别状态。"""
        self.frame_parts_window.clear()
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
        landmarks = []

        # 仅用于调试展示：把检测到的所有手部 landmarks 返回出去
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                one_hand_landmarks = []
                for lm in hand_landmarks.landmark:
                    one_hand_landmarks.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    })
                landmarks.append(one_hand_landmarks)

        # 构造双手单帧中间特征
        frame_parts = extract_two_hand_frame_parts(
            results,
            swap_handedness=False
        )

        if frame_parts is not None:
            has_valid_hand = True
            self.missing_frames = 0

            # 新版：窗口里存的是每一帧的双手中间特征
            self.frame_parts_window.append(frame_parts)

            # 窗口满 30 帧后开始预测
            if len(self.frame_parts_window) == self.window_size:
                sample = build_two_hand_sample(
                    list(self.frame_parts_window)
                )  # (30, 162)

                sample = np.expand_dims(sample, axis=0)  # (1, 30, 162)

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
                status = "no_hand"
            else:
                status = "missing_hand"

        return {
            "status": status,
            "label": self.predicted_label,
            "confidence": self.predicted_conf,
            "has_valid_hand": has_valid_hand,
            "valid_frames": len(self.frame_parts_window),
            "window_size": self.window_size,
            "landmarks": landmarks,
        }