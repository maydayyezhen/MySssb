from pathlib import Path
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    from src.config.gesture_config import (
        WINDOW_SIZE,
        MODEL_FILE_NAME,
        LABEL_MAP_FILE_NAME,
        SWAP_HANDEDNESS,
        SWAP_POSE_LR,
    )
    from src.utils.hand_features import extract_arm_pose_frame_parts, build_arm_pose_sample
except ImportError:
    from config.gesture_config import (
        WINDOW_SIZE,
        MODEL_FILE_NAME,
        LABEL_MAP_FILE_NAME,
        SWAP_HANDEDNESS,
        SWAP_POSE_LR,
    )
    from utils.hand_features import extract_arm_pose_frame_parts, build_arm_pose_sample


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / MODEL_FILE_NAME
DEFAULT_LABEL_MAP_PATH = ARTIFACTS_DIR / LABEL_MAP_FILE_NAME


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
        window_size: int = WINDOW_SIZE,
        confidence_threshold: float = 0.7,
        max_missing_frames: int = 10,
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames

        runtime_snapshot = {
            "modelVersionName": "default",
            "modelPath": "",
            "labelMapPath": "",
            "usingPublishedModel": False,
        }

        if model_path is None or label_map_path is None:
            from src.utils.runtime_model_registry import (
                get_runtime_model_paths,
                get_runtime_model_snapshot,
            )

            runtime_model_path, runtime_label_map_path = get_runtime_model_paths()
            runtime_snapshot = get_runtime_model_snapshot()

            if model_path is None:
                model_path = runtime_model_path if runtime_model_path else DEFAULT_MODEL_PATH

            if label_map_path is None:
                label_map_path = runtime_label_map_path if runtime_label_map_path else DEFAULT_LABEL_MAP_PATH

        model_path = Path(model_path)
        label_map_path = Path(label_map_path)

        self.model_version_name = runtime_snapshot.get("modelVersionName", "default")
        self.model_path = str(model_path)
        self.label_map_path = str(label_map_path)
        self.using_published_model = bool(runtime_snapshot.get("usingPublishedModel", False))

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")

        if not label_map_path.exists():
            raise FileNotFoundError(f"标签映射文件不存在：{label_map_path}")

        self.model = tf.keras.models.load_model(str(model_path))
        _, self.reverse_label_map = load_label_map(str(label_map_path))

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # 滑动窗口缓存，每项是一帧中间特征
        self.frame_parts_window = deque(maxlen=self.window_size)

        # 运行时状态
        self.missing_frames = 0
        self.predicted_label = "Waiting..."
        self.predicted_conf = 0.0

    def close(self):
        """释放 MediaPipe 资源。"""
        self.hands.close()
        self.pose.close()

    def reset(self):
        """清空窗口和当前识别状态。"""
        self.frame_parts_window.clear()
        self.missing_frames = 0
        self.predicted_label = "Waiting..."
        self.predicted_conf = 0.0

    def _draw_landmarks(self, frame, hand_results, pose_results):
        """在画面上绘制手部骨架 + 当前模型实际使用的 Pose 点位与连线。"""

        # 1. 画 Hands 骨架
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        # 2. 画 Pose 子集：左右肩、左右肘、左右腕
        if pose_results is None or pose_results.pose_landmarks is None:
            return

        pose_landmarks = pose_results.pose_landmarks.landmark
        PoseLandmark = self.mp_pose.PoseLandmark

        if not SWAP_POSE_LR:
            used_points = {
                "left_shoulder": PoseLandmark.LEFT_SHOULDER,
                "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
                "left_elbow": PoseLandmark.LEFT_ELBOW,
                "right_elbow": PoseLandmark.RIGHT_ELBOW,
                "left_wrist": PoseLandmark.LEFT_WRIST,
                "right_wrist": PoseLandmark.RIGHT_WRIST,
            }
        else:
            used_points = {
                "left_shoulder": PoseLandmark.RIGHT_SHOULDER,
                "right_shoulder": PoseLandmark.LEFT_SHOULDER,
                "left_elbow": PoseLandmark.RIGHT_ELBOW,
                "right_elbow": PoseLandmark.LEFT_ELBOW,
                "left_wrist": PoseLandmark.RIGHT_WRIST,
                "right_wrist": PoseLandmark.LEFT_WRIST,
            }

        point_map = {}

        frame_h, frame_w = frame.shape[:2]

        # 提取点坐标
        for name, enum_value in used_points.items():
            lm = pose_landmarks[enum_value.value]

            # visibility 太低就不画
            if getattr(lm, "visibility", 1.0) < 0.5:
                continue

            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            point_map[name] = (x, y)

        if "left_shoulder" in point_map and "right_shoulder" in point_map:
            shoulder_center = (
                int((point_map["left_shoulder"][0] + point_map["right_shoulder"][0]) / 2),
                int((point_map["left_shoulder"][1] + point_map["right_shoulder"][1]) / 2),
            )
            cv2.circle(frame, shoulder_center, 5, (255, 255, 0), -1)
            cv2.putText(
                frame,
                "shoulder_center",
                (shoulder_center[0] + 8, shoulder_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

        # 3. 连线
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
        ]

        for start_name, end_name in connections:
            if start_name in point_map and end_name in point_map:
                cv2.line(
                    frame,
                    point_map[start_name],
                    point_map[end_name],
                    (0, 255, 255),  # 黄色线
                    3
                )

        # 4. 画点
        point_colors = {
            "left_shoulder": (255, 0, 0),  # 蓝
            "right_shoulder": (255, 0, 0),
            "left_elbow": (0, 255, 0),  # 绿
            "right_elbow": (0, 255, 0),
            "left_wrist": (0, 0, 255),  # 红
            "right_wrist": (0, 0, 255),
        }

        for name, point in point_map.items():
            cv2.circle(frame, point, 6, point_colors.get(name, (255, 255, 255)), -1)
            cv2.putText(
                frame,
                name,
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    def _extract_debug_landmarks(self, hand_results):
        """提取调试用手部 landmarks。"""
        landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                one_hand_landmarks = []
                for lm in hand_landmarks.landmark:
                    one_hand_landmarks.append({
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                    })
                landmarks.append(one_hand_landmarks)
        return landmarks

    def process_frame(self, frame, draw_landmarks: bool = False):
        """处理单帧图像。

        :param frame: OpenCV BGR 图像
        :param draw_landmarks: 是否在 frame 上绘制骨架
        :return: 当前帧对应的识别结果字典
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        if draw_landmarks:
            self._draw_landmarks(frame, hand_results, pose_results)

        landmarks = self._extract_debug_landmarks(hand_results)

        has_valid_hand = False
        status = "no_hand"

        frame_parts = extract_arm_pose_frame_parts(
            hand_results,
            pose_results,
            self.mp_pose,
            swap_handedness=SWAP_HANDEDNESS
        )

        if frame_parts is not None:
            has_valid_hand = True
            self.missing_frames = 0

            self.frame_parts_window.append(frame_parts)

            if len(self.frame_parts_window) == self.window_size:
                sample = build_arm_pose_sample(list(self.frame_parts_window))  # (30, 166)
                sample = np.expand_dims(sample, axis=0)                        # (1, 30, 166)

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
            "model_version_name": self.model_version_name,
            "model_path": self.model_path,
            "label_map_path": self.label_map_path,
            "using_published_model": self.using_published_model,
        }