import cv2
import mediapipe as mp
import numpy as np
from utils.hand_features import build_frame_feature


def read_video_feature(path):

    # 初始化 MediaPipe Hands 方案
    mp_hands = mp.solutions.hands

    # 创建手部检测器
    hands = mp_hands.Hands(
        static_image_mode=False,         # False 表示视频流模式
        max_num_hands=2,                 # 最多检测两只手
        min_detection_confidence=0.5,    # 最小检测置信度
        min_tracking_confidence=0.5      # 最小跟踪置信度
    )

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        hands.close()
        raise ValueError(f"无法打开视频文件：{path}")

    window_size = 30  # 固定窗口长度：30帧
    frame_in_window = 0  # 当前窗口已经走过多少原始帧
    current_frames = []  # 当前窗口中成功提取到的单帧特征
    samples = []  # 保存这个视频中提取出的所有有效样本

    while True:
        # 读取一帧画面
        success, frame = cap.read()
        if not success:
            print("读取画面失败。")
            break

        frame_in_window += 1

        # OpenCV 默认是 BGR，MediaPipe 需要 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 进行手部检测
        results = hands.process(rgb_frame)

        if results.multi_hand_world_landmarks:

            # 提特征，改用 world landmarks
            first_hand_world_landmarks = results.multi_hand_world_landmarks[0]

            frame_feature = build_frame_feature(first_hand_world_landmarks)

            # 当前窗口只保存最终单帧特征
            current_frames.append(frame_feature)

        if frame_in_window == window_size:
            if len(current_frames) == window_size:
                sample = np.array(current_frames, dtype=np.float32)
                samples.append(sample)

            # 重置窗口，开始下一轮30帧
            frame_in_window = 0
            current_frames = []

    # 释放资源
    cap.release()
    hands.close()

    if len(samples) == 0:
        return np.empty((0, 30, 78), dtype=np.float32)

    return np.array(samples, dtype=np.float32)