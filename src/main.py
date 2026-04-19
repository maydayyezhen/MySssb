import cv2
import mediapipe as mp
import numpy as np
from utils.hand_features import build_frame_feature


def main():
    """主函数：打开摄像头，实时检测手部关键点并显示。"""

    # 初始化 MediaPipe Hands 方案
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 创建手部检测器
    hands = mp_hands.Hands(
        static_image_mode=False,         # False 表示视频流模式
        max_num_hands=2,                 # 最多检测两只手
        min_detection_confidence=0.5,    # 最小检测置信度
        min_tracking_confidence=0.5      # 最小跟踪置信度
    )

    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备或把 VideoCapture(0) 改成 1 试试。")
        return

    window_size = 30  # 固定窗口长度：30帧
    frame_in_window = 0  # 当前窗口已经走过多少原始帧
    current_frames = []  # 当前窗口中成功提取到的单帧特征

    while True:
        # 读取一帧画面
        success, frame = cap.read()
        if not success:
            print("读取摄像头画面失败。")
            break

        frame_in_window += 1

        # 水平翻转，显示更符合镜像习惯
        frame = cv2.flip(frame, 1)

        # OpenCV 默认是 BGR，MediaPipe 需要 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 进行手部检测
        results = hands.process(rgb_frame)

        # 如果检测到手，则绘制关键点和骨架
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
            # 先画图，还是用图像坐标
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

            # 提特征，改用 world landmarks
            first_hand_world_landmarks = results.multi_hand_world_landmarks[0]

            frame_feature = build_frame_feature(first_hand_world_landmarks)

            # 当前窗口只保存最终单帧特征
            current_frames.append(frame_feature)


        if frame_in_window == window_size:
            if len(current_frames) == window_size:
                sample = np.array(current_frames, dtype=np.float32)
                print("得到一个有效样本：", sample.shape)
                print("第一帧前63维坐标特征：")
                print(sample[0][:63])
                print("第一帧后15维角度余弦特征：")
                print(sample[0][63:])
            else:
                print(f"这一段30帧无效，丢弃。有效帧数：{len(current_frames)}/{window_size}")

            # 重置窗口，开始下一轮30帧
            frame_in_window = 0
            current_frames = []

        # 显示画面
        cv2.imshow("Hand Detection", frame)

        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()