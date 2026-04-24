import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

from utils.hand_features import extract_two_hand_frame_parts, build_two_hand_sample

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_next_sample_index(save_dir: Path) -> int:
    """获取下一个样本编号。"""
    existing_files = sorted(save_dir.glob("sample_*.npy"))
    if not existing_files:
        return 1

    max_index = 0
    for file in existing_files:
        stem = file.stem  # sample_001
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            max_index = max(max_index, int(parts[1]))

    return max_index + 1


def draw_info(frame, label_name: str, saved_count: int, state: str, collected_count: int, window_size: int):
    """在画面左上角绘制状态信息。"""
    lines = [
        f"Label: {label_name}",
        f"Saved Samples: {saved_count}",
        f"State: {state}",
        f"Frames: {collected_count}/{window_size}",
        "S: Start  Y: Save  N: Discard  Q: Quit"
    ]

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        y += 30


def main():
    """摄像头采集数据集主函数。"""

    # ========= 可改参数 =========
    window_size = 30
    camera_index = 0
    save_root = PROJECT_ROOT / "data_processed_twohand"

    label_name = input("请输入当前采集的手势标签：").strip()
    if not label_name:
        print("标签不能为空。")
        return
    # ==========================

    save_dir = save_root / label_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 已保存样本数
    saved_count = len(list(save_dir.glob("sample_*.npy")))

    # MediaPipe Hands 初始化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        hands.close()
        raise ValueError(f"无法打开摄像头：{camera_index}")

    # 状态机
    # idle: 待机
    # collecting: 采集中
    # confirm: 等待确认保存
    state = "idle"

    pending_sample = None
    current_frame_parts = []

    while True:
        success, frame = cap.read()
        if not success:
            print("读取摄像头画面失败。")
            break

        # 镜像显示，更符合摄像头习惯
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 绘制骨架，方便观察
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        # 采集中：只有同时检测到 image/world landmarks 才累计一帧
        if state == "collecting":
            frame_parts = extract_two_hand_frame_parts(results, swap_handedness=False)

            if frame_parts is not None:
                current_frame_parts.append(frame_parts)

                if len(current_frame_parts) == window_size:
                    pending_sample = build_two_hand_sample(current_frame_parts)
                    state = "confirm"

        # 状态文字
        display_state = {
            "idle": "Idle",
            "collecting": "Collecting",
            "confirm": "Confirm Save"
        }[state]

        draw_info(
            frame=frame,
            label_name=label_name,
            saved_count=saved_count,
            state=display_state,
            collected_count=len(current_frame_parts) if state == "collecting" else (window_size if state == "confirm" else 0),
            window_size=window_size
        )

        if state == "confirm":
            cv2.putText(
                frame,
                "Sample Ready! Press Y to save, N to discard.",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Create Dataset", frame)

        key = cv2.waitKey(1) & 0xFF

        # 退出
        if key == ord("q"):
            break

        # 待机 -> 开始采集
        elif key == ord("s") and state == "idle":
            current_frame_parts = []
            pending_sample = None
            state = "collecting"
            print(f"开始采集标签 [{label_name}] 的一个样本。")

        # 确认保存
        elif key == ord("y") and state == "confirm":
            sample_index = get_next_sample_index(save_dir)
            save_path = save_dir / f"sample_{sample_index:03d}.npy"
            np.save(save_path, pending_sample)

            saved_count += 1
            print(f"样本已保存：{save_path}")

            current_frame_parts = []
            pending_sample = None
            state = "idle"

        # 丢弃
        elif key == ord("n") and state == "confirm":
            print("当前样本已丢弃。")
            current_frame_parts = []
            pending_sample = None
            state = "idle"

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()