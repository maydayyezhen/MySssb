import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config.gesture_config import (
    WINDOW_SIZE,
    TARGET_FPS,
    SAMPLE_INTERVAL_SEC,
    DATA_DIR_NAME,
    SWAP_HANDEDNESS,
    SWAP_POSE_LR,
)
from utils.hand_features import extract_arm_pose_frame_parts, build_arm_pose_sample

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_next_sample_index(save_dir: Path) -> int:
    """获取下一个样本编号。"""
    existing_files = sorted(save_dir.glob("sample_*.npy"))
    if not existing_files:
        return 1

    max_index = 0
    for file in existing_files:
        stem = file.stem
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            max_index = max(max_index, int(parts[1]))

    return max_index + 1

def draw_pose_subset(frame, pose_results, mp_pose):
    """绘制当前模型实际使用的 Pose 点位与连线：左右肩、左右肘、左右腕。"""
    if pose_results is None or pose_results.pose_landmarks is None:
        return

    pose_landmarks = pose_results.pose_landmarks.landmark
    PoseLandmark = mp_pose.PoseLandmark

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

    for name, enum_value in used_points.items():
        lm = pose_landmarks[enum_value.value]
        if getattr(lm, "visibility", 1.0) < 0.5:
            continue

        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        point_map[name] = (x, y)

    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]

    for start_name, end_name in connections:
        if start_name in point_map and end_name in point_map:
            cv2.line(frame, point_map[start_name], point_map[end_name], (0, 255, 255), 3)

    point_colors = {
        "left_shoulder": (255, 0, 0),
        "right_shoulder": (255, 0, 0),
        "left_elbow": (0, 255, 0),
        "right_elbow": (0, 255, 0),
        "left_wrist": (0, 0, 255),
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

def draw_info(frame, label_name: str, saved_count: int, state: str, collected_count: int, window_size: int):
    """在画面左上角绘制状态信息。"""
    lines = [
        f"Label: {label_name}",
        f"Saved Samples: {saved_count}",
        f"State: {state}",
        f"Frames: {collected_count}/{window_size}",
        f"FPS: {TARGET_FPS}, Window: {window_size / TARGET_FPS:.1f}s",
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
    window_size = WINDOW_SIZE
    camera_index = 0
    save_root = PROJECT_ROOT / DATA_DIR_NAME
    prepare_seconds = 3.0

    label_name = input("请输入当前采集的手势标签：").strip()
    if not label_name:
        print("标签不能为空。")
        return
    # ==========================

    save_dir = save_root / label_name
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # MediaPipe Pose 初始化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        hands.close()
        pose.close()
        raise ValueError(f"无法打开摄像头：{camera_index}")

    state = "idle"
    pending_sample = None
    current_frame_parts = []
    last_sample_time = 0.0
    prepare_start_time = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            print("读取摄像头画面失败。")
            break

        # 镜像显示，更符合本地摄像头采集习惯
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        # 绘制手部骨架，方便观察
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        draw_pose_subset(frame, pose_results, mp_pose)

        # 准备阶段：按 S 后先倒计时 3 秒，让采集者摆好动作
        if state == "preparing":
            now = time.monotonic()
            elapsed = now - prepare_start_time
            remaining = max(0.0, prepare_seconds - elapsed)

            cv2.putText(
                frame,
                f"Get Ready: {remaining:.1f}s",
                (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )

            if elapsed >= prepare_seconds:
                current_frame_parts = []
                pending_sample = None
                last_sample_time = 0.0
                state = "collecting"
                print("准备结束，开始采集。")

        # 采集中：按 10FPS 抽帧，而不是摄像头来一帧就收一帧
        if state == "collecting":
            now = time.monotonic()

            if now - last_sample_time >= SAMPLE_INTERVAL_SEC:
                frame_parts = extract_arm_pose_frame_parts(
                    hand_results,
                    pose_results,
                    mp_pose,
                    swap_handedness=SWAP_HANDEDNESS
                )

                if frame_parts is not None:
                    current_frame_parts.append(frame_parts)
                    last_sample_time = now

                    if len(current_frame_parts) == window_size:
                        pending_sample = build_arm_pose_sample(current_frame_parts)
                        state = "confirm"

        display_state = {
            "idle": "Idle",
            "preparing": "Preparing",
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
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Create Dataset", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


        elif key == ord("s") and state == "idle":

            current_frame_parts = []

            pending_sample = None

            last_sample_time = 0.0

            prepare_start_time = time.monotonic()

            state = "preparing"

            print(f"准备采集标签 [{label_name}] 的一个样本，{prepare_seconds:.0f} 秒后开始。")

        elif key == ord("y") and state == "confirm":
            sample_index = get_next_sample_index(save_dir)
            save_path = save_dir / f"sample_{sample_index:03d}.npy"
            np.save(save_path, pending_sample)

            saved_count += 1
            print(f"样本已保存：{save_path}")

            state = "idle"
            pending_sample = None
            current_frame_parts = []
            last_sample_time = 0.0
            prepare_start_time = 0.0

        elif key == ord("n") and state == "confirm":
            print("当前样本已丢弃。")
            state = "idle"
            pending_sample = None
            current_frame_parts = []
            last_sample_time = 0.0
            prepare_start_time = 0.0
        elif key == ord("n") and state == "preparing":
            print("准备阶段已取消。")
            current_frame_parts = []
            pending_sample = None
            last_sample_time = 0.0
            prepare_start_time = 0.0
            state = "idle"

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()


if __name__ == "__main__":
    main()