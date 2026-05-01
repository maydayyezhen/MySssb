# -*- coding: utf-8 -*-
"""Raw dataset collection routes."""

import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.word_recognition.config.gesture_config import (
    RAW_PHONE_DATA_DIR_NAME,
    RAW_SAMPLE_FRAME_COUNT,
    SAMPLE_INTERVAL_SEC,
    SWAP_HANDEDNESS,
    SWAP_POSE_LR,
)
from src.word_recognition.utils.mediapipe_raw import extract_raw_mediapipe_frame
from src.word_recognition.utils.raw_dataset_scanner import scan_raw_dataset
from src.word_recognition.utils.raw_dataset_writer import save_raw_sample
from src.word_recognition.realtime.frame_decoder import decode_frontend_frame


router = APIRouter(tags=["dataset-collect"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = PROJECT_ROOT / RAW_PHONE_DATA_DIR_NAME

# Debug windows and per-frame disk writes must be opt-in for service deployments.
DEBUG_DATASET_WINDOW = False
SAVE_DEBUG_INPUT_FRAME = False


@router.get("/dataset/raw/samples")
async def list_raw_samples():
    """Scan raw dataset sample summaries for Spring Boot synchronization."""
    samples = scan_raw_dataset(RAW_DATA_ROOT)
    return {
        "rootDir": str(RAW_DATA_ROOT),
        "total": len(samples),
        "items": samples,
    }


def draw_hand_labels_for_debug(frame, hand_results, mp_hands) -> None:
    """Draw MediaPipe raw and project-used hand labels."""
    if hand_results is None:
        return
    if not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
        return

    frame_h, frame_w = frame.shape[:2]
    hand_count = min(
        len(hand_results.multi_hand_landmarks),
        len(hand_results.multi_handedness),
    )

    for index in range(hand_count):
        hand_landmarks = hand_results.multi_hand_landmarks[index]
        handedness = hand_results.multi_handedness[index].classification[0]

        raw_label = handedness.label
        used_label = raw_label
        if SWAP_HANDEDNESS:
            if raw_label == "Left":
                used_label = "Right"
            elif raw_label == "Right":
                used_label = "Left"

        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        x = int(wrist.x * frame_w)
        y = int(wrist.y * frame_h)

        cv2.putText(
            frame,
            f"Hand Raw:{raw_label} Used:{used_label}",
            (x + 10, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)


def draw_pose_subset_for_debug(frame, pose_results, mp_pose) -> None:
    """Draw the pose landmarks used by the current model."""
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
        point_map[name] = (int(lm.x * frame_w), int(lm.y * frame_h))

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
            cv2.LINE_AA,
        )

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
            cv2.LINE_AA,
        )


def show_dataset_debug_window(
    frame,
    hand_results,
    pose_results,
    mp_hands,
    mp_pose,
    label: str,
    valid_frames: int,
    window_size: int,
    frame_status: str,
) -> None:
    """Show a local debug window for raw dataset collection."""
    debug_frame = frame.copy()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    if hand_results is not None and hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=debug_frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )

    draw_hand_labels_for_debug(debug_frame, hand_results, mp_hands)
    draw_pose_subset_for_debug(debug_frame, pose_results, mp_pose)

    cv2.putText(
        debug_frame,
        f"Dataset Label: {label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug_frame,
        f"Valid Frames: {valid_frames}/{window_size}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug_frame,
        f"Frame Status: {frame_status}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255) if frame_status == "valid" else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Dataset Collect Debug", debug_frame)
    cv2.waitKey(1)


@router.websocket("/ws/dataset")
async def dataset_ws(websocket: WebSocket):
    """Collect raw MediaPipe frames from the phone client."""
    await websocket.accept()

    label = ""
    started = False
    raw_frames = []
    pending_raw_frames = []
    pending_label = ""
    waiting_save_confirm = False
    last_sample_time = 0.0

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            text_data = message.get("text")
            if text_data is not None:
                try:
                    payload = json.loads(text_data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "dataset_error",
                        "message": "无效的 JSON 文本消息",
                    })
                    continue

                msg_type = payload.get("type")

                if msg_type == "start_dataset":
                    label = str(payload.get("label", "")).strip()
                    if label == "":
                        await websocket.send_json({
                            "type": "dataset_error",
                            "message": "label 不能为空",
                        })
                        continue

                    raw_frames = []
                    pending_raw_frames = []
                    pending_label = ""
                    waiting_save_confirm = False
                    last_sample_time = 0.0
                    started = True

                    await websocket.send_json({
                        "type": "dataset_status",
                        "status": "ready",
                        "label": label,
                        "validFrames": 0,
                        "windowSize": RAW_SAMPLE_FRAME_COUNT,
                    })
                    continue

                if msg_type == "cancel_dataset":
                    raw_frames = []
                    pending_raw_frames = []
                    pending_label = ""
                    waiting_save_confirm = False
                    started = False
                    await websocket.send_json({
                        "type": "dataset_status",
                        "status": "cancelled",
                        "label": label,
                        "validFrames": 0,
                        "windowSize": RAW_SAMPLE_FRAME_COUNT,
                    })
                    continue

                if msg_type == "save_dataset":
                    has_pending_sample = (
                        waiting_save_confirm
                        and len(pending_raw_frames) == RAW_SAMPLE_FRAME_COUNT
                        and pending_label != ""
                    )
                    if not has_pending_sample:
                        await websocket.send_json({
                            "type": "dataset_error",
                            "message": "当前没有等待保存的样本",
                        })
                        continue

                    save_path = save_raw_sample(RAW_DATA_ROOT, pending_label, pending_raw_frames)

                    raw_frames = []
                    pending_raw_frames = []
                    label = pending_label
                    pending_label = ""
                    waiting_save_confirm = False
                    started = False

                    await websocket.send_json({
                        "type": "dataset_status",
                        "status": "saved",
                        "label": label,
                        "validFrames": RAW_SAMPLE_FRAME_COUNT,
                        "windowSize": RAW_SAMPLE_FRAME_COUNT,
                        "path": str(save_path),
                        "message": "样本已保存",
                    })
                    continue

                if msg_type == "discard_dataset":
                    raw_frames = []
                    pending_raw_frames = []
                    discarded_label = pending_label if pending_label != "" else label
                    pending_label = ""
                    waiting_save_confirm = False
                    started = False

                    await websocket.send_json({
                        "type": "dataset_status",
                        "status": "discarded",
                        "label": discarded_label,
                        "validFrames": 0,
                        "windowSize": RAW_SAMPLE_FRAME_COUNT,
                        "message": "样本已丢弃",
                    })
                    continue

                await websocket.send_json({
                    "type": "dataset_error",
                    "message": f"暂不支持的消息类型: {msg_type}",
                })
                continue

            bytes_data = message.get("bytes")
            if bytes_data is None:
                continue

            if not started or label == "":
                await websocket.send_json({
                    "type": "dataset_error",
                    "message": "请先发送 start_dataset 消息",
                })
                continue

            if waiting_save_confirm:
                continue

            now = time.monotonic()
            if last_sample_time > 0.0 and now - last_sample_time < SAMPLE_INTERVAL_SEC:
                continue

            frame = decode_frontend_frame(bytes_data)
            if frame is None:
                await websocket.send_json({
                    "type": "dataset_error",
                    "message": "图像解码失败",
                })
                continue

            if SAVE_DEBUG_INPUT_FRAME:
                debug_dir = PROJECT_ROOT / "debug_input_frames"
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / "frontend_ready_frame.jpg"), frame)

            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = hands.process(rgb_frame)
            pose_results = pose.process(rgb_frame)

            raw_frame = extract_raw_mediapipe_frame(
                hand_results=hand_results,
                pose_results=pose_results,
                mp_pose=mp_pose,
                timestamp_ms=time.time() * 1000.0,
                frame_width=frame_width,
                frame_height=frame_height,
                swap_handedness=SWAP_HANDEDNESS,
                swap_pose_lr=SWAP_POSE_LR,
            )

            if DEBUG_DATASET_WINDOW:
                show_dataset_debug_window(
                    frame=frame,
                    hand_results=hand_results,
                    pose_results=pose_results,
                    mp_hands=mp_hands,
                    mp_pose=mp_pose,
                    label=label,
                    valid_frames=len(raw_frames),
                    window_size=RAW_SAMPLE_FRAME_COUNT,
                    frame_status="valid" if raw_frame is not None else "invalid",
                )

            if raw_frame is None:
                await websocket.send_json({
                    "type": "dataset_status",
                    "status": "invalid_frame",
                    "label": label,
                    "validFrames": len(raw_frames),
                    "windowSize": RAW_SAMPLE_FRAME_COUNT,
                    "message": "未检测到有效手部或双肩",
                })
                continue

            raw_frames.append(raw_frame)
            last_sample_time = now

            if len(raw_frames) < RAW_SAMPLE_FRAME_COUNT:
                await websocket.send_json({
                    "type": "dataset_status",
                    "status": "collecting",
                    "label": label,
                    "validFrames": len(raw_frames),
                    "windowSize": RAW_SAMPLE_FRAME_COUNT,
                })
                continue

            pending_raw_frames = list(raw_frames)
            pending_label = label
            raw_frames = []
            waiting_save_confirm = True
            started = False

            await websocket.send_json({
                "type": "dataset_status",
                "status": "ready_to_save",
                "label": pending_label,
                "validFrames": RAW_SAMPLE_FRAME_COUNT,
                "windowSize": RAW_SAMPLE_FRAME_COUNT,
                "message": "样本采集完成，请确认是否保存",
            })

    except WebSocketDisconnect:
        pass
    finally:
        hands.close()
        pose.close()
        if DEBUG_DATASET_WINDOW:
            try:
                cv2.destroyWindow("Dataset Collect Debug")
            except cv2.error:
                pass
