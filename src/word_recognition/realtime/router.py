# -*- coding: utf-8 -*-
"""Single-word realtime recognition WebSocket route."""

import json

import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.word_recognition.config.gesture_config import WINDOW_SIZE
from src.word_recognition.predict import GesturePredictSession
from src.word_recognition.realtime.frame_decoder import decode_frontend_frame


router = APIRouter(tags=["realtime-word"])

# Debug windows must be opt-in for service deployments.
DEBUG_WS_WINDOW = False


def draw_ws_debug_overlay(frame, result: dict) -> None:
    """Draw local debug text on the frontend-supplied frame."""
    cv2.putText(
        frame,
        f"Status: {result['status']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Label: {result['label']}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {result['confidence']:.4f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Valid Frames: {result['validFrames']}/{result['windowSize']}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Has Valid Hand: {result['hasValidHand']}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )


@router.websocket("/ws/gesture")
async def gesture_ws(websocket: WebSocket):
    """Realtime gesture recognition WebSocket compatible with the legacy route."""
    await websocket.accept()

    started = False
    session = None

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
                        "type": "error",
                        "message": "无效的 JSON 文本消息",
                    })
                    continue

                msg_type = payload.get("type")

                if msg_type == "start":
                    started = True

                    if session is not None:
                        session.close()
                    session = GesturePredictSession()

                    print(
                        "[ws/gesture] session model:",
                        session.model_version_name,
                        session.using_published_model,
                        session.model_path,
                    )

                    await websocket.send_json({
                        "type": "result",
                        "status": "warming_up",
                        "label": "Waiting...",
                        "confidence": 0.0,
                        "validFrames": 0,
                        "windowSize": WINDOW_SIZE,
                        "hasValidHand": False,
                        "modelVersionName": session.model_version_name,
                        "modelPath": session.model_path,
                        "labelMapPath": session.label_map_path,
                        "usingPublishedModel": session.using_published_model,
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"暂不支持的消息类型: {msg_type}",
                    })

                continue

            bytes_data = message.get("bytes")
            if bytes_data is not None:
                if not started or session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先发送 start 消息",
                    })
                    continue

                frame = decode_frontend_frame(bytes_data)
                if frame is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "图像解码失败",
                    })
                    continue

                raw_result = session.process_frame(
                    frame,
                    draw_landmarks=DEBUG_WS_WINDOW,
                )

                result = {
                    "type": "result",
                    "status": raw_result["status"],
                    "label": raw_result["label"],
                    "confidence": round(float(raw_result["confidence"]), 4),
                    "validFrames": raw_result["valid_frames"],
                    "windowSize": raw_result["window_size"],
                    "hasValidHand": raw_result["has_valid_hand"],
                    "landmarks": raw_result.get("landmarks", []),
                    "modelVersionName": raw_result["model_version_name"],
                    "modelPath": raw_result["model_path"],
                    "labelMapPath": raw_result["label_map_path"],
                    "usingPublishedModel": raw_result["using_published_model"],
                }

                if DEBUG_WS_WINDOW:
                    debug_frame = frame.copy()
                    draw_ws_debug_overlay(debug_frame, result)
                    cv2.imshow("WebSocket Gesture Debug", debug_frame)
                    cv2.waitKey(1)

                await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            session.close()
        if DEBUG_WS_WINDOW:
            cv2.destroyAllWindows()
