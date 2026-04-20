from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import cv2
import numpy as np

from src.predict import GesturePredictSession

DEBUG_WS_WINDOW = True

def draw_ws_debug_overlay(frame, result: dict):
    """在前端传来的图像上绘制识别调试信息。"""
    cv2.putText(
        frame,
        f"Status: {result['status']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Label: {result['label']}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Confidence: {result['confidence']:.4f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Valid Frames: {result['validFrames']}/{result['windowSize']}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Has Valid Hand: {result['hasValidHand']}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 165, 255),
        2,
        cv2.LINE_AA
    )

app = FastAPI(title="MySssb Gesture Service")


@app.get("/health")
async def health():
    return {"ok": True}


@app.websocket("/ws/gesture")
async def gesture_ws(websocket: WebSocket):
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
                        "message": "无效的 JSON 文本消息"
                    })
                    continue

                msg_type = payload.get("type")

                if msg_type == "start":
                    started = True

                    # 每个连接在 start 时创建独立识别会话
                    if session is not None:
                        session.close()
                    session = GesturePredictSession()

                    await websocket.send_json({
                        "type": "result",
                        "status": "warming_up",
                        "label": "Waiting...",
                        "confidence": 0.0,
                        "validFrames": 0,
                        "windowSize": 30,
                        "hasValidHand": False
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"暂不支持的消息类型: {msg_type}"
                    })

                continue

            bytes_data = message.get("bytes")
            if bytes_data is not None:
                if not started or session is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先发送 start 消息"
                    })
                    continue

                arr = np.frombuffer(bytes_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "图像解码失败"
                    })
                    continue

                # 和本地 camera.py 一致，做镜像
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.flip(frame, 1)

                # 调用识别
                raw_result = session.process_frame(frame, draw_landmarks=True)

                result = {
                    "type": "result",
                    "status": raw_result["status"],
                    "label": raw_result["label"],
                    "confidence": round(float(raw_result["confidence"]), 4),
                    "validFrames": raw_result["valid_frames"],
                    "windowSize": raw_result["window_size"],
                    "hasValidHand": raw_result["has_valid_hand"],
                    "landmarks": raw_result["landmarks"]
                }

                # 本地调试窗口：显示前端输入图像 + 当前识别结果
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