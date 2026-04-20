from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI(title="MySssb Gesture Service")


@app.get("/health")
async def health():
    return {"ok": True}


@app.websocket("/ws/gesture")
async def gesture_ws(websocket: WebSocket):
    # 1. 接受连接
    await websocket.accept()

    started = False

    try:
        while True:
            # 2. 同时兼容文本消息和二进制消息
            message = await websocket.receive()

            # 前端主动断开
            if message["type"] == "websocket.disconnect":
                break

            # 3. 先处理 JSON 文本消息
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

            # 4. 再处理二进制帧
            bytes_data = message.get("bytes")
            if bytes_data is not None:
                if not started:
                    await websocket.send_json({
                        "type": "error",
                        "message": "请先发送 start 消息"
                    })
                    continue

                # 这里先不接预测逻辑，只验证链路
                await websocket.send_json({
                    "type": "result",
                    "status": "warming_up",
                    "label": "Waiting...",
                    "confidence": 0.0,
                    "validFrames": 0,
                    "windowSize": 30,
                    "hasValidHand": False,
                    "debugFrameBytes": len(bytes_data)
                })

    except WebSocketDisconnect:
        pass