# MySssb 手势识别服务 API 文档（当前联调版）

## 1. 文档说明

本文档用于前端与 Python 手势识别服务联调。

当前版本目标：

- 先打通 **前端 WebSocket -> Python 服务**
- 先验证 **文本控制消息 + 二进制图像帧消息**
- 当前返回结构已经确定
- 识别逻辑将后续接入，前端可先按本协议完成通信层开发

当前版本说明：

- 已确定服务框架为 **FastAPI**
- 已确定实时传输方式为 **WebSocket**
- 已确定图像帧采用 **JPEG 二进制**
- 已确定服务端返回的结果结构

---

## 2. 服务地址

本地开发默认地址：

```text
http://127.0.0.1:8000
```

健康检查接口：

```text
GET /health
```

WebSocket 地址：

```text
ws://127.0.0.1:8000/ws/gesture
```

---

## 3. 通信方式说明

当前实时识别使用 **WebSocket 长连接**。

消息分为两类：

### 3.1 文本消息（JSON）

用于：

- 初始化连接
- 发送控制指令
- 服务端返回识别结果
- 服务端返回错误信息

### 3.2 二进制消息（JPEG 帧）

用于：

- 前端持续发送摄像头截图帧
- 每条二进制消息对应一张 JPEG 图片

---

## 4. 健康检查接口

### 4.1 请求

```http
GET /health
```

### 4.2 响应

```json
{
  "ok": true
}
```

---

## 5. WebSocket 协议

## 5.1 连接地址

```text
ws://127.0.0.1:8000/ws/gesture
```

---

## 5.2 前端发送：start 初始化消息

前端在连接建立后，应先发送一条 JSON 文本消息，用于初始化本次实时识别。

### 消息类型

```json
{
  "type": "start",
  "format": "jpeg",
  "width": 320,
  "height": 240,
  "fps": 10
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| type | string | 是 | 固定为 `start` |
| format | string | 是 | 当前固定为 `jpeg` |
| width | number | 是 | 前端发送图像宽度 |
| height | number | 是 | 前端发送图像高度 |
| fps | number | 是 | 前端计划发送帧率 |

### 说明

- 当前版本建议前端先使用较低分辨率，如 `320x240`
- 当前版本建议帧率先控制在 `10~12 FPS`
- 服务端当前不会严格依赖这些参数，但前端仍应发送，便于后续扩展与调试

---

## 5.3 前端发送：图像帧消息

在发送 `start` 成功后，前端持续发送图像帧。

### 消息类型

- **WebSocket 二进制消息**
- 内容为一张 **JPEG 图片的 bytes**
- 不需要再外包 JSON
- 不使用 base64

### 说明

每一条二进制消息表示一帧图像，例如：

- 第 1 帧 JPEG
- 第 2 帧 JPEG
- 第 3 帧 JPEG

服务端收到后会按顺序处理。

---

## 5.4 服务端返回：识别结果消息

服务端处理完成后，返回 JSON 文本消息。

### 返回结构

```json
{
  "type": "result",
  "status": "warming_up",
  "label": "Waiting...",
  "confidence": 0.0,
  "validFrames": 12,
  "windowSize": 30,
  "hasValidHand": true
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| type | string | 是 | 固定为 `result` |
| status | string | 是 | 当前识别状态 |
| label | string | 是 | 当前识别标签 |
| confidence | number | 是 | 当前置信度 |
| validFrames | number | 是 | 当前窗口内有效帧数 |
| windowSize | number | 是 | 当前滑动窗口总长度，固定为 `30` |
| hasValidHand | boolean | 是 | 当前这一帧是否检测到有效手部 |

---

## 6. status 说明

当前约定以下状态值：

### 6.1 warming_up

表示：

- 当前窗口还没积满 30 帧
- 正在累积有效帧
- 暂时还不能输出正式识别结果

### 示例

```json
{
  "type": "result",
  "status": "warming_up",
  "label": "Waiting...",
  "confidence": 0.0,
  "validFrames": 12,
  "windowSize": 30,
  "hasValidHand": true
}
```

---

### 6.2 predicted

表示：

- 当前窗口已经满足预测条件
- 服务端已经给出当前识别标签

### 示例

```json
{
  "type": "result",
  "status": "predicted",
  "label": "hello",
  "confidence": 0.91,
  "validFrames": 30,
  "windowSize": 30,
  "hasValidHand": true
}
```

---

### 6.3 no_hand

表示：

- 当前这一帧没有检测到有效手部
- 前端可提示用户把手放进画面中

### 示例

```json
{
  "type": "result",
  "status": "no_hand",
  "label": "Waiting...",
  "confidence": 0.0,
  "validFrames": 8,
  "windowSize": 30,
  "hasValidHand": false
}
```

---

## 7. 服务端返回：错误消息

如果前端发送了非法消息，或未先发送 `start` 就直接发送二进制帧，服务端会返回错误消息。

### 返回结构

```json
{
  "type": "error",
  "message": "请先发送 start 消息"
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| type | string | 是 | 固定为 `error` |
| message | string | 是 | 错误说明 |

### 常见错误示例

#### 1）未发送 start 就先发帧

```json
{
  "type": "error",
  "message": "请先发送 start 消息"
}
```

#### 2）发送了无效 JSON

```json
{
  "type": "error",
  "message": "无效的 JSON 文本消息"
}
```

#### 3）发送了暂不支持的控制消息

```json
{
  "type": "error",
  "message": "暂不支持的消息类型: xxx"
}
```

---

## 8. 前端推荐接入流程

建议前端按以下顺序接入：

### 第一步：建立 WebSocket 连接

连接：

```text
ws://127.0.0.1:8000/ws/gesture
```

### 第二步：发送 start 初始化消息

```json
{
  "type": "start",
  "format": "jpeg",
  "width": 320,
  "height": 240,
  "fps": 10
}
```

### 第三步：周期性发送 JPEG 二进制帧

例如：

- 每 `80~100ms` 发送一帧
- 推荐初始帧率：`10~12 FPS`

### 第四步：接收服务端返回的 result

前端根据 `status`、`label`、`confidence` 等字段决定界面表现与业务判断。

---

## 9. 前端业务处理建议

服务端只负责返回“识别结果”，不负责判断“训练是否成功”。

建议前端自行处理以下逻辑：

- 当前目标手势是什么
- `label` 是否等于目标标签
- 是否需要连续多次命中才算成功
- 是否根据 `confidence` 再加一道判定
- 是否在 `no_hand` 状态下提示用户重新入镜

例如：

```javascript
if (result.status === "predicted" && result.label === expectedLabel) {
  // 前端判定为命中目标手势
}
```

---

## 10. 当前版本实现范围说明

当前文档对应的是**联调起步版协议**。

### 当前已经确定

- FastAPI 服务框架
- `/health` 健康检查
- `/ws/gesture` WebSocket 路径
- `start` 初始化消息格式
- JPEG 二进制帧传输方式
- `result` 返回结构
- `error` 返回结构

### 当前还会继续推进

- 服务端正式接入 OpenCV 图像解码
- 服务端正式接入手势预测逻辑
- 将滑动窗口预测结果通过本协议返回给前端

---

## 11. 建议的前端初始参数

前端初始联调建议参数：

```json
{
  "type": "start",
  "format": "jpeg",
  "width": 320,
  "height": 240,
  "fps": 10
}
```

建议原因：

- 分辨率较低，便于降低传输压力
- JPEG 体积较小，适合实时传输
- 10 FPS 对当前 30 帧滑动窗口已经足够起步联调

---

## 12. 文档版本

当前版本：

```text
v0.1 - WebSocket 联调协议草案
```
