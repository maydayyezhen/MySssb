# MySssb / HearBridge Gesture Service

HearBridge 的 Python 手势识别服务，负责视觉识别、数据采集、训练和模型运行时管理。业务编排、识别记录保存、DeepSeek 语义增强等能力由 Java / Spring Boot 侧完成。

## 1. 项目定位

本仓库是 HearBridge 后端体系中的 Python 识别服务：

- 使用 MediaPipe 提取手部 / 姿态关键点。
- 使用 TensorFlow / Keras 模型做单词级实时识别和句子视频离线识别。
- 通过 FastAPI 对 Java / HarmonyOS 暴露稳定接口。
- 保留 `experiments/` 作为实验脚本和模型探索目录，正式服务入口在 `src/`。

## 2. 当前服务能力

- 实时单词识别 WebSocket。
- 手机 raw dataset 采集 WebSocket。
- raw dataset 样本扫描。
- raw dataset 转 feature 数据集。
- 单词识别模型训练。
- 训练产物下载。
- 运行时模型状态查询和重载。
- 句子视频上传识别。
- gloss 中文展示映射。

## 3. 目录结构

```text
src/
├─ app.py                         # FastAPI 总入口
├─ app_legacy.py                  # 旧入口备份
├─ word_recognition/              # 单词级识别完整模块
│  ├─ realtime/                   # /ws/gesture
│  ├─ dataset/                    # /ws/dataset, raw 样本扫描
│  ├─ training/                   # raw->feature, 训练, 产物下载
│  ├─ model_runtime/              # 运行时模型状态与重载
│  ├─ config/
│  ├─ utils/
│  ├─ predict.py
│  └─ train.py
└─ sentence_video/                # 句子视频识别模块
   ├─ router.py
   ├─ service.py
   ├─ config.py
   ├─ runtime.py                  # 句子模型运行时缓存
   ├─ schemas.py
   ├─ zh_map.py
   ├─ video_io.py
   └─ wlasl_pipeline/             # 从实验目录复制的 WLASL 推理链
```

## 4. Python / Java / HarmonyOS 职责边界

Python：

- MediaPipe 关键点检测。
- TensorFlow 模型推理。
- 实时识别。
- 句子视频识别。
- 数据采集和训练。

Java / Spring Boot：

- 用户、课程、训练记录等业务数据。
- 文件上传业务入口。
- 调用 Python 识别服务。
- 调用 DeepSeek。
- 保存识别结果。
- 管理端权限和模型版本记录。

HarmonyOS：

- 相机采集。
- 视频上传。
- 数字人播放。
- 识别结果展示。

## 5. 接口列表

```text
GET  /health
WS   /ws/gesture
WS   /ws/dataset
GET  /dataset/raw/samples
POST /dataset/raw/convert-to-features
POST /model/train
GET  /artifacts/{run_name}/{file_name}
GET  /model/current
POST /model/reload
POST /model/reload-from-url
POST /api/sentence/recognize
```

## 6. 启动方式

```powershell
cd D:\MySssb
D:\MySssb\.venv\Scripts\python.exe -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

依赖安装：

```powershell
D:\MySssb\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 7. 句子视频识别配置

句子视频识别配置在 `src/sentence_video/config.py` 中，默认使用当前本机 WLASL-mini-v2-25 路径。Windows 路径可通过环境变量覆盖。

```text
SENTENCE_VIDEO_FEATURE_DIR
SENTENCE_VIDEO_MODEL_DIR
SENTENCE_VIDEO_TMP_ROOT
SENTENCE_VIDEO_TIMEOUT_SEC
SENTENCE_VIDEO_WINDOW_SIZE
SENTENCE_VIDEO_STRIDE
SENTENCE_VIDEO_CONFIDENCE_THRESHOLD
SENTENCE_VIDEO_MARGIN_THRESHOLD
SENTENCE_VIDEO_MIN_SEGMENT_WINDOWS
SENTENCE_VIDEO_MIN_SEGMENT_AVG_CONFIDENCE
SENTENCE_VIDEO_MIN_SEGMENT_MAX_CONFIDENCE
SENTENCE_VIDEO_SAME_LABEL_MERGE_GAP
SENTENCE_VIDEO_NMS_SUPPRESS_RADIUS
SENTENCE_VIDEO_MAX_UPLOAD_MB
SENTENCE_VIDEO_KEEP_TMP
```

示例：

```powershell
$env:SENTENCE_VIDEO_FEATURE_DIR="D:/datasets/WLASL-mini-v2-25/features_20f_plus"
$env:SENTENCE_VIDEO_MODEL_DIR="D:/datasets/WLASL-mini-v2-25/models_20f_plus"
$env:SENTENCE_VIDEO_MAX_UPLOAD_MB="50"
$env:SENTENCE_VIDEO_KEEP_TMP="false"
```

上传视频仅允许 `.mp4`、`.mov`、`.avi`、`.mkv`。默认请求结束后删除 `tmp/sentence_video/{request_id}`；调试时可设置 `SENTENCE_VIDEO_KEEP_TMP=true` 保留临时目录。

## 8. DeepSeek 语义增强说明

DeepSeek 语义增强不在 Python 服务中完成。

Python 只返回：

- `rawSequence`
- `rawDisplayZh`
- `rawTextZh`
- `segmentTopK`

Spring Boot 根据这些字段调用 DeepSeek 快速非思考模型进行 deletion-only 语义后处理，并负责保存业务识别结果。

## 9. 测试命令

健康检查：

```powershell
curl.exe "http://127.0.0.1:8000/health"
```

实时单词识别：

```text
ws://127.0.0.1:8000/ws/gesture
```

连接后先发送：

```json
{"type":"start"}
```

然后持续发送 JPEG bytes。

句子视频识别：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/sentence/recognize" `
  -F "file=@D:/datasets/WLASL-mini-v2-25/demo_videos_semantic/you_want_help_trimmed.mp4"
```

响应会包含：

```json
{
  "status": "recognized",
  "mode": "fast",
  "rawSequence": ["you", "want", "help"],
  "rawDisplayZh": ["你", "想要", "帮助"],
  "rawTextZh": "你 想要 帮助",
  "segmentTopK": [
    {
      "segmentIndex": 1,
      "rawLabel": "you",
      "rawLabelZh": "你",
      "topK": [
        {
          "label": "you",
          "labelZh": "你",
          "avgProb": 0.7,
          "maxProb": 0.9,
          "hitCount": 18
        }
      ]
    }
  ],
  "elapsedMs": 0
}
```

## 10. 当前限制与后续计划

- 句子视频模型当前是单进程内存缓存；高并发时后续可增加推理锁、队列或独立推理服务。
- 句子视频默认模型路径仍依赖本机 `D:/datasets/...`，部署时应使用环境变量覆盖。
- `sentence_video/wlasl_pipeline` 是从实验目录复制来的当前可用推理链，后续可继续瘦身为更小的生产推理模块。
- DeepSeek 语义后处理由 Spring Boot 完成，Python 不保存业务记录。
- 多客户端 WebSocket 并发和句子视频并发仍需做压力测试。
