# CE-CSL 中文连续手语 Gloss 识别实验

## 目标

本目录用于实验 CE-CSL 视频数据集的中文连续手语 gloss 序列识别。

目标流程：

CE-CSL 视频 → 视频帧读取 → MediaPipe 特征提取 → 时序模型 → CTC 解码 → gloss 序列 → AI 翻译中文

## 和原有模型的关系

原有模型用于数字人配套的单动作识别。

本实验模型用于连续手语短句识别。

两者相互独立，不混用训练数据，也不直接修改原有实时识别主链路。

## 数据集位置

CE-CSL 原始数据集不放入项目仓库。

当前本地数据集目录示例：

D:\CE-CSL\CE-CSL

数据集结构：

CE-CSL
- label
  - train.csv
  - dev.csv
  - test.csv
- video
  - train
  - dev
  - test

## 当前计划

1. 构建 manifest，整理视频路径和 gloss 标签。
2. 抽取少量视频测试读取。
3. 提取 MediaPipe 时序特征。
4. 训练 CTC baseline。
5. 输出 gloss 序列。
6. 接入 AI 翻译层。

## 注意事项

- 不提交 CE-CSL 原始视频。
- 不提交大体积特征文件。
- 不影响原有单动作识别模型。
- 实验跑通后，再考虑接入正式服务。