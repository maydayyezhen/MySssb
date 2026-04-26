# HearBridge 手势识别模型训练、发布与运行时加载流程

## 1. 目标

本文档说明 HearBridge 手势识别服务从 raw 样本采集、特征转换、模型训练、模型版本发布，到实时识别服务加载当前发布模型的完整流程。

## 2. 数据链路

当前数据链路如下：

```text
手机端采集 raw 样本
→ Python 保存 dataset_raw_phone_10fps/{label}/sample_XXX.npz
→ Python 扫描 raw 样本
→ Spring Boot 同步样本元数据到 MySQL
→ 管理端展示样本列表
→ Python 执行 raw → feature 转换
→ 生成 data_processed_arm_pose_10fps/{label}/sample_XXX.npy
→ Python 执行模型训练
→ 生成 artifacts/train_xxx 下的模型文件和 label_map
→ Spring Boot 登记 sign_model_version
→ 管理端发布模型版本
→ Spring Boot 调用 Python /model/reload
→ Python 新建实时识别会话时使用当前发布模型