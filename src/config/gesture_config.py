"""手势识别统一配置。

本文件集中管理模型窗口、采样帧率、特征维度、数据目录和模型文件名。
避免 window_size、feature_dim、模型路径散落在多个文件中。
"""

# 识别窗口配置：10FPS × 3秒 = 30帧
WINDOW_SIZE = 30
TARGET_FPS = 10
SAMPLE_INTERVAL_SEC = 1.0 / TARGET_FPS

# 新版特征维度：Hands 左右手 78×2 + Pose wrist/elbow 8 + elbow angle 2
FEATURE_DIM = 166
EXPECTED_SAMPLE_SHAPE = (WINDOW_SIZE, FEATURE_DIM)

# 新版数据目录与模型文件名
DATA_DIR_NAME = "data_processed_arm_pose_10fps"
MODEL_FILE_NAME = "gesture_cnn_arm_pose_10fps.keras"
LABEL_MAP_FILE_NAME = "label_map_arm_pose_10fps.json"

# MediaPipe handedness 是否需要左右对调。
# 如果实际观察到“我举左手却被标成 Right”，就设为 True。
SWAP_HANDEDNESS = True

# Pose 左右是否需要对调。
# 如果实际观察到“我真实左肩/左肘/左腕被画成 right_*”，就设为 True。
SWAP_POSE_LR = True

# Pose 点位可见度阈值
POSE_VISIBILITY_THRESHOLD = 0.5