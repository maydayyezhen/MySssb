WINDOW_SIZE = 30
TARGET_FPS = 10
SAMPLE_INTERVAL_SEC = 1.0 / TARGET_FPS

FEATURE_DIM = 166
EXPECTED_SAMPLE_SHAPE = (WINDOW_SIZE, FEATURE_DIM)

DATA_DIR_NAME = "data_processed_arm_pose_10fps"
MODEL_FILE_NAME = "gesture_cnn_arm_pose_10fps.keras"
LABEL_MAP_FILE_NAME = "label_map_arm_pose_10fps.json"

# Hands 左右是否交换。
# 当前手机 /ws/dataset 采集样本显示 Hands 槽位反了，因此先设为 False。
# 验收标准：slot 0 = 真实左手，slot 1 = 真实右手。
# 不再人为交换 MediaPipe Hands 左右。
# MediaPipe 返回 Left 就进入 slot 0，返回 Right 就进入 slot 1。
SWAP_HANDEDNESS = False

# Pose 左右点位交换。
# 注意：这只是 Pose 镜像规范化的一半，另一半是 x = 1 - x。
SWAP_POSE_LR = True

# Pose 坐标是否做水平镜像规范化。
# 前端发送镜像图时，Pose 需要翻回非镜像身体坐标系。
MIRROR_POSE_X = True

POSE_VISIBILITY_THRESHOLD = 0.5

RAW_PHONE_DATA_DIR_NAME = "dataset_raw_phone_10fps"
RAW_SAMPLE_FRAME_COUNT = WINDOW_SIZE
RAW_REQUIRE_SHOULDERS = True
RAW_REQUIRE_HAND = True