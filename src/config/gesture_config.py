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

# 不再人为交换 MediaPipe Pose 左右。
# Pose LEFT_* 就按 LEFT_* 使用，RIGHT_* 就按 RIGHT_* 使用。
SWAP_POSE_LR = False

POSE_VISIBILITY_THRESHOLD = 0.5

RAW_PHONE_DATA_DIR_NAME = "dataset_raw_phone_10fps"
RAW_SAMPLE_FRAME_COUNT = WINDOW_SIZE
RAW_REQUIRE_SHOULDERS = True
RAW_REQUIRE_HAND = True