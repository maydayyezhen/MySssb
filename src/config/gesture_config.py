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
SWAP_HANDEDNESS = True

# Pose 左右是否交换。
# 注意：Pose 要单独通过后端调试窗口确认，不要因为 Hands 反了就自动跟着改。
SWAP_POSE_LR = True

POSE_VISIBILITY_THRESHOLD = 0.5

RAW_PHONE_DATA_DIR_NAME = "dataset_raw_phone_10fps"
RAW_SAMPLE_FRAME_COUNT = WINDOW_SIZE
RAW_REQUIRE_SHOULDERS = True
RAW_REQUIRE_HAND = True