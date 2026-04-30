# CE-CSL Feature V1 特征规范

## 1. 目标

本文件定义 CE-CSL 中文连续手语 gloss 识别实验中的视频特征格式。

本实验不复用原有单动作识别模型的旧特征代码，而是重新定义一套独立、干净的特征提取规则。

目标输出：

* 单帧特征：166 维
* 单个视频特征：T × 166
* T 表示按时间采样后的帧数

整体流程：

CE-CSL 视频 → MediaPipe Holistic → CE-CSL Feature V1 → T × 166 → CTC → gloss 序列

## 2. 基本原则

1. 不翻转视频帧。
2. 不交换左右手。
3. 不使用手机前置镜像修正逻辑。
4. 不复用旧模型中的 mirror / swap / handedness 相关代码。
5. 不使用 MediaPipe hand world landmarks。
6. 使用 MediaPipe Holistic 输出的 normalized landmarks。
7. 所有 landmark 坐标先还原到当前处理帧的图像坐标系。
8. 再进行相对化和尺度归一化。
9. 最终输出固定 166 维单帧特征。
10. 某只手或 Pose 缺失时，对应部分填 0。

## 3. 输入来源

每帧输入为 CE-CSL 原始视频帧。

处理流程：

CE-CSL 原始视频帧 → 按时间采样 → 可选等比例缩放 → MediaPipe Holistic → pose_landmarks / left_hand_landmarks / right_hand_landmarks

当前约定：

* MIRROR_INPUT = False
* 即不对 CE-CSL 视频帧做水平翻转

## 4. 时间采样规则

由于 CE-CSL 视频的 FPS 不完全一致，因此不按固定帧间隔采样。

采用目标采样率：

* TARGET_FPS = 10

也就是尽量每 0.1 秒抽取一帧。

示例：

* 30 FPS 视频：约每 3 帧取 1 帧
* 20 FPS 视频：约每 2 帧取 1 帧
* 60 FPS 视频：约每 6 帧取 1 帧

最终每个视频得到可变长度特征：T × 166。CTC 模型支持可变长度输入。

## 5. 坐标还原规则

MediaPipe Holistic 的 hand / pose landmarks 使用 normalized 坐标：

* x：相对图像宽度
* y：相对图像高度
* z：相对深度

在做几何计算前，先还原到当前处理帧的图像坐标系。

设当前送入 Holistic 的图像尺寸为 width 和 height，则：

* x_img = x × width
* y_img = y × height
* z_img = z × width

说明：

* x_img 和 y_img 是图像坐标。
* z_img 不是物理深度，只是将 z 缩放到与 x 接近的量纲。
* 如果图像送入 Holistic 前做过缩放，则使用缩放后的 width 和 height。

## 6. 单帧特征总结构

单帧特征总维度为 166：

* 左手特征：78 维
* 右手特征：78 维
* 手臂位置特征：8 维
* 肘角特征：2 维

总计：78 + 78 + 8 + 2 = 166

拼接顺序固定为：

left_hand_78 → right_hand_78 → arm_position_8 → elbow_angle_2

## 7. 手部 78 维特征

每只手的特征为 78 维：

* 21 个手部点 × 3 维局部坐标 = 63 维
* 15 个手指关节角 = 15 维
* 总计：63 + 15 = 78

MediaPipe 手部关键点编号：

* 0：WRIST
* 1：THUMB_CMC
* 2：THUMB_MCP
* 3：THUMB_IP
* 4：THUMB_TIP
* 5：INDEX_FINGER_MCP
* 6：INDEX_FINGER_PIP
* 7：INDEX_FINGER_DIP
* 8：INDEX_FINGER_TIP
* 9：MIDDLE_FINGER_MCP
* 10：MIDDLE_FINGER_PIP
* 11：MIDDLE_FINGER_DIP
* 12：MIDDLE_FINGER_TIP
* 13：RING_FINGER_MCP
* 14：RING_FINGER_PIP
* 15：RING_FINGER_DIP
* 16：RING_FINGER_TIP
* 17：PINKY_MCP
* 18：PINKY_PIP
* 19：PINKY_DIP
* 20：PINKY_TIP

## 8. 手部坐标相对化

手部点先从 normalized 坐标还原为图像坐标。

然后以手腕点作为局部原点：

* wrist = point_0
* local_point_i = point_i - wrist

因此手腕点本身会变成：

* point_0 = 0, 0, 0

这一步用于去除手在画面中的绝对位置影响。

## 9. 手部尺度归一化

手部尺度使用掌长和掌宽共同估计。

掌长：

* palm_length = distance_2d(point_0, point_9)
* 即 WRIST 到 MIDDLE_FINGER_MCP 的二维距离

掌宽：

* palm_width = distance_2d(point_5, point_17)
* 即 INDEX_FINGER_MCP 到 PINKY_MCP 的二维距离

最终手部尺度：

* hand_scale = (palm_length + palm_width) / 2

如果 hand_scale 过小，则使用兜底值：

* hand_scale = 1.0

最终归一化坐标：

* normalized_point_i = (point_i - wrist) / hand_scale

一只手的 63 维坐标特征按点位顺序展开：

point_0.x, point_0.y, point_0.z, point_1.x, point_1.y, point_1.z, ... , point_20.x, point_20.y, point_20.z

## 10. 手指角度特征

每只手计算 15 个关节角度。

每根手指取 3 个角度：

拇指：

* 0-1-2
* 1-2-3
* 2-3-4

食指：

* 0-5-6
* 5-6-7
* 6-7-8

中指：

* 0-9-10
* 9-10-11
* 10-11-12

无名指：

* 0-13-14
* 13-14-15
* 14-15-16

小指：

* 0-17-18
* 17-18-19
* 18-19-20

角度计算使用三点夹角，其中中间点为夹角顶点。

角度归一化规则：

* angle_normalized = angle / π
* 取值范围约为 0 到 1

手指角度使用归一化后的局部手部坐标计算。

## 11. 手部缺失处理

如果某只手没有检测到，则该手的 78 维特征全部填 0。

例如：

* 左手缺失：left_hand_78 = 78 个 0
* 右手缺失：right_hand_78 = 78 个 0

当前版本不额外增加 hand_present 标记，以保持总维度为 166。

## 12. 手臂位置 8 维特征

手臂位置特征来自 Pose landmarks。

使用以下 Pose 点：

* 11：LEFT_SHOULDER
* 12：RIGHT_SHOULDER
* 13：LEFT_ELBOW
* 14：RIGHT_ELBOW
* 15：LEFT_WRIST
* 16：RIGHT_WRIST

先将 Pose 点从 normalized 坐标还原为图像坐标。

然后计算肩中心：

* shoulder_center = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2

计算肩宽：

* shoulder_width = distance_2d(LEFT_SHOULDER, RIGHT_SHOULDER)

如果 shoulder_width 过小，则使用兜底值：

* shoulder_width = 1.0

然后计算以下 4 个点相对肩中心的位置：

* LEFT_ELBOW
* RIGHT_ELBOW
* LEFT_WRIST
* RIGHT_WRIST

每个点只取 x / y：

* relative_xy = (point_xy - shoulder_center_xy) / shoulder_width

最终 8 维顺序为：

1. left_elbow_x
2. left_elbow_y
3. right_elbow_x
4. right_elbow_y
5. left_wrist_x
6. left_wrist_y
7. right_wrist_x
8. right_wrist_y

## 13. 肘角 2 维特征

左右肘角分别使用三点夹角计算。

左肘角：

* angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)

右肘角：

* angle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)

角度归一化：

* angle_normalized = angle / π

最终 2 维顺序：

1. left_elbow_angle
2. right_elbow_angle

## 14. Pose 缺失处理

如果 pose_landmarks 为空，则：

* arm_position_8 = 8 个 0
* elbow_angle_2 = 2 个 0

如果 Pose 存在但个别点异常，当前版本暂不做复杂修复，统一按检测结果计算。

后续如果发现异常样本较多，再增加质量过滤逻辑。

## 15. 最终单帧特征索引范围

最终 166 维顺序如下：

* 0 到 77：left_hand_78
* 78 到 155：right_hand_78
* 156 到 163：arm_position_8
* 164 到 165：elbow_angle_2

其中：

* 0 到 62：左手局部坐标
* 63 到 77：左手手指角度
* 78 到 140：右手局部坐标
* 141 到 155：右手手指角度
* 156：left_elbow_x
* 157：left_elbow_y
* 158：right_elbow_x
* 159：right_elbow_y
* 160：left_wrist_x
* 161：left_wrist_y
* 162：right_wrist_x
* 163：right_wrist_y
* 164：left_elbow_angle
* 165：right_elbow_angle

## 16. 视频级特征格式

一个视频保存为一个 .npy 文件。

形状为：

* T × 166

示例：

* train-00001.npy：64 × 166
* dev-00002.npy：103 × 166

## 17. 当前版本限制

CE-CSL Feature V1 有以下限制：

1. 不使用 hand world landmarks。
2. 不额外运行单独的 MediaPipe Hands 模型。
3. 不处理手机前置镜像。
4. 不和旧单动作模型共享特征代码。
5. 不额外加入 hand_present / pose_present 标记。
6. 不直接用于手机实时识别，先用于离线 CE-CSL gloss 识别实验。

## 18. 后续可能调整

如果实验效果不稳定，后续可以考虑：

1. 增加 left_hand_present / right_hand_present / pose_present 标记。
2. 使用单独 Hands 模型提取 hand_world_landmarks。
3. 增加手腕相对肩中心的 z 维信息。
4. 对低检测率视频做质量过滤。
5. 对手机端另行制定实时推理输入规范。
