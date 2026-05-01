# NationalCSL-DP Plus Blank Pipeline

本目录保存当前 NationalCSL-DP 小词表实验中效果最稳定的一版连续识别管线。

## 1. 当前版本定位

本版本用于验证：

```text
孤立词特征序列
  ↓
20帧滑动窗口词分类
  ↓
blank / uncertain 过滤
  ↓
NMS 去重
  ↓
tail 补帧
  ↓
gloss 序列输出
```

当前最佳特征目录：

```text
D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus_blank
```

当前最佳模型目录：

```text
D:/datasets/HearBridge-NationalCSL-mini/models_20f_plus_blank
```

## 2. 核心脚本

```text
build_features_20f_plus.py
```

用于从图片帧中提取增强版 20 帧特征。

当前每帧维度为 232：

```text
原始基础特征 166维
+ 静态相对位置特征 28维
+ 动态差分特征 38维
= 232维
```

```text
augment_with_transition_blank.py
```

用于在原始词样本基础上增加 synthetic transition blank 样本。

当前推荐参数：

```text
blank_per_participant = 8
min_split = 6
max_split = 14
```

```text
train_20f_classifier.py
```

用于训练 20 帧词级分类模型。

```text
predict_20f_test_samples.py
```

用于输出测试集 Top1 / Top3 诊断结果。

```text
pseudo_sentence_stream_test.py
```

用于将孤立词特征拼成伪连续句子，并测试滑动窗口连续识别效果。

## 3. 当前最佳训练结果

使用数据：

```text
features_20f_plus_blank
```

训练输出：

```text
models_20f_plus_blank
```

测试集结果：

```text
loss = 0.1882
accuracy = 0.9583
top3_accuracy = 1.0000
```

Top3 诊断结果：

```text
sample_count = 24
top1_correct_count = 23
top3_correct_count = 24
```

唯一 Top1 错误：

```text
真实=我
Top1=老师 0.673649
Top2=我 0.086291
Top3=今天 0.076427
Top3命中=1
```

## 4. 当前连续识别后处理参数

```text
confidence_threshold = 0.80
margin_threshold = 0.15
tail_frames = 12
tail_mode = repeat_last
nms_suppress_radius = 10
stable_frames = 1
blank_end_frames = 3
same_label_merge_gap = 8
min_segment_avg_confidence = 0.75
min_segment_max_confidence = 0.85
min_segment_duration = 3
short_segment_max_confidence = 0.90
```

短段过滤规则：

```text
duration < 3 且 max_confidence < 0.90
→ 丢弃该词段
```

该规则用于压制类似下面这种短促误报：

```text
老师 [20, 21] max=0.833989
```

## 5. 已验证的伪连续句子结果

### 5.1 我们 需要 帮助

```text
输入：我们 需要 帮助
输出：我们 需要 帮助
结果：成功
```

### 5.2 朋友 帮助 我们

```text
输入：朋友 帮助 我们
输出：朋友 帮助 我们
结果：成功
```

### 5.3 你 今天 学习

```text
输入：你 今天 学习
输出：今天 学习
结果：漏掉“你”，但不再错误输出“需要”
```

### 5.4 我 需要 帮助

```text
输入：我 需要 帮助
输出：需要 帮助
结果：漏掉“我”，但短段误报“老师”已被过滤
```

## 6. 当前结论

当前管线已经证明：

```text
20帧词级模型
+ 增强特征
+ blank 类
+ uncertain 过滤
+ NMS
+ tail 补帧
```

可以从伪连续手势流中恢复部分 gloss 序列。

当前最佳主线为：

```text
features_20f_plus_blank
models_20f_plus_blank
```

当前剩余问题主要集中在相似词分类边界：

```text
我 / 老师
你 / 需要
对不起 / 学习
```

## 7. 后续优化方向

```text
1. 增加易混词样本
2. 继续加强手部相对身体位置特征
3. 增加更真实的连续句子数据
4. 将 Top2 / Top3 候选交给上下文模块或 GPT 辅助判断
```

## 8. 常用命令

### 8.1 构建增强特征

```powershell
python experiments/national_csl_dp/plus_blank_pipeline/build_features_20f_plus.py --samples_csv "D:/datasets/HearBridge-NationalCSL-mini/samples.csv" --output_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus" --target_frames 20 --min_required_frames 8
```

### 8.2 增强 blank

```powershell
python experiments/national_csl_dp/plus_blank_pipeline/augment_with_transition_blank.py --feature_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus" --output_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus_blank" --blank_per_participant 8 --min_split 6 --max_split 14
```

### 8.3 训练模型

```powershell
python experiments/national_csl_dp/plus_blank_pipeline/train_20f_classifier.py --feature_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus_blank" --output_dir "D:/datasets/HearBridge-NationalCSL-mini/models_20f_plus_blank" --epochs 200 --batch_size 8
```

### 8.4 Top3 诊断

```powershell
python experiments/national_csl_dp/plus_blank_pipeline/predict_20f_test_samples.py --feature_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus_blank" --model_dir "D:/datasets/HearBridge-NationalCSL-mini/models_20f_plus_blank" --output_dir "D:/datasets/HearBridge-NationalCSL-mini/predict_20f_plus_blank" --mode test --top_k 3
```

### 8.5 伪连续句子测试

```powershell
python experiments/national_csl_dp/plus_blank_pipeline/pseudo_sentence_stream_test.py --feature_dir "D:/datasets/HearBridge-NationalCSL-mini/features_20f_plus_blank" --model_dir "D:/datasets/HearBridge-NationalCSL-mini/models_20f_plus_blank" --output_dir "D:/datasets/HearBridge-NationalCSL-mini/pseudo_sentence_20f_plus_uncertain" --sentence "朋友,帮助,我们" --participant "Participant_10" --gap_frames 0 --gap_mode none --tail_frames 12 --tail_mode repeat_last --confidence_threshold 0.80 --margin_threshold 0.15 --min_segment_avg_confidence 0.75 --min_segment_max_confidence 0.85 --nms_suppress_radius 10 --stable_frames 1 --blank_end_frames 3 --same_label_merge_gap 8
```

## 9. 验收点

执行：

```powershell
Get-ChildItem "experiments/national_csl_dp/plus_blank_pipeline"
```

预期至少看到：

```text
build_features_20f_plus.py
augment_with_transition_blank.py
train_20f_classifier.py
predict_20f_test_samples.py
pseudo_sentence_stream_test.py
README.md
```

执行伪连续句子测试时，预期：

```text
输入句子：朋友 帮助 我们
检测序列：朋友 帮助 我们
是否完全匹配：True
```