WLASL plus232 overlap pipeline
==============================

当前最终主线：
- 数据来源：Voxel51/WLASL
- 小词表：you, want, work, go, today, learn, help, friend, teacher, school, please, sorry, meet
- 特征：20帧 plus232
  - base166：左右手手型 + 基础身体
  - static_plus28：静态相对位置增强
  - dynamic_delta38：相邻帧动态差分
- 模型：Bidirectional GRU
- 句子推理：
  - trimmed demo video
  - 20帧滑窗
  - stride=2
  - confidence/margin过滤
  - 区间重叠 NMS

当前最优结果：
- 单词级测试集 accuracy=0.8400，top3_accuracy=0.9600
- 10条拼接句子 demo，7条完全匹配，完全匹配率=0.7000

主要目录：
- D:/datasets/WLASL-mini
- D:/datasets/WLASL-mini/features_20f_plus
- D:/datasets/WLASL-mini/models_20f_plus
- D:/datasets/WLASL-mini/demo_videos_trimmed
- D:/datasets/WLASL-mini/demo_eval_overlap_nms


一、构建 WLASL 小词表数据
---------------------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/build_wlasl_mini.py `
  --samples_json "D:/datasets/WLASL_HF_meta/samples.json" `
  --output_root "D:/datasets/WLASL-mini" `
  --labels "you,want,work,go,today,learn,help,friend,teacher,school,please,sorry,meet" `
  --max_per_label 10 `
  --overwrite


二、构建 plus232 特征
--------------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/build_wlasl_features_20f_plus.py `
  --samples_csv "D:/datasets/WLASL-mini/samples.csv" `
  --output_dir "D:/datasets/WLASL-mini/features_20f_plus" `
  --target_frames 20 `
  --padding 4 `
  --min_hand_frames 3


三、训练模型
------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/train_wlasl_20f_plus.py `
  --feature_dir "D:/datasets/WLASL-mini/features_20f_plus" `
  --model_dir "D:/datasets/WLASL-mini/models_20f_plus"


四、生成裁剪拼接版 demo 视频
----------------------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/compose_wlasl_sentence_video_trimmed.py `
  --samples_csv "D:/datasets/WLASL-mini/samples.csv" `
  --sentence "friend,meet,today" `
  --output_path "D:/datasets/WLASL-mini/demo_videos_trimmed/friend_meet_today_trimmed.mp4" `
  --sample_policy largest `
  --trim_padding 4 `
  --gap_frames 2 `
  --tail_frames 6

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/compose_wlasl_sentence_video_trimmed.py `
  --samples_csv "D:/datasets/WLASL-mini/samples.csv" `
  --sentence "sorry,teacher" `
  --output_path "D:/datasets/WLASL-mini/demo_videos_trimmed/sorry_teacher_trimmed.mp4" `
  --sample_policy largest `
  --trim_padding 4 `
  --gap_frames 2 `
  --tail_frames 6


五、单条视频推理
----------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/infer_wlasl_sentence_video.py `
  --video_path "D:/datasets/WLASL-mini/demo_videos_trimmed/friend_meet_today_trimmed.mp4" `
  --feature_dir "D:/datasets/WLASL-mini/features_20f_plus" `
  --model_dir "D:/datasets/WLASL-mini/models_20f_plus" `
  --output_dir "D:/datasets/WLASL-mini/demo_infer/friend_meet_today_final" `
  --expected "friend,meet,today" `
  --window_size 20 `
  --stride 2 `
  --confidence_threshold 0.45 `
  --margin_threshold 0.05 `
  --min_segment_windows 2 `
  --min_segment_avg_confidence 0.75 `
  --min_segment_max_confidence 0.85 `
  --same_label_merge_gap 8 `
  --nms_suppress_radius 6


六、批量评估 demo 句子
----------------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/evaluate_wlasl_demo_sentences.py `
  --feature_dir "D:/datasets/WLASL-mini/features_20f_plus" `
  --model_dir "D:/datasets/WLASL-mini/models_20f_plus" `
  --output_dir "D:/datasets/WLASL-mini/demo_eval_overlap_nms"

打开结果目录：

explorer "D:\datasets\WLASL-mini\demo_eval_overlap_nms"


七、单词级诊断
--------------

cd D:\MySssb

D:\MySssb\.venv\Scripts\python.exe experiments/asl_wlasl/plus232_overlap_pipeline/evaluate_wlasl_single_words.py `
  --feature_dir "D:/datasets/WLASL-mini/features_20f_plus" `
  --model_dir "D:/datasets/WLASL-mini/models_20f_plus" `
  --output_dir "D:/datasets/WLASL-mini/single_word_eval_plus232_test" `
  --mode test


备注
----

base166、static194、plus_blank 是消融实验分支，不作为当前主线。
当前主线固定使用：
- features_20f_plus
- models_20f_plus
- infer_wlasl_sentence_video.py 中的区间重叠 NMS
