from pathlib import Path

import numpy as np

try:
    from src.config.gesture_config import EXPECTED_SAMPLE_SHAPE
except ImportError:
    from config.gesture_config import EXPECTED_SAMPLE_SHAPE


def load_dataset(data_root: str):
    """读取数据目录下的所有样本，并合并成训练用数组。

    目录结构示例：
    data_processed_arm_pose_10fps/
    ├─ hello/
    │  ├─ sample_001.npy
    │  ├─ sample_002.npy
    ├─ thanks/
    │  ├─ sample_001.npy
    """

    root = Path(data_root)

    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(f"数据目录下没有找到任何标签文件夹：{data_root}")

    label_map = {name: idx for idx, name in enumerate(class_names)}

    x_list = []
    y_list = []

    for class_name in class_names:
        class_dir = root / class_name
        sample_files = sorted(class_dir.glob("sample_*.npy"))

        print(f"读取标签 [{class_name}]，样本数：{len(sample_files)}")

        for sample_file in sample_files:
            sample = np.load(sample_file)

            if sample.shape != EXPECTED_SAMPLE_SHAPE:
                print(f"跳过异常样本：{sample_file}，shape = {sample.shape}，期望 = {EXPECTED_SAMPLE_SHAPE}")
                continue

            x_list.append(sample.astype(np.float32))
            y_list.append(label_map[class_name])

    if not x_list:
        raise ValueError("没有读取到任何有效样本。")

    X = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    print("数据读取完成。")
    print("X shape =", X.shape)
    print("y shape =", y.shape)
    print("label_map =", label_map)

    return X, y, label_map


def split_dataset_stratified(X, y, train_ratio=0.8, seed=42):
    """按类别分层打乱并划分训练集、验证集。"""

    np.random.seed(seed)

    train_x_list = []
    train_y_list = []
    val_x_list = []
    val_y_list = []

    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        split_index = int(len(cls_indices) * train_ratio)

        if len(cls_indices) > 1:
            split_index = min(max(split_index, 1), len(cls_indices) - 1)

        train_indices = cls_indices[:split_index]
        val_indices = cls_indices[split_index:]

        train_x_list.append(X[train_indices])
        train_y_list.append(y[train_indices])
        val_x_list.append(X[val_indices])
        val_y_list.append(y[val_indices])

    X_train = np.concatenate(train_x_list, axis=0)
    y_train = np.concatenate(train_y_list, axis=0)
    X_val = np.concatenate(val_x_list, axis=0)
    y_val = np.concatenate(val_y_list, axis=0)

    train_perm = np.random.permutation(len(X_train))
    val_perm = np.random.permutation(len(X_val))

    X_train = X_train[train_perm]
    y_train = y_train[train_perm]
    X_val = X_val[val_perm]
    y_val = y_val[val_perm]

    return X_train, y_train, X_val, y_val


def print_label_distribution(name, y, label_map):
    """打印标签分布。"""
    reverse_label_map = {v: k for k, v in label_map.items()}
    unique, counts = np.unique(y, return_counts=True)

    print(f"{name} 标签分布：")
    for label_id, count in zip(unique, counts):
        print(f"  {reverse_label_map[label_id]}: {count}")