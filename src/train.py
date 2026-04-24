import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks

from src.utils.dataset_loader import (
    load_dataset,
    split_dataset_stratified,
    print_label_distribution
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def plot_training_history(history, save_dir: Path):
    """绘制并保存训练曲线图。"""
    train_acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history.history else "val_acc"

    plt.figure(figsize=(10, 6))

    # loss 曲线
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")

    # accuracy 曲线
    plt.plot(history.history[train_acc_key], label="train_acc")
    plt.plot(history.history[val_acc_key], label="val_acc")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = save_dir / "training_curve.png"
    plt.savefig(save_path, dpi=200)
    print(f"训练曲线图已保存到：{save_path}")

    # 本地跑脚本时弹窗展示
    plt.show()


def evaluate_on_validation(model, X_val, y_val, label_map, save_dir: Path):
    """在验证集上做简单评估，并保存结果与混淆矩阵。"""
    reverse_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    # 预测
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # 总体准确率
    overall_acc = float(np.mean(y_pred == y_val))

    # 混淆矩阵（numpy 版，不依赖 sklearn）
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_id, pred_id in zip(y_val, y_pred):
        cm[int(true_id), int(pred_id)] += 1

    # 每类准确率
    per_class_lines = []
    for class_id in range(num_classes):
        class_name = reverse_label_map[class_id]
        total = int(np.sum(cm[class_id]))
        correct = int(cm[class_id, class_id])
        class_acc = correct / total if total > 0 else 0.0
        per_class_lines.append((class_name, correct, total, class_acc))

    # 打印结果
    print("\n===== 验证集评估结果 =====")
    print(f"整体准确率: {overall_acc:.4f}")
    for class_name, correct, total, class_acc in per_class_lines:
        print(f"{class_name}: {correct}/{total} = {class_acc:.4f}")

    # 保存文字结果
    eval_txt_path = save_dir / "eval_result.txt"
    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write("===== 验证集评估结果 =====\n")
        f.write(f"整体准确率: {overall_acc:.4f}\n\n")
        for class_name, correct, total, class_acc in per_class_lines:
            f.write(f"{class_name}: {correct}/{total} = {class_acc:.4f}\n")
        f.write("\n===== 混淆矩阵 =====\n")
        f.write(str(cm))

    print(f"评估结果已保存到：{eval_txt_path}")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    class_names = [reverse_label_map[i] for i in range(num_classes)]
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)

    # 在格子里写数值
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    print(f"混淆矩阵图已保存到：{cm_path}")

    plt.show()

def build_model(input_shape, num_classes):
    """构建一个最小可跑的 1D CNN。"""
    model = models.Sequential([
        layers.Input(shape=input_shape),              # (30, 80)

        layers.Conv1D(96, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(160, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),

        layers.Dense(96, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    data_root = PROJECT_ROOT / "data_processed_twohand"

    # 1. 读取数据
    X, y, label_map = load_dataset(data_root)

    # 2. 分层划分
    X_train, y_train, X_val, y_val = split_dataset_stratified(X, y)

    print("\n===== 数据集信息 =====")
    print("X_train shape =", X_train.shape)
    print("y_train shape =", y_train.shape)
    print("X_val shape =", X_val.shape)
    print("y_val shape =", y_val.shape)
    print("label_map =", label_map)

    print()
    print_label_distribution("训练集", y_train, label_map)
    print()
    print_label_distribution("验证集", y_val, label_map)

    # 3. 建模
    input_shape = X_train.shape[1:]   # (30, 80)
    num_classes = len(label_map)

    model = build_model(input_shape, num_classes)
    model.summary()

    # 4. 训练
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[early_stop]
    )

    # 5. 创建输出目录
    save_dir = PROJECT_ROOT / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 6. 绘图
    plot_training_history(history, save_dir)

    # 7. 评估
    evaluate_on_validation(model, X_val, y_val, label_map, save_dir)

    # 8. 保存模型和标签映射
    model.save(save_dir / "gesture_cnn_twohand.keras")
    with open(save_dir / "label_map_twohand.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("\n模型已保存到：", save_dir / "gesture_cnn.keras")
    print("标签映射已保存到：", save_dir / "label_map.json")


if __name__ == "__main__":
    main()