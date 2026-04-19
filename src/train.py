import json
from pathlib import Path
from tensorflow.keras import layers, models, callbacks

from dataset_loader import (
    load_dataset,
    split_dataset_stratified,
    print_label_distribution
)


def build_model(input_shape, num_classes):
    """构建一个最小可跑的 1D CNN。"""
    model = models.Sequential([
        layers.Input(shape=input_shape),              # (30, 80)

        layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
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
    data_root = "data_processed"

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

    # 5. 保存模型和标签映射
    save_dir = Path("artifacts")
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save(save_dir / "gesture_cnn.keras")

    with open(save_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("\n模型已保存到：", save_dir / "gesture_cnn.keras")
    print("标签映射已保存到：", save_dir / "label_map.json")


if __name__ == "__main__":
    main()