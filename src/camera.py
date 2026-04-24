import cv2

from predict import GesturePredictSession


def draw_overlay(frame, result: dict):
    """在画面上绘制调试信息。"""
    predicted_label = result["label"]
    predicted_conf = result["confidence"]
    valid_frames = result["valid_frames"]
    window_size = result["window_size"]
    has_valid_hand = result["has_valid_hand"]

    cv2.putText(
        frame,
        f"Prediction: {predicted_label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Confidence: {predicted_conf:.3f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Valid Frames In Window: {valid_frames}/{window_size}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA
    )

    if not has_valid_hand:
        cv2.putText(
            frame,
            "No valid hand detected",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "Q: Quit",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA
    )


def main():
    # ========= 可改参数 =========
    camera_index = 0
    model_path = "../artifacts/gesture_cnn_twohand.keras"
    label_map_path = "../artifacts/label_map_twohand.json"
    window_size = 30
    confidence_threshold = 0.7
    max_missing_frames = 10
    # ==========================

    session = GesturePredictSession(
        model_path=model_path,
        label_map_path=label_map_path,
        window_size=window_size,
        confidence_threshold=confidence_threshold,
        max_missing_frames=max_missing_frames,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        session.close()
        raise ValueError(f"无法打开摄像头：{camera_index}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("读取摄像头画面失败。")
                break

            # 镜像显示，更符合本地摄像头习惯
            frame = cv2.flip(frame, 1)

            # 调用核心预测逻辑
            result = session.process_frame(frame, draw_landmarks=True)

            # 画调试文字
            draw_overlay(frame, result)

            cv2.imshow("Real-time Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        session.close()


if __name__ == "__main__":
    main()