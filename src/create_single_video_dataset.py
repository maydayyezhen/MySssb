import numpy as np
from read_video_feature import read_video_feature


def create_single_video_dataset(video_path, save_path):
    samples = read_video_feature(video_path)

    print("提取结果形状：", samples.shape)

    np.save(save_path, samples)
    print(f"数据已保存到：{save_path}")


if __name__ == "__main__":
    video_path = r"D:\MySssb\data\test.mp4"
    save_path = r"D:\MySssb\data\test_samples.npy"

    create_single_video_dataset(video_path, save_path)