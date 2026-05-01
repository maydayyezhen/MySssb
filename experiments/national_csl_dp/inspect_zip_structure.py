# -*- coding: utf-8 -*-
"""
检查 NationalCSL-DP 的 Participant 压缩包内部结构。

作用：
1. 不解压 zip
2. 直接列出 zip 内前若干个文件路径
3. 帮你判断图片目录命名规则
"""

import argparse
import zipfile
from pathlib import Path


def inspect_zip(zip_path: Path, limit: int = 80) -> None:
    """
    查看 zip 内部文件结构。

    :param zip_path: Participant_xx.zip 路径
    :param limit: 最多打印多少条内部路径
    """
    if not zip_path.exists():
        print(f"[错误] 文件不存在：{zip_path}")
        return

    print(f"[信息] 正在检查：{zip_path}")
    print(f"[信息] 文件大小：{zip_path.stat().st_size / 1024 / 1024:.2f} MB")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        print(f"[信息] zip 内文件/目录总数：{len(names)}")
        print("\n========== 前若干条路径 ==========")

        for name in names[:limit]:
            print(name)

        print("\n========== 顶层目录统计 ==========")

        top_level_names = {}

        for name in names:
            parts = name.replace("\\", "/").split("/")
            if len(parts) > 0 and parts[0]:
                top = parts[0]
                top_level_names[top] = top_level_names.get(top, 0) + 1

        for top, count in list(top_level_names.items())[:50]:
            print(f"{top}: {count}")


def main() -> None:
    """
    主入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip_path",
        required=True,
        help="Participant zip 路径，例如 D:/datasets/NationalCSL-DP/Pics/Participant_01.zip",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="打印前多少条路径",
    )

    args = parser.parse_args()

    inspect_zip(
        zip_path=Path(args.zip_path),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()