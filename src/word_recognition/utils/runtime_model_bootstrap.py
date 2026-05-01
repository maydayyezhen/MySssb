"""运行时模型启动加载工具。

Python 服务启动时：
1. 向 Spring Boot 查询当前发布模型；
2. 如果存在 published 版本，则自动 reload；
3. 如果失败，不阻塞 Python 服务启动。
"""

from typing import Dict

from src.word_recognition.utils.runtime_model_registry import reload_runtime_model
from src.word_recognition.utils.spring_boot_client import fetch_published_model_version

from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from src.word_recognition.config.gesture_config import MODEL_FILE_NAME, LABEL_MAP_FILE_NAME


def sanitize_runtime_name(value: str) -> str:
    """将模型版本名转换为安全目录名。"""
    text = str(value or "").strip()
    if text == "":
        text = "current"

    return "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in text
    )


def download_url_to_file(url: str, target_path: Path) -> None:
    """下载 URL 文件到本地。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=60) as response:
        with open(target_path, "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)


def bootstrap_runtime_model_from_backend() -> Dict:
    """从 Spring Boot 当前发布版本初始化 Python 运行时模型。

    Returns:
        启动加载结果。
    """
    version = fetch_published_model_version()

    if not version:
        return {
            "ok": False,
            "message": "no published model version from backend",
        }

    version_name = version.get("versionName") or version.get("runName") or ""

    model_path = version.get("modelPath")
    label_map_path = version.get("labelMapPath")


    if not model_path or not label_map_path:
        return {
            "ok": False,
            "message": "published model version missing modelPath or labelMapPath",
        }

    model_url = version.get("modelUrl")
    label_map_url = version.get("labelMapUrl")

    try:
        if model_url and label_map_url:
            project_root = Path(__file__).resolve().parents[3]
            runtime_dir = project_root / "runtime_models" / sanitize_runtime_name(version_name)

            local_model_path = runtime_dir / MODEL_FILE_NAME
            local_label_map_path = runtime_dir / LABEL_MAP_FILE_NAME

            download_url_to_file(model_url, local_model_path)
            download_url_to_file(label_map_url, local_label_map_path)

            result = reload_runtime_model(
                model_path=str(local_model_path),
                label_map_path=str(local_label_map_path),
                version_name=version_name,
            )
        else:
            result = reload_runtime_model(
                model_path=model_path,
                label_map_path=label_map_path,
                version_name=version_name,
            )

        return {
            "ok": True,
            "message": "runtime model bootstrapped from backend",
            "reloadResult": result,
        }

    except Exception as exception:
        return {
            "ok": False,
            "message": f"runtime model bootstrap failed: {exception}",
        }
