"""运行时模型启动加载工具。

Python 服务启动时：
1. 向 Spring Boot 查询当前发布模型；
2. 如果存在 published 版本，则自动 reload；
3. 如果失败，不阻塞 Python 服务启动。
"""

from typing import Dict

from src.utils.runtime_model_registry import reload_runtime_model
from src.utils.spring_boot_client import fetch_published_model_version


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

    model_path = version.get("modelPath")
    label_map_path = version.get("labelMapPath")
    version_name = version.get("versionName") or version.get("runName") or ""

    if not model_path or not label_map_path:
        return {
            "ok": False,
            "message": "published model version missing modelPath or labelMapPath",
        }

    try:
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