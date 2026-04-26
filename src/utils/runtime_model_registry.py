"""运行时模型注册表。

用于管理当前实时识别服务应该使用的模型路径。
第一版设计：
1. 通过 /model/reload 接口更新当前模型路径；
2. 新建 GesturePredictSession 时读取当前模型路径；
3. 已经创建的旧会话不强制热切换，避免线程与资源释放问题。
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf

from src.predict import load_label_map


_current_model_path: Optional[Path] = None
_current_label_map_path: Optional[Path] = None
_current_version_name: Optional[str] = None


def reload_runtime_model(
    model_path: str,
    label_map_path: str,
    version_name: str = "",
) -> Dict:
    """校验并注册当前运行时模型。

    注意：
    这里会真实加载一次模型，确保文件可用。
    但不会把模型对象全局缓存给所有会话复用。
    新 WebSocket 会话仍会在 GesturePredictSession 中加载模型。

    Args:
        model_path: 模型文件路径。
        label_map_path: 标签映射文件路径。
        version_name: 模型版本名称。

    Returns:
        重载结果。
    """
    global _current_model_path
    global _current_label_map_path
    global _current_version_name

    model = Path(model_path)
    label_map = Path(label_map_path)

    if not model.exists():
        raise FileNotFoundError(f"model file does not exist: {model}")

    if not label_map.exists():
        raise FileNotFoundError(f"label map file does not exist: {label_map}")

    # 真实加载一次，确保模型文件没有损坏。
    tf.keras.models.load_model(str(model))

    # 真实读取一次，确保 label_map JSON 可用。
    label_map_data, _ = load_label_map(str(label_map))

    _current_model_path = model
    _current_label_map_path = label_map
    _current_version_name = version_name

    return {
        "ok": True,
        "versionName": version_name,
        "modelPath": str(model),
        "labelMapPath": str(label_map),
        "classCount": len(label_map_data),
        "message": "model reloaded",
    }


def get_runtime_model_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """获取当前运行时模型路径。

    Returns:
        当前模型路径和 label_map 路径。
    """
    return _current_model_path, _current_label_map_path


def get_runtime_model_info() -> Dict:
    """获取当前运行时模型信息。"""
    return {
        "ok": _current_model_path is not None and _current_label_map_path is not None,
        "versionName": _current_version_name,
        "modelPath": str(_current_model_path) if _current_model_path else "",
        "labelMapPath": str(_current_label_map_path) if _current_label_map_path else "",
    }

def get_runtime_model_snapshot() -> Dict:
    """获取当前运行时模型快照。

    用于 GesturePredictSession 初始化时记录当前会话使用的模型版本。
    """
    return {
        "modelVersionName": _current_version_name or "default",
        "modelPath": str(_current_model_path) if _current_model_path else "",
        "labelMapPath": str(_current_label_map_path) if _current_label_map_path else "",
        "usingPublishedModel": _current_model_path is not None and _current_label_map_path is not None,
    }