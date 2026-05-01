"""Spring Boot 后端客户端。

用于 Python 手势识别服务启动时，从 Spring Boot 查询当前发布模型版本。
"""

import json
import os
from typing import Dict, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def get_spring_boot_base_url() -> str:
    """获取 Spring Boot 后端基础地址。

    可以通过环境变量覆盖：
    HEARBRIDGE_BACKEND_BASE_URL=http://127.0.0.1:8080
    """
    return os.environ.get("HEARBRIDGE_BACKEND_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def fetch_published_model_version() -> Optional[Dict]:
    """从 Spring Boot 查询当前发布模型版本。

    Returns:
        当前发布模型版本；如果没有发布版本或请求失败，返回 None。
    """
    url = f"{get_spring_boot_base_url()}/sign/model-versions/published"

    request = Request(
        url=url,
        method="GET",
        headers={
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8")

            if not body.strip():
                return None

            return json.loads(body)

    except HTTPError as exception:
        print(f"[spring_boot_client] request failed, status={exception.code}, url={url}")
        return None

    except URLError as exception:
        print(f"[spring_boot_client] backend unavailable: {exception}")
        return None

    except Exception as exception:
        print(f"[spring_boot_client] unexpected error: {exception}")
        return None