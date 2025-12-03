import json
from typing import Any

import httpx

from config import logger


def strip_trailing_slash(url: str) -> str:
    """去掉 URL 末尾的斜杠，避免重复拼接路径。"""
    if isinstance(url, str) and url.endswith("/"):
        return url[:-1]
    return url


def log_payload(label: str, payload: Any, limit: int = 2000) -> None:
    """调试用：记录请求或响应体，自动截断过长内容。"""
    try:
        if isinstance(payload, (dict, list)):
            text = json.dumps(payload, ensure_ascii=False)
        else:
            text = str(payload)
        if len(text) > limit:
            text = text[:limit] + "...(truncated)"
        logger.info("%s: %s", label, text)
    except Exception:
        logger.warning("记录 %s 时失败", label)


def strip_undefined_fields(payload: Any) -> Any:
    """递归清理值为 None 或 '[undefined]' 的字段，避免上游 400。"""
    placeholder = "[undefined]"
    if isinstance(payload, list):
        return [strip_undefined_fields(item) for item in payload]
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip() == placeholder:
                continue
            cleaned[key] = strip_undefined_fields(value)
        return cleaned
    return payload


def extract_error(resp: httpx.Response) -> Any:
    """尽量从响应中解析出错误信息（JSON 为主，失败则返回文本）。"""
    try:
        return resp.json()
    except Exception:
        return resp.text


def extract_payload(resp: httpx.Response) -> Any:
    """尽量把响应解析为 JSON，否则返回文本。"""
    try:
        return resp.json()
    except Exception:
        return resp.text

