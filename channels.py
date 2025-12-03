import json
from typing import Dict, Optional, Tuple

from fastapi import HTTPException

from config import NEWAPI_API_KEY, NEWAPI_BASE_URL, UPSTREAM_CHANNELS_RAW, BASE_DIR, logger
from overrides import OverrideRule
from utils import strip_trailing_slash


def load_upstream_channels_from_env(raw: str = UPSTREAM_CHANNELS_RAW) -> Dict[str, Dict[str, str]]:
    """从环境变量 UPSTREAM_CHANNELS 读取上游渠道配置。"""
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("UPSTREAM_CHANNELS 不是合法 JSON，已忽略：%s", raw)
        return {}
    if not isinstance(data, dict):
        logger.warning("UPSTREAM_CHANNELS 应该是对象类型，当前值：%s", data)
        return {}
    result: Dict[str, Dict[str, str]] = {}
    for name, cfg in data.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            continue
        result[name] = {
            "base_url": strip_trailing_slash(base_url),
            "api_key": api_key,
        }
    return result


def persist_upstream_channels_to_env(channels: Dict[str, Dict[str, str]]) -> None:
    """写入上游渠道配置到 .env 的 UPSTREAM_CHANNELS，保留其他行。"""
    env_path = BASE_DIR / ".env"
    payload: Dict[str, Dict[str, str]] = {}
    for name, cfg in (channels or {}).items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            continue
        payload[name] = {"base_url": base_url, "api_key": api_key}
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    line = f"UPSTREAM_CHANNELS={raw}"

    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if not ln.startswith("UPSTREAM_CHANNELS=")]
    lines.append(line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_upstream_for_request(
    rule: Optional[OverrideRule],
    upstream_channels: Dict[str, Dict[str, str]],
) -> Tuple[str, str, Optional[str]]:
    """根据覆写规则和渠道配置选择上游 base_url / api_key / X-Channel 名称。

    选择顺序：
    1. 若规则指定了渠道且配置存在，则优先使用该渠道；
    2. 否则，如果配置了 NEWAPI_*，作为默认上游；
    3. 再否则，从渠道列表中选择第一个合法渠道；
    4. 都没有时抛出 500。
    """
    # 若规则指定了渠道且配置存在，则优先使用该渠道
    if rule and rule.channel:
        cfg = upstream_channels.get(rule.channel)
        if isinstance(cfg, dict):
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            if isinstance(base_url, str) and isinstance(api_key, str):
                return strip_trailing_slash(base_url), api_key, rule.channel

    # 否则，如果配置了 NEWAPI_*，作为默认上游
    if NEWAPI_BASE_URL and NEWAPI_API_KEY:
        return strip_trailing_slash(NEWAPI_BASE_URL), NEWAPI_API_KEY, None

    # 再否则，从渠道列表中选择第一个合法渠道
    for name, cfg in upstream_channels.items():
        if not isinstance(cfg, dict):
            continue
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        if isinstance(base_url, str) and isinstance(api_key, str):
            return strip_trailing_slash(base_url), api_key, name

    raise HTTPException(status_code=500, detail="没有可用的上游渠道或 NEWAPI 配置")

