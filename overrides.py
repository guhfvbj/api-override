import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import BASE_DIR, MODEL_OVERRIDE_MAP_RAW, OVERRIDE_STORE_PATH, logger
from thinking import build_thinking_system_prefix, inject_thinking_system_prefix


@dataclass
class OverrideRule:
    """模型覆写规则：支持按别名重定向模型并绑定渠道。"""

    channel: Optional[str] = None  # 绑定的上游渠道名
    target_model: Optional[str] = None  # 上游实际模型 ID


def override_from_dict(model_id: str, cfg: Any) -> Optional[OverrideRule]:
    """将配置对象转换为 OverrideRule，兼容字符串与字典两种形式。"""
    if isinstance(cfg, str):
        # 简写："gpt-4o": "deepseek-chat"
        return OverrideRule(target_model=cfg)
    if not isinstance(cfg, dict):
        logger.warning("不支持的模型覆写配置 %s: %r", model_id, cfg)
        return None
    return OverrideRule(
        channel=cfg.get("channel"),
        target_model=cfg.get("target_model") or cfg.get("model") or model_id,
    )


def parse_override_map(raw: str) -> Dict[str, OverrideRule]:
    """解析 JSON 字符串，构造模型覆写规则字典。"""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("MODEL_OVERRIDE_MAP 不是合法 JSON，已忽略：%s", raw)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("MODEL_OVERRIDE_MAP 应为对象类型，当前值：%s", parsed)
        return {}

    result: Dict[str, OverrideRule] = {}
    for model_id, cfg in parsed.items():
        rule = override_from_dict(model_id, cfg)
        if rule:
            result[model_id] = rule
    return result


def load_persisted_overrides() -> Dict[str, OverrideRule]:
    """从本地持久化文件读取覆写规则。"""
    if not OVERRIDE_STORE_PATH.exists():
        return {}
    try:
        raw = OVERRIDE_STORE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        result: Dict[str, OverrideRule] = {}
        for mid, cfg in data.items():
            rule = override_from_dict(mid, cfg)
            if rule:
                result[mid] = rule
        return result
    except Exception as exc:
        logger.warning("读取覆写持久化文件失败：%s", exc)
        return {}


def persist_overrides(overrides: Dict[str, OverrideRule]) -> None:
    """写入覆写规则到本地文件，并同步到 .env 的 MODEL_OVERRIDE_MAP。"""
    serializable: Dict[str, Any] = {}
    for mid, rule in overrides.items():
        if not isinstance(rule, OverrideRule):
            continue
        serializable[mid] = {
            "channel": rule.channel,
            "target_model": rule.target_model,
        }

    # 1）写入本地持久化文件 override_store.json
    OVERRIDE_STORE_PATH.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2）同步写入 .env 中的 MODEL_OVERRIDE_MAP（便于通过环境变量恢复配置）
    env_path = BASE_DIR / ".env"
    raw = json.dumps(serializable, ensure_ascii=False, separators=(",", ":"))
    line = f"MODEL_OVERRIDE_MAP={raw}"

    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if not ln.startswith("MODEL_OVERRIDE_MAP=")]
    lines.append(line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def override_map_to_dict(overrides: Dict[str, OverrideRule]) -> Dict[str, Any]:
    """将 OverrideRule 映射转换为可序列化字典。"""
    result: Dict[str, Any] = {}
    for mid, rule in (overrides or {}).items():
        if not isinstance(rule, OverrideRule):
            continue
        result[mid] = {
            "channel": rule.channel,
            "target_model": rule.target_model,
        }
    return result


def load_initial_override_map() -> Dict[str, OverrideRule]:
    """根据本地持久化文件和环境变量加载初始覆写规则。"""
    env_overrides = parse_override_map(MODEL_OVERRIDE_MAP_RAW)
    persisted = load_persisted_overrides()
    return persisted or env_overrides


def apply_overrides(
    payload: Dict[str, Any],
    overrides: Dict[str, OverrideRule],
    thinking_max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """根据覆写规则修改模型 ID，并按需注入思考系统前缀。"""
    if not isinstance(payload, dict):
        return payload

    patched = dict(payload)
    model = patched.get("model")
    rule = overrides.get(model)
    if rule and rule.target_model:
        patched["model"] = rule.target_model

    # 思考系统前缀：仅当客户端显式提供预算时才注入
    if thinking_max_length is not None and thinking_max_length > 0:
        messages = patched.get("messages") or []
        if isinstance(messages, list):
            prefix = build_thinking_system_prefix(thinking_max_length)
            patched["messages"] = inject_thinking_system_prefix(messages, prefix)

    return patched
