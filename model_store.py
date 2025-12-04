import json
from typing import Any, Dict, List

from config import BASE_DIR, MODEL_STORE_PATH, MODEL_STORE_RAW, logger


def _normalize_item(item: Any, channel: str) -> Any:
    if not isinstance(item, dict):
        return None
    mid = item.get("id")
    if not isinstance(mid, str) or not mid.strip():
        return None
    model: Dict[str, Any] = {
        "id": mid.strip(),
        "object": item.get("object") or "model",
        "owned_by": item.get("owned_by") or "local",
    }
    alias_for = item.get("alias_for")
    if isinstance(alias_for, str) and alias_for.strip():
        model["alias_for"] = alias_for.strip()
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if channel:
        meta["channel"] = channel
    if meta:
        model["metadata"] = meta
    return model


def parse_model_store(raw: str) -> Dict[str, List[Dict[str, Any]]]:
    """解析统一模型/别名 JSON：{channel: [ {id, owned_by?, alias_for?, metadata?} ]}"""
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("MODEL_STORE 不是合法 JSON，已忽略：%s", raw)
        return {}
    if not isinstance(data, dict):
        logger.warning("MODEL_STORE 应为对象类型，当前值：%s", data)
        return {}

    result: Dict[str, List[Dict[str, Any]]] = {}
    for ch, items in data.items():
        channel = ch if isinstance(ch, str) else ""
        if not isinstance(items, list):
            continue
        normalized: List[Dict[str, Any]] = []
        for it in items:
            norm = _normalize_item(it, channel)
            if norm:
                normalized.append(norm)
        if normalized:
            result[channel] = normalized
    return result


def load_persisted_model_store() -> Dict[str, List[Dict[str, Any]]]:
    if not MODEL_STORE_PATH.exists():
        return {}
    try:
        raw = MODEL_STORE_PATH.read_text(encoding="utf-8")
        return parse_model_store(raw)
    except Exception as exc:
        logger.warning("读取 model_store 持久化文件失败：%s", exc)
        return {}


def persist_model_store(models: Dict[str, List[Dict[str, Any]]]) -> None:
    serializable: Dict[str, Any] = {}
    for ch, items in (models or {}).items():
        if not isinstance(items, list):
            continue
        serializable[ch] = []
        for it in items:
            if not isinstance(it, dict) or not it.get("id"):
                continue
            serializable[ch].append(it)

    MODEL_STORE_PATH.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env_path = BASE_DIR / ".env"
    raw = json.dumps(serializable, ensure_ascii=False, separators=(",", ":"))
    line = f"MODEL_STORE={raw}"
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if not ln.startswith("MODEL_STORE=")]
    lines.append(line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_initial_model_store() -> Dict[str, List[Dict[str, Any]]]:
    persisted = load_persisted_model_store()
    if persisted:
        return persisted
    return parse_model_store(MODEL_STORE_RAW)
