import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# 加载 .env 环境变量
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newapi-proxy")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_PORT = int(os.getenv("PORT", 14300))
OVERRIDE_STORE_PATH = BASE_DIR / "override_store.json"

# 默认上游配置（未指定渠道时使用）
NEWAPI_BASE_URL = os.getenv("NEWAPI_BASE_URL", "https://api.newapi.ai")
NEWAPI_API_KEY = os.getenv("NEWAPI_API_KEY")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")

# 模型覆写规则（环境变量）
MODEL_OVERRIDE_MAP_RAW = os.getenv("MODEL_OVERRIDE_MAP", "{}")

# 上游多渠道配置（JSON）：{"a":{"base_url":"...","api_key":"..."}, ...}
UPSTREAM_CHANNELS_RAW = os.getenv("UPSTREAM_CHANNELS", "{}")

# 思考覆写相关配置：由客户端请求体控制开关与预算
THINKING_SYSTEM_TEMPLATE = (
    "<antml:thinking_mode>interleaved</antml:thinking_mode>"
    "<antml:max_thinking_length>{max_length}</antml:max_thinking_length>"
)
THINKING_LENGTH_MULTIPLIER = 5


@dataclass
class OverrideRule:
    """模型覆写规则：支持渠道与模型重定向。"""

    # 渠道标识（可选），用于选择上游渠道；为空则使用默认上游
    channel: Optional[str] = None
    # 目标模型 ID（实际发往上游的模型），为空则默认为当前别名本身
    target_model: Optional[str] = None


def _strip_trailing_slash(url: str) -> str:
    """去掉末尾斜杠，避免重复拼接路径。"""
    return url[:-1] if url.endswith("/") else url


def _log_payload(label: str, payload: Any, limit: int = 2000) -> None:
    """调试用：记录请求/响应体（截断避免过大日志）。"""
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


NEWAPI_BASE_URL = _strip_trailing_slash(NEWAPI_BASE_URL)


def _load_upstream_channels_from_env(raw: str = UPSTREAM_CHANNELS_RAW) -> Dict[str, Dict[str, str]]:
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
            "base_url": _strip_trailing_slash(base_url),
            "api_key": api_key,
        }
    return result


def _persist_upstream_channels_to_env(channels: Dict[str, Dict[str, str]]) -> None:
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


def _override_from_dict(model_id: str, cfg: Any) -> Optional[OverrideRule]:
    """将任意配置对象转换为 OverrideRule，兼容字符串与字典两种形式。"""
    if isinstance(cfg, str):
        # 简写： "gpt-4o": "deepseek-chat"
        return OverrideRule(target_model=cfg)
    if not isinstance(cfg, dict):
        logger.warning("不支持的模型覆盖配置 %s: %r", model_id, cfg)
        return None
    return OverrideRule(
        channel=cfg.get("channel"),
        target_model=cfg.get("target_model") or cfg.get("model") or model_id,
    )


def _parse_override_map(raw: str) -> Dict[str, OverrideRule]:
    """解析环境变量中的 JSON 字符串，构造模型覆写规则字典。"""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MODEL_OVERRIDE_MAP 不是合法 JSON，已忽略：%s", raw)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("MODEL_OVERRIDE_MAP 应该是对象类型，当前值：%s", parsed)
        return {}

    result: Dict[str, OverrideRule] = {}
    for model_id, cfg in parsed.items():
        rule = _override_from_dict(model_id, cfg)
        if rule:
            result[model_id] = rule
    return result


override_map: Dict[str, OverrideRule] = _parse_override_map(MODEL_OVERRIDE_MAP_RAW)


def _load_persisted_overrides() -> Dict[str, OverrideRule]:
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
            rule = _override_from_dict(mid, cfg)
            if rule:
                result[mid] = rule
        return result
    except Exception as exc:
        logger.warning("读取覆写持久化文件失败：%s", exc)
        return {}


def _persist_overrides(overrides: Dict[str, OverrideRule]) -> None:
    """写入覆写规则到本地文件。"""
    serializable: Dict[str, Any] = {}
    for mid, rule in overrides.items():
        if not isinstance(rule, OverrideRule):
            continue
        serializable[mid] = {
            "channel": rule.channel,
            "target_model": rule.target_model,
        }
    OVERRIDE_STORE_PATH.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _require_upstream_key() -> None:
    """确保已配置默认上游 newapi 的鉴权 key。"""
    if not NEWAPI_API_KEY:
        raise RuntimeError("未设置 NEWAPI_API_KEY，无法向上游鉴权")


def _build_default_upstream_headers() -> Dict[str, str]:
    """构造发往默认上游 newapi 的基础请求头。"""
    return {
        "Authorization": f"Bearer {NEWAPI_API_KEY}",
        "Content-Type": "application/json",
    }


async def _safe_stream_thinking_v2(upstream_resp: httpx.Response):
    """基于 SSE JSON 结构，仅针对 thinking 段做覆写。

    - 聚合 `choices[].delta.content` 的前 10 个字符（原始 `<thinking>`）全部丢弃，只输出一次 `<think>`；
    - 首次出现 `</thinking>` 替换为 `</think>`；
    - 之后的内容全部原样透传。
    """
    open_tag_len = 10  # `<thinking>` 的长度
    dropped = 0  # 已丢弃的前缀字符数（跨 chunk 累积）
    prefix_emitted = False
    close_replaced = False

    def patch_content(text: Optional[str]) -> Optional[str]:
        nonlocal dropped, prefix_emitted, close_replaced
        if text is None or text == "":
            return text

        s = text
        out = ""

        # 1. 丢弃整体前 10 个字符，只输出一次 `<think>`
        if not prefix_emitted and dropped < open_tag_len:
            need = open_tag_len - dropped
            drop = min(len(s), need)
            s = s[drop:]
            dropped += drop
            if dropped >= open_tag_len:
                out += "<think>"
                prefix_emitted = True

        # 2. 将首次 `</thinking>` 替换为 `</think>`（只替一次）
        if prefix_emitted and not close_replaced and s:
            idx = s.find("</thinking>")
            if idx != -1:
                out += s[:idx]
                out += "</think>"
                close_replaced = True
                out += s[idx + len("</thinking>") :]
                return out

        # 3. 其余内容原样透传
        out += s
        return out

    try:
        async for line in upstream_resp.aiter_lines():
            # thinking 开头和结尾都已经处理完，后续行直接透传
            if prefix_emitted and close_replaced:
                yield (line + "\n").encode("utf-8")
                continue

            if not line.startswith("data:"):
                yield (line + "\n").encode("utf-8")
                continue

            raw = line[5:].lstrip()
            if raw.strip() == "[DONE]":
                yield (line + "\n").encode("utf-8")
                continue

            try:
                obj = json.loads(raw)
            except Exception:
                # 无法解析 JSON，保守透传
                yield (line + "\n").encode("utf-8")
                continue

            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str):
                        delta["content"] = patch_content(content)

            out_line = "data: " + json.dumps(obj, ensure_ascii=False)
            yield (out_line + "\n").encode("utf-8")
    except httpx.StreamClosed:
        logger.warning("上游流已关闭，提前结束推送")
    except Exception:
        logger.exception("转发流式响应时发生异常")
        raise


def _build_thinking_system_prefix(max_length: int) -> str:
    """根据思考长度构造 antml 思考系统前缀。"""
    return THINKING_SYSTEM_TEMPLATE.format(max_length=max_length)


def _inject_thinking_system_prefix(messages: Any, prefix: str):
    """在消息列表中为第一条 system 消息前置指定的思考系统前缀。"""
    if not isinstance(messages, list) or not prefix:
        return messages
    # 查找第一条 system 消息
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and not content.startswith(prefix):
                msg["content"] = prefix + content
            return messages
    # 没有 system 消息时，在最前面插入一条
    return [{"role": "system", "content": prefix}] + messages


def _strip_undefined_fields(payload: Any):
    """清理请求体中值为占位或 None 的字段，避免上游 400。"""
    placeholder = "[undefined]"
    if isinstance(payload, list):
        return [_strip_undefined_fields(item) for item in payload]
    if isinstance(payload, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in payload.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == placeholder:
                continue
            cleaned[k] = _strip_undefined_fields(v)
        return cleaned
    return payload


async def verify_proxy_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> None:
    """校验代理层的 API Key（支持 Authorization / X-Api-Key）。"""
    if not PROXY_API_KEY:
        return
    supplied: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1]
    if not supplied and x_api_key:
        supplied = x_api_key

    if supplied != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="代理 API Key 不匹配")


async def lifespan(app: FastAPI):
    """启动/关闭生命周期：检测配置并创建共享 HTTP 客户端。"""
    _require_upstream_key()
    timeout = httpx.Timeout(None, connect=20.0, read=None, write=None, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.http_client = client
        persisted = _load_persisted_overrides()
        app.state.override_map = persisted or override_map
        # 读取上游渠道配置
        app.state.upstream_channels = _load_upstream_channels_from_env()
        logger.info("上游 newapi 默认地址：%s", NEWAPI_BASE_URL)
        yield
        logger.info("代理已停止，HTTP 客户端已关闭")


app = FastAPI(title="newapi-openai-proxy", lifespan=lifespan)

# 挂载静态目录以提供 Web UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.info("未找到 static 目录，跳过 Web UI 挂载")


def _extract_thinking_max_length(payload: Dict[str, Any]) -> Optional[int]:
    """从请求体中提取思考预算，返回 max_thinking_length（支持顶层 thinking 和 providerOptions.*.thinking）。"""

    def _from_cfg(thinking_cfg: Any, source: str) -> Optional[int]:
        if not isinstance(thinking_cfg, dict):
            return None
        type_val = thinking_cfg.get("type")
        if not isinstance(type_val, str):
            return None
        t = type_val.lower()
        # 兼容 enabled / enable，其他（如 disabled/disable）视为关闭
        if t not in ("enabled", "enable"):
            return None
        budget = thinking_cfg.get("budget_tokens")
        try:
            budget_val = int(budget)
        except (TypeError, ValueError):
            logger.warning("thinking 配置 budget_tokens 非法：%r（source=%s）", budget, source)
            return None
        if budget_val <= 0:
            return None
        max_length = budget_val * THINKING_LENGTH_MULTIPLIER
        logger.info(
            "检测到 thinking 启用：source=%s, budget_tokens=%s, max_length=%s",
            source,
            budget_val,
            max_length,
        )
        return max_length

    # 1）优先支持 Cherry Studio 顶层字段：{"thinking": {type,budget_tokens}, ...}
    top_level = payload.get("thinking")
    max_len = _from_cfg(top_level, "thinking")
    if max_len is not None:
        return max_len

    # 2）兼容旧的 providerOptions.*.thinking 结构
    provider_options = payload.get("providerOptions")
    if not isinstance(provider_options, dict):
        return None

    for provider_id, provider_cfg in provider_options.items():
        if not isinstance(provider_cfg, dict):
            continue
        thinking_cfg = provider_cfg.get("thinking")
        max_len = _from_cfg(thinking_cfg, f"providerOptions[{provider_id}].thinking")
        if max_len is not None:
            return max_len

    return None


def apply_overrides(
    payload: Dict[str, Any],
    overrides: Dict[str, OverrideRule],
    thinking_max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """根据 override 规则修改模型 ID，并按需注入思考前缀。"""
    if not isinstance(payload, dict):
        return payload

    patched = dict(payload)
    model = patched.get("model")
    rule = overrides.get(model)
    if rule:
        # 模型别名重定向
        if rule.target_model:
            patched["model"] = rule.target_model

    # 思考系统前缀：仅当客户端显式开启 thinking 且提供预算时才注入
    if thinking_max_length is not None and thinking_max_length > 0:
        messages = patched.get("messages") or []
        if isinstance(messages, list):
            prefix = _build_thinking_system_prefix(thinking_max_length)
            patched["messages"] = _inject_thinking_system_prefix(messages, prefix)

    return patched


def _augment_models_response(upstream_payload: Any, overrides: Dict[str, OverrideRule]) -> Any:
    """在 /v1/models 返回值中追加代理定义的模型别名。"""
    if not isinstance(upstream_payload, dict):
        return upstream_payload
    data = upstream_payload.get("data")
    if not isinstance(data, list):
        return upstream_payload

    known_ids = {item.get("id") for item in data if isinstance(item, dict)}
    for alias, rule in overrides.items():
        if alias in known_ids:
            continue
        meta: Dict[str, Any] = {}
        if rule.channel:
            meta["channel"] = rule.channel

        model_obj: Dict[str, Any] = {
            "id": alias,
            "object": "model",
            "owned_by": "proxy-override",
            "alias_for": rule.target_model or alias,
        }
        if meta:
            model_obj["metadata"] = meta

        data.append(model_obj)

    upstream_payload["data"] = data
    return upstream_payload


def _extract_error(resp: httpx.Response) -> Any:
    """尽量提取上游错误消息为 JSON 内容。"""
    try:
        return resp.json()
    except Exception:
        return resp.text


def _extract_payload(resp: httpx.Response) -> Any:
    """尽量把响应解析为 JSON，否则返回文本。"""
    try:
        return resp.json()
    except Exception:
        return resp.text


def _override_map_to_dict(overrides: Dict[str, OverrideRule]) -> Dict[str, Any]:
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


@app.get("/health")
async def health() -> Dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Web UI 首页。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>newapi 代理</h2><p>未找到静态文件目录</p>")


@app.get("/v1/models")
async def list_models(_: None = Depends(verify_proxy_key), request: Request = None):
    """代理转发 /v1/models 请求并追加本地别名。"""
    client: httpx.AsyncClient = request.app.state.http_client
    url = f"{NEWAPI_BASE_URL}/v1/models"
    resp = await client.get(url, headers=_build_default_upstream_headers())
    payload = _augment_models_response(_extract_payload(resp), request.app.state.override_map)
    return JSONResponse(content=payload, status_code=resp.status_code)


@app.get("/overrides")
async def get_overrides(_: None = Depends(verify_proxy_key), request: Request = None):
    """获取当前覆写规则（持久化 + 内存）。"""
    return JSONResponse(content=_override_map_to_dict(request.app.state.override_map))


@app.post("/overrides")
async def save_overrides(
    overrides: Dict[str, Any] = Body(default_factory=dict),
    request: Request = None,
    _: None = Depends(verify_proxy_key),
):
    """保存覆写规则到本地文件，并更新内存配置。"""
    parsed = _parse_override_map(json.dumps(overrides))
    request.app.state.override_map = parsed
    _persist_overrides(parsed)
    logger.info("覆写规则已更新，条目数：%s", len(parsed))
    return JSONResponse(content={"status": "ok", "count": len(parsed)})


@app.get("/channels")
async def list_channels(_: None = Depends(verify_proxy_key), request: Request = None):
    """获取上游渠道配置（从内存，已与 .env 同步）。"""
    channels = getattr(request.app.state, "upstream_channels", {}) or {}
    return JSONResponse(content={"channels": channels})


@app.post("/channels")
async def save_channels(
    payload: Dict[str, Any] = Body(default_factory=dict),
    request: Request = None,
    _: None = Depends(verify_proxy_key),
):
    """保存上游渠道配置到内存与 .env。"""
    incoming = payload.get("channels") or {}
    if not isinstance(incoming, dict):
        raise HTTPException(status_code=400, detail="channels 字段应为对象")

    merged: Dict[str, Dict[str, str]] = {}
    for name, cfg in incoming.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            continue
        merged[name] = {
            "base_url": _strip_trailing_slash(base_url),
            "api_key": api_key,
        }

    request.app.state.upstream_channels = merged
    _persist_upstream_channels_to_env(merged)
    logger.info("上游渠道配置已更新，渠道数：%s", len(merged))
    return JSONResponse(content={"status": "ok", "count": len(merged)})


@app.post("/v1/chat/completions")
async def chat_completions(_: None = Depends(verify_proxy_key), request: Request = None):
    """代理转发 /v1/chat/completions，并根据客户端思考配置按需应用覆写。"""
    body = await request.json()

    # 1）从客户端请求体中提取 thinking 配置（是否启用 + 预算）
    thinking_max_length = _extract_thinking_max_length(body)

    # 2）按模型覆写规则 + 思考配置修改请求体
    overrides_map: Dict[str, OverrideRule] = request.app.state.override_map
    patched_body = apply_overrides(body, overrides_map, thinking_max_length)
    cleaned_body = _strip_undefined_fields(patched_body)
    is_stream = bool(cleaned_body.get("stream"))
    _log_payload("转发请求", cleaned_body)

    incoming_model = body.get("model")
    rule = overrides_map.get(incoming_model) or overrides_map.get(patched_body.get("model"))
    if incoming_model != patched_body.get("model"):
        logger.info("模型覆写: %s -> %s", incoming_model, patched_body.get("model"))

    # 3）根据渠道选择上游 base_url 和 api_key
    upstream_channels: Dict[str, Dict[str, str]] = getattr(request.app.state, "upstream_channels", {}) or {}
    base_url = NEWAPI_BASE_URL
    api_key = NEWAPI_API_KEY
    channel_name: Optional[str] = None
    if rule and rule.channel:
        channel_name = rule.channel
        cfg = upstream_channels.get(rule.channel)
        if isinstance(cfg, dict):
            if isinstance(cfg.get("base_url"), str):
                base_url = cfg["base_url"]
            if isinstance(cfg.get("api_key"), str):
                api_key = cfg["api_key"]

    base_url = _strip_trailing_slash(base_url or NEWAPI_BASE_URL)
    if not api_key:
        # 如果指定渠道缺少 api_key，仍然退回默认 NEWAPI_API_KEY
        api_key = NEWAPI_API_KEY
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if channel_name:
        headers["X-Channel"] = channel_name

    client: httpx.AsyncClient = request.app.state.http_client

    # 4）流式 / 非流式转发
    if is_stream:
        # 使用显式 send(stream=True) 保证在响应被消费完之前不会关闭上游连接
        request_obj = client.build_request("POST", url, headers=headers, json=cleaned_body)
        upstream_resp = await client.send(request_obj, stream=True)
        if upstream_resp.status_code >= 400:
            error_detail = await upstream_resp.aread()
            await upstream_resp.aclose()
            raise HTTPException(
                status_code=upstream_resp.status_code,
                detail=error_detail.decode("utf-8", errors="ignore"),
            )

        async def stream_upstream():
            try:
                # 仅当客户端显式开启 thinking 时，才对 <thinking> 标签进行覆写；否则完全透传
                if thinking_max_length is not None:
                    async for chunk in _safe_stream_thinking_v2(upstream_resp):
                        yield chunk
                else:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            stream_upstream(),
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            status_code=upstream_resp.status_code,
        )

    # 非流式
    resp = await client.post(url, headers=headers, json=cleaned_body)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_extract_error(resp))

    payload = _extract_payload(resp)
    _log_payload("上游响应", payload)

    media_type = resp.headers.get("content-type", "application/json")
    if isinstance(payload, (dict, list)):
        return JSONResponse(content=payload, status_code=resp.status_code, media_type=media_type)
    return Response(content=payload, status_code=resp.status_code, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)

