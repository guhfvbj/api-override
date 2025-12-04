import json
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from channels import (
    choose_upstream_for_request,
    load_upstream_channels_from_env,
    persist_upstream_channels_to_env,
)
from config import (
    DEFAULT_PORT,
    NEWAPI_API_KEY,
    NEWAPI_BASE_URL,
    PROXY_API_KEY,
    STATIC_DIR,
    logger,
)
from model_store import load_initial_model_store, persist_model_store
from overrides import (
    OverrideRule,
    apply_overrides,
    load_initial_override_map,
    override_map_to_dict,
    parse_override_map,
    persist_overrides,
)
from thinking import (
    extract_thinking_max_length,
    normalize_thinking_model_name,
    safe_stream_thinking,
)
from utils import (
    extract_error,
    extract_payload,
    log_payload,
    strip_trailing_slash,
    strip_undefined_fields,
)

# 带 -thinking/-think 后缀强制思考模式时的默认预算
FORCED_THINKING_DEFAULT_BUDGET = 16000


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
    """应用生命周期：初始化 HTTP 客户端和内存配置。"""
    # 启动时从环境变量加载上游渠道配置（可通过 /channels 更新）
    app.state.upstream_channels = load_upstream_channels_from_env()
    # 加载统一模型/别名列表（优先持久化文件，其次环境变量）
    app.state.model_store = load_initial_model_store()

    timeout = httpx.Timeout(None, connect=20.0, read=None, write=None, pool=None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.http_client = client

        # 覆写规则优先使用本地持久化文件，没有则回退到环境变量
        app.state.override_map = load_initial_override_map()

        logger.info(
            "已加载上游渠道：%s",
            ", ".join(sorted(app.state.upstream_channels.keys())) or "（暂无）",
        )
        yield
        logger.info("应用停止，HTTP 客户端已关闭")


app = FastAPI(title="newapi-openai-proxy", lifespan=lifespan)


# 挂载静态目录，提供 Web UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.info("未找到 static 目录，将不会提供 Web UI 静态文件")


@app.get("/health")
async def health() -> Dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Web UI 首页。"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>newapi 代理</h2><p>未找到静态文件目录</p>")


def _custom_models_as_list(custom_models: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """将按渠道分组的自定义模型转换为列表并补足 metadata.channel。"""
    result: List[Dict[str, Any]] = []
    if not isinstance(custom_models, dict):
        return result
    for ch_name, items in custom_models.items():
        if not isinstance(items, list):
            continue
        for m in items:
            if not isinstance(m, dict):
                continue
            model = dict(m)
            meta = model.get("metadata") if isinstance(model.get("metadata"), dict) else {}
            if ch_name:
                meta["channel"] = ch_name
            if meta:
                model["metadata"] = meta
            result.append(model)
    return result


@app.get("/v1/models")
async def list_models(
    _: None = Depends(verify_proxy_key),
    request: Request = None,
    channel: Optional[str] = Query(None, description="可选：仅拉取指定渠道的模型"),
) -> JSONResponse:
    """返回自定义模型列表（仅持久化 model_store）。"""
    model_store: Dict[str, List[Dict[str, Any]]] = getattr(request.app.state, "model_store", {}) or {}
    models = _custom_models_as_list(model_store)

    # 渠道过滤：若指定 channel 参数，则只返回该渠道的模型
    if channel:
        models = [m for m in models if (m.get("metadata", {}) or {}).get("channel") == channel]

    # 1）构建 {channel: 被别名指向的原始模型 ID 集合}
    alias_targets_by_channel: Dict[str, set] = {}
    for m in models:
        if not isinstance(m, dict):
            continue
        alias_for = m.get("alias_for")
        if not isinstance(alias_for, str) or not alias_for:
            continue
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        ch_name = meta.get("channel") or ""
        if ch_name not in alias_targets_by_channel:
            alias_targets_by_channel[ch_name] = set()
        alias_targets_by_channel[ch_name].add(alias_for)

    # 2）同一渠道内，如某个模型 id 已出现在 alias_for 集合中，则仅保留别名条目，隐藏原始条目
    filtered: List[Dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        ch_name = meta.get("channel") or ""
        mid = m.get("id")
        if (
            not m.get("alias_for")
            and isinstance(mid, str)
            and mid in alias_targets_by_channel.get(ch_name, set())
        ):
            continue
        filtered.append(m)

    payload: Dict[str, Any] = {"object": "list", "data": filtered}
    return JSONResponse(content=payload, status_code=200)


@app.get("/overrides")
async def get_overrides(_: None = Depends(verify_proxy_key), request: Request = None):
    """获取当前覆写规则（持久化 + 内存）。"""
    return JSONResponse(content=override_map_to_dict(request.app.state.override_map))


@app.post("/overrides")
async def save_overrides(
    overrides: Dict[str, Any] = Body(default_factory=dict),
    request: Request = None,
    _: None = Depends(verify_proxy_key),
):
    """保存覆写规则到本地文件，并更新内存配置。"""
    parsed = parse_override_map(json.dumps(overrides))
    request.app.state.override_map = parsed
    persist_overrides(parsed)
    logger.info("覆写规则已更新，条目数：%s", len(parsed))
    return JSONResponse(content={"status": "ok", "count": len(parsed)})


@app.get("/channels")
async def list_channels(_: None = Depends(verify_proxy_key), request: Request = None):
    """获取上游渠道配置（内存中的当前值）。"""
    channels = getattr(request.app.state, "upstream_channels", {}) or {}
    return JSONResponse(content={"channels": channels})


@app.post("/channels")
async def save_channels(
    payload: Dict[str, Any] = Body(default_factory=dict),
    request: Request = None,
    _: None = Depends(verify_proxy_key),
):
    """更新上游渠道配置到内存和 .env。

    额外逻辑：删除渠道时会同步清理绑定该渠道的别名配置。
    """
    incoming = payload.get("channels") or {}
    if not isinstance(incoming, dict):
        raise HTTPException(status_code=400, detail="channels 字段应为对象")

    old_channels: Dict[str, Dict[str, str]] = getattr(request.app.state, "upstream_channels", {}) or {}

    merged: Dict[str, Dict[str, str]] = {}
    for name, cfg in incoming.items():
        if not isinstance(name, str) or not isinstance(cfg, dict):
            continue
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            continue
        merged[name] = {
            "base_url": strip_trailing_slash(base_url),
            "api_key": api_key,
        }

    # 更新渠道配置
    request.app.state.upstream_channels = merged
    persist_upstream_channels_to_env(merged)

    # 计算被删除的渠道，并同步清理对应别名配置
    removed_channels = {name for name in old_channels.keys() if name not in merged}
    if removed_channels:
        overrides_map: Dict[str, OverrideRule] = getattr(request.app.state, "override_map", {}) or {}
        new_overrides: Dict[str, OverrideRule] = {}
        removed_count = 0
        for mid, rule in overrides_map.items():
            if isinstance(rule, OverrideRule) and rule.channel in removed_channels:
                removed_count += 1
                continue
            new_overrides[mid] = rule
        if removed_count:
            request.app.state.override_map = new_overrides
            persist_overrides(new_overrides)
            logger.info("因删除渠道 %s 已清理别名条目数：%s", ",".join(sorted(removed_channels)), removed_count)

    logger.info("上游渠道配置已更新，渠道数：%s", len(merged))
    return JSONResponse(content={"status": "ok", "count": len(merged)})


@app.get("/v1/upstream-models")
async def upstream_models(
    _: None = Depends(verify_proxy_key),
    request: Request = None,
    channel: Optional[str] = Query(None, description="可选：仅拉取指定渠道的模型"),
) -> JSONResponse:
    """从上游拉取模型列表（不包含本地自定义），用于导入。"""
    client: httpx.AsyncClient = request.app.state.http_client
    upstream_channels: Dict[str, Dict[str, str]] = getattr(request.app.state, "upstream_channels", {}) or {}

    upstreams: List[Tuple[str, str, Optional[str]]] = []
    if channel:
        cfg = upstream_channels.get(channel)
        if not isinstance(cfg, dict):
            raise HTTPException(status_code=400, detail=f"未知渠道: {channel}")
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key")
        if not isinstance(base_url, str) or not isinstance(api_key, str):
            raise HTTPException(status_code=400, detail=f"渠道 {channel} 缺少 base_url 或 api_key")
        upstreams.append((base_url, api_key, channel))
    else:
        for name, cfg in upstream_channels.items():
            if not isinstance(cfg, dict):
                continue
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            if isinstance(base_url, str) and isinstance(api_key, str):
                upstreams.append((base_url, api_key, name))
        if not upstreams and NEWAPI_BASE_URL and NEWAPI_API_KEY:
            upstreams.append((NEWAPI_BASE_URL, NEWAPI_API_KEY, None))

    if not upstreams:
        raise HTTPException(status_code=500, detail="没有可用的上游渠道或 NEWAPI 配置")

    aggregated_models: List[Dict[str, Any]] = []
    any_success = False
    for base_url, api_key, ch_name in upstreams:
        url = f"{strip_trailing_slash(base_url)}/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if ch_name:
            headers["X-Channel"] = ch_name
        try:
            resp = await client.get(url, headers=headers)
        except Exception as exc:
            logger.warning("拉取渠道 %s 模型失败（网络异常）：%s", ch_name or "default", exc)
            continue

        if resp.status_code >= 400:
            logger.warning(
                "拉取渠道 %s 模型失败，状态码=%s，响应=%s",
                ch_name or "default",
                resp.status_code,
                extract_error(resp),
            )
            continue

        any_success = True
        payload_any = extract_payload(resp)
        if not isinstance(payload_any, dict):
            continue
        data = payload_any.get("data")
        if not isinstance(data, list):
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            if ch_name:
                meta["channel"] = ch_name
            if meta:
                item["metadata"] = meta
            aggregated_models.append(item)

    if not any_success:
        raise HTTPException(status_code=502, detail="从任何上游获取 /v1/models 都失败了")

    return JSONResponse(content={"object": "list", "data": aggregated_models}, status_code=200)


@app.get("/custom-models")
async def get_custom_models(_: None = Depends(verify_proxy_key), request: Request = None):
    """兼容旧接口：返回统一模型存储。"""
    models = getattr(request.app.state, "model_store", {}) or {}
    return JSONResponse(content=models)


@app.post("/custom-models")
async def save_custom_models(
    payload: Dict[str, Any] = Body(default_factory=dict),
    request: Request = None,
    _: None = Depends(verify_proxy_key),
):
    """替换并持久化模型/别名列表（按渠道分组）。"""
    incoming: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(payload, dict):
        for ch, items in payload.items():
            if not isinstance(items, list):
                continue
            normalized: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                mid = it.get("id")
                if not isinstance(mid, str) or not mid.strip():
                    continue
                model_obj = {
                    "id": mid.strip(),
                    "object": it.get("object") or "model",
                    "owned_by": it.get("owned_by") or "local",
                }
                alias_for = it.get("alias_for")
                if isinstance(alias_for, str) and alias_for.strip():
                    model_obj["alias_for"] = alias_for.strip()
                meta = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
                if isinstance(ch, str) and ch:
                    meta["channel"] = ch
                if meta:
                    model_obj["metadata"] = meta
                normalized.append(model_obj)
            if normalized:
                incoming[ch if isinstance(ch, str) else ""] = normalized

    request.app.state.model_store = incoming
    persist_model_store(incoming)

    # 同步更新覆写规则 (OverrideMap)
    new_overrides: Dict[str, OverrideRule] = {}
    for ch, items in incoming.items():
        for it in items:
            mid = it.get("id")
            alias_for = it.get("alias_for")
            if mid:
                # 如果有 alias_for，则映射到 alias_for；否则保持原名
                target = alias_for if alias_for else mid
                # 只要该模型归属于某个渠道（ch），或者有别名定义，就建立路由规则
                # 注意：如果 ch 为空且没有 alias_for，则不建立规则（走默认路由）
                if ch or alias_for:
                    new_overrides[mid] = OverrideRule(channel=ch, target_model=target)
    
    request.app.state.override_map = new_overrides
    persist_overrides(new_overrides)

    logger.info("模型存储已更新：%s，覆写规则已同步：%s", sum(len(v) for v in incoming.values()), len(new_overrides))
    return JSONResponse(content={"status": "ok", "count": sum(len(v) for v in incoming.values())})


@app.post("/v1/chat/completions")
async def chat_completions(_: None = Depends(verify_proxy_key), request: Request = None):
    """代理转发 /v1/chat/completions，并根据客户端思考配置按需应用覆写。"""
    body = await request.json()

    # 0）规范化模型名：支持通过模型后缀 -thinking/-think 强制启用思考模式
    raw_model = body.get("model")
    normalized_model, force_thinking = normalize_thinking_model_name(raw_model)
    if force_thinking and normalized_model:
        body["model"] = normalized_model

    # 1）若通过模型后缀强制启用思考，则忽略请求体中的 thinking 配置，直接使用默认预算；
    #    否则按照请求体的 thinking/think 字段解析预算。
    if force_thinking:
        thinking_max_length = FORCED_THINKING_DEFAULT_BUDGET
    else:
        thinking_max_length = extract_thinking_max_length(body)

    # 2）按模型覆写规则 + 思考配置修改请求体
    overrides_map: Dict[str, OverrideRule] = request.app.state.override_map
    patched_body = apply_overrides(body, overrides_map, thinking_max_length)
    cleaned_body = strip_undefined_fields(patched_body)
    is_stream = bool(cleaned_body.get("stream"))

    # 預先記錄覆寫後的模型 ID，供後續多渠道隨機路由使用
    model_id_after_override = patched_body.get("model")
    same_model_after_override = model_id_after_override == body.get("model")
    candidate_channels_for_model: List[str] = []
    if same_model_after_override and isinstance(model_id_after_override, str):
        model_store: Dict[str, List[Dict[str, Any]]] = getattr(request.app.state, "model_store", {}) or {}
        for ch_name, items in model_store.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get("id") == model_id_after_override:
                    candidate_channels_for_model.append(ch_name)
                    break
    log_payload("转发请求", cleaned_body)

    incoming_model = body.get("model")
    rule = overrides_map.get(incoming_model) or overrides_map.get(patched_body.get("model"))
    # 日志中保留原始模型名，便于排查是否通过 -thinking/-think 触发
    log_from = raw_model if raw_model is not None else incoming_model
    if log_from != patched_body.get("model"):
        logger.info("模型覆写: %s -> %s", log_from, patched_body.get("model"))

    # 3）根据渠道选择上游 base_url / api_key
    upstream_channels: Dict[str, Dict[str, str]] = getattr(request.app.state, "upstream_channels", {}) or {}
    base_url, api_key, channel_name = choose_upstream_for_request(rule, upstream_channels)

    # 如果当前模型名覆写前后保持一致，说明仅影响路由渠道；
    # 当该模型在多个渠道下均可用时，在这些渠道中随机选择一个进行调用。
    if len(candidate_channels_for_model) > 1 and isinstance(model_id_after_override, str):
        valid_candidates = [ch for ch in candidate_channels_for_model if ch in upstream_channels]
        if len(valid_candidates) > 1:
            chosen_channel = random.choice(valid_candidates)
            cfg = upstream_channels.get(chosen_channel) or {}
            new_base_url = cfg.get("base_url")
            new_api_key = cfg.get("api_key")
            if isinstance(new_base_url, str) and isinstance(new_api_key, str):
                base_url = strip_trailing_slash(new_base_url)
                api_key = new_api_key
                channel_name = chosen_channel
                logger.info("模型 %s 在多个渠道可用，随机选择渠道：%s", model_id_after_override, chosen_channel)

    url = f"{strip_trailing_slash(base_url)}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if channel_name:
        headers["X-Channel"] = channel_name

    client: httpx.AsyncClient = request.app.state.http_client

    # 4）流式 / 非流式转发
    if is_stream:
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
                # 启用思考模式时，对 <thinking> 标签进行覆写；否则完全透传
                if thinking_max_length is not None:
                    async for chunk in safe_stream_thinking(upstream_resp):
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
        raise HTTPException(status_code=resp.status_code, detail=extract_error(resp))

    payload = extract_payload(resp)
    log_payload("上游响应", payload)

    media_type = resp.headers.get("content-type", "application/json")
    if isinstance(payload, (dict, list)):
        return JSONResponse(content=payload, status_code=resp.status_code, media_type=media_type)
    return Response(content=payload, status_code=resp.status_code, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)
