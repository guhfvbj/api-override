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

# 上游 newapi 相关配置
NEWAPI_BASE_URL = os.getenv("NEWAPI_BASE_URL", "https://api.newapi.ai")
NEWAPI_API_KEY = os.getenv("NEWAPI_API_KEY")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
MODEL_OVERRIDE_MAP_RAW = os.getenv("MODEL_OVERRIDE_MAP", "{}")


@dataclass
class OverrideRule:
    """模型覆盖规则，支持重定向模型、追加 system、强制参数、请求与响应替换。"""

    target_model: Optional[str] = None
    prepend_system: Optional[str] = None
    force_params: Dict[str, Any] = field(default_factory=dict)
    replacements: Dict[str, str] = field(default_factory=dict)
    response_replacements: Dict[str, str] = field(default_factory=dict)


def _parse_override_map(raw: str) -> Dict[str, OverrideRule]:
    """解析环境变量中的 JSON 字符串，构造成模型覆盖规则字典。"""
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
        if isinstance(cfg, str):
            result[model_id] = OverrideRule(target_model=cfg)
            continue

        if isinstance(cfg, dict):
            result[model_id] = OverrideRule(
                target_model=cfg.get("target_model") or cfg.get("model") or model_id,
                prepend_system=cfg.get("prepend_system"),
                force_params=cfg.get("force_params") or {},
                replacements=cfg.get("replace") or cfg.get("replacements") or {},
                response_replacements=cfg.get("response_replace")
                or cfg.get("response_replacements")
                or {},
            )
        else:
            logger.warning("不支持的模型覆盖配置 %s: %s", model_id, cfg)
    return result


override_map = _parse_override_map(MODEL_OVERRIDE_MAP_RAW)


def _strip_trailing_slash(url: str) -> str:
    """去掉末尾的斜杠，避免重复拼接路径。"""
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


def _load_persisted_overrides() -> Dict[str, OverrideRule]:
    """从本地持久化文件读取覆写规则。"""
    if not OVERRIDE_STORE_PATH.exists():
        return {}
    try:
        raw = OVERRIDE_STORE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                mid: OverrideRule(
                    target_model=cfg.get("target_model") or cfg.get("model") or mid,
                    prepend_system=cfg.get("prepend_system"),
                    force_params=cfg.get("force_params") or {},
                    replacements=cfg.get("replace") or cfg.get("replacements") or {},
                    response_replacements=cfg.get("response_replace")
                    or cfg.get("response_replacements")
                    or {},
                )
                for mid, cfg in data.items()
                if isinstance(cfg, dict)
            }
    except Exception as exc:
        logger.warning("读取覆写持久化文件失败：%s", exc)
    return {}


def _persist_overrides(overrides: Dict[str, OverrideRule]) -> None:
    """写入覆写规则到本地文件。"""
    serializable = {}
    for mid, rule in overrides.items():
        if not isinstance(rule, OverrideRule):
            continue
        serializable[mid] = {
            "target_model": rule.target_model,
            "prepend_system": rule.prepend_system,
            "force_params": rule.force_params,
            "replacements": rule.replacements,
            "response_replacements": rule.response_replacements,
        }
    OVERRIDE_STORE_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_upstream_key() -> None:
    """确保已配置上游 newapi 的鉴权 key。"""
    if not NEWAPI_API_KEY:
        raise RuntimeError("未设置 NEWAPI_API_KEY，无法向上游鉴权")


def _build_upstream_headers() -> Dict[str, str]:
    """构造发往上游 newapi 的基础请求头。"""
    return {
        "Authorization": f"Bearer {NEWAPI_API_KEY}",
        "Content-Type": "application/json",
    }


async def _safe_stream(upstream_resp: httpx.Response, replacements: Optional[Dict[str, str]] = None):
    """按 SSE data 行解析并覆写 JSON 字段（含 text），兼容 StreamClosed。"""
    if not replacements:
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        except httpx.StreamClosed:
            logger.warning("上游流已关闭，提前结束推送")
        except Exception:
            logger.exception("转发流式响应时发生异常")
            raise
        return

    try:
        async for line in upstream_resp.aiter_lines():
            if line.startswith("data:"):
                raw = line[5:].lstrip()
                if raw.strip() == "[DONE]":
                    yield (line + "\n").encode("utf-8")
                    continue

                try:
                    obj = json.loads(raw)
                    patched = _apply_replacements_to_any(obj, replacements)
                    out = "data: " + json.dumps(patched, ensure_ascii=False)
                except Exception:
                    patched_str = raw
                    for src, dst in replacements.items():
                        if src is None or dst is None:
                            continue
                        patched_str = patched_str.replace(str(src), str(dst))
                    out = "data: " + patched_str

                yield (out + "\n").encode("utf-8")
            else:
                patched = line
                for src, dst in replacements.items():
                    if src is None or dst is None:
                        continue
                    patched = patched.replace(str(src), str(dst))
                yield (patched + "\n").encode("utf-8")
    except httpx.StreamClosed:
        logger.warning("上游流已关闭，提前结束推送")
    except Exception:
        logger.exception("转发流式响应时发生异常")
        raise


def _apply_replacements_to_messages(messages: Any, replacements: Dict[str, str]):
    """对消息列表中的 content 逐条做字符串替换。"""
    return _apply_replacements_to_any(messages, replacements)


def _apply_replacements_to_any(payload: Any, replacements: Dict[str, str]):
    """在任意结构中替换字符串，兼容请求与响应。"""
    if not replacements:
        return payload

    def replace_text(text: str) -> str:
        patched = text
        for src, dst in replacements.items():
            if src is None or dst is None:
                continue
            patched = patched.replace(str(src), str(dst))
        return patched

    if isinstance(payload, str):
        return replace_text(payload)
    if isinstance(payload, list):
        return [_apply_replacements_to_any(item, replacements) for item in payload]
    if isinstance(payload, dict):
        return {k: _apply_replacements_to_any(v, replacements) for k, v in payload.items()}
    return payload


def _strip_undefined_fields(payload: Any):
    """清理请求体中值为占位符/None的字段，避免上游 400。"""
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
    authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)
) -> None:
    """校验代理层的 API Key（支持 Authorization/X-Api-Key）。"""
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
        logger.info("上游 newapi 地址：%s", NEWAPI_BASE_URL)
        yield
        logger.info("代理已停止，HTTP 客户端已关闭")


app = FastAPI(title="newapi-openai-proxy", lifespan=lifespan)

# 挂载静态目录以提供 Web UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.info("未找到 static 目录，跳过 Web UI 挂载")


def apply_overrides(payload: Dict[str, Any], overrides: Dict[str, OverrideRule]) -> Dict[str, Any]:
    """根据 override 规则修改模型 ID、追加 system、强制参数与内容替换。"""
    if not isinstance(payload, dict):
        return payload

    patched = dict(payload)
    model = patched.get("model")
    rule = overrides.get(model)
    if not rule:
        return patched

    if rule.target_model:
        patched["model"] = rule.target_model

    if rule.prepend_system:
        messages = patched.get("messages") or []
        if isinstance(messages, list):
            messages = [{"role": "system", "content": rule.prepend_system}] + messages
        patched["messages"] = messages

    for key, value in (rule.force_params or {}).items():
        patched[key] = value

    if rule.replacements:
        messages = patched.get("messages")
        patched["messages"] = _apply_replacements_to_messages(messages, rule.replacements)
        logger.info("请求替换已应用，模型=%s，规则数=%s", model, len(rule.replacements))

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
        data.append(
            {
                "id": alias,
                "object": "model",
                "owned_by": "proxy-override",
                "alias_for": rule.target_model or alias,
            }
        )
    upstream_payload["data"] = data
    return upstream_payload


def _extract_error(resp: httpx.Response) -> Any:
    """尽量提取上游错误消息的 JSON 内容。"""
    try:
        return resp.json()
    except Exception:
        return resp.text


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
    resp = await client.get(url, headers=_build_upstream_headers())
    payload = _augment_models_response(_extract_payload(resp), request.app.state.override_map)
    return JSONResponse(content=payload, status_code=resp.status_code)


@app.get("/overrides")
async def get_overrides(_: None = Depends(verify_proxy_key), request: Request = None):
    """获取当前覆写规则（持久化+内存）。"""
    return JSONResponse(content=_override_map_to_dict(request.app.state.override_map))


@app.post("/overrides")
async def save_overrides(
    overrides: Dict[str, Any] = Body(default_factory=dict), request: Request = None, _: None = Depends(verify_proxy_key)
):
    """保存覆写规则到本地文件，并更新内存配置。"""
    parsed = _parse_override_map(json.dumps(overrides))
    request.app.state.override_map = parsed
    _persist_overrides(parsed)
    logger.info("覆写规则已更新，条目数：%s", len(parsed))
    return JSONResponse(content={"status": "ok", "count": len(parsed)})


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
            "target_model": rule.target_model,
            "prepend_system": rule.prepend_system,
            "force_params": rule.force_params,
            "replacements": rule.replacements,
            "response_replacements": rule.response_replacements,
        }
    return result


@app.post("/v1/chat/completions")
async def chat_completions(_: None = Depends(verify_proxy_key), request: Request = None):
    """代理转发 /v1/chat/completions，并在发送前应用 override。"""
    body = await request.json()
    patched_body = apply_overrides(body, request.app.state.override_map)
    cleaned_body = _strip_undefined_fields(patched_body)
    is_stream = bool(cleaned_body.get("stream"))
    _log_payload("转发请求体", cleaned_body)
    incoming_model = body.get("model")
    rule = request.app.state.override_map.get(incoming_model) or request.app.state.override_map.get(
        patched_body.get("model")
    )
    if incoming_model != patched_body.get("model"):
        logger.info("模型覆写: %s -> %s", incoming_model, patched_body.get("model"))

    client: httpx.AsyncClient = request.app.state.http_client
    url = f"{NEWAPI_BASE_URL}/v1/chat/completions"
    headers = _build_upstream_headers()

    # ?????????????? JSON ?????????? text/thinking???????
    response_repls: Dict[str, str] = {}
    if rule:
        if rule.response_replacements:
            response_repls.update(rule.response_replacements)
        if rule.replacements:
            response_repls.update(rule.replacements)

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
                async for chunk in _safe_stream(upstream_resp, response_repls):
                    yield chunk
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(
            stream_upstream(),
            media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            status_code=upstream_resp.status_code,
        )

    resp = await client.post(url, headers=headers, json=cleaned_body)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_extract_error(resp))

    payload = _extract_payload(resp)
    _log_payload("上游响应体", payload)
    if response_repls:
        payload = _apply_replacements_to_any(payload, response_repls)
        logger.info("响应替换已应用，模型=%s，规则数=%s", incoming_model, len(response_repls))
    media_type = resp.headers.get("content-type", "application/json")
    if isinstance(payload, (dict, list)):
        return JSONResponse(content=payload, status_code=resp.status_code, media_type=media_type)
    return Response(content=payload, status_code=resp.status_code, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)
