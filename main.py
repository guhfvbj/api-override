import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Load .env for local dev
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newapi-proxy")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_PORT = int(os.getenv("PORT", 14300))

# Environment configs
NEWAPI_BASE_URL = os.getenv("NEWAPI_BASE_URL", "https://api.newapi.ai")
NEWAPI_API_KEY = os.getenv("NEWAPI_API_KEY")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
MODEL_OVERRIDE_MAP_RAW = os.getenv("MODEL_OVERRIDE_MAP", "{}")


@dataclass
class OverrideRule:
    target_model: Optional[str] = None
    prepend_system: Optional[str] = None
    force_params: Dict[str, Any] = field(default_factory=dict)
    replacements: Dict[str, str] = field(default_factory=dict)


def _parse_override_map(raw: str) -> Dict[str, OverrideRule]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MODEL_OVERRIDE_MAP is not valid JSON, ignored: %s", raw)
        return {}

    if not isinstance(parsed, dict):
        logger.warning("MODEL_OVERRIDE_MAP should be a dict, got: %s", parsed)
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
            )
        else:
            logger.warning("Ignore override item %s: %s", model_id, cfg)
    return result


override_map = _parse_override_map(MODEL_OVERRIDE_MAP_RAW)


def _strip_trailing_slash(url: str) -> str:
    return url[:-1] if url.endswith("/") else url


NEWAPI_BASE_URL = _strip_trailing_slash(NEWAPI_BASE_URL)


def _require_upstream_key() -> None:
    if not NEWAPI_API_KEY:
        raise RuntimeError("Missing NEWAPI_API_KEY for upstream forwarding")


def _build_upstream_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NEWAPI_API_KEY}",
        "Content-Type": "application/json",
    }


def _apply_replacements_to_messages(messages: Any, replacements: Dict[str, str]):
    if not replacements or not isinstance(messages, list):
        return messages

    patched_messages = []
    for msg in messages:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            text_val = msg["content"]
            for src, dst in replacements.items():
                if src is None or dst is None:
                    continue
                text_val = text_val.replace(str(src), str(dst))
            patched_messages.append({**msg, "content": text_val})
        else:
            patched_messages.append(msg)
    return patched_messages


async def verify_proxy_key(
    authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)
) -> None:
    if not PROXY_API_KEY:
        return  # allow if not set
    supplied: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1]
    if not supplied and x_api_key:
        supplied = x_api_key

    if supplied != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid proxy API Key")


async def lifespan(app: FastAPI):
    _require_upstream_key()
    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.http_client = client
        app.state.override_map = override_map
        logger.info("Proxy started, upstream: %s", NEWAPI_BASE_URL)
        yield
        logger.info("Proxy client closed")


app = FastAPI(title="newapi-openai-proxy", lifespan=lifespan)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.info("static directory not found, skip mount")


def apply_overrides(payload: Dict[str, Any], overrides: Dict[str, OverrideRule]) -> Dict[str, Any]:
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

    return patched


def _augment_models_response(upstream_payload: Any, overrides: Dict[str, OverrideRule]) -> Any:
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
    try:
        return resp.json()
    except Exception:
        return resp.text


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h2>newapi proxy</h2><p>static assets missing.</p>")


@app.get("/v1/models")
async def list_models(_: None = Depends(verify_proxy_key), request: Request = None):
    client: httpx.AsyncClient = request.app.state.http_client
    url = f"{NEWAPI_BASE_URL}/v1/models"
    resp = await client.get(url, headers=_build_upstream_headers())
    payload = _augment_models_response(_extract_payload(resp), request.app.state.override_map)
    return JSONResponse(content=payload, status_code=resp.status_code)


def _extract_payload(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


@app.post("/v1/chat/completions")
async def chat_completions(_: None = Depends(verify_proxy_key), request: Request = None):
    body = await request.json()
    patched_body = apply_overrides(body, request.app.state.override_map)
    is_stream = bool(patched_body.get("stream"))

    client: httpx.AsyncClient = request.app.state.http_client
    url = f"{NEWAPI_BASE_URL}/v1/chat/completions"
    headers = _build_upstream_headers()

    if is_stream:
        async with client.stream("POST", url, headers=headers, json=patched_body) as upstream_resp:
            if upstream_resp.status_code >= 400:
                error_detail = await upstream_resp.aread()
                raise HTTPException(
                    status_code=upstream_resp.status_code,
                    detail=error_detail.decode("utf-8", errors="ignore"),
                )

            return StreamingResponse(
                upstream_resp.aiter_raw(),
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
                status_code=upstream_resp.status_code,
            )

    resp = await client.post(url, headers=headers, json=patched_body)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_extract_error(resp))

    payload = _extract_payload(resp)
    media_type = resp.headers.get("content-type", "application/json")
    if isinstance(payload, (dict, list)):
        return JSONResponse(content=payload, status_code=resp.status_code, media_type=media_type)
    return Response(content=payload, status_code=resp.status_code, media_type=media_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)
