# newapi OpenAI-compatible proxy (api-override)

FastAPI-based lightweight proxy with OpenAI-compatible endpoints. Supports per-model override, keyword search/replace, independent proxy API key, and ships with a small Web UI (default port 14300).

## Features
- `/v1/models`: pass-through to upstream newapi, and append alias models defined in overrides.
- `/v1/chat/completions`: apply overrides (model remap, prepend system prompt, force params, keyword search/replace) before forwarding.
- Proxy auth: optional `PROXY_API_KEY` required via `Authorization: Bearer <key>` or `X-Api-Key`.
- Web UI at `/`: call proxy, view models, chat interactively.

## Quick start (Linux, default port 14300)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (example):
```bash
NEWAPI_BASE_URL=https://api.newapi.ai
NEWAPI_API_KEY=sk-xxxx
PROXY_API_KEY=proxy-123  # optional, empty means no auth
MODEL_OVERRIDE_MAP={
  "gpt-4o":{
    "target_model":"deepseek-chat",
    "prepend_system":"please reply in Chinese",
    "force_params":{"temperature":0.3},
    "replace":{"foo":"bar"}
  }
}
```

Run:
```bash
python main.py
# or uvicorn main:app --host 0.0.0.0 --port 14300
```

Call:
```bash
curl -H "Authorization: Bearer proxy-123" http://<host>:14300/v1/models
curl -H "Authorization: Bearer proxy-123" -H "Content-Type: application/json"   -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'   http://<host>:14300/v1/chat/completions
```

## Keyword search/replace
Use `replace` (or `replacements`) in a model entry under `MODEL_OVERRIDE_MAP` as shown above. Every message `content` string in the request will run ordered `.replace(old, new)` before forwarding.

## Deploy tips
- Keep `.env` with upstream keys; use `PROXY_API_KEY` to guard the proxy.
- Place Web UI assets under `static/`; root `/` will serve `static/index.html`.
- Behind nginx/caddy, forward `Authorization` headers and enable HTTP/2 where possible.
