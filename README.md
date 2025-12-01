# newapi OpenAI 代理（api-override）

基于 FastAPI 的轻量代理，兼容 OpenAI API，转发到 newapi.ai，并支持模型别名/参数覆盖/文本替换，附带 Web UI（默认端口 14300）。

## 功能特点
- 兼容 `/v1/models` 与 `/v1/chat/completions`，请求转发到上游 newapi。
- 通过 `MODEL_OVERRIDE_MAP` 配置模型别名、前置 system、强制参数、内容替换（支持 `replace`/`replacements`）。
- 代理层可选鉴权：设置 `PROXY_API_KEY` 后支持 `Authorization: Bearer <key>` 或 `X-Api-Key`。
- 根路径 `/` 提供内置 Web UI（静态文件在 `static/index.html`），便于拉取模型和发起聊天。
- 启动前强制检查 `NEWAPI_API_KEY`，避免漏配上游密钥。

## 快速开始（以 Linux/macOS 为例，默认端口 14300）
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

创建 `.env`（示例）
```bash
NEWAPI_BASE_URL=https://api.newapi.ai
NEWAPI_API_KEY=sk-xxxx
PROXY_API_KEY=proxy-123              # 可选：代理鉴权用
MODEL_OVERRIDE_MAP={
  "gpt-4o":{
    "target_model":"deepseek-chat",
    "prepend_system":"请始终用中文回答",
    "force_params":{"temperature":0.3},
    "replace":{"敏感词":"替换词"}
  }
}
```

启动服务
```bash
python main.py
# 或 uvicorn main:app --host 0.0.0.0 --port 14300
```

接口示例
```bash
curl -H "Authorization: Bearer proxy-123" http://<host>:14300/v1/models
curl -H "Authorization: Bearer proxy-123" -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"你好"}]}' \
  http://<host>:14300/v1/chat/completions
```

## 配置说明
- `NEWAPI_BASE_URL`：上游 newapi 地址，默认 `https://api.newapi.ai`，末尾无需 `/`。
- `NEWAPI_API_KEY`：上游鉴权必填，不设置会在启动时报错。
- `PROXY_API_KEY`：可选代理层密钥，不填则跳过校验。
- `MODEL_OVERRIDE_MAP`：JSON 对象字符串。
  - 直接字符串值：把别名映射到目标模型，例如 `"gpt-4o":"deepseek-chat"`。
  - 对象值支持：
    - `target_model`/`model`：转发时替换成的模型 ID。
    - `prepend_system`：在现有 `messages` 前注入一条 system。
    - `force_params`：强制附加或覆盖的请求参数。
    - `replace`/`replacements`：对 `messages` 内字符串内容做 `.replace(old, new)`。

## 注意事项
- 文本替换仅作用于消息体中字符串 `content` 字段，且为直接替换。
- 如经过 nginx/caddy 等反向代理，确保保留 `Authorization` 头并兼容 HTTP/2。
- `static/` 目录不存在时，根路径仅会返回提示页面。
