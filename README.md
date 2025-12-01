# newapi OpenAI 兼容转发器

使用 FastAPI 构建的轻量级代理，兼容 OpenAI 接口。支持模型覆写、独立代理密钥，并附带前端 Web UI（默认端口 14300）。

## 功能
- `/v1/models`：透传上游 newapi 的模型列表，并将代理层定义的别名模型一起返回（`alias_for` 标识指向真实模型）。
- `/v1/chat/completions`：转发聊天请求，支持在转发前覆写模型 ID、前置系统提示、强制参数。
- 代理侧鉴权：可设置独立的 `PROXY_API_KEY`，客户端需携带后才能访问代理。
- Web UI：默认根路径 `/`，可在浏览器直接调用代理接口、查看模型列表。

## 快速开始（Linux，默认端口 14300）
1. 准备环境
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 配置环境变量（可写入 `.env`）
```bash
export NEWAPI_BASE_URL=https://api.newapi.ai   # 上游 newapi 地址，末尾不带 /
export NEWAPI_API_KEY=sk-xxxx                  # 上游 newapi 密钥
export PROXY_API_KEY=proxy-123                 # 代理自身密钥，未设置则默认放行
# 模型覆写规则（JSON 字符串）
# 形式1：直接映射模型 ID
# MODEL_OVERRIDE_MAP={"gpt-4o":"deepseek-chat"}
# 形式2：携带额外参数
# MODEL_OVERRIDE_MAP={
#   "gpt-4o":{
#     "target_model":"deepseek-chat",
#     "prepend_system":"请始终使用中文回答",
#     "force_params":{"temperature":0.3}
#   }
# }
```

3. 启动（默认监听 0.0.0.0:14300）
```bash
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 14300
```

4. 访问
- Web UI：`http://<host>:14300/`（可填入代理地址、API Key、模型并直接对话）
- 健康检查：`http://<host>:14300/health`
- API 示例：
```bash
curl -X GET http://<host>:14300/v1/models \
  -H "Authorization: Bearer proxy-123"

curl -X POST http://<host>:14300/v1/chat/completions \
  -H "Authorization: Bearer proxy-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gpt-4o",
    "messages":[{"role":"user","content":"你好，简单介绍一下项目"}],
    "stream":false
  }'
```

## Web UI 说明
- 默认端口 14300，根路径即 UI；静态资源目录为 `static/`。
- UI 内部会在前端维护上下文，可随时点击“清空对话”重置。
- “拉取模型”会调用代理的 `/v1/models`，方便验证别名是否生效。

## 关键机制
- **模型覆写**：`MODEL_OVERRIDE_MAP` 支持为任意模型添加别名。转发前会根据配置修改 `model` 字段、前置系统提示、覆盖指定参数。
- **模型列表追加**：如果覆写配置暴露了新的别名模型，`/v1/models` 会在上游结果中追加对应条目。
- **鉴权**：设置 `PROXY_API_KEY` 后，客户端需要在 `Authorization: Bearer <PROXY_API_KEY>` 或 `X-Api-Key` 中携带。
- **流式响应**：当请求中 `stream=true` 时，保持上游 SSE/流式输出，不做本地拆包。

## 部署小贴士
- 生产环境建议在进程管理器（如 systemd、supervisor、docker）中运行，设置好 `NEWAPI_API_KEY`、`PROXY_API_KEY`、`MODEL_OVERRIDE_MAP`。
- 如需反向代理（Nginx/Caddy），确保转发 `Authorization` 头并开启 HTTP/2 以减少延迟。
- 若 Web UI 需要跨域访问，可在 `main.py` 中按需添加 CORS 中间件。

## 扩展
- 按同样模式可扩展 `embeddings`、`audio`、`images` 等端点，复用 `apply_overrides` 以保持覆写能力。
- 如需多路上游或动态路由，可将覆写规则拆分为独立配置文件，在启动时加载。
