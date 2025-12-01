# newapi OpenAI 兼容转发器

使用 FastAPI 构建的轻量级代理，兼容 OpenAI 接口。支持：

- `/v1/models`：向上游 newapi 拉取模型列表，并追加代理层定义的别名模型。
- `/v1/chat/completions`：在转发前对单个模型进行覆写（模型映射、插入系统提示、强制参数），再将请求转发到上游。
- 代理自身的 `PROXY_API_KEY` 校验，保证有独立的访问入口与密钥。

## 快速开始

1. 创建环境变量（可放在 `.env`）：

```
NEWAPI_BASE_URL=https://api.newapi.ai          # 上游 newapi 地址，末尾不带 /
NEWAPI_API_KEY=sk-xxxxxxx                      # 上游 newapi 密钥
PROXY_API_KEY=proxy-123                        # 代理自身的访问密钥，客户端需携带

# 模型覆写规则（JSON），支持字符串或对象形式
# 形式1：直接映射模型 ID
# MODEL_OVERRIDE_MAP={"gpt-4o":"deepseek-chat"}
# 形式2：包含更多覆写参数
MODEL_OVERRIDE_MAP={
  "gpt-4o":{
    "target_model":"deepseek-chat",
    "prepend_system":"请始终使用中文回答",
    "force_params":{"temperature":0.3}
  }
}
```

2. 安装依赖并启动：

```
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

3. 调用示例（客户端需携带代理密钥）：

```
curl -X GET http://localhost:8000/v1/models \
  -H "Authorization: Bearer proxy-123"

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer proxy-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gpt-4o",
    "messages":[{"role":"user","content":"你好，简单介绍一下项目"}],
    "stream":false
  }'
```

## 关键机制

- **模型覆写**：`MODEL_OVERRIDE_MAP` 支持为任意模型添加别名。转发前会根据配置修改 `model` 字段、前置系统提示、覆盖指定参数。
- **模型列表追加**：如果覆写配置暴露了新的别名模型，`/v1/models` 会在上游结果中追加对应条目（`alias_for` 字段指向真实上游模型）。
- **鉴权**：未设置 `PROXY_API_KEY` 时默认放行，设置后需使用 `Authorization: Bearer <PROXY_API_KEY>` 或 `X-Api-Key` 访问。
- **流式响应**：当请求中 `stream=true` 时，保持上游的 SSE/流式返回，不在本地拆包。

## 扩展建议

- 如需支持 `embeddings`、`audio` 等更多 OpenAI 兼容端点，可在 `main.py` 按同样模式添加路由并复用 `apply_overrides`。
- 如果需要多路上游或动态路由，可将覆写规则拆分为文件配置，并在启动时加载。
