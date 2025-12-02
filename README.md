# newapi OpenAI 代理（api-override）

基于 FastAPI 的轻量代理，兼容 OpenAI API，转发到 newapi.ai。当前版本专注于：

- 模型别名与重定向（`target_model`）
- 渠道区分（`channel`，通过请求头 `X-Channel` 传给上游）
- 追加系统提示词（`prepend_system`）
- 通过请求体 `thinking` 字段控制思考模式与思考预算，并覆写 `<thinking>` 标签为 `<think>`

自带 Web UI（`/`，默认端口 `14300`），用于可视化管理覆写规则。

---

## 功能特点

- 兼容 `POST /v1/chat/completions` 和 `GET /v1/models`，对外接口基本保持 OpenAI 兼容。
- 支持为任意模型 ID 定义本地覆写规则，包括：
  - 渠道标识 `channel`（可选）；
  - 模型重定向 `target_model`；
  - 追加系统提示词 `prepend_system`；
  - 固定请求参数 `force_params`（例如温度、top_p 等）。
- 当请求体中显式开启 thinking 时：
  - 自动在系统提示词中追加 antml 思考配置，思考长度 = `budget_tokens × 5`；
  - 流式响应中 `<thinking>...</thinking>` 自动覆写为 `<think>...</think>`。
- 代理层可选鉴权：支持 `Authorization: Bearer <PROXY_API_KEY>` 或 `X-Api-Key`。
- 覆写规则既可以通过环境变量 `MODEL_OVERRIDE_MAP` 配置，也可以通过 Web UI 管理并持久化到 `override_store.json`。

---

## 快速开始

以 Linux/macOS 为例（默认端口 14300）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

创建 `.env`（示例）：

```bash
NEWAPI_BASE_URL=https://api.newapi.ai
NEWAPI_API_KEY=sk-xxxx                    # 必填：上游 newapi 鉴权 Key
PROXY_API_KEY=proxy-123                   # 可选：代理鉴权用 Key

# 可选：启动时预置模型覆写规则（也可以通过 Web UI 管理）
MODEL_OVERRIDE_MAP={
  "gpt-4o": {
    "channel": "default",
    "target_model": "deepseek-chat",
    "prepend_system": "请始终用中文回答",
    "force_params": {"temperature": 0.3}
  }
}
```

启动服务：

```bash
python main.py
# 或：uvicorn main:app --host 0.0.0.0 --port 14300
```

接口示例：

```bash
# 获取模型列表（会附加本地定义的别名模型）
curl -H "Authorization: Bearer proxy-123" \
  http://<host>:14300/v1/models

# 发起聊天请求（不启用思考）
curl -H "Authorization: Bearer proxy-123" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o","messages":[{"role":"user","content":"你好"}]}' \
  http://<host>:14300/v1/chat/completions
```

---

## Thinking 模式与覆写

当前代理不会通过环境变量开关思考模式，而是完全由**客户端请求体**决定。

### 1）顶层 `thinking` 字段（推荐）

例如 Cherry Studio 当前的请求体：

```json
{
  "model": "claude-sonnet-4",
  "thinking": {
    "type": "enabled",
    "budget_tokens": 3276
  },
  "messages": [
    { "role": "user", "content": "你好" }
  ],
  "stream": true
}
```

代理会：

- 识别 `thinking.type = "enabled"` 且存在 `budget_tokens`；
- 计算 `max_thinking_length = budget_tokens * 5`（此例为 `3276 * 5 = 16380`）；
- 在请求体的消息列表中自动插入/前置 antml 系统提示词：

```text
<antml:thinking_mode>interleaved</antml:thinking_mode>
<antml:max_thinking_length>16380</antml:max_thinking_length>
```

- 流式响应中，针对 `<thinking>...</thinking>` 段：
  - 丢弃原始 `<thinking>` 标签；
  - 首次输出替换为 `<think>`；
  - 收尾 `</thinking>` 替换为 `</think>`；
  - 其余内容保持原样透传。

### 2）旧版 Cherry Studio `providerOptions.*.thinking`

如果请求体中没有顶层 `thinking`，但存在：

```json
{
  "providerOptions": {
    "some-provider-id": {
      "thinking": {
        "type": "enabled",
        "budget_tokens": 2048
      }
    }
  }
}
```

同样会被识别并应用上述逻辑（长度 = `2048 * 5`）。

### 3）未启用思考模式时

- 即没有 `thinking` 字段，或者 `type` 非 `enabled`/`enable`，或者预算无效时：
  - 不追加 antml 思考提示词；
  - 流式响应不做 `<thinking>` 标签覆写，直接透传上游数据；
  - 仅保留模型重定向、渠道头与系统提示词注入。

---

## 覆写规则结构（MODEL_OVERRIDE_MAP / /overrides）

无论是环境变量 `MODEL_OVERRIDE_MAP`，还是 Web UI 调用 `/overrides`，规则结构一致：

```json
{
  "<alias-model-id>": {
    "channel": "可选渠道标识",
    "target_model": "实际发往上游的模型 ID",
    "prepend_system": "追加的 system 提示词",
    "force_params": { "temperature": 0.2 }
  }
}
```

- 外层 key 为代理对外暴露的模型 ID（别名），即客户端在 `model` 字段中使用的值。
- `channel`（可选）：
  - 用于在代理内部区分不同“渠道”或“来源”；
  - 如果设置，将在转发到上游时通过请求头 `X-Channel: <channel>` 传递。
- `target_model`：
  - 若配置，代理会在转发前将请求体中的 `model` 字段替换为该值。
- `prepend_system`：
  - 若配置，会在现有 `messages` 列表最前插入一条 `{"role":"system","content": prepend_system}`。
- `force_params`：
  - 一个对象，键值会直接写入/覆盖请求体顶层字段（例如 `temperature`、`top_p` 等）。

> 注意：当前版本已移除“自定义文本查找/替换”功能，只保留通过系统提示词 + thinking 控制模型行为。

---

## Web UI 使用方式

1. 浏览器访问代理地址（默认 `http://localhost:14300/`）。
2. 顶部填写：
   - 代理地址（可留空使用当前域名）；
   - 代理密钥（如果设置了 `PROXY_API_KEY`）。
3. 点击“刷新模型”拉取 `/v1/models`，左侧展示所有上游模型以及本地定义的别名模型。
4. 点击任意模型，右侧可以编辑：
   - 渠道标识 `channel`；
   - 目标模型 ID `target_model`；
   - 追加系统提示词 `prepend_system`。
5. 点击“保存当前模型设置”会：
   - 调用 `POST /overrides` 将全部规则写入后端；
   - 同时更新本地 `override_store.json`，用于下次启动自动恢复。

下方“覆写配置预览”展示的 JSON，即当前全部规则的快照，可直接复制到环境变量 `MODEL_OVERRIDE_MAP` 中。

---

## 其它注意事项

- 如经 nginx/caddy 等反向代理，请务必：
  - 保留 `Authorization` 请求头；
  - 开启对 HTTP/2 的支持（如需要）。
- 若 `static/` 目录不存在，根路径 `/` 只会返回简单提示页面；
- 若同时配置了 `MODEL_OVERRIDE_MAP` 和 `override_store.json`，启动时优先使用本地持久化文件（即 Web UI 保存的配置）。如果想强制使用环境变量，可删除 `override_store.json` 后重启。 

