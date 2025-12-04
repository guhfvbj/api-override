# newapi OpenAI 代理（api-override）

基于 FastAPI 的轻量代理，兼容 OpenAI API 协议，转发到 newapi.ai 或其它 OpenAI 风格上游。

当前侧重点：

- 模型别名与重定向（`target_model`）
- 多上游渠道管理（`channel`，通过请求头 `X-Channel` 传给上游）
- 通过请求体 `thinking` / `think` 字段控制思考模式与预算，并在流式响应中将 `<thinking>` 标签覆写为 `<think>`
- 内置 Web UI（`/`，默认端口 `14300`）用于可视化管理覆写规则与渠道

---

## 功能概览

- 对外接口：
  - `POST /v1/chat/completions`
  - `GET /v1/models`
  - 基本保持与 OpenAI API 兼容
- 模型覆写：
  - 为任意模型 ID 定义本地别名；
  - 支持按别名选择上游渠道 `channel`；
  - 支持将别名重定向到实际模型 `target_model`。
- 思考模式（Thinking）：
  - 由客户端请求体完全决定是否启用；
  - 自动在系统提示词中注入 antml 思考配置；
  - 流式响应中自动将 `<thinking>...</thinking>` 改写为 `<think>...</think>`。
- 代理鉴权：
  - 可选开启代理层 API Key 校验；
  - 支持 `Authorization: Bearer <PROXY_API_KEY>` 与 `X-Api-Key`。
- 配置持久化：
  - 覆写规则既可以通过环境变量 `MODEL_OVERRIDE_MAP` 预置；
  - 也可以在 Web UI 中修改，并持久化到 `override_store.json` 与 `.env`。

---

## 快速开始

以 Linux/macOS 为例（默认端口 `14300`）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

创建 `.env`（示例）：

```bash
NEWAPI_BASE_URL=https://api.newapi.ai
NEWAPI_API_KEY=sk-xxxx                    # 必填：上游 newapi 的鉴权 Key
PROXY_API_KEY=proxy-123                   # 可选：代理层鉴权 Key

# 可选：启动时预置模型覆写规则（也可以通过 Web UI 管理）
MODEL_OVERRIDE_MAP={
  "gpt-4o": {
    "channel": "default",
    "target_model": "deepseek-chat"
  }
}
```

启动服务：

```bash
python main.py
# 或：
uvicorn main:app --host 0.0.0.0 --port 14300
```

简单调用示例：

```bash
# 获取模型列表（会附加本地定义的别名模型）
curl -H "Authorization: Bearer proxy-123" \
  http://localhost:14300/v1/models

# 发起聊天请求（不启用思考）
curl -H "Authorization: Bearer proxy-123" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o","messages":[{"role":"user","content":"你好"}]}' \
  http://localhost:14300/v1/chat/completions
```

---

## Thinking 模式与标签覆写

当前代理不会通过环境变量开关思考模式，而是完全由 **客户端请求体** 决定。

### 1）顶层 `thinking` / `think` 字段

推荐写法示例：

```json
{
  "model": "claude-sonnet-4",
  "thinking": {
    "budget_tokens": 3276
  },
  "messages": [
    { "role": "user", "content": "你好" }
  ],
  "stream": true
}
```

只要满足：

- 存在顶层 `thinking` 或 `think` 字段；
- 其中 `budget_tokens` 可以被解析为一个 **正整数**；

就会被视为“启用 thinking 模式”，并得到：

- `max_thinking_length = budget_tokens`（不再乘以额外系数）；
- 在请求体的消息列表中自动插入/前置 antml 系统提示词：

  ```text
  <antml:thinking_mode>interleaved</antml:thinking_mode>
  <antml:max_thinking_length>3276</antml:max_thinking_length>
  ```

- 在流式响应中，对 `<thinking>...</thinking>` 段：
  - 丢弃原始 `<thinking>` 标签，仅输出一次 `<think>`；
  - 首次 `</thinking>` 替换为 `</think>`；
  - 其余内容保持原样透传。

### 2）兼容旧结构：`providerOptions.*.(thinking|think)`

如果请求体中没有顶层 `thinking` / `think`，但存在：

```json
{
  "providerOptions": {
    "some-provider-id": {
      "thinking": {
        "budget_tokens": 2048
      }
    }
  }
}
```

同样会被识别，并按上述规则计算 `max_thinking_length = 2048`，注入 antml 前缀并改写流式标签。

### 3）未启用思考模式时

未启用的情况包括：

- 请求体中完全没有 `thinking` / `think` 字段；
- 或存在上述字段，但 `budget_tokens` 缺失、无法解析为整数或不大于 0。

此时：

- 不注入 antml 思考提示词；
- 流式响应不做 `<thinking>` 标签覆写，直接透传上游数据；
- 仍然保留模型重定向、渠道头与基本系统提示逻辑。

---

## 覆写规则结构（`MODEL_OVERRIDE_MAP` / `/overrides`）

无论是环境变量 `MODEL_OVERRIDE_MAP`，还是 Web UI 调用 `/overrides`，规则结构保持一致：

```json
{
  "<alias-model-id>": {
    "channel": "可选渠道标识",
    "target_model": "实际发往上游的模型 ID"
  }
}
```

- 外层 key 为代理对外暴露的模型 ID（别名），即客户端在 `model` 字段中使用的值；
- `channel`（可选）：
  - 用于在代理内部区分不同“渠道”或“来源”；
  - 如果设置，将在转发到上游时通过请求头 `X-Channel: <channel>` 传递；
- `target_model`：
  - 若配置，代理会在转发前将请求体中的 `model` 字段替换为该值。

> 当前版本已去掉“配置文本替换”等复杂行为，主要通过**别名 + 渠道 + thinking** 组合来控制模型行为。

---

## Web UI 使用方式

1. 浏览器访问代理地址（默认 `http://localhost:14300/`）；
2. 顶部填写：
   - 代理地址（可留空使用当前域名）；
   - 代理密钥（如果设置了 `PROXY_API_KEY`）；
3. 点击“刷新模型”拉取 `/v1/models`，左侧展示所有上游模型以及本地定义的别名模型；
4. 点击任意模型，右侧可以编辑：
   - 渠道标识 `channel`；
   - 目标模型 ID `target_model`；
5. 点击“保存当前模型设置”会：
   - 调用 `POST /overrides` 将全部规则写入后端；
   - 同时更新本地 `override_store.json`，用于下次启动自动恢复。

下方“覆写配置预览”展示的 JSON，即当前全部规则的快照，可直接复制到环境变量 `MODEL_OVERRIDE_MAP` 中。

---

## 其它注意事项

- 如经 nginx / caddy 等反向代理，请务必：
  - 保留 `Authorization` 请求头；
  - 视需要开启 HTTP/2 支持；
- 若 `static/` 目录不存在，根路径 `/` 只会返回简单提示页面；
- 若同时配置了 `MODEL_OVERRIDE_MAP` 和 `override_store.json`：
  - 启动时优先使用本地持久化文件（即 Web UI 保存的配置）；
  - 如果想强制使用环境变量，可删除 `override_store.json` 后重启。
