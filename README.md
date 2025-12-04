# newapi OpenAI 代理（api-override）

基于 FastAPI 的轻量代理，兼容 OpenAI API 协议，可转发到 newapi.ai 或其它 OpenAI 风格上游。

## 特性
- 模型别名/重定向：本地定义别名，转发前替换为 `target_model`，可绑定上游渠道。
- 多渠道：支持配置多个上游，通过请求头 `X-Channel` 传递，缺省回落到单一 `NEWAPI_*`。
- 思考模式：请求体决定是否启用，自动注入 antml 思考提示；流式响应提供 `<think>` 字符串和 `content_blocks` 双轨格式。
- 模型名后缀触发：`-thinking`/`-think` 后缀可强制启用思考模式（默认预算 16000）。
- 鉴权：可选代理层 API Key 校验（`Authorization: Bearer` 或 `X-Api-Key`）。
- Web UI：`/`（默认端口 `14300`）可视化管理模型覆写与渠道。

## 快速开始
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

示例 `.env`：
```env
NEWAPI_BASE_URL=https://api.newapi.ai    # 必填：上游地址
NEWAPI_API_KEY=sk-xxxx                   # 必填：上游鉴权
PROXY_API_KEY=proxy-123                  # 可选：代理层鉴权

# 可选：预置模型覆写（也可在 Web UI 编辑）
MODEL_OVERRIDE_MAP={
  "gpt-4o": { "channel": "default", "target_model": "deepseek-chat" }
}
```

启动：
```bash
python main.py
# 或
uvicorn main:app --host 0.0.0.0 --port 14300
```

## API 行为
- `GET /v1/models`：聚合上游模型并追加本地别名。
- `POST /v1/chat/completions`：
  - 覆写模型：按 `MODEL_OVERRIDE_MAP` 或 Web UI 配置替换 `model`，并按需带上 `X-Channel`。
  - 思考模式开关：
    - 顶层 `thinking`/`think` 或 `providerOptions.*.(thinking|think)`，若其中 `budget_tokens` 为正整数，则启用。
    - 模型名带 `-thinking`/`-think` 时强制启用，预算使用默认值 16000。
  - 思考提示注入：启用后，会在 messages 前置
    ```
    <antml:thinking_mode>interleaved</antml:thinking_mode>
    <antml:max_thinking_length>{budget}</antml:max_thinking_length>
    ```

## 流式输出格式（思考模式启用时）
- `choices[].delta.content`：字符串，原始 `<thinking>...</thinking>` 被改写为 `<think>...</think>`。
- `choices[].delta.content_blocks`：块列表，更易于客户端区分：
  ```json
  [{"type": "text", "text": "前置回复"}, {"type": "thinking", "text": "思维链"}]
  ```
- 在 `[DONE]` 前会补一条收尾 chunk，确保缓冲内容被下发。
- 未启用思考模式时，流式与非流式响应均原样透传。

## 覆写规则（`MODEL_OVERRIDE_MAP` / `/overrides`）
```json
{
  "<alias-model-id>": {
    "channel": "可选渠道标识",
    "target_model": "实际发往上游的模型 ID"
  }
}
```
外层 key 为客户端使用的模型 ID；`channel` 可选，`target_model` 为真正发往上游的模型。

## 渠道管理
- 环境变量 `UPSTREAM_CHANNELS` 或 Web UI `/channels`。
- 结构示例：
  ```json
  {
    "default": { "base_url": "https://api.newapi.ai", "api_key": "sk-xxx" },
    "backup":  { "base_url": "https://api.another.com", "api_key": "sk-yyy" }
  }
  ```
- 保存渠道时会同步清理绑定到已删除渠道的模型覆写。

## Web UI
1. 访问 `http://localhost:14300/`。
2. 填写代理地址与代理密钥（如启用）。
3. 刷新模型 -> 选择模型 -> 编辑 `channel` 与 `target_model`。
4. 保存后写入后端并持久化到 `override_store.json` 与 `.env`。

## 注意事项
- 代理层鉴权可选；如经反向代理，请保留 `Authorization` 头并视需要开启 HTTP/2。
- 缺少 `static/` 时根路径只返回简单提示。
- 同时存在 `MODEL_OVERRIDE_MAP` 与 `override_store.json` 时，启动优先使用本地持久化；若想强制使用环境变量，可删除 `override_store.json` 后重启。
