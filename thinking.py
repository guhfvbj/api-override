import json
from typing import Any, Dict, Optional, Tuple

import httpx

from config import THINKING_SYSTEM_TEMPLATE, logger

THINKING_START_TAG = "<thinking>"
THINKING_END_TAG = "</thinking>"
THINK_TAG = "<think>"
THINK_END_TAG = "</think>"


def build_thinking_system_prefix(max_length: int) -> str:
    """根据思考长度构造 antml 思考系统前缀。"""
    return THINKING_SYSTEM_TEMPLATE.format(max_length=max_length)


def inject_thinking_system_prefix(messages: Any, prefix: str) -> Any:
    """在消息列表中为第一条 system 消息前置指定的思考系统前缀。"""
    if not isinstance(messages, list) or not prefix:
        return messages
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and not content.startswith(prefix):
                msg["content"] = prefix + content
            return messages
    # 没有 system 消息时，在最前面插入一条
    return [{"role": "system", "content": prefix}] + messages


async def safe_stream_thinking(upstream_resp: httpx.Response):
    """流式响应中将 `<thinking>...</thinking>` 解析为 Claude 风格的分块内容：

    - 普通文本输出为 `{"type": "text", "text": "..."}`
    - 思维链输出为 `{"type": "thinking", "text": "..."}`
    - 向下游兼容：`delta.content` 仍然保持字符串（带 <think></think>），额外提供 `delta.content_blocks` 供富格式解析
    """
    buffer = ""
    in_thinking = False

    start_len = len(THINKING_START_TAG)
    end_len = len(THINKING_END_TAG)

    def patch_content(text: Optional[str]) -> tuple[str, list[Dict[str, str]]]:
        """将 content 字符串拆解为字符串输出 + 带 type 的内容块列表。"""
        nonlocal buffer, in_thinking

        if text is None or text == "":
            return "", []

        buffer += text
        parts: list[Dict[str, str]] = []
        str_out: list[str] = []

        while True:
            if in_thinking:
                end_idx = buffer.find(THINKING_END_TAG)
                if end_idx == -1:
                    # 未找到结束标签，先输出除末尾可能的半截标签外的思维内容
                    safe_len = max(0, len(buffer) - (end_len - 1))
                    if safe_len > 0:
                        chunk = buffer[:safe_len]
                        if chunk:
                            parts.append({"type": "thinking", "text": chunk})
                            str_out.append(chunk)
                        buffer = buffer[safe_len:]
                    break

                chunk = buffer[:end_idx]
                if chunk:
                    parts.append({"type": "thinking", "text": chunk})
                    str_out.append(chunk)
                buffer = buffer[end_idx + end_len :]
                in_thinking = False
                str_out.append(THINK_END_TAG)
            else:
                start_idx = buffer.find(THINKING_START_TAG)
                if start_idx == -1:
                    # 未找到起始标签，输出除末尾可能的半截标签外的普通文本
                    safe_len = max(0, len(buffer) - (start_len - 1))
                    if safe_len > 0:
                        chunk = buffer[:safe_len]
                        if chunk:
                            parts.append({"type": "text", "text": chunk})
                            str_out.append(chunk)
                        buffer = buffer[safe_len:]
                    break

                # 先输出标签前的文本
                chunk = buffer[:start_idx]
                if chunk:
                    parts.append({"type": "text", "text": chunk})
                    str_out.append(chunk)

                buffer = buffer[start_idx + start_len :]
                in_thinking = True
                str_out.append(THINK_TAG)

        return "".join(str_out), parts

    def flush_remaining_parts() -> tuple[str, list[Dict[str, str]]]:
        """在流结束前，将缓冲区剩余内容一次性输出。"""
        nonlocal buffer, in_thinking
        if not buffer:
            return "", []
        parts: list[Dict[str, str]] = []
        str_out: list[str] = []
        if buffer:
            parts.append(
                {
                    "type": "thinking" if in_thinking else "text",
                    "text": buffer,
                }
            )
            str_out.append(THINK_TAG if in_thinking else "")
            str_out.append(buffer)
            str_out.append(THINK_END_TAG if in_thinking else "")
        buffer = ""
        return "".join(str_out), parts

    try:
        async for line in upstream_resp.aiter_lines():
            if not line.startswith("data:"):
                yield (line + "\n").encode("utf-8")
                continue

            raw = line[5:].lstrip()
            if raw.strip() == "[DONE]":
                remaining_str, remaining_parts = flush_remaining_parts()
                if remaining_parts:
                    final_delta: Dict[str, Any] = {"content": remaining_str} if remaining_str else {}
                    final_delta["content_blocks"] = remaining_parts
                    final_obj = {"choices": [{"delta": final_delta}]}
                    final_line = "data: " + json.dumps(final_obj, ensure_ascii=False)
                    # 手动插入一个完整的 SSE 事件（含空行分隔）
                    yield (final_line + "\n").encode("utf-8")
                    yield b"\n"
                yield (line + "\n").encode("utf-8")
                continue

            try:
                obj = json.loads(raw)
            except Exception:
                # 无法解析 JSON，保守透传
                yield (line + "\n").encode("utf-8")
                continue

            choices = obj.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str):
                        content_str, parts = patch_content(content)
                        if parts:
                            delta["content_blocks"] = parts
                        if content_str:
                            delta["content"] = content_str
                        else:
                            delta.pop("content", None)

            out_line = "data: " + json.dumps(obj, ensure_ascii=False)
            yield (out_line + "\n").encode("utf-8")
    except httpx.StreamClosed:
        logger.warning("上游流已关闭，提前结束推送")
    except Exception:
        logger.exception("转发流式响应时发生异常")
        raise


def extract_thinking_max_length(payload: Dict[str, Any]) -> Optional[int]:
    """从请求体中提取思考预算，返回 max_thinking_length。

    规则：只要在 thinking/think 字段（顶层或 providerOptions.* 下）看到正的
    budget_tokens，就视为启用思考模式，并直接将 budget_tokens 作为 max_thinking_length。
    """

    def _from_cfg(thinking_cfg: Any, source: str) -> Optional[int]:
        if not isinstance(thinking_cfg, dict):
            return None
        budget = thinking_cfg.get("budget_tokens")
        try:
            budget_val = int(budget)
        except (TypeError, ValueError):
            logger.warning("thinking/think 配置 budget_tokens 非法: %r（source=%s）", budget, source)
            return None
        if budget_val <= 0:
            return None
        max_length = budget_val
        logger.info(
            "检测到 thinking/think 启用：source=%s, budget_tokens=%s, max_length=%s",
            source,
            budget_val,
            max_length,
        )
        return max_length

    # 1）顶层字段：thinking 或 think
    for key in ("thinking", "think"):
        top_level = payload.get(key)
        max_len = _from_cfg(top_level, key)
        if max_len is not None:
            return max_len

    # 2）兼容 providerOptions.*.(thinking|think)
    provider_options = payload.get("providerOptions")
    if isinstance(provider_options, dict):
        for provider_id, provider_cfg in provider_options.items():
            if not isinstance(provider_cfg, dict):
                continue
            for key in ("thinking", "think"):
                thinking_cfg = provider_cfg.get(key)
                max_len = _from_cfg(thinking_cfg, f"providerOptions[{provider_id}].{key}")
                if max_len is not None:
                    return max_len

    return None


def normalize_thinking_model_name(model: Optional[str]) -> Tuple[Optional[str], bool]:
    """规范化带 thinking/think 后缀的模型名。

    - 若模型名以 `-thinking` 或 `-think` 结尾，则去掉后缀和尾部多余短横线，返回（基础名, True）；
    - 否则返回（原始名, False）。
    """
    if not isinstance(model, str):
        return model, False

    lower = model.lower()
    for suffix in ("-thinking", "-think"):
        if lower.endswith(suffix):
            base = model[: -len(suffix)]
            # 兼容形如 "...-thinking" 的写法，去掉多余短横线
            base = base.rstrip("-")
            if not base:
                return model, False
            return base, True

    return model, False

