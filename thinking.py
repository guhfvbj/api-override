import json
from typing import Any, Dict, Optional

import httpx

from config import THINKING_SYSTEM_TEMPLATE, logger


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
    """基于 SSE JSON 结构，仅对 thinking 段做标签覆写。

    - 丢弃 choices[].delta.content 的前 10 个字符（假定为原始 `<thinking>`），只输出一次 `<think>`；
    - 首次出现 `</thinking>` 时替换为 `</think>`；
    - 之后的内容全部原样透传。
    """
    open_tag_len = 10  # `<thinking>` 的长度
    dropped = 0  # 已丢弃的前缀字符数（跨 chunk 累积）
    prefix_emitted = False
    close_replaced = False

    def patch_content(text: Optional[str]) -> Optional[str]:
        nonlocal dropped, prefix_emitted, close_replaced
        if text is None or text == "":
            return text

        s = text
        out = ""

        # 1. 丢弃整体前 10 个字符，只输出一次 `<think>`
        if not prefix_emitted and dropped < open_tag_len:
            need = open_tag_len - dropped
            drop = min(len(s), need)
            s = s[drop:]
            dropped += drop
            if dropped >= open_tag_len:
                out += "<think>"
                prefix_emitted = True

        # 2. 将首次 `</thinking>` 替换为 `</think>`（只替一次）
        if prefix_emitted and not close_replaced and s:
            idx = s.find("</thinking>")
            if idx != -1:
                out += s[:idx]
                out += "</think>"
                close_replaced = True
                out += s[idx + len("</thinking>") :]
                return out

        # 3. 其余内容原样透传
        out += s
        return out

    try:
        async for line in upstream_resp.aiter_lines():
            # thinking 开头和结尾都已经处理完，后续行直接透传
            if prefix_emitted and close_replaced:
                yield (line + "\n").encode("utf-8")
                continue

            if not line.startswith("data:"):
                yield (line + "\n").encode("utf-8")
                continue

            raw = line[5:].lstrip()
            if raw.strip() == "[DONE]":
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
                        delta["content"] = patch_content(content)

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

