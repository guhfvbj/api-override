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
    """流式响应中扫描 `<thinking>`/`</thinking>` 标签，将第一对改写为 `<think>`/`</think>`，其余内容直接透传。"""
    think_buffer = ""
    in_think_block = False
    prefix_emitted = False
    close_replaced = False

    def patch_content(text: Optional[str]) -> Optional[str]:
        nonlocal think_buffer, in_think_block, prefix_emitted, close_replaced

        if text is None or text == "" or close_replaced:
            return text

        think_buffer += text
        out_parts: list[str] = []
        pos = 0

        while pos < len(think_buffer):
            if not in_think_block:
                start_idx = think_buffer.find(THINKING_START_TAG, pos)
                if start_idx != -1:
                    before_text = think_buffer[pos:start_idx]
                    if before_text:
                        out_parts.append(before_text)

                    if not prefix_emitted:
                        out_parts.append(THINK_TAG)
                        prefix_emitted = True

                    in_think_block = True
                    pos = start_idx + len(THINKING_START_TAG)
                else:
                    remaining = think_buffer[pos:]
                    if remaining:
                        out_parts.append(remaining)
                    think_buffer = ""
                    break
            else:
                end_idx = think_buffer.find(THINKING_END_TAG, pos)
                if end_idx != -1:
                    thinking_text = think_buffer[pos:end_idx]
                    if thinking_text:
                        out_parts.append(thinking_text)

                    if not close_replaced:
                        out_parts.append(THINK_END_TAG)
                        close_replaced = True

                    in_think_block = False
                    pos = end_idx + len(THINKING_END_TAG)

                    # 第一对标签处理完成，后续内容直接透传
                    if close_replaced:
                        tail = think_buffer[pos:]
                        if tail:
                            out_parts.append(tail)
                        think_buffer = ""
                        break
                else:
                    remaining = think_buffer[pos:]
                    if remaining:
                        out_parts.append(remaining)
                    think_buffer = ""
                    break

        return "".join(out_parts)

    try:
        async for line in upstream_resp.aiter_lines():
            if close_replaced:
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

