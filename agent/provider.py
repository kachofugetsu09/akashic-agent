"""
LLM Provider — OpenAI 兼容格式
支持所有兼容 OpenAI Chat Completions API 的服务：DeepSeek、Qwen、OpenAI 等。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, cast
from openai import AsyncOpenAI

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

logger = logging.getLogger(__name__)
_LAST_PAYLOAD_PATH = Path(tempfile.gettempdir()) / "akashic-last-llm-payload.json"
StreamDelta = dict[str, str]

# 安全审查错误码（各厂商）
_SAFETY_ERROR_CODES = {
    "data_inspection_failed",  # Qwen / DashScope
    "content_filter",  # Azure OpenAI
    "content_policy_violation",  # OpenAI
}

_CONTEXT_LENGTH_KEYWORDS = (
    "range of input length",  # DashScope / Qwen
    "context_length_exceeded",  # OpenAI
    "maximum context length",  # OpenAI
    "context window exceeds limit",  # MiniMax
    "string too long",  # 通用
    "reduce the length",  # 通用
    "too many tokens",  # 通用
)


class ContentSafetyError(Exception):
    """LLM provider 因内容安全审查拒绝请求"""


class ContextLengthError(Exception):
    """LLM provider 因上下文超长拒绝请求"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


class LLMProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        system_prompt: str = "",
        extra_body: dict | None = None,
        request_timeout_s: float = 90.0,
        max_retries: int = 1,
    ) -> None:
        normalized_base_url = _normalize_openai_base_url(base_url)
        self._client = AsyncOpenAI(api_key=api_key, base_url=normalized_base_url)
        self._base_url = normalized_base_url or ""
        self._system = system_prompt
        self._extra_body = extra_body or {}
        self._request_timeout_s = max(1.0, float(request_timeout_s))
        self._max_retries = max(0, int(max_retries))

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        max_tokens: int,
        tool_choice: str | dict = "auto",
        extra_body: dict | None = None,
        on_content_delta: Callable[[StreamDelta], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        # 系统提示作为第一条消息（若 messages 已自带 system 消息则不再重复添加）
        already_has_system = messages and messages[0].get("role") == "system"
        full_messages = (
            [{"role": "system", "content": self._system}, *messages]
            if self._system and not already_has_system
            else messages
        )
        full_messages = _merge_leading_system_messages(full_messages)
        full_messages = _normalize_chat_messages(full_messages)
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=full_messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        merged_extra_body = dict(self._extra_body)
        if extra_body:
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        _LAST_PAYLOAD_PATH.write_text(
            json.dumps(kwargs, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        if on_content_delta is not None:
            return await self._chat_streaming(kwargs, on_content_delta)

        resp = cast(Any, await self._create_with_retry(kwargs))
        msg = resp.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        raw = msg.content
        thinking: str | None = None
        if raw:
            m = _THINK_RE.search(raw)
            if m:
                thinking = m.group(1).strip()
                raw = _THINK_RE.sub("", raw).strip() or None
        return LLMResponse(content=raw, tool_calls=tool_calls, thinking=thinking)

    async def _chat_streaming(
        self,
        kwargs: dict[str, Any],
        on_content_delta: Callable[[StreamDelta], Awaitable[None]],
    ) -> LLMResponse:
        stream = cast(Any, await self._create_with_retry({**kwargs, "stream": True}))
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_chunks: dict[int, dict[str, str]] = {}
        tool_call_seen = False

        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            reasoning_piece = _get_field(delta, "reasoning_content")
            if isinstance(reasoning_piece, str) and reasoning_piece:
                reasoning_parts.append(reasoning_piece)
                if not tool_call_seen:
                    await on_content_delta({"thinking_delta": reasoning_piece})

            for tc in _iter_tool_call_deltas(delta):
                tool_call_seen = True
                chunk_index = int(tc["index"])
                slot = tool_call_chunks.setdefault(chunk_index, {})
                tc_id = str(tc["id"])
                tc_name = str(tc["name"])
                tc_arguments = str(tc["arguments"])
                if tc_id:
                    slot["id"] = slot.get("id", "") + tc_id
                if tc_name:
                    slot["name"] = slot.get("name", "") + tc_name
                if tc_arguments:
                    slot["arguments"] = slot.get("arguments", "") + tc_arguments

            content_piece = _get_field(delta, "content")
            if isinstance(content_piece, str) and content_piece:
                content_parts.append(content_piece)
                if not tool_call_seen:
                    await on_content_delta({"content_delta": content_piece})

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_chunks):
            item = tool_call_chunks[idx]
            raw_args = item.get("arguments", "") or "{}"
            tool_calls.append(
                ToolCall(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=json.loads(raw_args),
                )
            )

        raw = "".join(content_parts).strip() or None
        thinking = "".join(reasoning_parts).strip() or None
        if raw:
            m = _THINK_RE.search(raw)
            if m:
                thinking = m.group(1).strip()
                raw = _THINK_RE.sub("", raw).strip() or None
        return LLMResponse(content=raw, tool_calls=tool_calls, thinking=thinking)

    async def _create_with_retry(self, kwargs: dict) -> object:
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._client.chat.completions.create(**kwargs),
                    timeout=self._request_timeout_s,
                )
            except Exception as e:
                last_err = e
                logger.warning(
                    "[llm.error] model=%s stream=%s base_url=%s tools=%d extra_body_keys=%s "
                    "err=%s",
                    kwargs.get("model"),
                    bool(kwargs.get("stream")),
                    self._base_url,
                    len(kwargs.get("tools") or []),
                    sorted((kwargs.get("extra_body") or {}).keys()),
                    e,
                )
                if self._is_safety_error(e):
                    raise ContentSafetyError(str(e)) from e
                if self._is_context_length_error(e):
                    raise ContextLengthError(str(e)) from e
                retryable = self._is_retryable(e)
                exhausted = attempt >= self._max_retries
                if (not retryable) or exhausted:
                    raise
                wait_s = min(8.0, 1.0 * (2**attempt))
                logger.warning(
                    "[llm] 请求失败，将重试 attempt=%d/%d wait=%.1fs err=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    wait_s,
                    type(e).__name__,
                )
                await asyncio.sleep(wait_s)
        if last_err:
            raise last_err
        raise RuntimeError("LLM request failed without exception")

    @staticmethod
    def _is_safety_error(err: Exception) -> bool:
        text = str(err)
        return any(code in text for code in _SAFETY_ERROR_CODES)

    @staticmethod
    def _is_context_length_error(err: Exception) -> bool:
        text = str(err).lower()
        return any(kw in text for kw in _CONTEXT_LENGTH_KEYWORDS)

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        if isinstance(err, TimeoutError):
            return True
        status_code = getattr(err, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        text = str(err).lower()
        keywords = (
            "429",
            "timeout",
            "timed out",
            "connect",
            "connection",
            "temporarily unavailable",
            "server error",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
        )
        return any(k in text for k in keywords)


def _get_field(delta: Any, name: str) -> Any:
    if isinstance(delta, dict):
        return delta.get(name)
    return getattr(delta, name, None)


def _iter_tool_call_deltas(delta: Any) -> list[dict[str, str | int]]:
    raw_items = _get_field(delta, "tool_calls") or []
    result: list[dict[str, str | int]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, dict):
            function = item.get("function") or {}
            result.append(
                {
                    "index": int(item.get("index", idx)),
                    "id": str(item.get("id", "") or ""),
                    "name": str(function.get("name", "") or ""),
                    "arguments": str(function.get("arguments", "") or ""),
                }
            )
            continue
        function = getattr(item, "function", None)
        result.append(
            {
                "index": int(getattr(item, "index", idx)),
                "id": str(getattr(item, "id", "") or ""),
                "name": str(getattr(function, "name", "") or ""),
                "arguments": str(getattr(function, "arguments", "") or ""),
            }
        )
    return result


def _summarize_roles(messages: list[dict]) -> str:
    roles = [str(msg.get("role", "?")) for msg in messages]
    if len(roles) <= 12:
        return ",".join(roles)
    head = ",".join(roles[:6])
    tail = ",".join(roles[-3:])
    return f"{head},...,{tail}"


def _summarize_message_shapes(messages: list[dict]) -> str:
    shapes: list[str] = []
    for msg in messages[:8]:
        keys = sorted(k for k in msg.keys() if k != "content")
        content = msg.get("content")
        if isinstance(content, str):
            content_kind = "str"
        elif isinstance(content, list):
            content_kind = "list"
        elif content is None:
            content_kind = "none"
        else:
            content_kind = type(content).__name__
        role = str(msg.get("role", "?"))
        extra = ",".join(keys) if keys else "-"
        shapes.append(f"{role}[content={content_kind};keys={extra}]")
    if len(messages) > 8:
        shapes.append("...")
    return " | ".join(shapes)


def _summarize_tool_names(tools: list[dict]) -> str:
    names = [
        str((tool.get("function") or {}).get("name", "?"))
        for tool in tools[:8]
    ]
    if len(tools) > 8:
        names.append("...")
    return ",".join(names)


def _merge_leading_system_messages(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    system_contents: list[str] = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") == "system":
        content = messages[idx].get("content")
        if isinstance(content, str) and content:
            system_contents.append(content)
        idx += 1
    if system_contents:
        merged.append({"role": "system", "content": "\n\n".join(system_contents)})
    merged.extend(messages[idx:])
    return merged if merged else list(messages)


def _normalize_chat_messages(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        role = str(item.get("role", "") or "")
        content = item.get("content")

        if role == "assistant" and item.get("tool_calls"):
            if content is None or (isinstance(content, str) and not content.strip()):
                tool_calls = item.get("tool_calls") or []
                first = tool_calls[0] if isinstance(tool_calls, list) and tool_calls else {}
                function = first.get("function") if isinstance(first, dict) else {}
                tool_name = ""
                if isinstance(function, dict):
                    tool_name = str(function.get("name", "") or "")
                item["content"] = (
                    f"调用工具 {tool_name}" if tool_name else "调用工具"
                )
        elif role in {"user", "assistant", "tool"}:
            if content is None:
                item["content"] = ""

        normalized.append(item)
    return normalized


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    text = (base_url or "").strip()
    if not text:
        return None
    parsed = urlsplit(text)
    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    if not path:
        path = ""
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))
