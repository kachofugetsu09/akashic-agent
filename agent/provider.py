"""
LLM Provider — OpenAI 兼容格式
支持所有兼容 OpenAI Chat Completions API 的服务：DeepSeek、Qwen、OpenAI 等。
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import itertools
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

import httpx
from openai import AsyncOpenAI

from agent.control.context import running_turn_id
from agent.llm_json import load_json_object_loose
from agent.model_runtime.auth.codex import CodexAuthDriver
from agent.model_runtime.auth.store import CredentialStore
from agent.model_runtime.errors import ContextWindowError
from agent.model_runtime.transports.responses import CodexResponsesTransport
from agent.model_runtime.types import (
    LLMResponse,
    ModelBackend,
    ModelRequest,
    ModelUsage,
    ToolCall,
    UsageCoverage,
)
from agent.model_runtime.usage import normalize_provider_usage
from core.common.diagnostic_log import turn_milestone
from core.error_context import (
    current_client_message_id,
    current_provider_attempt,
    current_provider_call_id,
    current_provider_operation,
    current_session_key,
)

if TYPE_CHECKING:
    from agent.config_models import ModelRuntimeConfig

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

logger = logging.getLogger(__name__)
_LLM_PAYLOAD_SNAPSHOT_ENABLED = False
_LAST_PAYLOAD_PATH = Path(tempfile.gettempdir()) / "akashic-last-llm-payload.json"
_PAYLOAD_SNAPSHOT_DIR = Path(tempfile.gettempdir()) / "akashic-llm-payloads"
_PAYLOAD_SNAPSHOT_SEQ = itertools.count(1)
_PAYLOAD_SNAPSHOT_MAX_FILES = 16
_PAYLOAD_SNAPSHOT_MAX_BYTES = 64 * 1024 * 1024
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


class LLMNetworkTimeoutError(TimeoutError):
    """LLM provider 在网络连接或读取边界超时。"""


class _ChatStreamReadError(Exception):
    """保留流读取原始错误及中断前是否已经收到有效 delta。"""

    def __init__(self, error: Exception, *, response_delta_seen: bool) -> None:
        super().__init__(str(error))
        self.error = error
        self.response_delta_seen = response_delta_seen


class _StreamHttpTelemetry:
    """流式调用传给 HTTP 重试层的观测上下文：稳定 span 与流重建序号。"""

    __slots__ = ("span_id", "stream_attempt")

    def __init__(self, *, span_id: str, stream_attempt: int) -> None:
        self.span_id = span_id
        self.stream_attempt = stream_attempt


class ProviderStrategy:
    def normalize_messages(self, messages: list[dict]) -> list[dict]:
        return _strip_reasoning_content(_normalize_chat_messages(messages))

    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        if disable_thinking:
            _drop_thinking_keys(extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body

    def extract_message(
        self,
        msg: Any,
        raw: str | None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        thinking: str | None = None
        if raw:
            m = _THINK_RE.search(raw)
            if m:
                thinking = m.group(1).strip()
                raw = _THINK_RE.sub("", raw).strip() or None
        return raw, thinking, {}

    def provider_fields_for_tool_call(
        self,
        fields: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return fields

    def prepare_stream_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {**kwargs, "stream": True}


class DeepSeekStrategy(ProviderStrategy):
    def normalize_messages(self, messages: list[dict]) -> list[dict]:
        return _strip_image_url_blocks(
            _normalize_chat_messages(messages, fill_tool_call_content=False)
        )

    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        thinking_enabled = extra_body.pop("enable_thinking", None)
        reasoning_effort = extra_body.pop("reasoning_effort", None)
        named_tool_choice = isinstance(kwargs.get("tool_choice"), dict)
        if disable_thinking or named_tool_choice:
            extra_body["thinking"] = {"type": "disabled"}
            reasoning_effort = None
            if named_tool_choice and not disable_thinking:
                logger.info("[deepseek] 命名 tool_choice 要求本次关闭 thinking")
        elif thinking_enabled is not None and "thinking" not in extra_body:
            extra_body["thinking"] = {
                "type": "enabled" if bool(thinking_enabled) else "disabled"
            }
        if reasoning_effort and not _deepseek_thinking_disabled(extra_body):
            kwargs["reasoning_effort"] = _normalize_deepseek_effort(
                str(reasoning_effort)
            )
        if not _deepseek_thinking_disabled(extra_body):
            messages = kwargs.get("messages")
            if isinstance(messages, list):
                kwargs["messages"] = _ensure_deepseek_reasoning_content(messages)
        if extra_body:
            kwargs["extra_body"] = extra_body

    def extract_message(
        self,
        msg: Any,
        raw: str | None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        reasoning = _get_field(msg, "reasoning_content")
        if reasoning is None:
            return raw, None, {}
        text = str(reasoning)
        return raw, text, {"reasoning_content": text}

    def provider_fields_for_tool_call(
        self,
        fields: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if _deepseek_thinking_disabled(dict(kwargs.get("extra_body") or {})):
            return fields
        if "reasoning_content" in fields:
            return fields
        return {**fields, "reasoning_content": ""}

    def prepare_stream_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        stream_kwargs = {**kwargs, "stream": True}
        stream_options = dict(stream_kwargs.get("stream_options") or {})
        stream_options["include_usage"] = True
        stream_kwargs["stream_options"] = stream_options
        return stream_kwargs


class DashScopeStrategy(ProviderStrategy):
    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        if disable_thinking:
            _drop_thinking_keys(extra_body)
            extra_body["enable_thinking"] = False
        if extra_body:
            kwargs["extra_body"] = extra_body


class OpenCodeGoStrategy(ProviderStrategy):
    def prepare_stream_request(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        stream_kwargs = {**kwargs, "stream": True}
        stream_options = dict(stream_kwargs.get("stream_options") or {})
        stream_options["include_usage"] = True
        stream_kwargs["stream_options"] = stream_options
        return stream_kwargs

    def extract_message(
        self,
        msg: Any,
        raw: str | None,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        reasoning = _get_field(msg, "reasoning_content")
        if reasoning is None:
            return super().extract_message(msg, raw)
        text = str(reasoning)
        return raw, text, {"reasoning_content": text}


class OpenCodeGoGLMStrategy(OpenCodeGoStrategy):
    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        thinking_enabled = extra_body.pop("enable_thinking", None)
        reasoning_effort = extra_body.pop("reasoning_effort", None)
        if (
            disable_thinking
            or thinking_enabled is False
            or _deepseek_thinking_disabled(extra_body)
        ):
            _drop_thinking_keys(extra_body)
        elif reasoning_effort or thinking_enabled:
            effort = str(reasoning_effort or "high").strip().lower()
            kwargs["reasoning_effort"] = (
                "max" if effort in {"xhigh", "max", "ultra"} else "high"
            )
        if extra_body:
            kwargs["extra_body"] = extra_body


class OpenCodeGoKimiStrategy(OpenCodeGoStrategy):
    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        thinking_enabled = extra_body.pop("enable_thinking", None)
        reasoning_effort = extra_body.pop("reasoning_effort", None)
        if (
            disable_thinking
            or thinking_enabled is False
            or _deepseek_thinking_disabled(extra_body)
        ):
            extra_body["thinking"] = {"type": "disabled"}
        elif reasoning_effort:
            extra_body.pop("thinking", None)
            effort = str(reasoning_effort).strip().lower()
            kwargs["reasoning_effort"] = (
                "high" if effort in {"xhigh", "max", "ultra"} else effort
            )
        elif thinking_enabled is True and "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            kwargs["extra_body"] = extra_body


class OpenCodeGoMiMoStrategy(OpenCodeGoStrategy):
    def prepare_request(
        self,
        kwargs: dict[str, Any],
        extra_body: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        if "max_tokens" in kwargs:
            kwargs["max_tokens"] = min(int(kwargs["max_tokens"]), 131_072)
        super().prepare_request(
            kwargs,
            extra_body,
            disable_thinking=disable_thinking,
        )


class ChatCompletionsRuntime:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        extra_body: dict | None = None,
        read_timeout_s: float = 90.0,
        max_retries: int = 3,
        provider_name: str = "",
        usage_provider_name: str = "",
        force_disable_thinking: bool = False,
        payload_snapshot_enabled: bool | None = None,
    ) -> None:
        normalized_base_url = _normalize_openai_base_url(base_url)
        network_timeout = httpx.Timeout(
            connect=30.0,
            read=max(0.001, float(read_timeout_s)),
            write=30.0,
            pool=30.0,
        )
        self._client = AsyncOpenAI(
            api_key=api_key or "credential-store",
            base_url=normalized_base_url,
            timeout=network_timeout,
            max_retries=0,
        )
        self._base_url = normalized_base_url or ""
        self._provider_name = provider_name
        self._usage_provider_name = usage_provider_name or provider_name
        self._extra_body = extra_body or {}
        self._max_retries = max(0, int(max_retries))
        self._force_disable_thinking = force_disable_thinking
        self._payload_snapshot_enabled = (
            _LLM_PAYLOAD_SNAPSHOT_ENABLED
            if payload_snapshot_enabled is None
            else bool(payload_snapshot_enabled)
        )

    def _milestone(
        self,
        event: str,
        *,
        model: str,
        started_at: float | None = None,
        outcome: str = "",
        kind: str = "",
        span_id: str = "",
        stream_attempt: int | None = None,
        http_attempt: int | None = None,
        response_id: str = "",
    ) -> None:
        """provider 侧观测里程碑：不记录正文，span/attempts/provider/model 进 counts。

        中性身份（provider_call_id/provider_attempt/provider_operation）从
        core.error_context 读，与高层 logical call 保持一致，不靠日志位置猜。
        """

        counts = f"provider={self._provider_name or '-'} model={model or '-'}"
        if span_id:
            counts += f" span_id={span_id}"
        if stream_attempt is not None:
            counts += f" stream_attempt={stream_attempt}"
        if http_attempt is not None:
            counts += f" http_attempt={http_attempt}"
        if response_id:
            counts += f" response_id={response_id}"
        if kind:
            counts += f" kind={kind}"
        counts += (
            f" provider_call_id={current_provider_call_id.get() or '-'} "
            f"provider_attempt={current_provider_attempt.get()} "
            f"provider_operation={current_provider_operation.get() or '-'}"
        )
        turn_milestone(
            logger,
            event,
            session_id=current_session_key.get() or "",
            turn_id=running_turn_id.get(),
            client_message_id=current_client_message_id.get(),
            duration_ms=(
                (time.monotonic() - started_at) * 1_000
                if started_at is not None
                else None
            ),
            outcome=outcome,
            counts=counts,
        )

    async def send(self, request: ModelRequest) -> LLMResponse:
        """把统一请求转换为 Chat Completions 并返回统一响应。"""
        strategy = _select_provider_strategy(
            provider_name=self._provider_name,
            base_url=self._base_url,
            model=request.model,
        )
        # 系统提示作为第一条消息（若 messages 已自带 system 消息则不再重复添加）。
        full_messages = _assemble_chat_messages(
            request.system_prompt,
            request.messages,
        )
        full_messages = strategy.normalize_messages(full_messages)
        kwargs: dict = dict(model=request.model, messages=full_messages)
        if request.max_output_tokens > 0:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = request.tool_choice
        merged_extra_body = dict(self._extra_body)
        merged_extra_body.update(request.extra_body)
        strategy.prepare_request(
            kwargs,
            merged_extra_body,
            disable_thinking=self._force_disable_thinking or request.disable_thinking,
        )

        if request.on_delta is not None:
            return await self._chat_streaming(kwargs, request.on_delta, strategy)

        # non-stream 总 span：覆盖真正 provider await 与 adapter 解析，恰好一个终态；
        # 内部 HTTP 重试仍由 _create_with_retry 静默处理，不展开为通用 attempt。
        nonstream_span_id = uuid.uuid4().hex
        nonstream_started = time.monotonic()
        self._milestone(
            "tl:provider.nonstream.start",
            span_id=nonstream_span_id,
            model=str(kwargs.get("model") or ""),
        )
        try:
            resp = cast(Any, await self._create_with_retry(kwargs))
            choice = resp.choices[0]
            msg = choice.message
            raw_finish_reason = getattr(choice, "finish_reason", None)
            finish_reason = (
                str(raw_finish_reason) if raw_finish_reason is not None else None
            )

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=_parse_tool_arguments(tc.function.arguments),
                        )
                    )

            raw, thinking, provider_fields = strategy.extract_message(msg, msg.content)
            cache_prompt_tokens, cache_hit_tokens = _extract_cache_usage(
                getattr(resp, "usage", None)
            )
            usage = _normalize_chat_usage(
                resp,
                provider_id=self._usage_provider_name,
                provider_api_url=self._base_url,
            )
            if tool_calls:
                provider_fields = strategy.provider_fields_for_tool_call(
                    provider_fields,
                    kwargs,
                )
            response = LLMResponse(
                content=raw,
                tool_calls=tool_calls,
                thinking=thinking,
                finish_reason=finish_reason,
                provider_fields=provider_fields,
                cache_prompt_tokens=cache_prompt_tokens,
                cache_hit_tokens=cache_hit_tokens,
                usage=usage,
            )
        except asyncio.CancelledError:
            self._milestone(
                "tl:provider.nonstream.cancelled",
                span_id=nonstream_span_id,
                model=str(kwargs.get("model") or ""),
                started_at=nonstream_started,
                outcome="cancelled",
            )
            raise
        except Exception:
            self._milestone(
                "tl:provider.nonstream.error",
                span_id=nonstream_span_id,
                model=str(kwargs.get("model") or ""),
                started_at=nonstream_started,
                outcome="error",
            )
            raise
        self._milestone(
            "tl:provider.nonstream.done",
            span_id=nonstream_span_id,
            model=str(kwargs.get("model") or ""),
            started_at=nonstream_started,
            outcome="done",
        )
        return response

    async def _chat_streaming(
        self,
        kwargs: dict[str, Any],
        on_content_delta: Callable[[StreamDelta], Awaitable[None]],
        strategy: ProviderStrategy,
    ) -> LLMResponse:
        """有限重试首个有效 delta 前断开的流，并组装最终响应。"""

        # 1. 每次重试都重建完整请求；收到有效 delta 后禁止重放。
        stream_kwargs = strategy.prepare_stream_request(kwargs)
        transport_span_id = uuid.uuid4().hex
        for attempt in range(self._max_retries + 1):
            stream_attempt = attempt + 1
            started_at = time.monotonic()
            self._milestone(
                "tl:provider.transport.start",
                span_id=transport_span_id,
                stream_attempt=stream_attempt,
                model=str(kwargs.get("model") or ""),
            )
            try:
                stream = cast(
                    Any,
                    await self._create_with_retry(
                        stream_kwargs,
                        telemetry=_StreamHttpTelemetry(
                            span_id=transport_span_id,
                            stream_attempt=stream_attempt,
                        ),
                    ),
                )
            except asyncio.CancelledError:
                self._milestone(
                    "tl:provider.transport.cancelled",
                    span_id=transport_span_id,
                    stream_attempt=stream_attempt,
                    model=str(kwargs.get("model") or ""),
                    started_at=started_at,
                    outcome="cancelled",
                )
                raise
            except Exception:
                self._milestone(
                    "tl:provider.transport.error",
                    span_id=transport_span_id,
                    stream_attempt=stream_attempt,
                    model=str(kwargs.get("model") or ""),
                    started_at=started_at,
                    outcome="error",
                )
                raise
            try:
                response = await self._consume_chat_stream(
                    stream,
                    kwargs,
                    on_content_delta,
                    strategy,
                    request_started_at=started_at,
                    transport_span_id=transport_span_id,
                    stream_attempt=stream_attempt,
                )
            except asyncio.CancelledError:
                self._milestone(
                    "tl:provider.transport.cancelled",
                    span_id=transport_span_id,
                    stream_attempt=stream_attempt,
                    model=str(kwargs.get("model") or ""),
                    started_at=started_at,
                    outcome="cancelled",
                )
                raise
            except _ChatStreamReadError as interrupted:
                error = interrupted.error
                retryable = self._is_retryable(error)
                exhausted = attempt >= self._max_retries
                if interrupted.response_delta_seen or not retryable or exhausted:
                    self._milestone(
                        "tl:provider.transport.error",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=started_at,
                        outcome="error",
                    )
                    if self._is_network_timeout(error):
                        raise LLMNetworkTimeoutError("LLM 流读取网络超时") from error
                    raise error
                wait_s = min(8.0, 1.0 * (2**attempt))
                logger.warning(
                    "[llm] 流读取中断，将重试 attempt=%d/%d wait=%.1fs err=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    wait_s,
                    type(error).__name__,
                )
                self._milestone(
                    "tl:provider.transport.retry",
                    span_id=transport_span_id,
                    stream_attempt=stream_attempt,
                    model=str(kwargs.get("model") or ""),
                    started_at=started_at,
                    outcome="retry",
                )
                try:
                    await asyncio.sleep(wait_s)
                except asyncio.CancelledError:
                    # retry 已是本 transport attempt 的终态；backoff 被取消
                    # 记独立事件，不得再给同一 attempt 记 cancelled。
                    self._milestone(
                        "tl:provider.transport.backoff_cancelled",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=started_at,
                        outcome="cancelled",
                    )
                    raise
                continue
            except Exception:
                self._milestone(
                    "tl:provider.transport.error",
                    span_id=transport_span_id,
                    stream_attempt=stream_attempt,
                    model=str(kwargs.get("model") or ""),
                    started_at=started_at,
                    outcome="error",
                )
                raise
            self._milestone(
                "tl:provider.transport.done",
                span_id=transport_span_id,
                stream_attempt=stream_attempt,
                model=str(kwargs.get("model") or ""),
                started_at=started_at,
                outcome="done",
            )
            return response
        raise RuntimeError("LLM stream failed without exception")

    async def _consume_chat_stream(
        self,
        stream: Any,
        kwargs: dict[str, Any],
        on_content_delta: Callable[[StreamDelta], Awaitable[None]],
        strategy: ProviderStrategy,
        *,
        request_started_at: float,
        transport_span_id: str,
        stream_attempt: int,
    ) -> LLMResponse:
        """消费一次 provider stream，并在读取中断时保留重试边界。"""

        # 1. 收集一次流的完整响应；只包装 SDK iterator 的读取错误。
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_chunks: dict[int, dict[str, str]] = {}
        tool_call_seen = False
        response_delta_seen = False
        first_any_logged = False
        first_thinking_logged = False
        first_answer_logged = False
        cache_prompt_tokens: int | None = None
        cache_hit_tokens: int | None = None
        usage: ModelUsage | None = None
        finish_reason: str | None = None

        try:
            stream_iter = aiter(stream)
            while True:
                try:
                    chunk = await anext(stream_iter)
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    raise _ChatStreamReadError(
                        exc,
                        response_delta_seen=response_delta_seen,
                    ) from exc
                raw_usage = getattr(chunk, "usage", None)
                prompt_tokens, hit_tokens = _extract_cache_usage(raw_usage)
                if prompt_tokens is not None:
                    cache_prompt_tokens = prompt_tokens
                    cache_hit_tokens = hit_tokens
                    usage = _normalize_chat_usage(
                        {"model": kwargs.get("model"), "usage": _dump_value(raw_usage)},
                        provider_id=self._usage_provider_name,
                        provider_api_url=self._base_url,
                    )
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                raw_finish_reason = getattr(choice, "finish_reason", None)
                if raw_finish_reason is not None:
                    finish_reason = str(raw_finish_reason)
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                reasoning_piece = _get_field(delta, "reasoning_content")
                tool_call_deltas = _iter_tool_call_deltas(delta)
                content_piece = _get_field(delta, "content")
                response_id = str(getattr(chunk, "id", "") or "")
                has_thinking = isinstance(reasoning_piece, str) and bool(
                    reasoning_piece
                )
                has_tool = bool(tool_call_deltas)
                has_content = isinstance(content_piece, str) and bool(content_piece)
                if has_tool and not first_any_logged:
                    first_any_logged = True
                    self._milestone(
                        "tl:provider.raw.first_any",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=request_started_at,
                        kind="tool",
                        response_id=response_id,
                    )
                elif has_thinking and not first_any_logged:
                    first_any_logged = True
                    self._milestone(
                        "tl:provider.raw.first_any",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=request_started_at,
                        kind="thinking",
                        response_id=response_id,
                    )
                elif has_content and not first_any_logged:
                    first_any_logged = True
                    self._milestone(
                        "tl:provider.raw.first_any",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=request_started_at,
                        kind="answer",
                        response_id=response_id,
                    )
                if has_thinking and not first_thinking_logged:
                    first_thinking_logged = True
                    self._milestone(
                        "tl:provider.raw.first_thinking",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=request_started_at,
                        response_id=response_id,
                    )
                if has_content and not first_answer_logged:
                    first_answer_logged = True
                    self._milestone(
                        "tl:provider.raw.first_answer",
                        span_id=transport_span_id,
                        stream_attempt=stream_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=request_started_at,
                        response_id=response_id,
                    )
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    response_delta_seen = True
                    reasoning_parts.append(reasoning_piece)
                    if not tool_call_seen:
                        await on_content_delta({"thinking_delta": reasoning_piece})

                if tool_call_deltas:
                    response_delta_seen = True
                for tc in tool_call_deltas:
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

                if isinstance(content_piece, str) and content_piece:
                    response_delta_seen = True
                    content_parts.append(content_piece)
                    if not tool_call_seen:
                        await on_content_delta({"content_delta": content_piece})
        finally:
            await stream.close()

        # 2. 将完整 tool-call 参数恢复成内部对象
        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_chunks):
            item = tool_call_chunks[idx]
            raw_args = item.get("arguments", "") or "{}"
            tool_calls.append(
                ToolCall(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    arguments=_parse_tool_arguments(raw_args),
                )
            )

        # 3. 组装文本、推理和 provider 字段
        raw = "".join(content_parts).strip() or None
        thinking = "".join(reasoning_parts).strip() or None
        raw, parsed_thinking, provider_fields = strategy.extract_message(
            {"reasoning_content": thinking} if thinking is not None else {},
            raw,
        )
        thinking = parsed_thinking if parsed_thinking is not None else thinking
        if tool_calls:
            provider_fields = strategy.provider_fields_for_tool_call(
                provider_fields,
                kwargs,
            )
        return LLMResponse(
            content=raw,
            tool_calls=tool_calls,
            thinking=thinking,
            finish_reason=finish_reason,
            provider_fields=provider_fields,
            cache_prompt_tokens=cache_prompt_tokens,
            cache_hit_tokens=cache_hit_tokens,
            usage=usage,
        )

    async def _create_with_retry(
        self,
        kwargs: dict,
        *,
        telemetry: _StreamHttpTelemetry | None = None,
    ) -> object:
        _save_llm_payload_snapshot(kwargs, enabled=self._payload_snapshot_enabled)
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            http_attempt = attempt + 1
            started_at = time.monotonic() if telemetry is not None else None
            if telemetry is not None:
                self._milestone(
                    "tl:provider.http.start",
                    span_id=telemetry.span_id,
                    stream_attempt=telemetry.stream_attempt,
                    http_attempt=http_attempt,
                    model=str(kwargs.get("model") or ""),
                )
            try:
                resp = await self._client.chat.completions.create(**kwargs)
            except asyncio.CancelledError:
                if telemetry is not None:
                    self._milestone(
                        "tl:provider.http.cancelled",
                        span_id=telemetry.span_id,
                        stream_attempt=telemetry.stream_attempt,
                        http_attempt=http_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=started_at,
                        outcome="cancelled",
                    )
                raise
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
                safety = self._is_safety_error(e)
                context_exceeded = self._is_context_length_error(e)
                retryable = self._is_retryable(e)
                exhausted = attempt >= self._max_retries
                if safety or context_exceeded or (not retryable) or exhausted:
                    if telemetry is not None:
                        self._milestone(
                            "tl:provider.http.error",
                            span_id=telemetry.span_id,
                            stream_attempt=telemetry.stream_attempt,
                            http_attempt=http_attempt,
                            model=str(kwargs.get("model") or ""),
                            started_at=started_at,
                            outcome="error",
                        )
                    if safety:
                        raise ContentSafetyError(str(e)) from e
                    if context_exceeded:
                        raise ContextLengthError(str(e)) from e
                    if self._is_network_timeout(e):
                        raise LLMNetworkTimeoutError("LLM 请求网络超时") from e
                    raise
                if telemetry is not None:
                    self._milestone(
                        "tl:provider.http.retry",
                        span_id=telemetry.span_id,
                        stream_attempt=telemetry.stream_attempt,
                        http_attempt=http_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=started_at,
                        outcome="retry",
                    )
                wait_s = min(8.0, 1.0 * (2**attempt))
                logger.warning(
                    "[llm] 请求失败，将重试 attempt=%d/%d wait=%.1fs err=%s",
                    attempt + 1,
                    self._max_retries + 1,
                    wait_s,
                    type(e).__name__,
                )
                await asyncio.sleep(wait_s)
            else:
                if telemetry is not None:
                    self._milestone(
                        "tl:provider.http.done",
                        span_id=telemetry.span_id,
                        stream_attempt=telemetry.stream_attempt,
                        http_attempt=http_attempt,
                        model=str(kwargs.get("model") or ""),
                        started_at=started_at,
                        outcome="done",
                    )
                return resp
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
        if ChatCompletionsRuntime._is_network_timeout(err) or isinstance(
            err, httpx.TransportError
        ):
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

    @staticmethod
    def _is_network_timeout(err: Exception) -> bool:
        return (
            isinstance(
                err,
                (TimeoutError, httpx.TimeoutException),
            )
            or type(err).__name__ == "APITimeoutError"
        )


class LLMProvider:
    """把稳定调用签名收敛为统一模型请求，并隐藏后端差异。"""

    @classmethod
    def from_runtime(
        cls,
        runtime: ModelRuntimeConfig,
        *,
        system_prompt: str,
        credential_store: CredentialStore | None = None,
        extra_body: dict[str, object] | None = None,
        read_timeout_s: float = 90.0,
        force_disable_thinking: bool = False,
        payload_snapshot_enabled: bool | None = None,
    ) -> LLMProvider:
        """从已校验配置构建 provider，不向调用层暴露后端参数。"""
        body = dict(extra_body or {})
        if runtime.reasoning_effort and not force_disable_thinking:
            body.setdefault("reasoning_effort", runtime.reasoning_effort)
        return cls(
            api_key=runtime.api_key,
            base_url=runtime.base_url or None,
            system_prompt=system_prompt,
            extra_body=body,
            read_timeout_s=read_timeout_s,
            provider_name=runtime.provider,
            usage_provider_name=runtime.catalog_provider_id or runtime.provider,
            auth_id=runtime.auth,
            credential_store=credential_store,
            runtime_id=runtime.runtime_id,
            context_window=runtime.context_window,
            use_responses_lite=runtime.use_responses_lite,
            supports_parallel_tool_calls=runtime.supports_parallel_tool_calls,
            reasoning_summary=runtime.reasoning_summary,
            force_disable_thinking=force_disable_thinking,
            payload_snapshot_enabled=payload_snapshot_enabled,
        )

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        system_prompt: str = "",
        extra_body: dict | None = None,
        read_timeout_s: float = 90.0,
        max_retries: int = 3,
        provider_name: str = "",
        usage_provider_name: str = "",
        auth_id: str = "",
        credential_store: CredentialStore | None = None,
        runtime_id: str = "main",
        context_window: int = 0,
        use_responses_lite: bool = False,
        supports_parallel_tool_calls: bool = True,
        reasoning_summary: str = "none",
        force_disable_thinking: bool = False,
        payload_snapshot_enabled: bool | None = None,
    ) -> None:
        self._system = system_prompt
        self._runtime_id = runtime_id
        self._extra_body = dict(extra_body or {})
        self._context_window = int(context_window)
        self._force_disable_thinking = force_disable_thinking
        if self._context_window < 0:
            raise ValueError("context_window 不能小于 0")
        self._backend: ModelBackend
        if provider_name.lower() == "codex":
            auth = CodexAuthDriver(credential_store or CredentialStore(), auth_id)
            self._backend = CodexResponsesTransport(
                auth,
                runtime_id=runtime_id,
                base_url=base_url or "https://chatgpt.com/backend-api/codex",
                read_timeout_s=read_timeout_s,
                use_responses_lite=use_responses_lite,
                supports_parallel_tool_calls=supports_parallel_tool_calls,
                reasoning_summary=reasoning_summary,
            )
        else:
            self._backend = ChatCompletionsRuntime(
                api_key=api_key,
                base_url=base_url,
                extra_body=extra_body,
                read_timeout_s=read_timeout_s,
                max_retries=max_retries,
                provider_name=provider_name,
                usage_provider_name=usage_provider_name,
                force_disable_thinking=force_disable_thinking,
                payload_snapshot_enabled=payload_snapshot_enabled,
            )

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        max_tokens: int,
        tool_choice: str | dict = "auto",
        extra_body: dict | None = None,
        disable_thinking: bool = False,
        on_content_delta: Callable[[StreamDelta], Awaitable[None]] | None = None,
        cache_namespace: str = "",
    ) -> LLMResponse:
        merged_extra = {**self._extra_body, **(extra_body or {})}
        effort = merged_extra.get("reasoning_effort")
        request = ModelRequest(
            messages=messages,
            tools=tools,
            model=model,
            max_output_tokens=max_tokens,
            system_prompt=self._system,
            tool_choice=tool_choice,
            reasoning_effort=(
                None
                if self._force_disable_thinking or disable_thinking
                else str(effort or "") or None
            ),
            prompt_cache_key=(
                _stable_prompt_cache_key(self._runtime_id, model, cache_namespace)
                if cache_namespace
                else None
            ),
            on_delta=on_content_delta,
            extra_body=dict(extra_body or {}),
            disable_thinking=self._force_disable_thinking or disable_thinking,
        )
        try:
            return await self._backend.send(request)
        except ContextWindowError as exc:
            raise ContextLengthError(str(exc)) from exc

    @property
    def context_window(self) -> int:
        return self._context_window

    @property
    def runtime_id(self) -> str:
        """Return the stable runtime identity used by durable compaction receipts."""

        return self._runtime_id

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        """Estimate the complete provider input owned by this runtime."""
        return _estimate_context_tokens(self._system, messages, tools)

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        """Estimate messages appended after an exact provider usage sample."""
        return _estimate_message_tokens(messages)


def _estimate_context_tokens(
    system_prompt: str, messages: list[dict], tools: list[dict]
) -> int:
    """估算文本与图片块预算，避免把 data URI 当作文本 token。"""
    full_messages = _assemble_chat_messages(system_prompt, messages)
    fixed_chars = len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
    return max(1, fixed_chars // 3 + _estimate_message_tokens(full_messages))


def _assemble_chat_messages(
    system_prompt: str,
    messages: list[dict],
) -> list[dict]:
    """统一发送与估算共用的首条 system 消息组装规则。"""

    already_has_system = bool(messages) and messages[0].get("role") == "system"
    full_messages = (
        [{"role": "system", "content": system_prompt}, *messages]
        if system_prompt and not already_has_system
        else messages
    )
    return _merge_leading_system_messages(full_messages)


def _estimate_message_tokens(messages: list[dict]) -> int:
    """估算消息增量，保持与完整 provider input 相同的编码口径。"""
    text_chars = 0
    image_tokens = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {
                    "image_url",
                    "input_image",
                }:
                    detail = block.get("detail")
                    image = block.get("image_url")
                    if isinstance(image, dict):
                        detail = image.get("detail", detail)
                    image_tokens += 1024 if detail == "low" else 8192
                    continue
                text_chars += len(
                    json.dumps(block, ensure_ascii=False, separators=(",", ":"))
                )
        elif content is not None:
            text_chars += len(str(content))
        text_chars += len(
            json.dumps(
                {key: value for key, value in message.items() if key != "content"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if not messages:
        return 0
    return max(1, text_chars // 3 + image_tokens)


def _stable_prompt_cache_key(runtime_id: str, model: str, namespace: str) -> str:
    """生成不泄露 session 标识的稳定缓存路由键。"""
    raw = f"{runtime_id}\0{model}\0{namespace}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _get_field(delta: Any, name: str) -> Any:
    if isinstance(delta, dict):
        return delta.get(name)
    return getattr(delta, name, None)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """解析 provider 工具参数并保持内部对象契约。"""

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = load_json_object_loose(raw_arguments)
        if arguments is None:
            raise
        logger.warning(
            "[llm] repaired malformed tool arguments original_chars=%d",
            len(raw_arguments),
        )
    if not isinstance(arguments, dict):
        raise TypeError("LLM 工具调用参数必须是 JSON 对象")
    return arguments


def _save_llm_payload_snapshot(
    kwargs: dict,
    *,
    enabled: bool | None = None,
) -> Path | None:
    """保存完整请求快照，并在写入前回收最旧快照。"""

    if not (_LLM_PAYLOAD_SNAPSHOT_ENABLED if enabled is None else enabled):
        return None
    try:
        # 1. 序列化后先为新快照腾出空间，避免把共享 /tmp 写满。
        payload = json.dumps(
            kwargs,
            ensure_ascii=False,
            indent=2,
            default=str,
        ).encode("utf-8")
        if len(payload) > _PAYLOAD_SNAPSHOT_MAX_BYTES:
            logger.warning(
                "[LLM请求快照] 跳过超限快照 bytes=%d max_bytes=%d",
                len(payload),
                _PAYLOAD_SNAPSHOT_MAX_BYTES,
            )
            return None
        _PAYLOAD_SNAPSHOT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = _PAYLOAD_SNAPSHOT_DIR / ".rotation.lock"
        with lock_path.open("a+b") as lock_file:
            lock_path.chmod(0o600)
            fcntl.flock(lock_file, fcntl.LOCK_EX)

            # 2. 回收崩溃残留和旧快照，再原子发布完整新快照。
            for stale_path in _PAYLOAD_SNAPSHOT_DIR.glob(".*.tmp"):
                stale_path.unlink()
            for stale_link in _LAST_PAYLOAD_PATH.parent.glob(
                f".{_LAST_PAYLOAD_PATH.name}.*.tmp"
            ):
                stale_link.unlink()
            _LAST_PAYLOAD_PATH.unlink(missing_ok=True)
            _prune_llm_payload_snapshots(required_bytes=len(payload))
            seq = next(_PAYLOAD_SNAPSHOT_SEQ)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            path = _PAYLOAD_SNAPSHOT_DIR / f"{ts}-{os.getpid()}-{seq:06d}.json"
            pending_path = path.with_name(f".{path.name}.tmp")
            pending_path.write_bytes(payload)
            pending_path.chmod(0o600)
            os.replace(pending_path, path)

            # 3. last 与最新快照共享 inode，保留入口但不重复占空间。
            latest_link = _LAST_PAYLOAD_PATH.with_name(
                f".{_LAST_PAYLOAD_PATH.name}.{os.getpid()}-{seq:06d}.tmp"
            )
            os.link(path, latest_link)
            os.replace(latest_link, _LAST_PAYLOAD_PATH)
        logger.info("[LLM请求快照] saved=%s", path)
        return path
    except Exception as exc:
        logger.warning("[LLM请求快照] 保存失败: %s", exc)
        return None


def _prune_llm_payload_snapshots(*, required_bytes: int) -> None:
    """删除最旧快照，为下一份快照保留数量和字节预算。"""

    snapshots = sorted(
        (entry for entry in _PAYLOAD_SNAPSHOT_DIR.glob("*.json") if entry.is_file()),
        key=lambda entry: (entry.stat().st_mtime_ns, entry.name),
    )
    retained_bytes = sum(entry.stat().st_size for entry in snapshots)
    while snapshots and (
        len(snapshots) >= _PAYLOAD_SNAPSHOT_MAX_FILES
        or retained_bytes + required_bytes > _PAYLOAD_SNAPSHOT_MAX_BYTES
    ):
        oldest = snapshots.pop(0)
        retained_bytes -= oldest.stat().st_size
        oldest.unlink()


def _extract_cache_usage(usage: Any) -> tuple[int | None, int | None]:
    hit_tokens = _coerce_int(_get_field(usage, "prompt_cache_hit_tokens"))
    miss_tokens = _coerce_int(_get_field(usage, "prompt_cache_miss_tokens"))
    if hit_tokens is not None or miss_tokens is not None:
        hit = hit_tokens or 0
        miss = miss_tokens or 0
        return hit + miss, hit

    prompt_tokens = _coerce_int(_get_field(usage, "prompt_tokens"))
    prompt_details = _get_field(usage, "prompt_tokens_details")
    cached_tokens = _coerce_int(_get_field(prompt_details, "cached_tokens"))
    if prompt_tokens is None or cached_tokens is None:
        return None, None
    return prompt_tokens, cached_tokens


def _extract_model_usage(usage: Any) -> ModelUsage | None:
    """把 Chat Completions usage 映射为规范化用量。"""
    if usage is None:
        return None
    prompt_tokens = _coerce_int(_get_field(usage, "prompt_tokens"))
    completion_tokens = _coerce_int(_get_field(usage, "completion_tokens"))
    prompt_details = _get_field(usage, "prompt_tokens_details")
    completion_details = _get_field(usage, "completion_tokens_details")
    cached_tokens = _coerce_int(_get_field(prompt_details, "cached_tokens"))
    cache_write_tokens = _coerce_int(_get_field(prompt_details, "cache_write_tokens"))
    if cached_tokens is None:
        _, cached_tokens = _extract_cache_usage(usage)
    reasoning_tokens = _coerce_int(_get_field(completion_details, "reasoning_tokens"))
    return ModelUsage(
        input_tokens=prompt_tokens,
        cache_write_input_tokens=cache_write_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=completion_tokens,
        reasoning_output_tokens=reasoning_tokens,
        covered_request_count=(
            1 if prompt_tokens is not None and completion_tokens is not None else 0
        ),
        coverage=(
            UsageCoverage.EXACT
            if prompt_tokens is not None and completion_tokens is not None
            else (
                UsageCoverage.PARTIAL
                if prompt_tokens is not None or completion_tokens is not None
                else UsageCoverage.UNAVAILABLE
            )
        ),
    )


def _normalize_chat_usage(
    response: Any,
    *,
    provider_id: str,
    provider_api_url: str,
) -> ModelUsage | None:
    """优先用成熟 extractor 归一化，未知兼容格式沿用 wire parser。"""

    data = _dump_value(response)
    raw_usage = _get_field(response, "usage")
    if isinstance(response, dict):
        raw_usage = response.get("usage")
    completion_details = _get_field(raw_usage, "completion_tokens_details")
    reasoning_tokens = _coerce_int(_get_field(completion_details, "reasoning_tokens"))
    fallback = _extract_model_usage(raw_usage)
    normalized = normalize_provider_usage(
        data,
        provider_id=provider_id,
        provider_api_url=provider_api_url,
        api_flavor="chat",
        reasoning_output_tokens=reasoning_tokens,
    )
    if normalized is None:
        return fallback
    if fallback is None:
        return normalized
    input_tokens = (
        normalized.input_tokens
        if normalized.input_tokens is not None
        else fallback.input_tokens
    )
    output_tokens = (
        normalized.output_tokens
        if normalized.output_tokens is not None
        else fallback.output_tokens
    )
    covered = int(input_tokens is not None and output_tokens is not None)
    coverage = (
        UsageCoverage.EXACT
        if covered
        else (
            UsageCoverage.PARTIAL
            if input_tokens is not None or output_tokens is not None
            else UsageCoverage.UNAVAILABLE
        )
    )
    return ModelUsage(
        input_tokens=input_tokens,
        cache_write_input_tokens=(
            normalized.cache_write_input_tokens
            if normalized.cache_write_input_tokens is not None
            else fallback.cache_write_input_tokens
        ),
        cached_input_tokens=(
            normalized.cached_input_tokens
            if normalized.cached_input_tokens is not None
            else fallback.cached_input_tokens
        ),
        output_tokens=output_tokens,
        reasoning_output_tokens=(
            normalized.reasoning_output_tokens
            if normalized.reasoning_output_tokens is not None
            else fallback.reasoning_output_tokens
        ),
        request_count=max(normalized.request_count, fallback.request_count),
        covered_request_count=covered,
        coverage=coverage,
    )


def _dump_value(value: Any) -> Any:
    if isinstance(value, dict) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


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


def _select_provider_strategy(
    *,
    provider_name: str,
    base_url: str,
    model: str,
) -> ProviderStrategy:
    if provider_name.strip().lower() == "opencode-go":
        normalized_model = model.strip().lower()
        if normalized_model.startswith("deepseek-"):
            return DeepSeekStrategy()
        if normalized_model.startswith("glm-"):
            return OpenCodeGoGLMStrategy()
        if normalized_model.startswith("kimi-"):
            return OpenCodeGoKimiStrategy()
        if normalized_model.startswith("mimo-v2.5-pro"):
            return OpenCodeGoMiMoStrategy()
        return OpenCodeGoStrategy()
    provider_text = f"{provider_name} {base_url} {model}".lower()
    if "deepseek" in provider_text:
        return DeepSeekStrategy()
    if (
        "dashscope.aliyuncs.com" in provider_text
        or "dashscope" in provider_text
        or "xiaomimimo.com" in provider_text
    ):
        return DashScopeStrategy()
    return ProviderStrategy()


def _drop_thinking_keys(extra_body: dict[str, Any]) -> None:
    for key in ("enable_thinking", "thinking", "reasoning_effort"):
        extra_body.pop(key, None)


def _deepseek_thinking_disabled(extra_body: dict[str, Any]) -> bool:
    thinking = extra_body.get("thinking")
    if not isinstance(thinking, dict):
        return False
    return str(thinking.get("type", "") or "").lower() == "disabled"


def _normalize_deepseek_effort(value: str) -> str:
    effort = value.strip().lower()
    if effort == "xhigh":
        return "max"
    return effort


def _ensure_deepseek_reasoning_content(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        if item.get("role") == "assistant" and "reasoning_content" not in item:
            item["reasoning_content"] = ""
        normalized.append(item)
    return normalized


def _normalize_chat_messages(
    messages: list[dict],
    *,
    fill_tool_call_content: bool = True,
) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        role = str(item.get("role", "") or "")
        content = item.get("content")

        if fill_tool_call_content and role == "assistant" and item.get("tool_calls"):
            if content is None or (isinstance(content, str) and not content.strip()):
                tool_calls = item.get("tool_calls") or []
                first = (
                    tool_calls[0] if isinstance(tool_calls, list) and tool_calls else {}
                )
                function = first.get("function") if isinstance(first, dict) else {}
                tool_name = ""
                if isinstance(function, dict):
                    tool_name = str(function.get("name", "") or "")
                item["content"] = f"调用工具 {tool_name}" if tool_name else "调用工具"
        elif role in {"user", "assistant", "tool"}:
            if content is None:
                item["content"] = ""

        normalized.append(item)
    return normalized


def _strip_reasoning_content(messages: list[dict]) -> list[dict]:
    # 非 DeepSeek provider 不应发送 reasoning_content 字段
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


def _strip_image_url_blocks(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for msg in messages:
        item = dict(msg)
        content = item.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            image_count = 0
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                elif block_type == "image_url":
                    image_count += 1
            if image_count:
                text_parts.append(
                    f"[已移除 {image_count} 个 image_url 图片块：DeepSeek 当前接口只接受文本消息。]"
                )
            item["content"] = "\n".join(text_parts)
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
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )
