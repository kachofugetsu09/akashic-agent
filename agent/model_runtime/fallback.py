from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import httpx
import openai

from agent.model_runtime.errors import RateLimitError, TransportError
from agent.provider import LLMProvider, LLMResponse, StreamDelta

logger = logging.getLogger("agent.model_runtime.fallback")


def _openai_error_types(*names: str) -> tuple[type[BaseException], ...]:
    return tuple(
        error_type
        for name in names
        if isinstance((error_type := getattr(openai, name, None)), type)
        and issubclass(error_type, BaseException)
    )


_OPENAI_CONNECTION_ERROR_TYPES = _openai_error_types(
    "APIConnectionError",
    "APITimeoutError",
)
_OPENAI_STATUS_ERROR_TYPES = _openai_error_types("APIStatusError")
_RUNTIME_FAILURE_TYPES = (
    TimeoutError,
    httpx.TimeoutException,
    httpx.ConnectError,
    RateLimitError,
    TransportError,
    *_OPENAI_CONNECTION_ERROR_TYPES,
    *_OPENAI_STATUS_ERROR_TYPES,
)


class ResilientLightProvider(LLMProvider):
    """让独立轻量模型仅在可恢复运行故障时回退主模型。"""

    def __init__(
        self,
        *,
        primary: LLMProvider,
        primary_runtime_id: str,
        primary_model: str,
        fallback: LLMProvider,
        fallback_model: str,
    ) -> None:
        self.primary = primary
        self.primary_runtime_id = primary_runtime_id
        self.primary_model = primary_model
        self.fallback = fallback
        self.fallback_model = fallback_model

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
        """先调用轻量模型；未输出 delta 的可恢复故障才切换主模型。"""

        # 1. 代理 delta 并记录是否已经向调用方产生可见输出。
        emitted = False

        async def track_delta(delta: StreamDelta) -> None:
            nonlocal emitted
            emitted = emitted or _has_visible_delta(delta)
            if on_content_delta is not None:
                await on_content_delta(delta)

        callback = track_delta if on_content_delta is not None else None

        # 2. 主路径保留独立 light model，其余调用参数原样传递。
        try:
            return await self.primary.chat(
                messages=messages,
                tools=tools,
                model=self.primary_model,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                extra_body=extra_body,
                disable_thinking=disable_thinking,
                on_content_delta=callback,
                cache_namespace=cache_namespace,
            )
        except _RUNTIME_FAILURE_TYPES as exc:
            if emitted or not _is_recoverable_runtime_error(exc):
                raise

            # 3. 只替换模型和 provider，避免改变上层任务语义。
            logger.warning(
                "[light.fallback] primary_runtime=%s primary_model=%s err_type=%s "
                "fallback_model=%s",
                self.primary_runtime_id,
                self.primary_model,
                type(exc).__name__,
                self.fallback_model,
            )
            return await self.fallback.chat(
                messages=messages,
                tools=tools,
                model=self.fallback_model,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                extra_body=extra_body,
                disable_thinking=disable_thinking,
                on_content_delta=on_content_delta,
                cache_namespace=cache_namespace,
            )


def _has_visible_delta(delta: StreamDelta) -> bool:
    if isinstance(delta, str):
        return bool(delta)
    return any(
        isinstance(delta.get(key), str) and bool(delta[key])
        for key in ("content_delta", "thinking_delta")
    )


def _is_recoverable_runtime_error(exc: BaseException) -> bool:
    """只认可超时、连接、限流和服务端瞬时故障。"""
    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, _OPENAI_CONNECTION_ERROR_TYPES):
        return True
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, TransportError):
        return _is_connection_transport_error(exc)
    if isinstance(exc, _OPENAI_STATUS_ERROR_TYPES):
        status_code = int(getattr(exc, "status_code"))
        if status_code == 429:
            return not _looks_like_quota_error(exc)
        return 500 <= status_code < 600
    return False


def _is_connection_transport_error(exc: TransportError) -> bool:
    cause = exc.__cause__
    if isinstance(
        cause,
        (
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
            *_OPENAI_CONNECTION_ERROR_TYPES,
        ),
    ):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "连接失败",
            "connection",
            "timeout",
            "超时",
            "暂时失败",
        )
    )


def _looks_like_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    body = getattr(exc, "body", None)
    combined = f"{text} {body!r}".lower()
    return any(
        marker in combined
        for marker in ("insufficient_quota", "quota exceeded", "billing", "credits")
    )


__all__ = ["ResilientLightProvider"]
