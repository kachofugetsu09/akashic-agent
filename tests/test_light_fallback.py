from __future__ import annotations

import asyncio
from typing import cast

import httpx
import openai
import pytest

from agent.model_runtime.errors import (
    QuotaError,
    RateLimitError,
    RetryableTransportError,
    TransportError,
)
from agent.model_runtime.fallback import ResilientLightProvider
from agent.provider import ContentSafetyError, ContextLengthError, LLMProvider, LLMResponse


class _Provider:
    def __init__(self, outcome: LLMResponse | BaseException, *, emit: bool = False) -> None:
        self.outcome = outcome
        self.emit = emit
        self.calls: list[dict] = []

    async def chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        callback = kwargs.get("on_content_delta")
        if self.emit and callback:
            await callback({"content_delta": "partial"})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _HangingProvider(_Provider):
    async def chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _status(code: int, body: object | None = None) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://light.example/v1")
    response = httpx.Response(code, request=request)
    return openai.APIStatusError(f"status={code}", response=response, body=body)


def _resilient(primary: _Provider, fallback: _Provider) -> ResilientLightProvider:
    return ResilientLightProvider(
        primary=cast(LLMProvider, primary),
        primary_runtime_id="fast",
        primary_model="fast-model",
        fallback=cast(LLMProvider, fallback),
        fallback_model="main-model",
    )


async def _chat(provider: ResilientLightProvider, **kwargs) -> LLMResponse:
    return await provider.chat(
        messages=kwargs.get("messages", [{"role": "user", "content": "hi"}]),
        tools=kwargs.get("tools", []),
        model="ignored",
        max_tokens=321,
        tool_choice="required",
        disable_thinking=True,
        on_content_delta=kwargs.get("on_content_delta"),
        cache_namespace="session",
    )


@pytest.mark.asyncio
async def test_light_success_does_not_call_main() -> None:
    primary = _Provider(LLMResponse(content="fast"))
    fallback = _Provider(LLMResponse(content="main"))

    result = await _chat(_resilient(primary, fallback))

    assert result.content == "fast"
    assert primary.calls[0]["model"] == "fast-model"
    assert fallback.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        httpx.ConnectError("connection"),
        _status(429),
        _status(503),
        RateLimitError("limited"),
        RetryableTransportError("disconnected"),
    ],
)
async def test_light_recoverable_failure_replays_same_request(error: BaseException) -> None:
    fallback = _Provider(LLMResponse(content="main"))
    messages = [{"role": "user", "content": "same"}]
    tools = [{"type": "function", "function": {"name": "x"}}]

    result = await _chat(
        _resilient(_Provider(error), fallback), messages=messages, tools=tools
    )

    assert result.content == "main"
    call = fallback.calls[0]
    assert (call["messages"], call["tools"], call["model"]) == (
        messages,
        tools,
        "main-model",
    )
    assert (call["max_tokens"], call["tool_choice"], call["disable_thinking"]) == (
        321,
        "required",
        True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _status(400),
        _status(429, {"error": {"code": "insufficient_quota"}}),
        ContextLengthError("too long"),
        ContentSafetyError("unsafe"),
        QuotaError("quota"),
        TransportError("invalid protocol"),
    ],
)
async def test_light_nonrecoverable_failure_stays_fail_loud(error: BaseException) -> None:
    fallback = _Provider(LLMResponse(content="main"))

    with pytest.raises(type(error)):
        await _chat(_resilient(_Provider(error), fallback))

    assert fallback.calls == []


@pytest.mark.asyncio
async def test_light_never_replays_after_output_or_outer_cancellation() -> None:
    fallback = _Provider(LLMResponse(content="main"))
    deltas: list[dict[str, str]] = []

    async def on_delta(delta: dict[str, str]) -> None:
        deltas.append(delta)

    with pytest.raises(TimeoutError):
        await _chat(
            _resilient(_Provider(TimeoutError("timeout"), emit=True), fallback),
            on_content_delta=on_delta,
        )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            _chat(_resilient(_HangingProvider(LLMResponse(content=None)), fallback)),
            timeout=0.01,
        )

    assert deltas == [{"content_delta": "partial"}]
    assert fallback.calls == []
