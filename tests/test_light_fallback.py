from __future__ import annotations

import asyncio
import inspect
from typing import cast

import httpx
import pytest

import agent.model_runtime.fallback as fallback_module
from agent.model_runtime.errors import RateLimitError, TransportError
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
        if self.emit and callback is not None:
            await callback({"content_delta": "partial"})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _HangingProvider(_Provider):
    async def chat(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _StatusError(Exception):
    def __init__(self, status_code: int, body: object | None = None) -> None:
        super().__init__(f"status={status_code}")
        self.status_code = status_code
        self.body = body


@pytest.fixture(autouse=True)
def _register_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fallback_module, "_OPENAI_STATUS_ERROR_TYPES", (_StatusError,))
    monkeypatch.setattr(
        fallback_module,
        "_RUNTIME_FAILURE_TYPES",
        (*fallback_module._RUNTIME_FAILURE_TYPES, _StatusError),
    )


def _provider(
    primary: _Provider, fallback: _Provider
) -> ResilientLightProvider:
    return ResilientLightProvider(
        primary=cast(LLMProvider, primary),
        primary_runtime_id="fast",
        primary_model="fast-model",
        fallback=cast(LLMProvider, fallback),
        fallback_model="main-model",
    )


def test_light_wrapper_does_not_own_a_total_deadline() -> None:
    source = inspect.getsource(ResilientLightProvider.chat)
    assert "wait_for" not in source
    assert "create_task" not in source
    assert ".cancel(" not in source


async def _chat(provider: ResilientLightProvider, **kwargs) -> LLMResponse:
    return await provider.chat(
        messages=kwargs.get("messages", [{"role": "user", "content": "hi"}]),
        tools=kwargs.get("tools", [{"type": "function"}]),
        model="caller-model",
        max_tokens=kwargs.get("max_tokens", 128),
        tool_choice=kwargs.get("tool_choice", "required"),
        disable_thinking=kwargs.get("disable_thinking", True),
        on_content_delta=kwargs.get("on_content_delta"),
    )


@pytest.mark.asyncio
async def test_light_primary_success_does_not_call_main() -> None:
    primary = _Provider(LLMResponse(content="fast"))
    fallback = _Provider(LLMResponse(content="main"))

    result = await _chat(_provider(primary, fallback))

    assert result.content == "fast"
    assert primary.calls[0]["model"] == "fast-model"
    assert fallback.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        httpx.ConnectError(
            "connection", request=httpx.Request("POST", "https://light.example/v1")
        ),
        _StatusError(429),
        _StatusError(503),
        RateLimitError("Codex 请求被限流"),
        TransportError("Codex Responses 连接失败"),
    ],
    ids=["timeout", "connection", "429", "5xx", "codex-rate-limit", "codex-transport"],
)
async def test_light_recoverable_failure_falls_back_with_main_model(
    error: BaseException,
) -> None:
    primary = _Provider(error)
    fallback = _Provider(LLMResponse(content="main"))
    messages = [{"role": "user", "content": "same"}]
    tools = [{"type": "function", "function": {"name": "x"}}]

    result = await _chat(
        _provider(primary, fallback),
        messages=messages,
        tools=tools,
        max_tokens=321,
    )

    assert result.content == "main"
    call = fallback.calls[0]
    assert call["model"] == "main-model"
    assert call["messages"] == messages
    assert call["tools"] == tools
    assert call["max_tokens"] == 321
    assert call["tool_choice"] == "required"
    assert call["disable_thinking"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        _StatusError(401),
        _StatusError(400),
        _StatusError(404),
        _StatusError(429, body={"error": {"code": "insufficient_quota"}}),
        ContextLengthError("too long"),
        ContentSafetyError("unsafe"),
        TransportError("响应 JSON 损坏"),
    ],
    ids=[
        "401",
        "invalid-request",
        "unknown-model",
        "quota",
        "context",
        "safety",
        "protocol",
    ],
)
async def test_light_nonrecoverable_failure_stays_fail_loud(
    error: BaseException,
) -> None:
    fallback = _Provider(LLMResponse(content="main"))

    with pytest.raises(type(error)):
        await _chat(_provider(_Provider(error), fallback))

    assert fallback.calls == []


@pytest.mark.asyncio
async def test_light_does_not_replay_after_visible_delta() -> None:
    primary = _Provider(TimeoutError("timeout"), emit=True)
    fallback = _Provider(LLMResponse(content="main"))
    deltas: list[dict[str, str]] = []

    async def on_delta(delta: dict[str, str]) -> None:
        deltas.append(delta)

    with pytest.raises(TimeoutError):
        await _chat(_provider(primary, fallback), on_content_delta=on_delta)

    assert deltas == [{"content_delta": "partial"}]
    assert fallback.calls == []


@pytest.mark.asyncio
async def test_upper_deadline_cancellation_does_not_trigger_fallback() -> None:
    primary = _HangingProvider(LLMResponse(content="unused"))
    fallback = _Provider(LLMResponse(content="main"))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(_chat(_provider(primary, fallback)), timeout=0.01)

    assert fallback.calls == []
