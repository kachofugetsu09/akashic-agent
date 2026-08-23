from __future__ import annotations
from typing import Any, cast

import asyncio
import httpx
import json
import logging
import runpy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent.provider as provider_module
from agent.config_models import Config as ConfigModel
from agent.provider import (
    ContextLengthError,
    ContentSafetyError,
    LLMNetworkTimeoutError,
    LLMProvider,
    _normalize_openai_base_url,
)
from agent.tool_runtime import append_assistant_tool_calls
from infra.channels.group_filter import DefaultGroupFilter, strip_at_segments
from bootstrap.app import AppRuntime
from bootstrap.providers import build_providers, build_vl_provider
from bus.event_bus import EventBus
from session.manager import Session


class _Response:
    def __init__(
        self,
        content: str = "ok",
        tool_calls: list | None = None,
        reasoning_content: str | None = None,
        usage: object | None = None,
        finish_reason: str | None = None,
    ) -> None:
        message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        if reasoning_content is not None:
            message.reasoning_content = reasoning_content
        self.choices = [SimpleNamespace(message=message, finish_reason=finish_reason)]
        self.usage = usage


class _ToolCall:
    def __init__(self, id: str, name: str, arguments: dict) -> None:
        self.id = id
        self.function = SimpleNamespace(
            name=name, arguments=json.dumps(arguments, ensure_ascii=False)
        )


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeStream:
    def __init__(self, chunks: list[object], delay_s: float = 0.0) -> None:
        self._chunks = list(chunks)
        self._delay_s = delay_s
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        chunk = self._chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_opencode_go_request_mappings_cross_real_http_boundary() -> None:
    payloads: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payloads.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payloads[-1]["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                                "reasoning_content": "thought",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = server.server_address[0]
        port = server.server_address[1]
        base_url = f"http://{host}:{port}/v1"
        cases = [
            ("glm-5.99", {"reasoning_effort": "xhigh"}, 200_000),
            (
                "glm-5.98",
                {"reasoning_effort": "high", "thinking": {"type": "disabled"}},
                200_000,
            ),
            ("kimi-k3", {"enable_thinking": True}, 200_000),
            (
                "kimi-k2.6",
                {"reasoning_effort": "high", "thinking": {"type": "disabled"}},
                200_000,
            ),
            ("deepseek-v4-pro", {"reasoning_effort": "xhigh"}, 200_000),
            ("mimo-v2.5-pro", {}, 200_000),
            ("grok-4.5", {}, 200_000),
        ]
        for model, extra_body, max_tokens in cases:
            result = await LLMProvider(
                api_key="secret",
                base_url=base_url,
                provider_name="opencode-go",
                extra_body=extra_body,
                max_retries=0,
            ).chat([], [], model, max_tokens)
            assert result.content == "ok"
            assert result.thinking == "thought"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    by_model = {payload["model"]: payload for payload in payloads}
    assert by_model["glm-5.99"]["reasoning_effort"] == "max"
    assert "reasoning_effort" not in by_model["glm-5.98"]
    assert "thinking" not in by_model["glm-5.98"]
    assert by_model["kimi-k3"]["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in by_model["kimi-k3"]
    assert by_model["kimi-k2.6"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in by_model["kimi-k2.6"]
    assert by_model["deepseek-v4-pro"]["reasoning_effort"] == "max"
    assert by_model["mimo-v2.5-pro"]["max_tokens"] == 131_072


@pytest.mark.asyncio
async def test_provider_chat_and_retry_paths(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient(
        [
            httpx.ReadTimeout("request idle"),
            _Response(
                content="done",
                tool_calls=[_ToolCall("1", "search", {"q": "x"})],
                finish_reason="tool_calls",
            ),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    slept = []

    async def _sleep(sec: float) -> None:
        slept.append(sec)

    monkeypatch.setattr("agent.provider.asyncio.sleep", _sleep)
    provider = LLMProvider(
        api_key="k",
        base_url="https://example.com",
        system_prompt="system",
        extra_body={"x": 1},
        max_retries=1,
    )
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        model="m",
        max_tokens=10,
    )
    assert result.content == "done"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].arguments == {"q": "x"}
    assert fake.calls[-1]["messages"][0]["role"] == "system"
    assert fake.calls[-1]["extra_body"] == {"x": 1}
    assert slept == [1.0]

    fake = _FakeClient(
        [
            _Response(
                content="cache-ok",
                usage=SimpleNamespace(
                    prompt_cache_hit_tokens=12,
                    prompt_cache_miss_tokens=28,
                ),
            )
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    result = await LLMProvider(api_key="k", provider_name="deepseek").chat(
        [], [], "deepseek-v4-flash", 1
    )
    assert result.cache_prompt_tokens == 40
    assert result.cache_hit_tokens == 12

    fake = _FakeClient(
        [
            _Response(
                content="mimo-cache-ok",
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=76),
                ),
            )
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    result = await LLMProvider(api_key="k").chat([], [], "mimo-v2.5", 1)
    assert result.cache_prompt_tokens == 100
    assert result.cache_hit_tokens == 76

    fake = _FakeClient(
        [
            RuntimeError("Error code: 429"),
            _Response(content="retry-ok"),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    slept = []
    monkeypatch.setattr("agent.provider.asyncio.sleep", _sleep)
    result = await LLMProvider(api_key="k", max_retries=1).chat([], [], "m", 1)
    assert result.content == "retry-ok"
    assert slept == [1.0]

    fake = _FakeClient(
        [
            RuntimeError("Error code: 503"),
            RuntimeError("Error code: 503"),
            RuntimeError("Error code: 503"),
            _Response(content="busy-recovered"),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    slept = []
    monkeypatch.setattr("agent.provider.asyncio.sleep", _sleep)
    result = await LLMProvider(api_key="k").chat([], [], "m", 1)
    assert result.content == "busy-recovered"
    assert slept == [1.0, 2.0, 4.0]

    fake = _FakeClient([RuntimeError("content_policy_violation")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(ContentSafetyError):
        await LLMProvider(api_key="k").chat([], [], "m", 1)

    fake = _FakeClient([RuntimeError("maximum context length exceeded")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(ContextLengthError):
        await LLMProvider(api_key="k").chat([], [], "m", 1)

    fake = _FakeClient([RuntimeError("invalid_parameter_error")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(RuntimeError):
        await LLMProvider(api_key="k", max_retries=0).chat([], [], "m", 1)

    fake = _FakeClient([RuntimeError("bad request")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(RuntimeError):
        await LLMProvider(api_key="k", max_retries=0).chat([], [], "m", 1)


@pytest.mark.asyncio
async def test_chat_completions_omits_zero_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="done")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    result = await LLMProvider(api_key="k").chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="m",
        max_tokens=0,
    )

    assert result.content == "done"
    assert "max_tokens" not in fake.calls[0]


@pytest.mark.asyncio
async def test_provider_outer_deadline_cancels_without_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def _blocking_create(**_kwargs):
        nonlocal calls
        calls += 1
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_blocking_create))
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k", max_retries=2)

    task = asyncio.create_task(provider.chat([], [], "m", 1))
    await started.wait()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, timeout=0.01)

    assert cancelled.is_set()
    assert calls == 1


def test_normalize_openai_base_url_trims_endpoint_suffix():
    assert (
        _normalize_openai_base_url("https://pro.nasdw.top:888/v1/chat/completions")
        == "https://pro.nasdw.top:888/v1"
    )
    assert (
        _normalize_openai_base_url("https://example.com/v1/responses")
        == "https://example.com/v1"
    )
    assert _normalize_openai_base_url("https://example.com") == "https://example.com"


@pytest.mark.asyncio
async def test_provider_payload_snapshot_switch_default_off_and_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    snapshot_dir = tmp_path / "payloads"
    last_payload = tmp_path / "last.json"
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(provider_module, "_LAST_PAYLOAD_PATH", last_payload)

    stream = _FakeStream([SimpleNamespace(choices=[])])
    fake = _FakeClient([_Response(content="off"), _Response(content="ok"), stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    provider = LLMProvider(api_key="k")
    await provider.chat(
        messages=[{"role": "user", "content": "off"}],
        tools=[],
        model="m",
        max_tokens=10,
    )

    assert not snapshot_dir.exists()
    assert not last_payload.exists()

    monkeypatch.setattr(provider_module, "_LLM_PAYLOAD_SNAPSHOT_ENABLED", True)
    provider_enabled = LLMProvider(api_key="k")
    await provider_enabled.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="m",
        max_tokens=10,
    )
    await provider_enabled.chat(
        messages=[{"role": "user", "content": "stream"}],
        tools=[],
        model="m",
        max_tokens=10,
        on_content_delta=lambda chunk: _collect_delta([], chunk),
    )

    files = sorted(snapshot_dir.glob("*.json"))
    assert len(files) == 2
    first_payload = json.loads(files[0].read_text(encoding="utf-8"))
    second_payload = json.loads(files[1].read_text(encoding="utf-8"))
    assert first_payload["messages"][0]["content"] == "hi"
    assert second_payload["messages"][0]["content"] == "stream"
    assert second_payload["stream"] is True


@pytest.mark.asyncio
async def test_provider_payload_snapshot_can_enable_per_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    snapshot_dir = tmp_path / "payloads"
    last_payload = tmp_path / "last.json"
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(provider_module, "_LAST_PAYLOAD_PATH", last_payload)
    monkeypatch.setattr(provider_module, "_LLM_PAYLOAD_SNAPSHOT_ENABLED", False)

    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    provider = LLMProvider(api_key="k", payload_snapshot_enabled=True)
    await provider.chat(
        messages=[{"role": "user", "content": "dev"}],
        tools=[],
        model="m",
        max_tokens=10,
    )

    files = sorted(snapshot_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["messages"][0]["content"] == "dev"


def test_provider_payload_snapshot_rotates_by_count_and_reuses_latest_inode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    snapshot_dir = tmp_path / "payloads"
    last_payload = tmp_path / "last.json"
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(provider_module, "_LAST_PAYLOAD_PATH", last_payload)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_FILES", 3)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_BYTES", 1024 * 1024)

    for index in range(5):
        provider_module._save_llm_payload_snapshot(
            {"request": index},
            enabled=True,
        )

    files = sorted(snapshot_dir.glob("*.json"))
    assert len(files) == 3
    assert [json.loads(item.read_text())["request"] for item in files] == [2, 3, 4]
    assert json.loads(last_payload.read_text())["request"] == 4
    assert last_payload.stat().st_ino == files[-1].stat().st_ino


def test_provider_payload_snapshot_rotates_by_total_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    snapshot_dir = tmp_path / "payloads"
    last_payload = tmp_path / "last.json"
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(provider_module, "_LAST_PAYLOAD_PATH", last_payload)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_FILES", 16)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_BYTES", 700)

    for index in range(4):
        provider_module._save_llm_payload_snapshot(
            {"request": index, "content": "x" * 300},
            enabled=True,
        )

    files = sorted(snapshot_dir.glob("*.json"))
    assert len(files) == 2
    assert sum(item.stat().st_size for item in files) <= 700
    assert [json.loads(item.read_text())["request"] for item in files] == [2, 3]


def test_provider_payload_snapshot_rejects_oversize_without_losing_latest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    snapshot_dir = tmp_path / "payloads"
    last_payload = tmp_path / "last.json"
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(provider_module, "_LAST_PAYLOAD_PATH", last_payload)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_BYTES", 256)

    saved = provider_module._save_llm_payload_snapshot({"request": 1}, enabled=True)
    rejected = provider_module._save_llm_payload_snapshot(
        {"content": "x" * 300},
        enabled=True,
    )

    assert saved is not None
    assert rejected is None
    assert json.loads(last_payload.read_text())["request"] == 1
    assert "跳过超限快照" in caplog.text


def test_provider_payload_snapshot_serializes_rotation_and_cleans_stale_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    snapshot_dir = tmp_path / "payloads"
    snapshot_dir.mkdir()
    stale_temp = snapshot_dir / ".interrupted.json.tmp"
    stale_temp.write_text("partial")
    last_payload = tmp_path / "last.json"
    stale_link = tmp_path / ".last.json.123-000001.tmp"
    stale_link.write_text("stale")
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_DIR", snapshot_dir)
    monkeypatch.setattr(provider_module, "_LAST_PAYLOAD_PATH", last_payload)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_FILES", 4)
    monkeypatch.setattr(provider_module, "_PAYLOAD_SNAPSHOT_MAX_BYTES", 1024 * 1024)

    with ThreadPoolExecutor(max_workers=4) as executor:
        saved = list(
            executor.map(
                lambda index: provider_module._save_llm_payload_snapshot(
                    {"request": index},
                    enabled=True,
                ),
                range(12),
            )
        )

    files = sorted(snapshot_dir.glob("*.json"))
    assert all(item is not None for item in saved)
    assert len(files) == 4
    assert not stale_temp.exists()
    assert not stale_link.exists()
    assert last_payload.stat().st_ino in {item.stat().st_ino for item in files}
    assert all(item.stat().st_mode & 0o777 == 0o600 for item in files)


@pytest.mark.asyncio
async def test_provider_chat_stream_parses_content_reasoning_and_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="你", reasoning_content="想", tool_calls=[]
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="好", reasoning_content="法", tool_calls=[]
                        ),
                        finish_reason="stop",
                    )
                ]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_cache_hit_tokens=16,
                    prompt_cache_miss_tokens=48,
                ),
            ),
        ]
    )
    fake = _FakeClient([stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    deltas: list[dict[str, str]] = []
    provider = LLMProvider(api_key="k")
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="m",
        max_tokens=10,
        on_content_delta=lambda chunk: _collect_delta(deltas, chunk),
    )
    assert result.content == "你好"
    assert result.thinking == "想法"
    assert result.finish_reason == "stop"
    content_deltas = [d["content_delta"] for d in deltas if "content_delta" in d]
    thinking_deltas = [d["thinking_delta"] for d in deltas if "thinking_delta" in d]
    assert content_deltas == ["你", "好"]
    assert thinking_deltas == ["想", "法"]
    assert fake.calls[0]["stream"] is True
    assert result.cache_prompt_tokens == 64
    assert result.cache_hit_tokens == 16
    assert stream.closed is True


@pytest.mark.asyncio
async def test_provider_chat_stream_extracts_openai_cached_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="好", tool_calls=[]))
                ]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    prompt_tokens_details={"cached_tokens": 80},
                ),
            ),
        ]
    )
    fake = _FakeClient([stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k")

    result = await provider.chat(
        messages=[],
        tools=[],
        model="mimo-v2.5",
        max_tokens=10,
        on_content_delta=lambda chunk: _collect_delta([], chunk),
    )

    assert result.content == "好"
    assert result.cache_prompt_tokens == 100
    assert result.cache_hit_tokens == 80


@pytest.mark.asyncio
async def test_opencode_go_stream_requests_usage_chunk(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="好", tool_calls=[]))
                ]
            ),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=100,
                    prompt_tokens_details={"cached_tokens": 80},
                ),
            ),
        ]
    )
    fake = _FakeClient([stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k", provider_name="opencode-go")

    result = await provider.chat(
        messages=[],
        tools=[],
        model="kimi-k3",
        max_tokens=10,
        on_content_delta=lambda chunk: _collect_delta([], chunk),
    )

    assert fake.calls[0]["stream_options"] == {"include_usage": True}
    assert result.cache_prompt_tokens == 100
    assert result.cache_hit_tokens == 80


@pytest.mark.asyncio
async def test_opencode_go_endpoint_normalizes_tool_schema_even_when_provider_is_deepseek(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response()])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "lookup",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "default": "all",
                        "minLength": 1,
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1, "maximum": 10},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
    provider = LLMProvider(
        api_key="k",
        base_url="https://opencode.ai/zen/go/v1",
        provider_name="deepseek",
        max_retries=0,
    )

    await provider.chat([], [tool], "deepseek-v4-flash", 10)

    assert provider.max_tool_schemas == 16
    sent_function = fake.calls[0]["tools"][0]["function"]
    assert "strict" not in sent_function
    sent_schema = sent_function["parameters"]
    assert sent_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }


@pytest.mark.asyncio
async def test_official_deepseek_endpoint_keeps_tool_schema_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response()])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "default": 5}
                },
                "additionalProperties": False,
            },
        },
    }
    provider = LLMProvider(
        api_key="k",
        base_url="https://api.deepseek.com/v1",
        provider_name="deepseek",
        max_retries=0,
    )

    await provider.chat([], [tool], "deepseek-chat", 10)

    assert provider.max_tool_schemas == 0
    assert fake.calls[0]["tools"] == [tool]


@pytest.mark.asyncio
async def test_provider_chat_stream_propagates_sdk_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _FakeStream([httpx.ReadTimeout("stream idle")])
    fake = _FakeClient([stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    provider = LLMProvider(api_key="k", read_timeout_s=0.01, max_retries=0)
    with pytest.raises(LLMNetworkTimeoutError, match="流读取网络超时") as exc_info:
        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="m",
            max_tokens=10,
            on_content_delta=lambda chunk: _collect_delta([], chunk),
        )
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)
    assert stream.closed is True


@pytest.mark.asyncio
async def test_provider_chat_stream_retries_transport_error_before_first_delta(
    monkeypatch: pytest.MonkeyPatch,
):
    interrupted = _FakeStream(
        [
            httpx.RemoteProtocolError(
                "peer closed connection without sending complete message body"
            )
        ]
    )
    recovered = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="完成", tool_calls=[]),
                        finish_reason="stop",
                    )
                ]
            )
        ]
    )
    fake = _FakeClient([interrupted, recovered])
    sleep = AsyncMock()
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)

    result = await LLMProvider(api_key="k", max_retries=1).chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="m",
        max_tokens=10,
        on_content_delta=lambda chunk: _collect_delta([], chunk),
    )

    assert result.content == "完成"
    assert result.finish_reason == "stop"
    assert len(fake.calls) == 2
    sleep.assert_awaited_once_with(1.0)
    assert interrupted.closed is True
    assert recovered.closed is True


@pytest.mark.asyncio
async def test_provider_chat_stream_does_not_retry_after_response_delta(
    monkeypatch: pytest.MonkeyPatch,
):
    error = httpx.RemoteProtocolError("incomplete chunked read")
    interrupted = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="部分", tool_calls=[])
                    )
                ]
            ),
            error,
        ]
    )
    unused = _FakeStream([])
    fake = _FakeClient([interrupted, unused])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    deltas: list[dict[str, str]] = []

    with pytest.raises(httpx.RemoteProtocolError, match="incomplete chunked read"):
        await LLMProvider(api_key="k", max_retries=1).chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="m",
            max_tokens=10,
            on_content_delta=lambda chunk: _collect_delta(deltas, chunk),
        )

    assert deltas == [{"content_delta": "部分"}]
    assert len(fake.calls) == 1
    assert interrupted.closed is True
    assert unused.closed is False


@pytest.mark.asyncio
async def test_provider_rejects_non_object_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient(
        [
            _Response(
                content="",
                tool_calls=[
                    SimpleNamespace(
                        id="1",
                        function=SimpleNamespace(name="search", arguments="[]"),
                    )
                ],
            )
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    with pytest.raises(TypeError, match="JSON 对象"):
        await LLMProvider(api_key="k").chat([], [], "m", 1)


@pytest.mark.asyncio
async def test_provider_stream_rejects_non_object_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="1",
                                    function=SimpleNamespace(
                                        name="search", arguments="[]"
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        ]
    )
    fake = _FakeClient([stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    with pytest.raises(TypeError, match="JSON 对象"):
        await LLMProvider(api_key="k").chat(
            [], [], "m", 1, on_content_delta=lambda chunk: _collect_delta([], chunk)
        )
    assert stream.closed is True


@pytest.mark.asyncio
async def test_provider_stream_closes_when_delta_callback_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="boom", tool_calls=[])
                    )
                ]
            )
        ]
    )
    fake = _FakeClient([stream])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)

    async def _raise_callback(_chunk: dict[str, str]) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        await LLMProvider(api_key="k").chat(
            [], [], "m", 1, on_content_delta=_raise_callback
        )
    assert stream.closed is True


def test_bootstrap_providers_set_network_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    created: list[dict] = []

    class _ProviderConfig:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

    monkeypatch.setattr("bootstrap.providers.LLMProvider", _ProviderConfig)
    cfg = ConfigModel(
        model="main",
        api_key="main-key",
        base_url="https://example.com/v1",
        system_prompt="system",
        extra_body={},
        provider="openai",
        dev_mode=False,
        light_model="light",
        light_api_key="light-key",
        light_base_url="https://light.example.com/v1",
        agent_model="agent",
        agent_api_key="agent-key",
        agent_base_url="https://agent.example.com/v1",
        multimodal=False,
        vl_model="vl",
        vl_api_key="vl-key",
        vl_base_url="https://vl.example.com/v1",
    )

    build_providers(cfg)
    build_vl_provider(cfg)

    assert [item["read_timeout_s"] for item in created] == [
        120.0,
        60.0,
        120.0,
        120.0,
    ]


@pytest.mark.asyncio
async def test_deepseek_strategy_maps_thinking_config(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(
        api_key="k",
        provider_name="deepseek",
        extra_body={"enable_thinking": True, "reasoning_effort": "xhigh"},
    )

    await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    assert fake.calls[-1]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert fake.calls[-1]["reasoning_effort"] == "max"


@pytest.mark.asyncio
async def test_deepseek_strategy_disables_thinking(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(
        api_key="k",
        provider_name="deepseek",
        extra_body={
            "enable_thinking": True,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        },
    )

    await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
        disable_thinking=True,
    )

    assert fake.calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in fake.calls[-1]


@pytest.mark.asyncio
async def test_deepseek_named_tool_choice_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(
        api_key="k",
        provider_name="deepseek",
        extra_body={"enable_thinking": True, "reasoning_effort": "high"},
    )

    await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "probe"}}],
        model="deepseek-v4-pro",
        max_tokens=10,
        tool_choice={"type": "function", "function": {"name": "probe"}},
    )

    assert fake.calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in fake.calls[-1]


@pytest.mark.asyncio
async def test_token_plan_strategy_disables_thinking(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(
        api_key="k",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        force_disable_thinking=True,
    )

    await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="mimo-v2.5",
        max_tokens=10,
    )

    assert fake.calls[-1]["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_deepseek_strategy_strips_image_url_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k", provider_name="deepseek")

    await provider.chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                    {"type": "text", "text": "看看这张图"},
                ],
            }
        ],
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    content = fake.calls[-1]["messages"][0]["content"]
    assert isinstance(content, str)
    assert "看看这张图" in content
    assert "image_url" in content


@pytest.mark.asyncio
async def test_deepseek_tool_call_round_trips_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient(
        [
            _Response(
                content="",
                tool_calls=[_ToolCall("1", "search", {"q": "x"})],
                reasoning_content="先查资料",
            )
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k", provider_name="deepseek")

    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    messages: list[dict] = []
    append_assistant_tool_calls(
        messages,
        content=result.content,
        tool_calls=result.tool_calls,
        provider_fields=result.provider_fields,
    )

    assert result.thinking == "先查资料"
    assert messages[0]["reasoning_content"] == "先查资料"


@pytest.mark.asyncio
async def test_deepseek_explicit_thinking_patches_dirty_history(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(
        api_key="k",
        provider_name="deepseek",
        extra_body={"enable_thinking": True},
    )

    await provider.chat(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "again"},
        ],
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    assert fake.calls[-1]["messages"][1]["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_deepseek_default_thinking_patches_only_missing_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k", provider_name="deepseek")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "content": "tool reply",
            "reasoning_content": "已有推理",
        },
        {"role": "user", "content": "continue"},
    ]

    await provider.chat(
        messages=messages,
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    sent = fake.calls[-1]["messages"]
    assert sent[1]["reasoning_content"] == ""
    assert sent[3]["reasoning_content"] == "已有推理"
    assert "reasoning_content" not in messages[1]


@pytest.mark.asyncio
async def test_deepseek_explicit_disabled_thinking_keeps_history_unpatched(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(
        api_key="k",
        provider_name="deepseek",
        extra_body={"enable_thinking": False},
    )

    await provider.chat(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "again"},
        ],
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    sent = fake.calls[-1]
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_content" not in sent["messages"][1]


@pytest.mark.asyncio
async def test_deepseek_default_thinking_patches_session_tool_chain_replay(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeClient([_Response(content="ok")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    provider = LLMProvider(api_key="k", provider_name="deepseek")
    session = Session("cli:deepseek-tool-replay")
    session.add_message("user", "完成长任务")
    session.add_message(
        "assistant",
        "已完成",
        tool_chain=[
            {
                "text": "",
                "calls": [
                    {
                        "call_id": "call-1",
                        "name": "probe",
                        "arguments": {},
                        "result": "ok",
                    }
                ],
            }
        ],
    )
    history = session.get_history()

    await provider.chat(
        messages=history,
        tools=[],
        model="deepseek-v4-pro",
        max_tokens=10,
    )

    sent_assistant = fake.calls[-1]["messages"][1]
    assert sent_assistant["tool_calls"][0]["function"]["name"] == "probe"
    assert sent_assistant["reasoning_content"] == ""
    assert "reasoning_content" not in history[1]


async def _collect_delta(bucket: list, chunk) -> None:
    bucket.append(chunk)


@pytest.mark.asyncio
async def test_app_runtime_start_passes_markdown_store_to_memory_optimizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    engine = MagicMock(name="engine")
    markdown_store = MagicMock(name="markdown_store")
    memory_runtime = SimpleNamespace(
        engine=engine,
        markdown=SimpleNamespace(store=markdown_store),
        aclose=AsyncMock(),
    )
    startup_order: list[str] = []
    plugin_manager = MagicMock()
    plugin_manager.bind_core_channel_definitions = AsyncMock(
        side_effect=lambda definitions: startup_order.append("bindings")
    )
    plugin_manager.run_runtime_services = AsyncMock()
    core = SimpleNamespace(
        loop=SimpleNamespace(
            run=lambda: "loop-task",
            bind_plugin_rollout_fact_provider=MagicMock(),
        ),
        bus=SimpleNamespace(dispatch_outbound=lambda: "bus-task"),
        event_bus=EventBus(),
        tools=MagicMock(),
        push_tool=MagicMock(),
        session_manager=MagicMock(),
        provider=MagicMock(),
        light_provider=MagicMock(),
        memory_runtime=memory_runtime,
        channel_attachment_store=MagicMock(),
        presence=MagicMock(),
        plugin_manager=plugin_manager,
        bind_conversation_runtime=MagicMock(),
        start=AsyncMock(),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(
        "bootstrap.app.build_core_runtime", lambda *args, **kwargs: core
    )
    channel_host = SimpleNamespace(
        start_all=AsyncMock(side_effect=lambda: startup_order.append("providers")),
        stop_all=AsyncMock(),
        bind_plugin_channels=MagicMock(),
        swap_plugin_channels=AsyncMock(),
        channels=(),
    )
    monkeypatch.setattr(
        "bootstrap.app.start_channels",
        AsyncMock(return_value=channel_host),
    )
    memory_optimizer = MagicMock()
    build_memory_optimizer_task = MagicMock(return_value=([], memory_optimizer))
    monkeypatch.setattr(
        "bootstrap.app.build_memory_optimizer_task", build_memory_optimizer_task
    )
    monkeypatch.setattr(
        "bootstrap.app.build_dashboard_server",
        lambda **kwargs: SimpleNamespace(
            should_exit=False,
            serve=AsyncMock(return_value=None),
            manual_memory_optimizer=kwargs["manual_memory_optimizer"],
        ),
    )

    app = AppRuntime(
        config=cast(
            Any,
            SimpleNamespace(
                app_server=SimpleNamespace(enabled=False),
                channels=SimpleNamespace(chat=SimpleNamespace(enabled=False)),
                mobile_realtime=SimpleNamespace(enabled=False),
            ),
        ),
        workspace=tmp_path,
    )
    await app.start()

    build_memory_optimizer_task.assert_called_once()
    assert (
        build_memory_optimizer_task.call_args.kwargs["memory_store"] is markdown_store
    )
    assert app.dashboard_server.manual_memory_optimizer is memory_optimizer
    assert startup_order == ["bindings", "providers"]
    await app.shutdown()


@pytest.mark.asyncio
async def test_group_filter_paths() -> None:
    group = SimpleNamespace(group_id="1", allow_from=["42"], require_at=True)
    event = SimpleNamespace(user_id="42", raw_message="[CQ:at,qq=10001] hi")

    assert (
        await DefaultGroupFilter("10001").should_process(event, cast(Any, group))
        is True
    )
    assert strip_at_segments("x [CQ:at,qq=10001] y") == "x  y".strip()

    bad_user = SimpleNamespace(user_id="9", raw_message="hi")
    assert (
        await DefaultGroupFilter("10001").should_process(bad_user, cast(Any, group))
        is False
    )


@pytest.mark.asyncio
async def test_bootstrap_trigger_and_entrypoints_cover_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    from agent.migrations import MigrationOutcome

    supervisor_calls: list[tuple[Path, Path]] = []

    def _fake_supervisor(
        *,
        config_path: Path,
        workspace: Path,
        readiness_timeout_s: float = 15.0,
    ) -> int:
        supervisor_calls.append((config_path, workspace))
        return 0

    def _fake_migration(config_path: Path, workspace: Path) -> MigrationOutcome:
        return MigrationOutcome(state="current")

    monkeypatch.setattr("agent.supervisor.run_supervisor", _fake_supervisor)
    monkeypatch.setattr("agent.migrations.migrate_installation", _fake_migration)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--config", "missing.json", "--workspace", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")
    assert exc.value.code == 0
    assert supervisor_calls == [(Path("missing.json"), tmp_path)]

    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    supervisor_calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--workspace", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")
    assert exc.value.code == 0
    assert supervisor_calls == [(Path("config.toml"), tmp_path)]
