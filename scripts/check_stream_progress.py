"""用合成 SSE 验证真实 driver 的进展期限，不访问模型或正式 workspace。"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

sys.path.insert(0, str(Path.cwd()))

from plugins.models.contract import ModelRequest
from plugins.models.contract import (
    BoundModelDescriptor,
    CapabilitySources,
    ModelCapabilities,
)
from plugins.models.contract import (
    ModelTimeoutError,
    ModelUnavailableError,
    ModelError,
)
from core.net.http import HttpClient
from plugins.codex import responses as codex
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.openai_compatible import driver as compatible
from plugins.opencode_go import driver as opencode


class Stream(httpx.AsyncByteStream):
    """以独立于 parser 的节奏发送 SSE，记录取消与关闭。"""

    def __init__(self, events: list[tuple[float, str]]) -> None:
        self.events = events
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for delay, event in self.events:
            await asyncio.sleep(delay)
            yield (event + "\n\n").encode()

    async def aclose(self) -> None:
        self.closed = True


class HeldStream(Stream):
    """首个增量已由 parser 消费后才通知取消，不依赖 sleep。"""

    def __init__(self, prefix: str) -> None:
        super().__init__([])
        self.prefix = prefix
        self.consumed = asyncio.Event()
        self.release = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield (self.prefix + "\n\n").encode()
        self.consumed.set()
        await self.release.wait()


def data(value: object) -> str:
    return "data: " + json.dumps(value)


def delta(driver: str, kind: str, text: str) -> str:
    if driver == "codex":
        name = {"content": "output_text", "reasoning_content": "reasoning_text",
                "arguments": "function_call_arguments"}[kind]
        return data({"type": f"response.{name}.delta", "delta": text, "item_id": "tool"})
    value = ({"tool_calls": [{"index": 0, "function": {"arguments": text}}]}
             if kind == "arguments" else {kind: text})
    return data({"choices": [{"delta": value}]})


async def consume(driver: str, stream: Stream, *, slow_callback: bool = False):
    """同一夹具也可在修复前源码运行，以证明失败而非只验证新实现。"""
    module = {"compatible": compatible, "opencode": opencode, "codex": codex}[driver]
    kwargs = ({"progress_timeout": 0.1}
              if "progress_timeout" in inspect.signature(module._consume_stream).parameters else {})

    async def receive(_value: dict[str, str]) -> None:
        if slow_callback:
            await asyncio.sleep(0.15)

    response = httpx.Response(200, stream=stream)
    try:
        if driver == "codex":
            return await codex._consume_stream(response, ModelRequest([], on_delta=receive), "binding", (), **kwargs)
        return await module._consume_stream(response, receive, **kwargs)
    finally:
        await response.aclose()


class LocalCredential:
    """仅供内存 transport 使用的凭据，禁止触发外部刷新。"""

    connection_id = "scenario"
    auth_identity = "scenario"

    async def read(self) -> Mapping[str, str]:
        return {"driver": "codex", "access_token": "scenario", "account_id": "scenario",
                "expires_at": "2099-01-01T00:00:00+00:00"}

    async def refresh(self, payload: Mapping[str, str]) -> None:
        raise AssertionError("scenario must not refresh credentials")

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        yield


async def check_codex_receipt() -> None:
    """穿过完整 driver 和 Models，核对单次预算耗尽后的部分响应与禁止隐式重发。"""
    descriptor = BoundModelDescriptor(
        binding_id="scenario", plugin_snapshot_id="scenario", model_revision=0,
        model_id="scenario", connection_id="scenario", driver_id="codex",
        driver_contract_version="scenario", auth_identity="scenario", model="scenario",
        role="default", reasoning_effort=None, capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(), capability_digest="scenario",
    )
    stream = Stream([(0, delta("codex", "content", "partial"))]
                    + [(0.01, ": heartbeat")] * 30)
    sent = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(200, stream=stream)

    http = HttpClient(lambda: httpx.AsyncClient(
        base_url="https://scenario.invalid", transport=httpx.MockTransport(respond),
    ))
    driver = codex.CodexResponses(http=http, credential=LocalCredential(),
                                 descriptor=descriptor, config={}, progress_timeout=0.1)
    with TemporaryDirectory(prefix="stream-progress-") as directory:
        root = Path(directory)
        store = ModelsStore(root / "models.db", root / "backups")
        store.initialize()
        try:
            bound = _BoundChat(descriptor, driver, store, max_attempts=1)
            request = ModelRequest([], request_key="stream-progress")
            try:
                await bound.complete(request)
            except (RuntimeError, TimeoutError) as error:
                if not (ModelError.matches(error, ModelTimeoutError)):
                    raise
                assert (value := ModelError.read(error)) is not None and value.response_delta_seen
                assert "没有有效进展" in str(error)
            else:
                raise AssertionError("partial stream did not time out")
            records = store.calls_for_key("stream-progress")
            assert len(records) == 1
            assert records[0]["partial_response"]
            assert records[0]["send_evidence"] is None
            assert records[0]["next_attempt_at"] is None
            try:
                await bound.complete(request)
            except (RuntimeError, TimeoutError) as _model_error:
                if not (ModelError.matches(_model_error, ModelUnavailableError)):
                    raise
                pass
            else:
                raise AssertionError("uncertain request was sent again")
            assert sent == 1 and stream.closed
        finally:
            await http.aclose()
            store.close()


async def main() -> None:
    """验证停滞、合法进展、慢回调和取消四类可观察行为。"""
    checked = 0
    for driver in ("compatible", "opencode", "codex"):
        terminal = data({"type": "response.completed", "response": {}}) if driver == "codex" else "data: [DONE]"
        # 1. 心跳、空内容和用量事件都不能无限延长一次请求。
        for prefix, idle in (([], ": heartbeat"),
                             ([(0, delta(driver, "content", "partial"))], ": heartbeat"),
                             ([], delta(driver, "content", "")),
                             ([], data({"usage": {}, "choices": []}))):
            stream = Stream(prefix + [(0.01, idle)] * 30 + [(0, terminal)])
            try:
                await consume(driver, stream)
            except Exception as error:
                failure = getattr(error, "error", error)
                value = ModelError.read(failure, ModelTimeoutError)
                assert value is not None, repr(error)
                assert "没有有效进展" in str(failure)
                assert value.send_evidence is None
                if driver != "codex":
                    module = compatible if driver == "compatible" else opencode
                    assert not module._retryable(module._map_error(error))
            else:
                raise AssertionError(f"{driver}: idle stream was accepted")
            assert stream.closed
            checked += 1

        # 2. 合法正文、推理、工具参数增量可以超过总时长预算。
        for kind in ("content", "reasoning_content", "arguments"):
            events = [(0.04, delta(driver, kind, " " if kind == "arguments" else "x"))] * 6
            if kind == "arguments":
                if driver == "codex":
                    events.append((0, data({"type": "response.output_item.done", "item": {
                        "type": "function_call", "id": "tool", "call_id": "call", "name": "run", "arguments": "{}"}})))
                else:
                    events.append((0, data({"choices": [{"delta": {"tool_calls": [{"index": 0,
                        "id": "call", "function": {"name": "run", "arguments": "{}"}}]}}]})))
            stream = Stream(events + [(0, terminal)])
            result = await consume(driver, stream)
            if kind == "arguments":
                assert len(result.tool_calls) == 1
            else:
                assert (result.content if kind == "content" else result.thinking) == "xxxxxx"
            assert stream.closed
            checked += 1

        # 3. 下游慢消费者不应被误报成上游停滞；取消仍传递。
        stream = Stream([(0, delta(driver, "content", "x")), (0, terminal)])
        await consume(driver, stream, slow_callback=True)
        checked += 1
        stream = Stream([(0.01, ": heartbeat")] * 100)
        task = asyncio.create_task(consume(driver, stream))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("caller cancellation was swallowed")
        assert stream.closed
        checked += 1
    # 4. 完成型内容也续期；相同完成项的重复投递不能伪装成进展。
    terminal = data({"type": "response.completed", "response": {}})
    for item in (
        {"type": "function_call", "id": "tool", "call_id": "call", "name": "run", "arguments": "{}"},
        {"type": "reasoning", "encrypted_content": "opaque-reasoning"},
    ):
        done = data({"type": "response.output_item.done", "item": item})
        prefix = [(0, delta("codex", "arguments", ""))] if item["type"] == "function_call" else []
        await consume("codex", Stream(prefix + [(0.06, done), (0.06, terminal)]))
        checked += 1
        try:
            await consume("codex", Stream([(0.04, done)] * 6 + [(0, terminal)]))
        except (RuntimeError, TimeoutError) as _model_error:
            if not (ModelError.matches(_model_error, ModelTimeoutError)):
                raise
            pass
        else:
            raise AssertionError("duplicate completed items extended progress deadline")
        checked += 1
    await check_codex_receipt()
    checked += 1
    # 5. 取消保留实际进展，纯心跳取消不能伪造已收到部分响应。
    for driver in ("compatible", "opencode", "codex"):
        for kind in ("content", "reasoning_content", "arguments", None):
            prefix = ": heartbeat" if kind is None else delta(driver, kind, "x")
            stream = HeldStream(prefix)
            task = asyncio.create_task(consume(driver, stream))
            await asyncio.wait_for(stream.consumed.wait(), 1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as error:
                assert bool(getattr(error, "response_delta_seen", False)) == (kind is not None), (
                    driver, kind, "取消丢失或伪造已接收响应的事实"
                )
            else:
                raise AssertionError("caller cancellation was swallowed")
            assert stream.closed
            checked += 1
    print(f"passed: {checked} driver scenarios; no live provider or workspace access")


if __name__ == "__main__":
    asyncio.run(main())
