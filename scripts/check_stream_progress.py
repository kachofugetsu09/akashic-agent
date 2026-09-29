"""用合成 SSE 验证真实 driver 的进展期限，不访问模型或正式 workspace。"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

sys.path.insert(0, str(Path.cwd()))

from agent.plugin_composition import ModelRequest, ModelTimeoutError
from plugins.codex import responses as codex
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
                assert isinstance(failure, ModelTimeoutError), repr(error)
                assert "没有有效进展" in str(failure)
                assert getattr(failure, "send_evidence", None) is None
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
    print(f"passed: {checked} driver scenarios; no live provider or workspace access")


if __name__ == "__main__":
    asyncio.run(main())
