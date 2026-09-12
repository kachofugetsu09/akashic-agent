"""验证 Akashic 客户端监听器的真实就绪和失败传播边界。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from plugins.akashic_clients.channel import (
    _GenerationAkashicAdapter,
    _SERVER_START_TIMEOUT_SECONDS,
)


class _FakeServer:
    def __init__(self, *, start: bool = True, failure: BaseException | None = None) -> None:
        self.started = False
        self.should_exit = False
        self._start = start
        self._failure = failure
        self.stopped = asyncio.Event()

    async def serve(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._start:
            self.started = True
        await self.stopped.wait()


def _adapter() -> _GenerationAkashicAdapter:
    async def spawn_owned(coroutine, *, name):
        return asyncio.create_task(coroutine, name=name)

    adapter = object.__new__(_GenerationAkashicAdapter)
    adapter._context = SimpleNamespace(spawn_owned=spawn_owned)
    adapter._servers = []
    return adapter


@pytest.mark.asyncio
async def test_start_server_returns_only_after_listener_is_ready() -> None:
    adapter = _adapter()
    server = _FakeServer()

    await adapter._start_server(server, name="test-web")

    assert server.started is True
    assert len(adapter._servers) == 1
    assert adapter._servers[0][0] is server
    server.should_exit = True
    server.stopped.set()
    await adapter._servers[0][1]


@pytest.mark.asyncio
async def test_start_server_propagates_failure_before_listener_ready() -> None:
    adapter = _adapter()
    failure = OSError("address already in use")
    server = _FakeServer(failure=failure)

    with pytest.raises(OSError, match="address already in use") as raised:
        await adapter._start_server(server, name="test-web")

    assert raised.value is failure
    assert server.should_exit is True
    assert adapter._servers == []


@pytest.mark.asyncio
async def test_start_server_cancels_listener_when_ready_timeout_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    server = _FakeServer(start=False)
    monkeypatch.setattr(
        "plugins.akashic_clients.channel._SERVER_START_TIMEOUT_SECONDS",
        min(_SERVER_START_TIMEOUT_SECONDS, 0.01),
    )

    with pytest.raises(TimeoutError):
        await adapter._start_server(server, name="test-web")

    assert server.should_exit is True
    assert adapter._servers == []
