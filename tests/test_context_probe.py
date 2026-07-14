from __future__ import annotations

import asyncio

import pytest

from docker.debug.context_probe import _ensure_successful_reply, _send_and_read


class _Handle:
    async def result(self) -> dict[str, object]:
        return {"finalResponse": "formal reply"}


class _Client:
    def __init__(self) -> None:
        self.request: tuple[str, str] | None = None

    async def start_turn(self, thread_id: str, text: str) -> _Handle:
        self.request = (thread_id, text)
        return _Handle()


def test_context_probe_rejects_runtime_failure_reply() -> None:
    with pytest.raises(RuntimeError, match="turn 3 返回运行时失败回复"):
        _ensure_successful_reply("处理消息时出错，请稍后再试。", 3)


def test_context_probe_accepts_normal_reply() -> None:
    _ensure_successful_reply("记住了，你喝茶不加糖。", 1)


def test_context_probe_uses_formal_turn_rpc() -> None:
    client = _Client()

    reply = asyncio.run(_send_and_read(client, "thread-1", "hello", 3))  # type: ignore[arg-type]

    assert reply == "formal reply"
    assert client.request == ("thread-1", "hello")
