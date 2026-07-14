from __future__ import annotations

import asyncio

from docker.debug.shell_background_probe import _run_turn


class _Handle:
    async def result(self) -> dict[str, object]:
        return {"finalResponse": "job complete"}


class _Client:
    def __init__(self) -> None:
        self.request: tuple[str, str] | None = None

    async def start_turn(self, thread_id: str, text: str) -> _Handle:
        self.request = (thread_id, text)
        return _Handle()


def test_shell_background_probe_uses_formal_turn_rpc() -> None:
    client = _Client()

    response = asyncio.run(_run_turn(client, "thread-2", "run job", 5.0))  # type: ignore[arg-type]

    assert response == {"content": "job complete", "metadata": {}}
    assert client.request == ("thread-2", "run job")
