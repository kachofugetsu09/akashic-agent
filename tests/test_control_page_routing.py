from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Mapping, cast

import pytest

from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService


@pytest.mark.asyncio
async def test_only_message_read_uses_explicit_page_sender() -> None:
    regular: list[dict[str, object]] = []
    pages: list[tuple[dict[str, object], Mapping[str, object]]] = []
    page: dict[str, object] = {
        "version": 2,
        "session_id": "session:a",
        "items": [],
        "after_seq": -1,
        "through_seq": -1,
        "next_after_seq": -1,
        "has_more": False,
    }
    # Router only consumes this narrow service surface in the page-routing fixture.
    service = cast(ControlService, SimpleNamespace(
        methods={},
        initialize=lambda _params: {"ok": True},
        status=lambda: {"ready": True},
        read_messages=lambda *_args: page,
    ))

    async def send(frame: dict[str, object]) -> None:
        regular.append(frame)

    async def send_page(frame: dict[str, object], value: Mapping[str, object]) -> None:
        pages.append((frame, value))

    router = ConnectionRouter(service, send, send_message_page=send_page)
    await router.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2.0",
            "clientInfo": {"name": "test", "version": "1"},
        },
    }).encode())
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')
    await router.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 2, "method": "message/read",
        "params": {"session_id": "session:a"},
    }).encode())
    await router.handle_line(json.dumps({
        "jsonrpc": "2.0", "id": 3, "method": "server/status",
        "params": {},
    }).encode())

    assert len(pages) == 1
    assert pages[0][0]["id"] == 2
    assert pages[0][1] is page
    assert [frame["id"] for frame in regular] == [1, 3]
