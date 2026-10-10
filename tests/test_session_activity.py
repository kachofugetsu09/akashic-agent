from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.reply.status import ReplyState


class _StubTask:
    """ReplyState.open 只依赖 handle、active 与 on_close 三个成员。"""

    def __init__(self, handle: str) -> None:
        self.handle = handle
        self.active = True
        self._closers: list[Callable[[], None]] = []

    def on_close(self, callback: Callable[[], None]) -> None:
        self._closers.append(callback)

    def revoke(self) -> None:
        self.active = False
        for callback in self._closers:
            callback()


def _open(state: ReplyState, task: _StubTask, session_id: str):
    return state.open(task, session_id, "user")  # pyright: ignore[reportArgumentType]


def test_active_sessions_tracks_each_running_reply() -> None:
    state = ReplyState()
    first, second = _StubTask("a"), _StubTask("b")
    assert state.read.active_sessions() == frozenset()
    with _open(state, first, "akashic:one"), _open(state, second, "akashic:two"):
        assert state.read.active_sessions() == {"akashic:one", "akashic:two"}
    assert state.read.active_sessions() == frozenset()


def test_revoked_reply_is_not_running_while_it_drains() -> None:
    state = ReplyState()
    task = _StubTask("a")
    with _open(state, task, "akashic:one"):
        task.revoke()
        assert state.read.active_sessions() == frozenset()


def test_one_session_with_two_replies_is_listed_once() -> None:
    state = ReplyState()
    first, second = _StubTask("a"), _StubTask("b")
    with _open(state, first, "akashic:one"), _open(state, second, "akashic:one"):
        assert state.read.active_sessions() == {"akashic:one"}


def _client(tmp_path: Path, reader: Callable[[], object] | None) -> TestClient:
    async def active() -> frozenset[str] | None:
        return reader() if reader is not None else None  # pyright: ignore[reportReturnType]

    app = create_chat_app(
        workspace=tmp_path, channel=WebChatChannel("akashic"),
        active_sessions=active if reader is not None else None,
    )
    return TestClient(app)


def test_activity_route_lists_only_this_channel_sorted(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda: frozenset({"akashic:b", "akashic:a", "telegram:9"}))
    response = client.get("/api/chat/sessions/activity")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"version": 1, "available": True, "active": ["akashic:a", "akashic:b"]}


@pytest.mark.parametrize("reader", [None, lambda: None])
def test_activity_route_reports_unavailable_without_reply_plugin(
    tmp_path: Path, reader: Callable[[], object] | None,
) -> None:
    response = _client(tmp_path, reader).get("/api/chat/sessions/activity")
    assert response.json() == {"version": 1, "available": False, "active": []}
