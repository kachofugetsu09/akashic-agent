import asyncio
from collections.abc import AsyncGenerator, Callable
from types import MappingProxyType

import pytest

from plugins.akashic_clients.session_activity import follow_session_activity
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.reply.status import ReplyState
from plugins.ledger.log import MessageLog, SessionAttributes
from plugins.ledger.contract import Input
from tests.test_message_log import text_schema


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


# ---- ReplyState：活动集合 ----

def test_active_sessions_tracks_each_running_reply() -> None:
    state = ReplyState()
    first, second = _StubTask("a"), _StubTask("b")
    assert state.active_sessions() == frozenset()
    with _open(state, first, "akashic:one"), _open(state, second, "akashic:two"):
        assert state.active_sessions() == {"akashic:one", "akashic:two"}
    assert state.active_sessions() == frozenset()


def test_revoked_reply_is_not_running_while_it_drains() -> None:
    state = ReplyState()
    task = _StubTask("a")
    with _open(state, task, "akashic:one"):
        task.revoke()
        assert state.active_sessions() == frozenset()


def test_one_session_with_two_replies_is_listed_once() -> None:
    state = ReplyState()
    first, second = _StubTask("a"), _StubTask("b")
    with _open(state, first, "akashic:one"), _open(state, second, "akashic:one"):
        assert state.active_sessions() == {"akashic:one"}


@pytest.mark.asyncio
async def test_follow_active_sessions_is_not_woken_by_draft_tokens() -> None:
    state = ReplyState()
    frames: list[frozenset[str]] = []

    async def consume() -> None:
        async for value in state.read.follow_active_sessions():
            frames.append(value)

    follower = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    with _open(state, _StubTask("h"), "akashic:a") as preview:
        await asyncio.sleep(0.01)
        with preview("m1") as delta:
            for _ in range(100):
                await delta({"content_delta": "x"})
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.01)
    follower.cancel()
    assert frames == [frozenset(), frozenset({"akashic:a"}), frozenset()]


# ---- follow_session_activity：水位与活动合并 ----

async def _queued(queue: asyncio.Queue):
    while True:
        yield await queue.get()


def _heads(**values: int):
    return MappingProxyType({key.replace("__", ":"): seq for key, seq in values.items()})


@pytest.mark.asyncio
async def test_activity_publishes_snapshot_then_only_changes() -> None:
    heads: asyncio.Queue = asyncio.Queue()
    active: asyncio.Queue = asyncio.Queue()
    active.put_nowait(frozenset())
    frames = follow_session_activity(_queued(heads), _queued(active), prefix="akashic:")

    async def next_frame():
        return await asyncio.wait_for(anext(frames), 1)

    heads.put_nowait(_heads(akashic__a=1))
    first = await next_frame()
    heads.put_nowait(_heads(akashic__a=1, akashic__b=0))
    second = await next_frame()
    heads.put_nowait(_heads(akashic__a=1, akashic__b=0))  # 完全相同：不应产生帧
    await asyncio.sleep(0.05)
    heads.put_nowait(_heads(akashic__b=0))
    third = await next_frame()
    active.put_nowait(frozenset({"akashic:b"}))
    fourth = await next_frame()
    await frames.aclose()
    assert first["snapshot"] and first["heads"] == {"akashic:a": 1}
    assert not second["snapshot"] and second["heads"] == {"akashic:b": 0}
    assert third["removed"] == ["akashic:a"] and third["heads"] == {}
    assert fourth["active"] == ["akashic:b"] and fourth["heads"] == {}


@pytest.mark.asyncio
async def test_activity_reports_unavailable_without_reply_plugin() -> None:
    heads: asyncio.Queue = asyncio.Queue()
    active: asyncio.Queue = asyncio.Queue()
    heads.put_nowait(_heads(akashic__a=1))
    active.put_nowait(None)
    frames = follow_session_activity(_queued(heads), _queued(active), prefix="akashic:")
    frame = await asyncio.wait_for(anext(frames), 1)
    await frames.aclose()
    assert frame["available"] is False and frame["active"] == []


@pytest.mark.asyncio
async def test_activity_ignores_running_sessions_outside_the_catalog() -> None:
    heads: asyncio.Queue = asyncio.Queue()
    active: asyncio.Queue = asyncio.Queue()
    heads.put_nowait(_heads(akashic__a=1))
    active.put_nowait(frozenset({"akashic:a", "akashic:ghost", "telegram:1"}))
    frames = follow_session_activity(_queued(heads), _queued(active), prefix="akashic:")
    frame = await asyncio.wait_for(anext(frames), 1)
    await frames.aclose()
    assert frame["active"] == ["akashic:a"]


# ---- 订阅失败不拖垮聊天连接 ----

class _FakeSocket:
    def __init__(self, fail_after: int | None = None) -> None:
        self.sent: list[dict[str, object]] = []
        self._fail_after = fail_after

    async def send_json(self, value: dict[str, object]) -> None:
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            raise RuntimeError("socket closed")
        self.sent.append(value)


@pytest.mark.asyncio
async def test_failed_activity_feed_stops_quietly_and_clears_running() -> None:
    channel = WebChatChannel("akashic")

    async def broken() -> AsyncGenerator[dict[str, object], None]:
        yield {"version": 1, "snapshot": True, "available": True, "active": ["akashic:a"],
               "heads": {"akashic:a": 1}, "removed": []}
        raise RuntimeError("会话日志订阅已结束，需要重新连接")

    channel.bind_session_activity(broken)
    socket = _FakeSocket()
    await channel._send_session_activity(socket)  # pyright: ignore[reportArgumentType,reportPrivateUsage]
    assert len(socket.sent) == 2 and socket.sent[0]["active"] == ["akashic:a"]
    assert socket.sent[1]["available"] is False and socket.sent[1]["active"] == []


@pytest.mark.asyncio
async def test_activity_feed_survives_a_closed_socket() -> None:
    channel = WebChatChannel("akashic")

    async def one() -> AsyncGenerator[dict[str, object], None]:
        yield {"version": 1, "snapshot": True, "available": True, "active": [], "heads": {}, "removed": []}

    channel.bind_session_activity(one)
    await channel._send_session_activity(_FakeSocket(fail_after=0))  # pyright: ignore[reportArgumentType,reportPrivateUsage]


# ---- 目录水位：过滤、软删与缓存失效 ----

def _append(log: MessageLog, session: str, message_id: str) -> None:
    log.writer(session, author="user", source="conversation", body_types=(Input,),
               content={"text": text_schema}).append(message_id, Input(()))


def test_snapshot_heads_filters_prefix_visibility_and_soft_deleted(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        _append(log, "akashic:a", "a1")
        _append(log, "akashic:b", "b1")
        _append(log, "telegram:c", "c1")
        log.ensure_session("akashic:hidden", SessionAttributes(visibility="internal"))
        _append(log, "akashic:hidden", "h1")
        catalog = log.catalog()
        assert set(catalog.snapshot_heads(prefix="akashic:", visibility="listed")) == {"akashic:a", "akashic:b"}
        assert "akashic:hidden" in catalog.snapshot_heads(prefix="akashic:")
        _ = log.set_session_deleted("akashic:b", deleted=True)
        # 软删对任何过滤读取一律排除；无过滤的全量目录保持原语义。
        assert set(catalog.snapshot_heads(prefix="akashic:", visibility="listed")) == {"akashic:a"}
        assert "akashic:b" not in catalog.snapshot_heads(prefix="akashic:")
        assert "akashic:b" in catalog.snapshot_heads()
        _ = log.set_session_deleted("akashic:b", deleted=False)
        assert set(catalog.snapshot_heads(prefix="akashic:", visibility="listed")) == {"akashic:a", "akashic:b"}
    finally:
        log.close()


def test_snapshot_heads_cache_is_per_filter_and_follows_new_commits(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        _append(log, "akashic:a", "a1")
        catalog = log.catalog()
        assert catalog.snapshot_heads(prefix="akashic:", visibility="listed") == {"akashic:a": 0}
        assert catalog.snapshot_heads(prefix="telegram:", visibility="listed") == {}
        _append(log, "akashic:a", "a2")
        assert catalog.snapshot_heads(prefix="akashic:", visibility="listed") == {"akashic:a": 1}
        assert catalog.snapshot_heads(prefix="telegram:", visibility="listed") == {}
    finally:
        log.close()
