"""跨设备已读水位：存储、侧栏摘要、手机通知流与标记接口。"""
import asyncio
import json
from contextlib import asynccontextmanager, nullcontext, suppress
from types import MappingProxyType
from typing import cast

import httpx
import pytest

from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.notifications import NotificationFeed, notification_events
from plugins.akashic_clients.read_marks import ReadMarks
from plugins.akashic_clients.services import MessageCatalogPort
from plugins.akashic_clients.session_activity import follow_session_activity
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.ledger.contract import ContentPart, Input, Output
from plugins.ledger.log import MessageLog
from tests.test_message_log import text_schema


def _marks(log: MessageLog) -> ReadMarks:
    return ReadMarks(lambda: nullcontext(log.owner("plugin:akashic_clients")))


def _say(log: MessageLog, session: str, message_id: str, text: str = "hi") -> None:
    log.writer(session, author="assistant", source="conversation", body_types=(Output,),
               content={"text": text_schema}).append(message_id, Output((ContentPart("text", text),), "complete"))


def _ask(log: MessageLog, session: str, message_id: str) -> None:
    log.writer(session, author="user", source="conversation", body_types=(Input,),
               content={"text": text_schema}).append(message_id, Input((ContentPart("text", "q"),)))


# ---- 存储：基线一次、只前进、分页读全 ----

def test_baseline_is_taken_once_and_marks_only_advance(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        marks = _marks(log)
        assert not marks.has_baseline() and marks.read_all() == {}
        marks.ensure_baseline({"akashic:a": 3, "akashic:b": 0})
        # 第二次基线不覆盖已有水位，也不给之后的新会话补记录。
        marks.ensure_baseline({"akashic:a": 9, "akashic:c": 5})
        assert marks.has_baseline() and marks.read_all() == {"akashic:a": 3, "akashic:b": 0}
        assert marks.mark("akashic:a", 5) == 5
        assert marks.mark("akashic:a", 2) == 5
        assert marks.mark("akashic:c", -1) == -1
        assert _marks(log).read_all() == {"akashic:a": 5, "akashic:b": 0, "akashic:c": -1}
    finally:
        log.close()


def test_read_all_pages_past_one_scan(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        marks = _marks(log)
        heads = {f"akashic:{index:04d}": index for index in range(1203)}
        marks.ensure_baseline(heads)
        assert marks.read_all() == heads
    finally:
        log.close()


@pytest.mark.asyncio
async def test_follow_wakes_only_on_committed_changes(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        marks = _marks(log)
        frames = marks.follow()
        assert await asyncio.wait_for(anext(frames), 1) == {}
        pending = asyncio.ensure_future(anext(frames))
        _ = marks.mark("akashic:a", 0)
        assert await asyncio.wait_for(pending, 1) == {"akashic:a": 0}
        # 没有前进的标记不唤醒订阅者。
        pending = asyncio.ensure_future(anext(frames))
        _ = marks.mark("akashic:a", 0)
        await asyncio.sleep(0.05)
        assert not pending.done()
        _ = pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await frames.aclose()
    finally:
        log.close()


# ---- 侧栏摘要：已读快照与增量 ----

async def _queued(queue: asyncio.Queue):
    while True:
        yield await queue.get()


@pytest.mark.asyncio
async def test_activity_carries_read_marks_for_listed_sessions() -> None:
    heads: asyncio.Queue = asyncio.Queue()
    active: asyncio.Queue = asyncio.Queue()
    reads: asyncio.Queue = asyncio.Queue()
    active.put_nowait(frozenset())
    heads.put_nowait(MappingProxyType({"akashic:a": 2, "akashic:b": 4}))
    reads.put_nowait({"akashic:a": 2, "akashic:gone": 7})
    frames = follow_session_activity(_queued(heads), _queued(active), prefix="akashic:", reads=_queued(reads))
    first = await asyncio.wait_for(anext(frames), 1)
    reads.put_nowait({"akashic:a": 2, "akashic:b": 4, "akashic:gone": 7})
    second = await asyncio.wait_for(anext(frames), 1)
    await frames.aclose()
    assert first["snapshot"] and first["read_available"] and first["read"] == {"akashic:a": 2}
    assert not second["snapshot"] and second["read"] == {"akashic:b": 4} and second["heads"] == {}


@pytest.mark.asyncio
async def test_activity_without_read_marks_reports_unavailable() -> None:
    heads: asyncio.Queue = asyncio.Queue()
    active: asyncio.Queue = asyncio.Queue()
    active.put_nowait(frozenset())
    heads.put_nowait(MappingProxyType({"akashic:a": 1}))
    frames = follow_session_activity(_queued(heads), _queued(active), prefix="akashic:")
    frame = await asyncio.wait_for(anext(frames), 1)
    await frames.aclose()
    assert frame["read_available"] is False and frame["read"] == {}


# ---- 通知流：别处已读不提醒，声明后推送 read 事件 ----

def _feed(log: MessageLog) -> NotificationFeed:
    @asynccontextmanager
    async def open_catalog():
        yield cast(MessageCatalogPort, log.catalog())
    return NotificationFeed(open_catalog, prefix="akashic:")


async def _events(stream, count: int) -> list[tuple[str, dict[str, object]]]:
    events: list[tuple[str, dict[str, object]]] = []
    while len(events) < count:
        chunk = (await asyncio.wait_for(anext(stream), 1)).decode()
        name, data = chunk.strip().split("\n")
        event = (name.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        if event[0] not in {"cursor", "heartbeat"}:
            events.append(event)
    return events


@pytest.mark.asyncio
async def test_notifications_skip_read_messages_and_push_read_events(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        marks = _marks(log)
        _say(log, "akashic:a", "a0", "read elsewhere")
        _say(log, "akashic:b", "b0", "still unread")
        _ = marks.mark("akashic:a", 0)
        stream = notification_events(_feed(log), {}, reads=marks.read_all, read_events=True, poll_interval=0.01)
        ready, message = await _events(stream, 2)
        assert ready == ("ready", {"cursor": {}, "read": {"akashic:a": 0}})
        assert message[0] == "message" and message[1]["message_id"] == "b0"
        _ = marks.mark("akashic:b", 0)
        assert await _events(stream, 1) == [("read", {"session_id": "akashic:b", "seq": 0})]
        await stream.aclose()
    finally:
        log.close()


@pytest.mark.asyncio
async def test_old_clients_never_receive_read_events(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        marks = _marks(log)
        _say(log, "akashic:a", "a0")
        stream = notification_events(_feed(log), {}, reads=marks.read_all, poll_interval=0.01)
        ready, message = await _events(stream, 2)
        assert ready == ("ready", {"cursor": {}}) and message[1]["message_id"] == "a0"
        # 已读变化之后的下一条提醒之前，旧客户端不会收到任何 read 事件。
        _ = marks.mark("akashic:a", 0)
        await asyncio.sleep(0.05)
        _say(log, "akashic:a", "a1")
        assert [name for name, _data in await _events(stream, 1)] == ["message"]
        await stream.aclose()
    finally:
        log.close()


# ---- 标记接口：首次基线、截断到已提交水位、拒绝外部会话 ----

@pytest.mark.asyncio
async def test_mark_endpoint_takes_baseline_and_clamps_to_head(tmp_path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    try:
        _ask(log, "akashic:old", "o0")
        _say(log, "akashic:old", "o1")
        marks = _marks(log)
        app = create_chat_app(workspace=tmp_path, channel=WebChatChannel("akashic"),
                              messages=cast(MessageCatalogPort, log.catalog()), read_marks=marks)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            _ask(log, "akashic:new", "n0")
            response = await client.post("/api/chat/sessions/read", json={"session_id": "akashic:new", "seq": 99})
            assert response.status_code == 200 and response.json() == {"session_id": "akashic:new", "seq": 0}
            # 首次使用时已有会话按当时水位视为已读。
            assert marks.read_all() == {"akashic:old": 1, "akashic:new": 0}
            missing = await client.post("/api/chat/sessions/read", json={"session_id": "akashic:none", "seq": 0})
            foreign = await client.post("/api/chat/sessions/read", json={"session_id": "telegram:1", "seq": 0})
            assert missing.status_code == 404 and foreign.status_code == 400
    finally:
        log.close()
