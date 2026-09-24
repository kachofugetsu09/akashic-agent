import asyncio
import json
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from plugins.akashic_clients.notifications import NotificationFeed, notification_events
from plugins.akashic_clients.services import MessageCatalogPort
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input, Output


def _writer(log: MessageLog, session: str, author: str):
    return log.writer(session, author=author, source="conversation", body_types=(Input, Output),
                      content={"text": lambda _: ContentReferences()})


def _feed(log: MessageLog) -> NotificationFeed:
    @asynccontextmanager
    async def opener():
        yield cast(MessageCatalogPort, log.catalog())
    return NotificationFeed(opener, prefix="akashic:")


def _text(value: str) -> tuple[ContentPart, ...]:
    return (ContentPart("text", value),)


@pytest.mark.asyncio
async def test_scan_reports_only_completed_replies_once_and_advances(tmp_path):
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        start = datetime.now(UTC) - timedelta(seconds=1)
        user, assistant = _writer(log, "akashic:a", "user"), _writer(log, "akashic:a", "assistant")
        user.append("u0", Input(_text("明早提醒我开会")))
        assistant.append("step", Output(_text("正在查日程"), "continue"))
        assistant.append("quiet", Output(_text("后台整理"), "quiet"))
        assistant.append("done", Output(_text("好的，  明早 9 点\n提醒你。"), "complete"))
        _writer(log, "telegram:x", "assistant").append("other", Output(_text("别的渠道"), "complete"))
        feed = _feed(log)

        first = await feed.scan(start)
        assert [(item.message_id, item.title, item.preview) for item in first.items] == [
            ("done", "明早提醒我开会", "好的， 明早 9 点 提醒你。"),
        ]
        assert (await feed.scan(first.cursor)).items == ()

        _writer(log, "akashic:b", "assistant").append("wake", Output(_text("主动来找你聊天"), "complete"))
        later = await feed.scan(first.cursor)
        assert [item.message_id for item in later.items] == ["wake"]
        assert later.cursor > first.cursor


@pytest.mark.asyncio
async def test_reconnect_from_cursor_replays_messages_missed_while_offline(tmp_path):
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        assistant = _writer(log, "akashic:a", "assistant")
        assistant.append("seen", Output(_text("第一条"), "complete"))
        cursor = (await _feed(log).scan(datetime.now(UTC) - timedelta(seconds=1))).cursor
        assistant.append("missed", Output(_text("离线时的定时提醒"), "complete"))

        stream = notification_events(_feed(log), cursor, poll_interval=0.01, heartbeat=3600)
        try:
            ready = await anext(stream)
            message = await asyncio.wait_for(anext(stream), 2)
        finally:
            await stream.aclose()
        assert ready.startswith(b"event: ready\n")
        name, data = message.decode().strip().split("\n")
        assert name == "event: message"
        assert json.loads(data.removeprefix("data: "))["message_id"] == "missed"
