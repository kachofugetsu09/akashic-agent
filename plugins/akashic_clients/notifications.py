from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from agent.plugin_contracts.message import ContentPart, Input, Message, Output

from .services import MessageCatalogPort, SessionEntryPort

POLL_INTERVAL_SECONDS = 2.0
HEARTBEAT_SECONDS = 15.0
_TAIL_LIMIT = 30
_SESSION_PAGE_LIMIT = 50
_MAX_SESSION_PAGES = 4
_PREVIEW_CHARS = 160
_TITLE_CHARS = 40

type CatalogOpener = Callable[[], AbstractAsyncContextManager[MessageCatalogPort]]


@dataclass(frozen=True, slots=True)
class NotificationItem:
    """一条值得提醒的已提交助手回复；正文仍以 Session 为准。"""

    session_id: str
    message_id: str
    seq: int
    recorded_at: str
    title: str
    preview: str


@dataclass(frozen=True, slots=True)
class NotificationScan:
    items: tuple[NotificationItem, ...]
    cursor: datetime


# 只提醒完成的可见回复；工具中间步骤与 quiet 被动消息不打扰用户。
def is_notifiable(message: Message) -> bool:
    body = message.body
    return isinstance(body, Output) and body.finish == "complete" and bool(_text_of(body.parts))


def _text_of(parts: tuple[object, ...]) -> str:
    texts = [
        part.value for part in parts
        if isinstance(part, ContentPart) and part.kind == "text" and isinstance(part.value, str)
    ]
    return "\n".join(text.strip() for text in texts if text.strip())


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _as_utc(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class NotificationFeed:
    """按时间游标扫描 Session 目录；每次扫描只短暂借用消息目录。"""

    def __init__(self, open_catalog: CatalogOpener, *, prefix: str) -> None:
        self._open_catalog = open_catalog
        self._prefix = prefix

    # 返回 (cursor, 本轮目录最新时间] 之间提交的回复，并推进游标。
    async def scan(self, cursor: datetime) -> NotificationScan:
        async with self._open_catalog() as catalog:
            changed = self._changed_sessions(catalog, cursor)
            if not changed:
                return NotificationScan((), cursor)
            horizon = max(updated for _, updated in changed)
            items = [
                item
                for entry, _ in changed
                for item in self._session_items(catalog, entry, cursor, horizon)
            ]
        items.sort(key=lambda item: (item.recorded_at, item.session_id, item.seq))
        return NotificationScan(tuple(items), horizon)

    # 目录按最近活跃倒序排列，遇到不晚于游标的会话即可停止翻页。
    def _changed_sessions(
        self, catalog: MessageCatalogPort, cursor: datetime,
    ) -> list[tuple[SessionEntryPort, datetime]]:
        changed: list[tuple[SessionEntryPort, datetime]] = []
        after: tuple[str, str] | None = None
        for _ in range(_MAX_SESSION_PAGES):
            page = catalog.sessions(prefix=self._prefix, visibility="listed", after=after, limit=_SESSION_PAGE_LIMIT)
            for entry in page.items:
                updated = _as_utc(entry.updated_at)
                if updated is None or updated <= cursor:
                    return changed
                changed.append((entry, updated))
            if page.next_cursor is None:
                return changed
            after = page.next_cursor
        return changed

    def _session_items(
        self, catalog: MessageCatalogPort, entry: SessionEntryPort, cursor: datetime, horizon: datetime,
    ) -> list[NotificationItem]:
        page = catalog.reader(entry.session_id).read_tail(before_seq=None, through_seq=None, limit=_TAIL_LIMIT)
        title = _session_title(entry)
        items: list[NotificationItem] = []
        for message in page.messages:
            recorded = _as_utc(message.recorded_at)
            if recorded is None or not cursor < recorded <= horizon or not is_notifiable(message):
                continue
            body = message.body
            assert isinstance(body, Output)
            items.append(NotificationItem(
                session_id=message.session_id,
                message_id=message.message_id,
                seq=message.seq,
                recorded_at=recorded.isoformat(),
                title=title,
                preview=_clip(_text_of(body.parts), _PREVIEW_CHARS),
            ))
        return items


def _session_title(entry: SessionEntryPort) -> str:
    body = None if entry.first_message is None else entry.first_message.body
    text = _text_of(body.parts) if isinstance(body, (Input, Output)) else ""
    return _clip(text, _TITLE_CHARS) if text else "Akashic"


def parse_cursor(raw: str | None, now: datetime) -> datetime:
    if not raw:
        return now
    parsed = _as_utc(datetime.fromisoformat(raw))
    if parsed is None:
        raise ValueError("since 必须是 ISO 时间")
    return min(parsed, now)


def _event(name: str, payload: dict[str, object]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode("utf-8")


# 输出 SSE：ready 给出起点，message 逐条提醒，heartbeat 维持连接并推进游标。
async def notification_events(
    feed: NotificationFeed,
    cursor: datetime,
    *,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    heartbeat: float = HEARTBEAT_SECONDS,
    clock: Callable[[], float] | None = None,
) -> AsyncGenerator[bytes, None]:
    loop_clock = clock or asyncio.get_running_loop().time
    yield _event("ready", {"cursor": cursor.isoformat(), "poll_seconds": poll_interval})
    last_beat = loop_clock()
    while True:
        scan = await feed.scan(cursor)
        cursor = scan.cursor
        for item in scan.items:
            yield _event("message", {**asdict(item), "cursor": item.recorded_at})
        if scan.items:
            yield _event("cursor", {"cursor": cursor.isoformat()})
        if loop_clock() - last_beat >= heartbeat:
            last_beat = loop_clock()
            yield _event("heartbeat", {"cursor": cursor.isoformat()})
        await asyncio.sleep(poll_interval)
