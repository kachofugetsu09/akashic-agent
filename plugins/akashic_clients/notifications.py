from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from agent.plugin_contracts.message import ContentPart, Input, Message, Output

from .services import MessageCatalogPort, MessagePagePort, SessionEntryPort

POLL_INTERVAL_SECONDS = 2.0
HEARTBEAT_SECONDS = 15.0
_PAGE_SIZE = 100

type CatalogOpener = Callable[[], AbstractAsyncContextManager[MessageCatalogPort]]
type Cursor = dict[str, int]


class NotificationRequest(BaseModel):
    """进度只属于通知客户端；空值表示从首次连接时的已有消息之后开始。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    cursor: dict[str, Annotated[StrictInt, Field(ge=-1)]] | None = None


def _text_of(parts: tuple[object, ...]) -> str:
    return "\n".join(
        part.value.strip() for part in parts
        if isinstance(part, ContentPart) and part.kind == "text"
        and isinstance(part.value, str) and part.value.strip()
    )


def is_notifiable(message: Message) -> bool:
    return isinstance(message.body, Output) and message.body.finish == "complete" and bool(_text_of(message.body.parts))


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def _item(message: Message, entry: SessionEntryPort) -> dict[str, object]:
    body = message.body
    assert isinstance(body, Output)
    first = None if entry.first_message is None else entry.first_message.body
    title = _text_of(first.parts) if isinstance(first, (Input, Output)) else ""
    return {
        "session_id": message.session_id,
        "message_id": message.message_id,
        "seq": message.seq,
        "recorded_at": message.recorded_at.isoformat(),
        "title": _clip(title, 40) or "Akashic",
        "preview": _clip(_text_of(body.parts), 160),
    }


class NotificationFeed:
    """只读现有目录和消息页，每页读取后释放插件作用域。"""

    def __init__(self, open_catalog: CatalogOpener, *, prefix: str) -> None:
        self._open_catalog = open_catalog
        self._prefix = prefix

    async def sessions(self) -> AsyncGenerator[SessionEntryPort, None]:
        """完整遍历 live 目录；并发移动的条目由下一轮按 seq 补齐。"""
        after: tuple[str, str] | None = None
        while True:
            async with self._open_catalog() as catalog:
                page = catalog.sessions(prefix=self._prefix, visibility="listed", after=after, limit=_PAGE_SIZE)
            for entry in page.items:
                yield entry
            if page.next_cursor is None:
                return
            after = page.next_cursor

    async def baseline(self) -> Cursor:
        return {entry.session_id: entry.head_seq async for entry in self.sessions()}

    async def read(self, entry: SessionEntryPort, after: int) -> MessagePagePort | None:
        """页面使用当前 head；会话被用户删除时本轮跳过，绝不写回消息。"""
        async with self._open_catalog() as catalog:
            try:
                return catalog.reader(entry.session_id).read_page(after_seq=after, limit=_PAGE_SIZE)
            except KeyError:
                return None


def _event(name: str, payload: dict[str, object]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode("utf-8")


async def notification_events(
    feed: NotificationFeed,
    cursor: Cursor,
    *,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    heartbeat: float = HEARTBEAT_SECONDS,
) -> AsyncGenerator[bytes, None]:
    """按会话 seq 重放；提醒先于进度，客户端展示成功后才能保存进度。"""
    # 1. ready 保存首次基线；重连时客户端仍是进度的唯一 owner。
    cursor = dict(cursor)
    yield _event("ready", {"cursor": cursor})
    clock = asyncio.get_running_loop().time
    last_beat = clock()
    while True:
        # 2. 每轮完整扫目录，页大小只控制内存，不裁掉离线消息。
        async for entry in feed.sessions():
            after = cursor.get(entry.session_id, -1)
            while after < entry.head_seq:
                page = await feed.read(entry, after)
                if page is None:
                    break
                for message in page.messages:
                    if is_notifiable(message):
                        yield _event("message", _item(message, entry))
                after = page.messages[-1].seq if page.has_more else page.through_seq
                cursor[entry.session_id] = after
                yield _event("cursor", {"session_id": entry.session_id, "seq": after})
                # 网络发送期间不持有 catalog 或 generation lease。
                await asyncio.sleep(0)
        # 3. 心跳只保活，不凭时间推进消费进度。
        if clock() - last_beat >= heartbeat:
            last_beat = clock()
            yield _event("heartbeat", {})
        await asyncio.sleep(poll_interval)
