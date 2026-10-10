from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime
from typing import TypeVar, cast

from agent.plugin_composition import Context, EmitEventKey, ServiceKey
from plugins.eventmail.contract import (BoundContentSource, ContentSourceServices, BoundAlertSource, AlertSourceServices,
    BoundContextSource, ContextSourceServices, EVENTMAIL_CONTENT_SOURCE, EVENTMAIL_ALERT_SOURCE, EVENTMAIL_CONTEXT_SOURCE)
from plugins.eventmail.contract import EVENTMAIL_DELIVERY_V2 as EVENTMAIL_DELIVERY, EVENTMAIL_WAKE_V2 as EVENTMAIL_WAKE, ContentWakeServicesV2 as ContentWakeServices

from core.common.file_io import run_file_io

from .store import EventMailStore

api_version = 3
name = "eventmail"
version = "4.1.0"
desc = "Immutable Content, Alert, and Context mailbox"
author = "Akashic Core"
inject = ()
workspace_roots = ()
workspace_files = ()


EVENTMAIL_CHANGED = EmitEventKey[None]("eventmail.changed")


T = TypeVar("T")


class _StoreIO:
    """同一 EventMail 发布中的写入和首次发送检查共享提交顺序。"""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()

    async def write(self, call: Callable[[], T], changed: Callable[[], None] | None = None) -> T:
        async with self.lock:
            committed = False

            def commit() -> T:
                nonlocal committed
                result = call()
                committed = True
                return result

            try:
                return await run_file_io(commit)
            finally:
                # 取消会等物理事务退出；已经提交的事实仍在原 loop 发出通知。
                if committed and changed is not None:
                    changed()


class _SourceBinding:
    """Release one source ID only after its owning Fiber drains its calls."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._release()
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("EventMail source binding 已关闭")


class _BoundSource(_SourceBinding):
    def __init__(
        self,
        store: EventMailStore, io: _StoreIO,
        source_id: str,
        changed: Callable[[], None],
        release: Callable[[], None],
    ) -> None:
        super().__init__(release)
        self._store = store
        self._io = io
        self._source_id = source_id
        self._changed = changed

    async def submit(
        self, batch_id: str, items: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]:
        items = tuple(deepcopy(dict(item)) for item in items)
        self._check_open()
        receipt = await self._io.write(lambda: self._store.submit(self._source_id, batch_id, items), self._changed)
        return receipt

    async def read_submission(self, batch_id: str) -> Mapping[str, object] | None:
        self._check_open()
        return await run_file_io(lambda: self._store.read_submission(self._source_id, batch_id))

    async def read_revision(self, item_id: str, revision: str) -> Mapping[str, object] | None:
        self._check_open()
        return await run_file_io(lambda: self._store.read_revision(self._source_id, item_id, revision))

    async def unsettled(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        self._check_open()
        return await run_file_io(lambda: self._store.unsettled(self._source_id, limit))

    async def ack(self, settlement_ref: str) -> Mapping[str, object]:
        self._check_open()
        return await self._io.write(lambda: self._store.ack(self._source_id, settlement_ref))


class _SourceServices:
    def __init__(self, store: EventMailStore, io: _StoreIO, changed: Callable[[], None]) -> None:
        self._store = store
        self._io = io
        self._changed = changed
        self._bound: dict[str, _BoundSource] = {}

    def bind(self, source_id: str) -> BoundContentSource:
        if not source_id or source_id.strip() != source_id:
            raise ValueError("Content source_id 必须非空且无首尾空白")
        if source_id in self._bound:
            raise RuntimeError(f"Content source_id 已有 owner: {source_id}")
        def release() -> None:
            if self._bound.get(source_id) is not bound:
                raise RuntimeError(f"Content source_id owner 已改变: {source_id}")
            del self._bound[source_id]

        bound = _BoundSource(self._store, self._io, source_id, self._changed, release)
        self._bound[source_id] = bound
        return bound


class _BoundAlertSource(_SourceBinding):
    def __init__(
        self, store: EventMailStore, io: _StoreIO, source_id: str, changed: Callable[[], None],
        release: Callable[[], None],
    ) -> None:
        super().__init__(release)
        self._store = store
        self._io = io
        self._source_id = source_id
        self._changed = changed

    async def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        payload = deepcopy(dict(payload))
        self._check_open()
        receipt = await self._io.write(lambda: self._store.report_alert(
            source_id=self._source_id,
            event_id=event_id,
            payload=payload,
            observed_at=observed_at,
            expires_at=expires_at,
        ), self._changed)
        return receipt

    async def status(self, *, event_id: str) -> str | None:
        self._check_open()
        return await run_file_io(lambda: self._store.alert_status(self._source_id, event_id))


class _AlertSourceServices:
    def __init__(self, store: EventMailStore, io: _StoreIO, changed: Callable[[], None]) -> None:
        self._store = store
        self._io = io
        self._changed = changed
        self._bound: dict[str, _BoundAlertSource] = {}

    def bind(self, source_id: str) -> BoundAlertSource:
        source = _source_id(source_id)
        if source in self._bound:
            raise RuntimeError(f"EventMail Alert source_id 已有 owner: {source}")
        def release() -> None:
            if self._bound.get(source) is not bound:
                raise RuntimeError(f"EventMail Alert source_id owner 已改变: {source}")
            del self._bound[source]

        bound = _BoundAlertSource(self._store, self._io, source, self._changed, release)
        self._bound[source] = bound
        return bound


class _BoundContextSource(_SourceBinding):
    def __init__(
        self, store: EventMailStore, io: _StoreIO, source_id: str, changed: Callable[[], None],
        release: Callable[[], None],
    ) -> None:
        super().__init__(release)
        self._store = store
        self._io = io
        self._source_id = source_id
        self._changed = changed

    async def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        payload = deepcopy(dict(payload))
        self._check_open()
        receipt = await self._io.write(lambda: self._store.report_context(
            source_id=self._source_id,
            event_id=event_id,
            payload=payload,
            observed_at=observed_at,
            expires_at=expires_at,
        ), self._changed)
        return receipt


class _ContextSourceServices:
    def __init__(self, store: EventMailStore, io: _StoreIO, changed: Callable[[], None]) -> None:
        self._store = store
        self._io = io
        self._changed = changed
        self._bound: dict[str, _BoundContextSource] = {}

    def bind(self, source_id: str) -> BoundContextSource:
        source = _source_id(source_id)
        if source in self._bound:
            raise RuntimeError(f"EventMail Context source_id 已有 owner: {source}")
        def release() -> None:
            if self._bound.get(source) is not bound:
                raise RuntimeError(f"EventMail Context source_id owner 已改变: {source}")
            del self._bound[source]

        bound = _BoundContextSource(self._store, self._io, source, self._changed, release)
        self._bound[source] = bound
        return bound


class _WakeServices:
    def __init__(self, store: EventMailStore, io: _StoreIO) -> None:
        self._store = store
        self._io = io

    async def snapshot(self, now: datetime) -> Mapping[str, object]:
        return await run_file_io(lambda: self._store.snapshot(now))

    async def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return await run_file_io(lambda: self._store.selected(limit))

    async def expire(
        self,
        item_refs: Sequence[Mapping[str, object]],
        now: datetime,
    ) -> Mapping[str, object]:
        item_refs = tuple(deepcopy(dict(item)) for item in item_refs)
        return await self._io.write(lambda: self._store.expire(item_refs, now))

    async def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        accepted_turn = deepcopy(dict(accepted_turn))
        return await run_file_io(lambda: self._store.selection(accepted_turn))

    async def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]:
        accepted_turn = deepcopy(dict(accepted_turn))
        item_ref = deepcopy(dict(item_ref))
        return await self._io.write(lambda: self._store.select(item_ref, snapshot_seq, accepted_turn, now))

    async def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]:
        item_refs = tuple(deepcopy(dict(item)) for item in item_refs)
        accepted_turn = deepcopy(dict(accepted_turn))
        return await self._io.write(lambda: self._store.select_batch(item_refs, snapshot_seq, accepted_turn, now))

    async def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]:
        selected_refs = None if selected_refs is None else tuple(dict(item) for item in selected_refs)
        allowed = {
            "ready_for_delivery",
            "release",
            "defer",
            "await_change",
            "invalidated",
            "abandoned",
            "failed",
            "expired",
        }
        if action not in allowed:
            raise ValueError(f"Content Wake capability 不拥有 transition: {action}")
        return await self._io.write(lambda: self._store.transition(
            selection_token,
            action,
            not_before=not_before,
            selected_refs=selected_refs,
        ))

    async def mail_watermark(self) -> int:
        return await run_file_io(lambda: self._store.mail_watermark())

    async def alert_deadline(self, now: datetime) -> datetime | None:
        return await self._io.write(lambda: self._store.alert_deadline(now))

    async def alert_status(self, source_id: str, event_id: str, *, mail_id: str | None = None) -> str | None:
        return await run_file_io(lambda: self._store.alert_status(source_id, event_id, mail_id=mail_id))

    async def change_alert(self, item_ref: Mapping[str, object], accepted_turn: Mapping[str, object],
                     action: str, now: datetime, *, not_before: datetime | None = None) -> bool:
        accepted_turn = deepcopy(dict(accepted_turn))
        item_ref = deepcopy(dict(item_ref))
        return await self._io.write(lambda: self._store.change_alert(item_ref, accepted_turn, action, now, not_before=not_before))

    async def peek_alert(self, now: datetime) -> Mapping[str, object] | None:
        return await run_file_io(lambda: self._store.peek_alert(now))

    async def select_alert(
        self, accepted_turn: Mapping[str, object], now: datetime, *, item_ref: Mapping[str, object] | None = None,
    ) -> Mapping[str, object] | None:
        accepted_turn = deepcopy(dict(accepted_turn))
        item_ref = None if item_ref is None else dict(item_ref)
        return await self._io.write(lambda: self._store.select_alert(accepted_turn, now, item_ref=item_ref))

    async def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        accepted_turn = deepcopy(dict(accepted_turn))
        return await run_file_io(lambda: self._store.selected_alert(accepted_turn))

    async def selected_alerts(self) -> tuple[Mapping[str, object], ...]:
        return await run_file_io(lambda: self._store.selected_alerts())

    async def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool:
        return await self._io.write(lambda: self._store.expire_alert(source_id, event_id, now))

    async def defer_alert(
        self, source_id: str, event_id: str, not_before: datetime
    ) -> None:
        await self._io.write(lambda: self._store.defer_alert(source_id, event_id, not_before))

    async def close_alert(self, source_id: str, event_id: str, status: str) -> None:
        await self._io.write(lambda: self._store.close_alert(source_id, event_id, status))

    async def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]:
        return await run_file_io(lambda: self._store.active_context(now))


    @asynccontextmanager
    async def alert_start(self, item_ref: Mapping[str, object], expires_at: datetime | None,
                          now: Callable[[], datetime]) -> AsyncIterator[str | None]:
        """保护原告警检查到首次 started 提交，不持有网络发送锁。"""
        source, event, mail = (cast(str, item_ref[key]) for key in ("source_id", "event_id", "mail_id"))
        async with self._io.lock:
            status = await run_file_io(lambda: self._store.alert_status(source, event, mail_id=mail))
            if status != "selected":
                yield "原告警版本已结束"
            elif expires_at is not None and expires_at <= now():
                yield "告警在发送前已过期"
            else:
                yield None


class _DeliveryServices:
    def __init__(self, store: EventMailStore, io: _StoreIO) -> None:
        self._store = store
        self._io = io

    async def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return await run_file_io(lambda: self._store.pending_delivery(limit))

    async def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        accepted_turn = deepcopy(dict(accepted_turn))
        return await run_file_io(lambda: self._store.delivery(accepted_turn))

    async def settle(self, selection_token: str, settlement_ref: str) -> Mapping[str, object]:
        return await self._io.write(lambda: self._store.settle_delivery(selection_token, settlement_ref))


async def apply(ctx: Context) -> None:
    """Publish typed source and consumer views over one EventMail store."""

    store = EventMailStore(ctx.data_root / "eventmail.sqlite3")
    await run_file_io(store.initialize)
    io = _StoreIO()
    _ = await ctx.provide(
        EVENTMAIL_CONTENT_SOURCE,
        _SourceServices(store, io, lambda: ctx.emit(EVENTMAIL_CHANGED, None)),
    )
    _ = await ctx.provide(
        EVENTMAIL_ALERT_SOURCE,
        _AlertSourceServices(store, io, lambda: ctx.emit(EVENTMAIL_CHANGED, None)),
    )
    _ = await ctx.provide(
        EVENTMAIL_CONTEXT_SOURCE,
        _ContextSourceServices(store, io, lambda: ctx.emit(EVENTMAIL_CHANGED, None)),
    )
    _ = await ctx.provide(EVENTMAIL_WAKE, _WakeServices(store, io))
    _ = await ctx.provide(EVENTMAIL_DELIVERY, _DeliveryServices(store, io))


def _source_id(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError("EventMail source_id 必须非空且无首尾空白")
    return value
