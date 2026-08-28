from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import Context, EmitEventKey, ServiceKey
from .store import EventMailStore

api_version = 3
name = "eventmail"
version = "4.1.0"
desc = "Immutable Content, Alert, and Context mailbox"
author = "Akashic Core"
inject = ()
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


class BoundContentSource(Protocol):
    def submit(
        self, batch_id: str, items: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]: ...

    def read_submission(self, batch_id: str) -> Mapping[str, object] | None:
        """Read a checkpointed receipt during offline handoff verification."""
        ...

    def read_revision(self, item_id: str, revision: str) -> Mapping[str, object] | None:
        """Read a checkpointed revision during offline handoff verification."""
        ...

    def unsettled(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def ack(self, settlement_ref: str) -> Mapping[str, object]: ...


class ContentSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContentSource: ...


class BoundAlertSource(Protocol):
    def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]: ...

    def status(self, *, event_id: str) -> str | None: ...


class AlertSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundAlertSource: ...


class BoundContextSource(Protocol):
    def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]: ...


class ContextSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContextSource: ...


class ContentWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def expire(
        self,
        item_refs: Sequence[Mapping[str, object]],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]: ...

    def mail_watermark(self) -> int: ...

    def alert_deadline(self, now: datetime) -> datetime | None: ...

    def alert_status(self, source_id: str, event_id: str) -> str | None: ...

    def select_alert(
        self, accepted_turn: Mapping[str, object], now: datetime
    ) -> Mapping[str, object] | None: ...

    def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def selected_alerts(self) -> tuple[Mapping[str, object], ...]: ...

    def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool: ...

    def defer_alert(
        self, source_id: str, event_id: str, not_before: datetime
    ) -> None: ...

    def close_alert(self, source_id: str, event_id: str, status: str) -> None: ...

    def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]: ...


class ContentDeliveryServices(Protocol):
    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...


EVENTMAIL_CONTENT_SOURCE = ServiceKey[ContentSourceServices]("eventmail.content_source.v1")
EVENTMAIL_ALERT_SOURCE = ServiceKey[AlertSourceServices]("eventmail.alert_source.v1")
EVENTMAIL_CONTEXT_SOURCE = ServiceKey[ContextSourceServices]("eventmail.context_source.v1")
EVENTMAIL_WAKE = ServiceKey[ContentWakeServices]("eventmail.wake.v1")
EVENTMAIL_DELIVERY = ServiceKey[ContentDeliveryServices]("eventmail.delivery.v1")
EVENTMAIL_ALERT_DELIVERY = ServiceKey[object]("eventmail.alert_delivery.v1")
EVENTMAIL_CHANGED = EmitEventKey[None]("eventmail.changed")


class _BoundSource:
    def __init__(
        self,
        store: EventMailStore,
        source_id: str,
        changed: Callable[[], None],
    ) -> None:
        self._store = store
        self._source_id = source_id
        self._changed = changed

    def submit(
        self, batch_id: str, items: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]:
        receipt = self._store.submit(self._source_id, batch_id, items)
        self._changed()
        return receipt

    def read_submission(self, batch_id: str) -> Mapping[str, object] | None:
        return self._store.read_submission(self._source_id, batch_id)

    def read_revision(self, item_id: str, revision: str) -> Mapping[str, object] | None:
        return self._store.read_revision(self._source_id, item_id, revision)

    def unsettled(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return self._store.unsettled(self._source_id, limit)

    def ack(self, settlement_ref: str) -> Mapping[str, object]:
        return self._store.ack(self._source_id, settlement_ref)


class _SourceServices:
    def __init__(self, store: EventMailStore, changed: Callable[[], None]) -> None:
        self._store = store
        self._changed = changed
        self._bound: dict[str, _BoundSource] = {}

    def bind(self, source_id: str) -> BoundContentSource:
        if not source_id or source_id.strip() != source_id:
            raise ValueError("Content source_id 必须非空且无首尾空白")
        if source_id in self._bound:
            raise RuntimeError(f"Content source_id 已有 owner: {source_id}")
        bound = _BoundSource(self._store, source_id, self._changed)
        self._bound[source_id] = bound
        return bound


class _BoundAlertSource:
    def __init__(
        self, store: EventMailStore, source_id: str, changed: Callable[[], None]
    ) -> None:
        self._store = store
        self._source_id = source_id
        self._changed = changed

    def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        receipt = self._store.report_alert(
            source_id=self._source_id,
            event_id=event_id,
            payload=payload,
            observed_at=observed_at,
            expires_at=expires_at,
        )
        self._changed()
        return receipt

    def status(self, *, event_id: str) -> str | None:
        return self._store.alert_status(self._source_id, event_id)


class _AlertSourceServices:
    def __init__(self, store: EventMailStore, changed: Callable[[], None]) -> None:
        self._store = store
        self._changed = changed
        self._bound: dict[str, _BoundAlertSource] = {}

    def bind(self, source_id: str) -> BoundAlertSource:
        source = _source_id(source_id)
        if source in self._bound:
            raise RuntimeError(f"EventMail Alert source_id 已有 owner: {source}")
        bound = _BoundAlertSource(self._store, source, self._changed)
        self._bound[source] = bound
        return bound


class _BoundContextSource:
    def __init__(
        self, store: EventMailStore, source_id: str, changed: Callable[[], None]
    ) -> None:
        self._store = store
        self._source_id = source_id
        self._changed = changed

    def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]:
        receipt = self._store.report_context(
            source_id=self._source_id,
            event_id=event_id,
            payload=payload,
            observed_at=observed_at,
            expires_at=expires_at,
        )
        self._changed()
        return receipt


class _ContextSourceServices:
    def __init__(self, store: EventMailStore, changed: Callable[[], None]) -> None:
        self._store = store
        self._changed = changed
        self._bound: dict[str, _BoundContextSource] = {}

    def bind(self, source_id: str) -> BoundContextSource:
        source = _source_id(source_id)
        if source in self._bound:
            raise RuntimeError(f"EventMail Context source_id 已有 owner: {source}")
        bound = _BoundContextSource(self._store, source, self._changed)
        self._bound[source] = bound
        return bound


class _WakeServices:
    def __init__(self, store: EventMailStore) -> None:
        self._store = store

    def snapshot(self, now: datetime) -> Mapping[str, object]:
        return self._store.snapshot(now)

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return self._store.selected(limit)

    def expire(
        self,
        item_refs: Sequence[Mapping[str, object]],
        now: datetime,
    ) -> Mapping[str, object]:
        return self._store.expire(item_refs, now)

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return self._store.selection(accepted_turn)

    def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]:
        return self._store.select(item_ref, snapshot_seq, accepted_turn, now)

    def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]:
        return self._store.select_batch(item_refs, snapshot_seq, accepted_turn, now)

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]:
        allowed = {
            "ready_for_delivery",
            "release",
            "defer",
            "await_change",
            "invalidated",
            "abandoned",
            "expired",
        }
        if action not in allowed:
            raise ValueError(f"Content Wake capability 不拥有 transition: {action}")
        return self._store.transition(
            selection_token,
            action,
            not_before=not_before,
            selected_refs=selected_refs,
        )

    def mail_watermark(self) -> int:
        return self._store.mail_watermark()

    def alert_deadline(self, now: datetime) -> datetime | None:
        return self._store.alert_deadline(now)

    def alert_status(self, source_id: str, event_id: str) -> str | None:
        return self._store.alert_status(source_id, event_id)

    def select_alert(
        self, accepted_turn: Mapping[str, object], now: datetime
    ) -> Mapping[str, object] | None:
        return self._store.select_alert(accepted_turn, now)

    def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return self._store.selected_alert(accepted_turn)

    def selected_alerts(self) -> tuple[Mapping[str, object], ...]:
        return self._store.selected_alerts()

    def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool:
        return self._store.expire_alert(source_id, event_id, now)

    def defer_alert(
        self, source_id: str, event_id: str, not_before: datetime
    ) -> None:
        self._store.defer_alert(source_id, event_id, not_before)

    def close_alert(self, source_id: str, event_id: str, status: str) -> None:
        self._store.close_alert(source_id, event_id, status)

    def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]:
        return self._store.active_context(now)


class _DeliveryServices:
    def __init__(self, store: EventMailStore) -> None:
        self._store = store

    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return self._store.pending_delivery(limit)

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return self._store.delivery(accepted_turn)

    def settle(self, selection_token: str, settlement_ref: str) -> Mapping[str, object]:
        return self._store.settle_delivery(selection_token, settlement_ref)


async def apply(ctx: Context, config: object) -> None:
    """Publish typed source and consumer views over one EventMail store."""

    _ = config
    store = EventMailStore(
        ctx.data_root / "eventmail.sqlite3",
        data_access=ctx.data_access,
    )
    store.initialize()
    _ = await ctx.provide(
        EVENTMAIL_CONTENT_SOURCE,
        _SourceServices(store, lambda: ctx.emit(EVENTMAIL_CHANGED, None)),
    )
    _ = await ctx.provide(
        EVENTMAIL_ALERT_SOURCE,
        _AlertSourceServices(store, lambda: ctx.emit(EVENTMAIL_CHANGED, None)),
    )
    _ = await ctx.provide(
        EVENTMAIL_CONTEXT_SOURCE,
        _ContextSourceServices(store, lambda: ctx.emit(EVENTMAIL_CHANGED, None)),
    )
    _ = await ctx.provide(EVENTMAIL_WAKE, _WakeServices(store))
    _ = await ctx.provide(EVENTMAIL_DELIVERY, _DeliveryServices(store))
    _ = await ctx.provide(EVENTMAIL_ALERT_DELIVERY, object())


def _source_id(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError("EventMail source_id 必须非空且无首尾空白")
    return value
