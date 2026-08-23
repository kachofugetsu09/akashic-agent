from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import Context, EmitEventKey, ServiceKey
from plugins.content.store import ContentStore

api_version = 3
name = "content"
version = "3.0.0"
desc = "Durable source-neutral Content inbox"
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

    def read_revision(
        self, item_id: str, revision: str
    ) -> Mapping[str, object] | None:
        """Read a checkpointed revision during offline handoff verification."""
        ...

    def unsettled(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def ack(self, settlement_ref: str) -> Mapping[str, object]: ...


class ContentSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContentSource: ...


class ContentWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

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

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
    ) -> Mapping[str, object]: ...


class ContentDeliveryServices(Protocol):
    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...

CONTENT_SOURCE = ServiceKey[ContentSourceServices]("content.source.v1")
CONTENT_WAKE = ServiceKey[ContentWakeServices]("content.wake.v1")
CONTENT_DELIVERY = ServiceKey[ContentDeliveryServices]("content.delivery.v1")
CONTENT_CHANGED = EmitEventKey[None]("content.changed")


class _BoundSource:
    def __init__(
        self,
        store: ContentStore,
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
    def __init__(self, store: ContentStore, changed: Callable[[], None]) -> None:
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


class _WakeServices:
    def __init__(self, store: ContentStore) -> None:
        self._store = store

    def snapshot(self, now: datetime) -> Mapping[str, object]:
        return self._store.snapshot(now)

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return self._store.selected(limit)

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

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
    ) -> Mapping[str, object]:
        allowed = {
            "ready_for_delivery",
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
        )


class _DeliveryServices:
    def __init__(self, store: ContentStore) -> None:
        self._store = store

    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return self._store.pending_delivery(limit)

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return self._store.delivery(accepted_turn)

    def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]:
        return self._store.settle_delivery(selection_token, settlement_ref)


async def apply(ctx: Context, config: object) -> None:
    """Publish three narrow views over one generation-scoped Content store."""

    _ = config
    store = ContentStore(
        ctx.data_root / "content.sqlite3",
        data_access=ctx.data_access,
    )
    store.initialize()
    _ = await ctx.provide(
        CONTENT_SOURCE,
        _SourceServices(store, lambda: ctx.emit(CONTENT_CHANGED, None)),
    )
    _ = await ctx.provide(CONTENT_WAKE, _WakeServices(store))
    _ = await ctx.provide(CONTENT_DELIVERY, _DeliveryServices(store))
