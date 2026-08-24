from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import Context, ServiceKey
from plugins.drift.store import DriftStore

api_version = 3
name = "drift"
version = "3.0.0"
desc = "Durable Drift proposal state"
author = "Akashic Core"
inject = ()
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


class DriftWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def select(
        self,
        ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def transition(self, token: str, action: str) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...


DRIFT_WAKE = ServiceKey[DriftWakeServices]("drift.wake.v1")


class DriftProposalServices(Protocol):
    def propose(
        self,
        proposal_id: str,
        revision: str,
        payload: Mapping[str, object],
        due_at: datetime,
        *,
        next_due: datetime | None = None,
    ) -> Mapping[str, object]: ...


DRIFT_PROPOSALS = ServiceKey[DriftProposalServices]("drift.proposals.v1")


class DriftDeliveryServices(Protocol):
    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...


DRIFT_DELIVERY = ServiceKey[DriftDeliveryServices]("drift.delivery.v1")


class _WakeServices:
    def __init__(self, store: DriftStore) -> None:
        self._store = store

    def snapshot(self, now: datetime) -> Mapping[str, object]:
        return self._store.snapshot(now)

    def select(
        self,
        ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]:
        return self._store.select(ref, accepted_turn, now)

    def transition(self, token: str, action: str) -> Mapping[str, object]:
        return self._store.transition(token, action)

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]:
        return self._store.selected(limit)

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return self._store.selection(accepted_turn)


class _ProposalServices:
    def __init__(self, store: DriftStore) -> None:
        self._store = store

    def propose(
        self,
        proposal_id: str,
        revision: str,
        payload: Mapping[str, object],
        due_at: datetime,
        *,
        next_due: datetime | None = None,
    ) -> Mapping[str, object]:
        return self._store.propose(
            proposal_id,
            revision,
            payload,
            due_at,
            next_due=next_due,
        )


class _DeliveryServices:
    def __init__(self, store: DriftStore) -> None:
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
    """Publish the narrow Drift view over one generation-scoped store."""

    _ = config
    store = DriftStore(
        ctx.data_root / "drift.sqlite3",
        data_access=ctx.data_access,
    )
    store.initialize()
    _ = await ctx.provide(DRIFT_PROPOSALS, _ProposalServices(store))
    _ = await ctx.provide(DRIFT_WAKE, _WakeServices(store))
    _ = await ctx.provide(DRIFT_DELIVERY, _DeliveryServices(store))
