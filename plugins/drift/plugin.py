from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from agent.plugin_composition import Context
from plugins.drift.contract import DRIFT_DELIVERY_V2, DRIFT_WAKE_V2

from core.common.file_io import run_file_io

from .store import DriftStore

api_version = 3
name = "drift"
version = "3.0.0"
desc = "Durable Drift proposal state"
author = "Akashic Core"
inject = ()
workspace_roots = ()
workspace_files = ()


class _AsyncWakeServices:
    def __init__(self, store: DriftStore) -> None:
        self._store = store

    async def snapshot(self, now: datetime) -> Mapping[str, object]:
        return await run_file_io(lambda: self._store.snapshot(now))

    async def select(
        self,
        ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]:
        return await run_file_io(lambda: self._store.select(ref, accepted_turn, now))

    async def transition(self, token: str, action: str) -> Mapping[str, object]:
        return await run_file_io(lambda: self._store.transition(token, action))

    async def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return await run_file_io(lambda: self._store.selection(accepted_turn))


class _AsyncDeliveryServices:
    def __init__(self, store: DriftStore) -> None:
        self._store = store

    async def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        return await run_file_io(lambda: self._store.delivery(accepted_turn))

    async def settle(self, selection_token: str, settlement_ref: str) -> Mapping[str, object]:
        return await run_file_io(lambda: self._store.settle_delivery(selection_token, settlement_ref))


async def apply(ctx: Context) -> None:
    """Publish the narrow Drift view over one generation-scoped store."""

    store = DriftStore(ctx.data_root / "drift.sqlite3")
    await run_file_io(store.initialize)
    _ = await ctx.provide(DRIFT_WAKE_V2, _AsyncWakeServices(store))
    _ = await ctx.provide(DRIFT_DELIVERY_V2, _AsyncDeliveryServices(store))
