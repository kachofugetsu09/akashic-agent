from __future__ import annotations

from agent.plugin_composition import ModelCatalogSnapshot, ModelChange, SettingsReceipt
from agent.plugins.snapshot import (
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.plugin_composition.model_settings_http import (
    BoundModelControl,
    ModelControlUnavailable,
)


class RuntimeModelControl:
    """Run each control request against one leased committed plugin Root."""

    def __init__(self, snapshot_store: RuntimeSnapshotStore) -> None:
        self._snapshot_store = snapshot_store
        self._bound = BoundModelControl()

    async def catalog(self) -> ModelCatalogSnapshot:
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            return await self._bound.catalog()
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def apply(self, command: ModelChange) -> SettingsReceipt:
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            return await self._bound.apply(command)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()


__all__ = ["ModelControlUnavailable", "RuntimeModelControl"]
