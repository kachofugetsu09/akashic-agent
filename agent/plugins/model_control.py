from __future__ import annotations

from agent.plugin_composition import (
    MODEL_CATALOG,
    MODEL_SETTINGS,
    ModelCatalogSnapshot,
    ModelChange,
    SettingsReceipt,
)
from agent.plugins.snapshot import (
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)


class ModelControlUnavailable(RuntimeError):
    """The committed plugin snapshot lacks the model control services."""


class RuntimeModelControl:
    """Run each control request against one leased committed plugin Root."""

    def __init__(self, snapshot_store: RuntimeSnapshotStore) -> None:
        self._snapshot_store = snapshot_store

    async def catalog(self) -> ModelCatalogSnapshot:
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise ModelControlUnavailable("RuntimeSnapshot 缺少插件组合 Root")
            catalog = root.context.get(MODEL_CATALOG)
            if catalog is None:
                raise ModelControlUnavailable("models 插件未提供模型目录")
            return catalog.snapshot()
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def apply(self, command: ModelChange) -> SettingsReceipt:
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise ModelControlUnavailable("RuntimeSnapshot 缺少插件组合 Root")
            settings = root.context.get(MODEL_SETTINGS)
            if settings is None:
                raise ModelControlUnavailable("models 插件未提供模型设置")
            return await settings.apply(command)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()


__all__ = ["ModelControlUnavailable", "RuntimeModelControl"]
