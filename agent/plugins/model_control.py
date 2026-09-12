from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.plugin_composition import ModelCatalogSnapshot
from agent.plugin_composition.model import ServiceKey
from agent.plugins.snapshot import (
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_composition.model_settings_http import ModelControlUnavailable
from agent.plugin_composition.models import (
    MODEL_CALL_STATS,
    MODEL_CATALOG,
    ChatModelSelection,
    ModelCallStats,
)


class _ModelSelectionReader(Protocol):
    def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection: ...


_MODEL_SELECTION = ServiceKey[_ModelSelectionReader]("models.selection.v1")


class RuntimeModelControl:
    """Run each control request against one leased committed plugin Root."""

    def __init__(self, snapshot_store: RuntimeSnapshotStore) -> None:
        self._snapshot_store = snapshot_store

    async def _acquire(self) -> RuntimeSnapshotLease:
        """Acquire one stable snapshot and map absence to the public error."""

        try:
            return await self._snapshot_store.acquire()
        except RuntimeError as error:
            raise ModelControlUnavailable("模型控制服务尚未就绪") from error

    async def call_stats(self, call_id: str) -> ModelCallStats:
        lease = await self._acquire()
        token = bind_runtime_snapshot(lease)
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise ModelControlUnavailable("模型控制服务尚未绑定插件组合 Root")
            read = root.context.get(MODEL_CALL_STATS)
            if read is None:
                raise ModelControlUnavailable("models 插件未提供调用统计")
            return read(call_id)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection:
        """在当前 snapshot lease 内解析并读取会话模型选择。"""
        lease = await self._acquire()
        token = bind_runtime_snapshot(lease)
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise ModelControlUnavailable("模型控制服务尚未绑定插件组合 Root")
            reader = root.context.get(_MODEL_SELECTION)
            if reader is None:
                raise ModelControlUnavailable("models 插件未提供模型选择")
            return reader.read_saved(metadata)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def catalog(self) -> ModelCatalogSnapshot:
        lease = await self._acquire()
        token = bind_runtime_snapshot(lease)
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise ModelControlUnavailable("模型控制服务尚未绑定插件组合 Root")
            catalog = root.context.get(MODEL_CATALOG)
            if catalog is None:
                raise ModelControlUnavailable("models 插件未提供模型目录")
            return catalog.snapshot()
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def invoke_rpc(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> object:
        """Resolve and invoke one plugin RPC while holding its exact lease."""
        lease = await self._acquire()
        token = bind_runtime_snapshot(lease)
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise ModelControlUnavailable("模型控制服务尚未绑定插件组合 Root")
            operation = root.context.get(rpc_method_key(method))
            if operation is None:
                raise ModelControlUnavailable("models 插件未提供模型 HTTP 服务")
            typed = operation.params.model_validate(params)
            return await operation.invoke(typed, None)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()


__all__ = ["ModelControlUnavailable", "RuntimeModelControl"]
