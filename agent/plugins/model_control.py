from __future__ import annotations

from collections.abc import Mapping

from agent.plugin_composition import ModelCatalogSnapshot
from agent.plugins.snapshot import (
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_composition.model_settings_http import (
    BoundModelControl,
    ModelControlUnavailable,
)
from agent.plugin_composition.models import ChatModelSelection, ModelCallStats


class RuntimeModelControl:
    """Run each control request against one leased committed plugin Root."""

    def __init__(self, snapshot_store: RuntimeSnapshotStore) -> None:
        self._snapshot_store = snapshot_store
        self._bound = BoundModelControl()

    async def call_stats(self, call_id: str) -> ModelCallStats:
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            return await self._bound.call_stats(call_id)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection:
        """在当前 snapshot lease 内解析并读取会话模型选择。"""
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            return await self._bound.read_saved(metadata)
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def catalog(self) -> ModelCatalogSnapshot:
        lease = await self._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            return await self._bound.catalog()
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    async def invoke_rpc(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> object:
        """Resolve and invoke one plugin RPC while holding its exact lease."""
        try:
            lease = await self._snapshot_store.acquire()
        except RuntimeError as error:
            raise ModelControlUnavailable("模型控制服务尚未就绪") from error
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
