"""Runtime-bound model control primitives used by Core adapters.

The models plugin owns payloads, HTTP routes, RPC method schemas, and provider
error mapping. This module only resolves the public model services from the
current generation lease so Core consumers cannot retain a provider method.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import (
    MODEL_CATALOG,
    MODEL_CALL_STATS,
    ChatModelSelection,
    ModelCallStats,
    ModelCatalogSnapshot,
)


class ModelControlUnavailable(RuntimeError):
    """The bound plugin snapshot does not provide model control services."""


class ModelSelectionReader(Protocol):
    def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection: ...


MODEL_SELECTION = ServiceKey[ModelSelectionReader]("models.selection.v1")


class ModelControl(Protocol):
    async def call_stats(self, call_id: str) -> ModelCallStats: ...

    async def catalog(self) -> ModelCatalogSnapshot: ...

    async def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection: ...


class BoundModelControl:
    """Resolve model services from the exact snapshot bound to this request."""

    async def call_stats(self, call_id: str) -> ModelCallStats:
        read = _bound_root().context.get(MODEL_CALL_STATS)
        if read is None:
            raise ModelControlUnavailable("models 插件未提供调用统计")
        return read(call_id)

    async def catalog(self) -> ModelCatalogSnapshot:
        root = _bound_root()
        catalog = root.context.get(MODEL_CATALOG)
        if catalog is None:
            raise ModelControlUnavailable("models 插件未提供模型目录")
        return catalog.snapshot()

    async def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection:
        """在当前已绑定 Root 内即时解析模型选择 owner。"""
        reader = _bound_root().context.get(MODEL_SELECTION)
        if reader is None:
            raise ModelControlUnavailable("models 插件未提供模型选择")
        return reader.read_saved(metadata)


def _bound_root():
    from agent.plugins.snapshot import get_current_runtime_snapshot

    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        raise ModelControlUnavailable("请求未绑定插件组合 Root")
    return snapshot.composition_root


__all__ = [
    "BoundModelControl",
    "MODEL_SELECTION",
    "ModelControl",
    "ModelControlUnavailable",
    "ModelSelectionReader",
]
