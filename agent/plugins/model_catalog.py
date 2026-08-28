from __future__ import annotations

from agent.plugin_composition import (
    MODEL_CATALOG,
    ModelAvailability,
    ModelCatalogSnapshot,
    ModelKind,
)
from agent.plugins.snapshot import RuntimeSnapshotStore


class ModelCatalogUnavailable(RuntimeError):
    """The committed plugin snapshot does not provide a model catalog."""


class RuntimeModelCatalogReader:
    """Read the model catalog from one leased committed plugin snapshot."""

    def __init__(self, snapshot_store: RuntimeSnapshotStore) -> None:
        self._snapshot_store = snapshot_store

    async def read(self) -> ModelCatalogSnapshot:
        lease = await self._snapshot_store.acquire()
        async with lease as snapshot:
            root = snapshot.composition_root
            if root is None:
                raise ModelCatalogUnavailable("RuntimeSnapshot 缺少插件组合 Root")
            catalog = root.context.get(MODEL_CATALOG)
            if catalog is None:
                raise ModelCatalogUnavailable("models 插件未提供模型目录")
            return catalog.snapshot()


def project_chat_runtimes(snapshot: ModelCatalogSnapshot) -> list[dict[str, object]]:
    """Project the provider-neutral catalog into the existing client DTO."""

    roles_by_model: dict[str, list[str]] = {}
    for role, model_id in snapshot.role_bindings.items():
        roles_by_model.setdefault(model_id, []).append(role.value)
    connections = {
        connection.connection_id: connection for connection in snapshot.connections
    }
    result: list[dict[str, object]] = []
    for model in snapshot.models:
        if (
            model.kind is not ModelKind.CHAT
            or model.availability is not ModelAvailability.AVAILABLE
        ):
            continue
        connection = connections[model.connection_id]
        capabilities = model.capabilities
        sources = model.capability_sources
        result.append(
            {
                "id": model.model_id,
                "provider": connection.driver_id,
                "catalogProvider": connection.driver_id,
                "model": model.model,
                "reasoningEffort": model.default_reasoning_effort or "",
                "supportedReasoningEfforts": list(
                    capabilities.supported_reasoning_efforts
                ),
                "sourceId": connection.connection_id,
                "sourceName": connection.name,
                "contextWindow": capabilities.context_window or 0,
                "maxOutputTokens": capabilities.max_output_tokens or 0,
                "inputModalities": list(capabilities.input_modalities),
                "capabilitySource": sources.context_window,
                "capabilitySources": {
                    "contextWindow": sources.context_window,
                    "maxOutputTokens": sources.max_output_tokens,
                    "inputModalities": sources.input_modalities,
                },
                "roles": sorted(roles_by_model.get(model.model_id, [])),
            }
        )
    return result


def default_chat_model_id(snapshot: ModelCatalogSnapshot) -> str:
    """Return the configured default model without inventing a fallback."""

    for role, model_id in snapshot.role_bindings.items():
        if role.value == "default":
            return model_id
    return ""


__all__ = [
    "RuntimeModelCatalogReader",
    "ModelCatalogUnavailable",
    "default_chat_model_id",
    "project_chat_runtimes",
]
