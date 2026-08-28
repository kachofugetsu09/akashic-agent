from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    MODEL_CATALOG,
    CapabilitySources,
    ConnectionDescriptor,
    ModelAvailability,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelDescriptor,
    ModelKind,
    ModelRole,
)
from agent.plugins.model_catalog import (
    ModelCatalogUnavailable,
    RuntimeModelCatalogReader,
    default_chat_model_id,
    project_chat_runtimes,
)


def _catalog() -> ModelCatalogSnapshot:
    return ModelCatalogSnapshot(
        revision=9,
        connections=(
            ConnectionDescriptor(
                connection_id="connection-a",
                name="Account A",
                driver_id="driver-a",
                endpoint="https://example.test/v1",
                auth_identity="account-a",
                availability=ModelAvailability.AVAILABLE,
            ),
        ),
        models=(
            ModelDescriptor(
                model_id="chat-a",
                connection_id="connection-a",
                kind=ModelKind.CHAT,
                model="wire-a",
                default_reasoning_effort="high",
                capabilities=ModelCapabilities(
                    context_window=32_000,
                    supported_reasoning_efforts=("medium", "high"),
                ),
                capability_sources=CapabilitySources(context_window="catalog"),
                availability=ModelAvailability.AVAILABLE,
            ),
            ModelDescriptor(
                model_id="chat-disabled",
                connection_id="connection-a",
                kind=ModelKind.CHAT,
                model="wire-disabled",
                default_reasoning_effort=None,
                capabilities=ModelCapabilities(),
                capability_sources=CapabilitySources(),
                availability=ModelAvailability.DISABLED,
            ),
        ),
        role_bindings={ModelRole.DEFAULT: "chat-a", ModelRole.AGENT: "chat-a"},
        default_embedding_model_id=None,
    )


def test_catalog_projection_keeps_client_shape_without_unavailable_models() -> None:
    snapshot = _catalog()
    assert default_chat_model_id(snapshot) == "chat-a"
    assert project_chat_runtimes(snapshot) == [
        {
            "id": "chat-a",
            "provider": "driver-a",
            "catalogProvider": "driver-a",
            "model": "wire-a",
            "reasoningEffort": "high",
            "supportedReasoningEfforts": ["medium", "high"],
            "sourceId": "connection-a",
            "sourceName": "Account A",
            "contextWindow": 32_000,
            "maxOutputTokens": 0,
            "inputModalities": ["text"],
            "capabilitySource": "catalog",
            "capabilitySources": {
                "contextWindow": "catalog",
                "maxOutputTokens": "unknown",
                "inputModalities": "unknown",
            },
            "roles": ["agent", "default"],
        }
    ]


@pytest.mark.asyncio
async def test_reader_holds_one_snapshot_lease_for_the_read() -> None:
    catalog = _catalog()
    released = False

    class Lease:
        async def __aenter__(self) -> object:
            service = SimpleNamespace(snapshot=lambda: catalog)
            context = SimpleNamespace(
                get=lambda key: service if key is MODEL_CATALOG else None
            )
            return SimpleNamespace(composition_root=SimpleNamespace(context=context))

        async def __aexit__(self, *_args: object) -> None:
            nonlocal released
            released = True

    class Store:
        async def acquire(self) -> Lease:
            return Lease()

    reader = RuntimeModelCatalogReader(cast(Any, Store()))
    assert await reader.read() is catalog
    assert released


@pytest.mark.asyncio
async def test_reader_reports_missing_models_service_and_releases_lease() -> None:
    released = False
    service: object | None = SimpleNamespace(snapshot=_catalog)

    class Lease:
        async def __aenter__(self) -> object:
            context = SimpleNamespace(get=lambda _key: service)
            return SimpleNamespace(composition_root=SimpleNamespace(context=context))

        async def __aexit__(self, *_args: object) -> None:
            nonlocal released
            released = True

    class Store:
        async def acquire(self) -> Lease:
            return Lease()

    reader = RuntimeModelCatalogReader(cast(Any, Store()))
    assert (await reader.read()).revision == 9
    service = None
    released = False
    with pytest.raises(ModelCatalogUnavailable, match="models 插件"):
        await reader.read()
    assert released
