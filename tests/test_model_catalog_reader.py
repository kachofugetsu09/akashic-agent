from __future__ import annotations

from agent.plugin_composition import (
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
