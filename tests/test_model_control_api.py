from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from agent.plugin_composition import (
    AddConnection,
    MODEL_CATALOG,
    MODEL_SETTINGS,
    CapabilitySources,
    ConnectionDescriptor,
    CreateConnectionWithModel,
    DiscoveredModel,
    ModelAvailability,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelDescriptor,
    ModelKind,
    ModelRole,
    SetDefaultModel,
    SettingsReceipt,
    UpdateConnection,
)
from agent.plugins.model_control import ModelControlUnavailable, RuntimeModelControl
from agent.plugins.snapshot import get_current_runtime_snapshot
from bootstrap.chat_api import create_chat_app
from infra.channels.web_chat_channel import WebChatChannel


def _catalog() -> ModelCatalogSnapshot:
    return ModelCatalogSnapshot(
        revision=7,
        connections=(
            ConnectionDescriptor(
                connection_id="account-a",
                name="Account A",
                driver_id="openai-compatible",
                auth_identity="key-a",
                availability=ModelAvailability.AVAILABLE,
            ),
        ),
        models=(
            ModelDescriptor(
                model_id="chat-a",
                connection_id="account-a",
                kind=ModelKind.CHAT,
                model="wire-chat",
                default_reasoning_effort="high",
                capabilities=ModelCapabilities(
                    context_window=64_000,
                    input_modalities=("text", "image"),
                    supports_tool_calls=True,
                ),
                capability_sources=CapabilitySources(context_window="catalog"),
                availability=ModelAvailability.AVAILABLE,
            ),
        ),
        role_bindings={ModelRole.DEFAULT: "chat-a"},
        default_embedding_model_id=None,
    )


class _Lease:
    def __init__(self, root: object) -> None:
        self.snapshot = SimpleNamespace(composition_root=root)
        self.active = True

    def fork(self) -> _Lease:
        return self

    async def release(self) -> None:
        self.active = False


@pytest.mark.asyncio
async def test_runtime_model_control_binds_and_releases_exact_snapshot() -> None:
    catalog = _catalog()
    commands: list[object] = []

    class Settings:
        async def discover(
            self,
            connection: AddConnection,
        ) -> tuple[DiscoveredModel, ...]:
            assert get_current_runtime_snapshot() is lease.snapshot
            return (
                DiscoveredModel(
                    kind=ModelKind.CHAT,
                    model=connection.name,
                    capabilities=ModelCapabilities(),
                    capability_sources=CapabilitySources(),
                ),
            )

        async def apply(self, command: object) -> SettingsReceipt:
            assert get_current_runtime_snapshot() is lease.snapshot
            commands.append(command)
            return SettingsReceipt(revision=8, status="committed")

    def read_catalog() -> ModelCatalogSnapshot:
        assert get_current_runtime_snapshot() is lease.snapshot
        return catalog

    services = {
        MODEL_CATALOG: SimpleNamespace(snapshot=read_catalog),
        MODEL_SETTINGS: Settings(),
    }
    root = SimpleNamespace(context=SimpleNamespace(get=services.get))
    lease = _Lease(root)

    class Store:
        async def acquire(self) -> _Lease:
            lease.active = True
            return lease

    control = RuntimeModelControl(cast(Any, Store()))
    assert await control.catalog() is catalog
    assert not lease.active
    preview = AddConnection(
        expected_revision=7,
        connection_id="preview",
        name="preview-model",
        driver_id="openai-compatible",
        endpoint="https://example.test/v1",
        auth_identity="preview",
        credential={"access_token": "temporary"},
    )
    assert (await control.discover(preview))[0].model == "preview-model"
    assert not lease.active
    receipt = await control.apply(SetDefaultModel(7, ModelRole.DEFAULT, "chat-a"))
    assert receipt.revision == 8
    assert commands == [SetDefaultModel(7, ModelRole.DEFAULT, "chat-a")]
    assert not lease.active
    assert get_current_runtime_snapshot() is None


@pytest.mark.asyncio
async def test_runtime_model_control_reports_missing_service_and_releases() -> None:
    root = SimpleNamespace(context=SimpleNamespace(get=lambda _key: None))
    lease = _Lease(root)

    class Store:
        async def acquire(self) -> _Lease:
            return lease

    control = RuntimeModelControl(cast(Any, Store()))
    with pytest.raises(ModelControlUnavailable, match="模型目录"):
        await control.catalog()
    assert not lease.active
    assert get_current_runtime_snapshot() is None


@pytest.mark.asyncio
async def test_cancelled_discovery_releases_runtime_snapshot() -> None:
    started = asyncio.Event()

    class Settings:
        async def discover(self, _connection: AddConnection):
            started.set()
            await asyncio.Event().wait()

    services = {MODEL_SETTINGS: Settings()}
    root = SimpleNamespace(context=SimpleNamespace(get=services.get))
    lease = _Lease(root)

    class Store:
        async def acquire(self) -> _Lease:
            lease.active = True
            return lease

    control = RuntimeModelControl(cast(Any, Store()))
    connection = AddConnection(
        expected_revision=0,
        connection_id="preview",
        name="Preview",
        driver_id="openai-compatible",
        endpoint="https://example.test/v1",
        auth_identity="preview",
        credential={"access_token": "temporary"},
    )
    task = asyncio.create_task(control.discover(connection))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not lease.active
    assert get_current_runtime_snapshot() is None


def test_model_settings_http_projects_catalog_and_validates_command(tmp_path) -> None:
    applied: list[object] = []

    class Control:
        async def catalog(self) -> ModelCatalogSnapshot:
            return _catalog()

        async def discover(
            self,
            connection: AddConnection,
        ) -> tuple[DiscoveredModel, ...]:
            assert connection.endpoint == "https://example.test/v1"
            assert connection.credential == {"access_token": "temporary"}
            return (
                DiscoveredModel(
                    kind=ModelKind.CHAT,
                    model="deepseek-v4-flash-vision",
                    capabilities=ModelCapabilities(
                        context_window=1_000_000,
                        input_modalities=("text", "image"),
                        supports_tool_calls=True,
                    ),
                    capability_sources=CapabilitySources(
                        context_window="litellm-remote@sha256:test",
                        input_modalities="litellm-remote@sha256:test",
                        tool_calls="litellm-remote@sha256:test",
                    ),
                    driver_config={"format_version": 1},
                ),
            )

        async def apply(self, command: object) -> SettingsReceipt:
            applied.append(command)
            return SettingsReceipt(revision=8, status="committed")

    app = create_chat_app(
        workspace=tmp_path,
        channel=WebChatChannel(),
        model_control=cast(Any, Control()),
    )
    client = TestClient(app)

    catalog = client.get("/api/chat/model-settings/catalog")
    discovered = client.post(
        "/api/chat/model-settings/discover",
        json={
            "expected_revision": 7,
            "connection_id": "temporary-account",
            "name": "Temporary",
            "driver_id": "openai-compatible",
            "endpoint": "https://example.test/v1",
            "auth_identity": "temporary-account",
            "credential": {"access_token": "temporary"},
            "driver_config": {"catalog_provider_id": "deepseek"},
        },
    )
    response = client.post(
        "/api/chat/model-settings/command",
        json={
            "type": "set_default",
            "expected_revision": 7,
            "role": "vision",
            "model_id": "chat-a",
        },
    )

    assert catalog.status_code == 200
    assert discovered.status_code == 200
    assert discovered.json()["models"][0] == {
        "kind": "chat",
        "model": "deepseek-v4-flash-vision",
        "defaultReasoningEffort": None,
        "capabilities": {
            "contextWindow": 1_000_000,
            "maxOutputTokens": None,
            "inputModalities": ["text", "image"],
            "supportsToolCalls": True,
            "supportsParallelToolCalls": None,
            "supportedReasoningEfforts": [],
            "embeddingDimensions": None,
            "embeddingNormalization": None,
        },
        "capabilitySources": {
            "contextWindow": "litellm-remote@sha256:test",
            "maxOutputTokens": "unknown",
            "inputModalities": "litellm-remote@sha256:test",
            "toolCalls": "litellm-remote@sha256:test",
            "parallelToolCalls": "unknown",
            "reasoningEfforts": "unknown",
            "embeddingDimensions": "unknown",
            "embeddingNormalization": "unknown",
        },
        "driverConfig": {"format_version": 1},
    }
    invalid_discovery = client.post(
        "/api/chat/model-settings/discover",
        json={
            "expected_revision": 7,
            "connection_id": "temporary-account",
            "name": "Temporary",
            "driver_id": "openai-compatible",
            "endpoint": "https://example.test/v1",
            "auth_identity": "temporary-account",
            "credential": {"access_token": "must-not-leak"},
            "unexpected": True,
        },
    )
    assert invalid_discovery.status_code == 422
    assert "must-not-leak" not in invalid_discovery.text
    unsupported_discovery = client.post(
        "/api/chat/model-settings/discover",
        json={
            "expected_revision": 7,
            "connection_id": "temporary-account",
            "name": "Temporary",
            "driver_id": "opencode-go",
            "endpoint": "https://example.test/v1",
            "auth_identity": "temporary-account",
            "credential": {"access_token": "must-not-leak"},
            "driver_config": {
                "max_retries": 1_000_000,
                "connect_timeout": 1_000_000,
                "read_timeout": 1_000_000,
            },
        },
    )
    assert unsupported_discovery.status_code == 422
    assert unsupported_discovery.json()["detail"] == (
        "模型预览仅支持 openai-compatible"
    )
    assert "must-not-leak" not in unsupported_discovery.text
    assert catalog.json()["connections"][0]["driverId"] == "openai-compatible"
    assert "endpoint" not in catalog.json()["connections"][0]
    assert catalog.json()["models"][0]["capabilities"]["inputModalities"] == [
        "text",
        "image",
    ]
    assert response.json() == {
        "revision": 8,
        "status": "committed",
        "attemptId": None,
        "challenge": None,
    }
    assert applied == [SetDefaultModel(7, ModelRole.VISION, "chat-a")]

    retained = client.post(
        "/api/chat/model-settings/command",
        json={
            "type": "update_connection",
            "expected_revision": 8,
            "connection_id": "account-a",
            "name": "Renamed",
            "auth_identity": "key-a",
        },
    )
    assert retained.status_code == 200
    assert applied[-1] == UpdateConnection(
        expected_revision=8,
        connection_id="account-a",
        name="Renamed",
        auth_identity="key-a",
        endpoint=None,
    )

    invalid = client.post(
        "/api/chat/model-settings/command",
        json={
            "type": "set_default",
            "expected_revision": 8,
            "role": "default",
            "model_id": "chat-a",
            "provider": "special-case",
        },
    )
    assert invalid.status_code == 422

    secret_invalid = client.post(
        "/api/chat/model-settings/command",
        json={
            "type": "add_connection",
            "expected_revision": 8,
            "connection_id": "account-b",
            "name": "Account B",
            "driver_id": "openai-compatible",
            "endpoint": "https://example.test/v1",
            "auth_identity": "account-b",
            "credential": {"access_token": "must-not-leak"},
            "unexpected": True,
        },
    )
    assert secret_invalid.status_code == 422
    assert "must-not-leak" not in secret_invalid.text

    created = client.post(
        "/api/chat/model-settings/command",
        json={
            "type": "create_connection_with_model",
            "connection": {
                "expected_revision": 8,
                "connection_id": "account-b",
                "name": "Account B",
                "driver_id": "openai-compatible",
                "endpoint": "https://example.test/v1",
                "auth_identity": "account-b",
                "credential": {"access_token": "safe"},
            },
            "model": {
                "expected_revision": 8,
                "model_id": "chat-b",
                "connection_id": "account-b",
                "kind": "chat",
                "model": "wire-b",
                "capabilities": {
                    "input_modalities": ["text", "image"],
                    "supported_reasoning_efforts": ["high"],
                },
                "capability_sources": {},
            },
        },
    )
    assert created.status_code == 200
    assert isinstance(applied[-1], CreateConnectionWithModel)
    assert applied[-1].model.capabilities.input_modalities == ("text", "image")
    assert applied[-1].model.capabilities.supported_reasoning_efforts == ("high",)
