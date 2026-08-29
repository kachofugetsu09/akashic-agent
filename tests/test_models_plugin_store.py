from __future__ import annotations

import asyncio
import stat
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CapabilitySources,
    DisableConnection,
    DiscoveredModel,
    ModelCapabilities,
    ModelDriverDefinition,
    ModelKind,
    ModelUnavailableError,
    RevisionConflictError,
)
from agent.model_runtime.auth.store import Credential
from agent.model_runtime.store import ModelRegistryStore
from plugins.openai_compatible.driver import definition as openai_driver_definition
from plugins.models.store import ModelsStore
from plugins.models.state import ModelsState
import plugins.models.state as models_state_module


def _connection(
    revision: int,
    connection_id: str,
    *,
    token: str,
    endpoint: str = "https://example.test/v1",
) -> AddConnection:
    return AddConnection(
        expected_revision=revision,
        connection_id=connection_id,
        name=connection_id,
        driver_id="fake",
        endpoint=endpoint,
        auth_identity="shared-account",
        credential={"driver": "api_key", "access_token": token},
    )


@pytest.mark.asyncio
async def test_saved_model_service_rejects_another_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ModelsState(
        ModelsStore(
            tmp_path / "model-registry.sqlite3",
            backup_dir=tmp_path / "backups",
        ),
        root_instance_token=object(),
    )

    class Lease:
        def __init__(self, service: object) -> None:
            context = SimpleNamespace(
                root_instance_token=state.root_instance_token,
                get=lambda _key: service,
            )
            self.snapshot = SimpleNamespace(
                snapshot_id="other",
                composition_root=SimpleNamespace(context=context),
            )
            self.released = False

        async def release(self) -> None:
            self.released = True

    lease = Lease(object())
    monkeypatch.setattr(
        models_state_module,
        "lease_current_runtime_snapshot",
        lambda: lease,
    )

    with pytest.raises(RuntimeError, match="不属于当前 runtime snapshot"):
        async with state.chat_models.execution():
            pass
    assert lease.released is True


@pytest.mark.asyncio
async def test_model_service_requires_owner_task_snapshot_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ModelsState(
        ModelsStore(
            tmp_path / "model-registry.sqlite3",
            backup_dir=tmp_path / "backups",
        ),
        root_instance_token=object(),
    )
    monkeypatch.setattr(
        models_state_module,
        "lease_current_runtime_snapshot",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="当前 task"):
        async with state.embeddings.bind():
            pass


@pytest.mark.asyncio
async def test_legacy_openai_provider_ids_upgrade_to_ordinary_driver(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    path = workspace / "model-registry.sqlite3"
    legacy = ModelRegistryStore(path)
    revision = legacy.replace_from_llm_config(
        {
            "main": "openai-chat",
            "fast": "deepseek-chat",
            "agent": "qwen-chat",
            "runtimes": {
                "openai-chat": {
                    "provider": "openai",
                    "model": "gpt-test",
                    "source_id": "openai-source",
                    "auth": "openai-auth",
                    "base_url": "https://api.openai.com/v1",
                },
                "deepseek-chat": {
                    "provider": "deepseek",
                    "model": "deepseek-test",
                    "source_id": "deepseek-source",
                    "auth": "deepseek-auth",
                    "base_url": "https://api.deepseek.com/v1",
                },
                "qwen-chat": {
                    "provider": "qwen",
                    "model": "qwen-test",
                    "source_id": "qwen-source",
                    "auth": "qwen-auth",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                },
            },
        },
        credentials={
            auth_id: Credential(driver="api_key", access_token="secret")
            for auth_id in ("openai-auth", "deepseek-auth", "qwen-auth")
        },
    )
    assert revision == 1

    store = ModelsStore(
        path,
        backup_dir=workspace / "runtime" / "model-backups",
    )
    store.initialize()
    snapshot = store.read_snapshot()
    assert snapshot is not None
    assert {connection.driver_id for connection in snapshot.connections.values()} == {
        "openai-compatible"
    }

    state = ModelsState(store, root_instance_token=object())
    definition = openai_driver_definition()
    state._driver_registrations[definition.driver_id] = definition  # noqa: SLF001
    await state.seal(None)  # type: ignore[arg-type]
    assert all(
        connection.availability.value == "available"
        for connection in state.catalog_snapshot().connections
    )
    assert len(tuple(store.backup_dir.glob("*.sqlite3"))) == 1


@pytest.mark.asyncio
async def test_historical_text_vision_binding_fails_before_driver_open(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    revision = store.add_connection(
        _connection(
            0,
            "connection",
            token="one",
            endpoint="https://user:public-catalog-secret@example.test/v1?key=hidden",
        )
    )
    _ = store.add_model(
        AddModel(
            expected_revision=revision,
            model_id="text-chat",
            connection_id="connection",
            kind=ModelKind.CHAT,
            model="text-wire",
            capabilities=ModelCapabilities(input_modalities=("text",)),
            capability_sources=CapabilitySources(input_modalities="legacy"),
        )
    )
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "INSERT INTO model_role_bindings(role, model_id, reasoning_effort) "
            "VALUES ('vision', 'text-chat', '')"
        )
        connection.commit()

    opens = 0

    async def open_driver(*_args: object) -> Any:
        nonlocal opens
        opens += 1
        return object()

    state = ModelsState(store, root_instance_token=object())
    state._driver_registrations["fake"] = ModelDriverDefinition(  # noqa: SLF001
        driver_id="fake",
        contract_version="1",
        open=open_driver,
    )

    with pytest.raises(ModelUnavailableError, match="image-capable"):
        await state.seal(None)  # type: ignore[arg-type]

    assert opens == 0
    assert state.sealed is False


@pytest.mark.asyncio
async def test_credentials_are_connection_scoped_and_refresh_keeps_revision(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    revision = store.add_connection(_connection(0, "first", token="one"))
    revision = store.add_connection(_connection(revision, "second", token="two"))
    first = store.credential_handle("first", "shared-account")
    second = store.credential_handle("second", "shared-account")

    await first.refresh({"driver": "api_key", "access_token": "rotated"})

    assert (await first.read())["access_token"] == "rotated"
    assert (await second.read())["access_token"] == "two"
    assert store.read_snapshot().revision == revision  # type: ignore[union-attr]

    revision = store.disable_connection(
        DisableConnection(expected_revision=revision, connection_id="first")
    )
    await first.refresh({"driver": "api_key", "access_token": "draining"})
    assert (await first.read())["access_token"] == "draining"
    assert store.read_snapshot().revision == revision  # type: ignore[union-attr]

    with pytest.raises(RevisionConflictError):
        store.disable_connection(
            DisableConnection(expected_revision=0, connection_id="second")
        )

    database_mode = stat.S_IMODE(store.path.stat().st_mode)
    assert database_mode == 0o600
    backups = tuple(store.backup_dir.glob("*.sqlite3"))
    assert len(backups) >= 5
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in backups)


@pytest.mark.asyncio
async def test_credential_exclusive_is_cross_process_and_cancel_safe(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    _ = store.add_connection(_connection(0, "connection", token="one"))
    handle = store.credential_handle("connection", "shared-account")
    lock_path = store.path.with_name(f"{store.path.name}.credentials.lock")
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys; "
                "f=pathlib.Path(sys.argv[1]).open('a+'); "
                "fcntl.flock(f.fileno(), fcntl.LOCK_EX); "
                "print('locked', flush=True); input(); f.close()"
            ),
            str(lock_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "locked"

    entered = asyncio.Event()

    async def wait_for_lock() -> None:
        async with handle.exclusive():
            entered.set()

    waiter = asyncio.create_task(wait_for_lock())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not entered.is_set()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert child.stdin is not None
    child.stdin.write("\n")
    child.stdin.flush()
    assert child.wait(timeout=5) == 0
    child.stdin.close()
    child.stdout.close()
    async with handle.exclusive():
        assert True

    read_only = ModelsStore(
        store.path,
        backup_dir=store.backup_dir,
        writable=False,
    ).credential_handle("connection", "shared-account")
    lock_path.unlink()
    with pytest.raises(PermissionError, match="read-only"):
        async with read_only.exclusive():
            pass
    assert not lock_path.exists()


def test_model_capabilities_sources_and_driver_config_round_trip(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    revision = store.add_connection(_connection(0, "connection", token="one"))
    capabilities = ModelCapabilities(
        context_window=1234,
        max_output_tokens=234,
        input_modalities=("text", "image"),
        supports_tool_calls=False,
        supports_parallel_tool_calls=False,
        supported_reasoning_efforts=("low", "high"),
    )
    sources = CapabilitySources(
        context_window="context-source",
        max_output_tokens="output-source",
        input_modalities="modalities-source",
        tool_calls="tool-source",
        parallel_tool_calls="parallel-source",
        reasoning_efforts="reasoning-source",
    )
    revision = store.add_model(
        AddModel(
            expected_revision=revision,
            model_id="chat",
            connection_id="connection",
            kind=ModelKind.CHAT,
            model="wire-chat",
            capabilities=capabilities,
            capability_sources=sources,
            default_reasoning_effort="high",
            driver_config={
                "use_responses_lite": True,
                "reasoning_summary": "auto",
                "nested": {"value": [1, 2]},
            },
        )
    )
    revision = store.add_model(
        AddModel(
            expected_revision=revision,
            model_id="embedding",
            connection_id="connection",
            kind=ModelKind.EMBEDDING,
            model="wire-embedding",
            capabilities=ModelCapabilities(
                embedding_dimensions=3,
                embedding_normalization="l2",
            ),
            capability_sources=CapabilitySources(
                embedding_dimensions="dimension-source",
                embedding_normalization="normalization-source",
            ),
            driver_config={"batch_size": 8},
        )
    )

    snapshot = store.read_snapshot()
    assert snapshot is not None and snapshot.revision == revision
    chat = snapshot.models["chat"]
    assert chat.capabilities == capabilities
    assert chat.capability_sources == sources
    assert chat.default_reasoning_effort == "high"
    assert chat.driver_config["nested"]["value"] == (1, 2)  # type: ignore[index]
    embedding = snapshot.models["embedding"]
    assert embedding.capabilities.embedding_normalization == "l2"
    assert (
        embedding.capability_sources.embedding_normalization == "normalization-source"
    )
    assert snapshot.connections["connection"].driver_config == {}
    public_catalog = ModelsState(
        store,
        root_instance_token=object(),
    ).catalog_snapshot()
    assert "public-catalog-secret" not in repr(public_catalog)
    assert not hasattr(public_catalog.connections[0], "endpoint")

    with pytest.raises(ValueError, match="finite"):
        store.add_model(
            AddModel(
                expected_revision=revision,
                model_id="invalid",
                connection_id="connection",
                kind=ModelKind.CHAT,
                model="invalid",
                capabilities=ModelCapabilities(),
                capability_sources=CapabilitySources(),
                driver_config={"bad": float("nan")},
            )
        )


def test_connection_legacy_provider_column_matches_driver_config(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    revision = store.add_connection(
        AddConnection(
            expected_revision=0,
            connection_id="connection",
            name="connection",
            driver_id="openai-compatible",
            endpoint="https://example.test/v1",
            auth_identity="account",
            credential={"driver": "api_key", "access_token": "secret"},
            driver_config={"catalog_provider_id": "deepseek"},
        )
    )

    snapshot = store.read_snapshot()
    assert snapshot is not None and snapshot.revision == revision
    assert snapshot.connections["connection"].driver_config == {
        "catalog_provider_id": "deepseek"
    }
    with closing(sqlite3.connect(store.path)) as connection:
        row = connection.execute(
            "SELECT catalog_provider_id FROM model_connections WHERE id = 'connection'"
        ).fetchone()
    assert row == ("deepseek",)
    with pytest.raises(ValueError, match="outer whitespace"):
        store.add_connection(
            AddConnection(
                expected_revision=revision,
                connection_id="invalid",
                name="invalid",
                driver_id="openai-compatible",
                endpoint="https://example.test/v1",
                auth_identity="account",
                credential={"driver": "api_key", "access_token": "secret"},
                driver_config={"catalog_provider_id": " deepseek "},
            )
        )


def test_model_ids_are_unique_across_kinds_and_corruption_fails_loud(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    revision = store.add_connection(_connection(0, "connection", token="one"))
    revision = store.add_model(
        AddModel(
            expected_revision=revision,
            model_id="same",
            connection_id="connection",
            kind=ModelKind.CHAT,
            model="chat-wire",
            capabilities=ModelCapabilities(),
            capability_sources=CapabilitySources(),
        )
    )
    with pytest.raises(ValueError, match="already exists"):
        store.add_model(
            AddModel(
                expected_revision=revision,
                model_id="same",
                connection_id="connection",
                kind=ModelKind.EMBEDDING,
                model="embedding-wire",
                capabilities=ModelCapabilities(embedding_dimensions=3),
                capability_sources=CapabilitySources(),
            )
        )

    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute(
            "INSERT INTO embedding_models(id, connection_id, model, dimensions) "
            "VALUES ('same', 'connection', 'embedding-wire', 3)"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="duplicate model id across kinds"):
        store.read_snapshot()


def test_discovery_sync_is_one_revision_and_preserves_store_owned_id(
    tmp_path: Path,
) -> None:
    store = ModelsStore(
        tmp_path / "workspace" / "model-registry.sqlite3",
        backup_dir=tmp_path / "workspace" / "runtime" / "model-backups",
    )
    revision = store.add_connection(_connection(0, "connection", token="one"))
    first = DiscoveredModel(
        kind=ModelKind.CHAT,
        model="wire-model",
        capabilities=ModelCapabilities(context_window=100),
        capability_sources=CapabilitySources(context_window="provider"),
        driver_config={"profile": "first"},
    )
    revision = store.sync_models(revision, "connection", (first,))
    snapshot = store.read_snapshot()
    assert snapshot is not None and snapshot.revision == revision
    model_id = next(iter(snapshot.models))
    assert model_id == "discovered:10:connection4:chat10:wire-model"

    updated = DiscoveredModel(
        kind=ModelKind.CHAT,
        model="wire-model",
        capabilities=ModelCapabilities(context_window=200),
        capability_sources=CapabilitySources(context_window="provider-refresh"),
        driver_config={"profile": "second"},
    )
    revision = store.sync_models(revision, "connection", (updated,))
    snapshot = store.read_snapshot()
    assert snapshot is not None and snapshot.revision == revision
    assert tuple(snapshot.models) == (model_id,)
    assert snapshot.models[model_id].capabilities.context_window == 200
    assert snapshot.models[model_id].driver_config == {"profile": "second"}

    extra = DiscoveredModel(
        kind=ModelKind.CHAT,
        model="removed-wire",
        capabilities=ModelCapabilities(context_window=50),
        capability_sources=CapabilitySources(context_window="provider"),
    )
    revision = store.sync_models(revision, "connection", (updated, extra))
    removed_id = "discovered:10:connection4:chat12:removed-wire"
    assert store.read_snapshot().models[removed_id].enabled is True  # type: ignore[union-attr]
    revision = store.sync_models(revision, "connection", (updated,))
    assert store.read_snapshot().models[removed_id].enabled is False  # type: ignore[union-attr]
    backups_before = tuple(store.backup_dir.glob("*.sqlite3"))
    assert store.sync_models(revision, "connection", (updated,)) == revision
    assert tuple(store.backup_dir.glob("*.sqlite3")) == backups_before

    revision = store.add_model(
        AddModel(
            expected_revision=revision,
            model_id="manual",
            connection_id="connection",
            kind=ModelKind.CHAT,
            model="manual-wire",
            capabilities=ModelCapabilities(context_window=777),
            capability_sources=CapabilitySources(context_window="manual"),
            driver_config={"profile": "manual"},
        )
    )
    revision = store.sync_models(
        revision,
        "connection",
        (
            updated,
            DiscoveredModel(
                kind=ModelKind.CHAT,
                model="manual-wire",
                capabilities=ModelCapabilities(context_window=999),
                capability_sources=CapabilitySources(context_window="provider"),
                driver_config={"profile": "provider"},
            ),
        ),
    )
    snapshot = store.read_snapshot()
    assert snapshot is not None
    assert snapshot.models["manual"].capabilities.context_window == 777
    assert snapshot.models["manual"].driver_config == {"profile": "manual"}

    with pytest.raises(ValueError, match="duplicate model"):
        store.sync_models(revision, "connection", (updated, updated))
    assert store.read_snapshot().revision == revision  # type: ignore[union-attr]

    invalid_items = (
        (),
        (
            DiscoveredModel(
                kind="chat",  # type: ignore[arg-type]
                model="invalid-kind",
                capabilities=ModelCapabilities(),
                capability_sources=CapabilitySources(),
            ),
        ),
        (
            DiscoveredModel(
                kind=ModelKind.CHAT,
                model=" padded ",
                capabilities=ModelCapabilities(),
                capability_sources=CapabilitySources(),
            ),
        ),
        (
            DiscoveredModel(
                kind=ModelKind.CHAT,
                model="invalid-capabilities",
                capabilities={},  # type: ignore[arg-type]
                capability_sources=CapabilitySources(),
            ),
        ),
    )
    for invalid in invalid_items:
        with pytest.raises((TypeError, ValueError)):
            store.sync_models(revision, "connection", invalid)  # type: ignore[arg-type]

    revision = store.disable_connection(
        DisableConnection(expected_revision=revision, connection_id="connection")
    )
    with pytest.raises(ValueError, match="disabled"):
        store.sync_models(revision, "connection", (updated,))
