from __future__ import annotations

import asyncio
import stat
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CapabilitySources,
    DisableConnection,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
    RevisionConflictError,
)
from plugins.models.store import ModelsStore


def _connection(
    revision: int,
    connection_id: str,
    *,
    token: str,
) -> AddConnection:
    return AddConnection(
        expected_revision=revision,
        connection_id=connection_id,
        name=connection_id,
        driver_id="fake",
        endpoint="https://example.test/v1",
        auth_identity="shared-account",
        credential={"driver": "api_key", "access_token": token},
    )


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


def test_model_capabilities_sources_and_driver_config_round_trip(tmp_path: Path) -> None:
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
    assert embedding.capability_sources.embedding_normalization == "normalization-source"
    assert snapshot.connections["connection"].driver_config == {}

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


def test_connection_legacy_provider_column_matches_driver_config(tmp_path: Path) -> None:
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
