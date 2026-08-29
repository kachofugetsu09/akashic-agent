from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tomllib
from contextlib import closing
from pathlib import Path
from types import ModuleType

import yoyo
import pytest

from agent.migrations.context import bind_migration_context
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.store import ModelRegistryStore
from plugins.models.store import ModelsStore
from plugins.models.state import ModelsState
from plugins.openai_compatible.driver import definition as openai_driver_definition

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/yoyo/20260829_03_retire_core_model_config.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("retire_core_model_config", MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _legacy_registry(workspace: Path) -> None:
    _ = ModelRegistryStore.for_workspace(workspace).replace_from_llm_config(
        {
            "main": "chat",
            "runtimes": {
                "chat": {
                    "provider": "openai",
                    "model": "test-model",
                    "source_id": "source",
                    "auth": "source",
                    "base_url": "https://example.test/v1",
                }
            },
        },
        credentials={
            "source": Credential(driver="api_key", access_token="chat-secret")
        },
    )


def _legacy_sidecars(workspace: Path) -> tuple[Path, Path]:
    memory_root = workspace / "memory"
    memory_root.mkdir()
    paths = memory_root / "akasha-v2-index.db", memory_root / "akasha.db"
    for path in paths:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE state(value TEXT NOT NULL)")
            connection.execute("INSERT INTO state VALUES ('preserved-in-backup')")
            connection.commit()
    return paths


@pytest.mark.asyncio
async def test_retires_llm_and_memory_with_recoverable_backup(
    tmp_path: Path,
) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "config.toml"
    original = b"""# keep this operator comment
[agent]
max_iterations = 4
[llm]
registry = "workspace"
[memory]
enabled = true
[memory.embedding]
model = "text-embedding-v4"
base_url = "https://embedding.example.test/v1"
api_key = "embedding-secret"
"""
    config.write_bytes(original)
    os.chmod(config, 0o640)
    _legacy_registry(workspace)
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        connection.execute(
            "CREATE TABLE message_embeddings(model TEXT NOT NULL, dim INTEGER NOT NULL)"
        )
        connection.execute(
            "INSERT INTO message_embeddings(model, dim) VALUES (?, ?)",
            ("text-embedding-v4", 1024),
        )
        connection.commit()
    index_path, memory_path = _legacy_sidecars(workspace)

    with bind_migration_context(config_path=config, workspace=workspace):
        module.retire_core_model_config(object())

    assert tomllib.loads(config.read_text(encoding="utf-8")) == {
        "agent": {"max_iterations": 4}
    }
    assert "# keep this operator comment" in config.read_text(encoding="utf-8")
    assert config.stat().st_mode & 0o777 == 0o640
    backup_roots = tuple((workspace / "backups/retire-core-model-config").iterdir())
    assert len(backup_roots) == 1
    config_backup = backup_roots[0] / "config.toml.before"
    registry_backup = backup_roots[0] / "model-registry.before.sqlite3"
    assert config_backup.read_bytes() == original
    assert config_backup.stat().st_mode & 0o777 == 0o600
    assert ModelRegistryStore(registry_backup).list_embedding_models() == ()
    assert (backup_roots[0] / "sessions.before.sqlite3").is_file()
    assert not (backup_roots[0] / "akasha-index.before.sqlite3").exists()
    assert not (backup_roots[0] / "akasha-memory.before.sqlite3").exists()

    store = ModelsStore(
        workspace / "model-registry.sqlite3",
        backup_dir=workspace / "runtime/model-backups",
    )
    store.initialize()
    migrated = store.read_snapshot()
    assert migrated is not None
    assert migrated.default_embedding_model_id == "legacy_memory_embedding"
    embedding = migrated.models["legacy_memory_embedding"]
    assert embedding.model == "text-embedding-v4"
    assert embedding.capabilities.embedding_dimensions == 1024
    assert (
        migrated.connections[embedding.connection_id].driver_id == "openai-compatible"
    )
    assert (
        CredentialStore.for_workspace(workspace).api_key("legacy_memory_embedding")
        == "embedding-secret"
    )
    state = ModelsState(store, root_instance_token=object())
    driver = openai_driver_definition()
    state._driver_registrations[driver.driver_id] = driver  # noqa: SLF001
    await state.seal(None)  # type: ignore[arg-type]
    identity = state.describe_embedding(None).identity
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        assert connection.execute(
            "SELECT DISTINCT model FROM message_embeddings"
        ).fetchall() == [(identity,)]
    for path in (index_path, memory_path):
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute("SELECT value FROM state").fetchone() == (
                "preserved-in-backup",
            )


def test_preserves_config_symlink_identity(tmp_path: Path) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "shared.toml"
    target.write_text(
        '[agent]\nmax_iterations = 4\n[llm]\nregistry = "workspace"\n', encoding="utf-8"
    )
    _legacy_registry(workspace)
    config = workspace / "config.toml"
    config.symlink_to(target)

    with bind_migration_context(config_path=config, workspace=workspace):
        module.retire_core_model_config(object())

    assert config.is_symlink() and os.readlink(config) == str(target)
    assert "llm" not in tomllib.loads(config.read_text(encoding="utf-8"))


def test_rejects_malformed_llm_without_backup_or_mutation(tmp_path: Path) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "config.toml"
    original = b"llm = []\n[agent]\nmax_iterations = 4\n"
    config.write_bytes(original)

    with bind_migration_context(config_path=config, workspace=workspace):
        with pytest.raises(RuntimeError, match="handoff"):
            module.retire_core_model_config(object())

    assert config.read_bytes() == original
    assert not (workspace / "backups/retire-core-model-config").exists()


def test_rejects_inline_embedding_without_an_exact_dimension(tmp_path: Path) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "config.toml"
    original = b"""[llm]
registry = "workspace"
[memory]
enabled = true
[memory.embedding]
model = "custom-embedding"
base_url = "https://embedding.example.test/v1"
api_key = "secret"
"""
    config.write_bytes(original)
    _legacy_registry(workspace)

    with bind_migration_context(config_path=config, workspace=workspace):
        with pytest.raises(RuntimeError, match="不能确定唯一维度"):
            module.retire_core_model_config(object())

    assert config.read_bytes() == original
    assert ModelRegistryStore.for_workspace(workspace).list_embedding_models() == ()
    assert not (workspace / "backups/retire-core-model-config").exists()


def test_restores_registry_when_config_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "config.toml"
    original = b"""[llm]
registry = "workspace"
[memory]
enabled = true
[memory.embedding]
model = "text-embedding-v4"
base_url = "https://embedding.example.test/v1"
api_key = "secret"
output_dimensionality = 1024
"""
    config.write_bytes(original)
    _legacy_registry(workspace)
    original_write = module._write_atomic
    failed = False

    def fail_config_publish(path: Path, content: bytes, mode: int) -> None:
        nonlocal failed
        if path == config and b"[memory]" not in content and not failed:
            failed = True
            raise OSError("publish failed")
        original_write(path, content, mode)

    monkeypatch.setattr(module, "_write_atomic", fail_config_publish)
    with bind_migration_context(config_path=config, workspace=workspace):
        with pytest.raises(OSError, match="publish failed"):
            module.retire_core_model_config(object())

    assert config.read_bytes() == original
    store = ModelRegistryStore.for_workspace(workspace)
    assert store.list_embedding_models() == ()
    with closing(sqlite3.connect(store.path)) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(model_registry_meta)")
        }
    assert "default_embedding_model_id" not in columns
