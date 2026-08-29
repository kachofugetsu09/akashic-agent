from __future__ import annotations

import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import yoyo
import pytest

from agent.migrations.context import bind_migration_context
from agent.model_runtime.store import ModelRegistryStore

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
                    "base_url": "https://example.test/v1",
                }
            },
        }
    )


def test_retires_llm_and_memory_with_recoverable_backup(tmp_path: Path) -> None:
    module = _module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "config.toml"
    original = b'[agent]\nmax_iterations = 4\n[llm]\nregistry = "workspace"\n[memory]\nenabled = true\n'
    config.write_bytes(original)
    os.chmod(config, 0o640)
    _legacy_registry(workspace)

    with bind_migration_context(config_path=config, workspace=workspace):
        module.retire_core_model_config(object())

    assert tomllib.loads(config.read_text(encoding="utf-8")) == {
        "agent": {"max_iterations": 4}
    }
    assert config.stat().st_mode & 0o777 == 0o640
    backups = tuple(
        (workspace / "backups/retire-core-model-config").rglob("config.toml.before")
    )
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert backups[0].stat().st_mode & 0o777 == 0o600


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
