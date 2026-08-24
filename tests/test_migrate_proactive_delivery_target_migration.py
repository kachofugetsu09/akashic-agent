from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tomllib
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT
    / "migrations"
    / "yoyo"
    / "20260825_01_migrate_proactive_delivery_target.py"
)


def _load_migration():
    """Load the migration callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "migrate_proactive_delivery_target_under_test",
        _MIGRATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载迁移: {_MIGRATION_PATH}")
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.migrate_proactive_delivery_target(None)


def _legacy_config(*, enabled: bool = True) -> bytes:
    return (
        '[runtime]\nworkspace = "/state/workspace"\n\n'
        "[proactive]\n"
        f"enabled = {str(enabled).lower()}\n"
        'profile = "daily"\n\n'
        "[proactive.target]\n"
        'channel = "mobile"\n'
        'chat_id = "conversation-id"\n\n'
        "[proactive.agent]\n"
        "max_steps = 35\n\n"
        "[agent]\n"
        'system_prompt = "preserve"\n'
    ).encode("utf-8")


def _wake_path(workspace: Path) -> Path:
    return workspace / "plugin-data/wake-builtin/config.local.toml"


def _latest_backup(workspace: Path) -> Path:
    roots = sorted((workspace / "backups/migrate-proactive-delivery-target").iterdir())
    assert len(roots) == 1
    return roots[0]


def test_enabled_mobile_target_moves_to_wake_and_preserves_root_config(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original = _legacy_config()
    config.write_bytes(original)

    _run(module, config, workspace)

    migrated = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "proactive" not in migrated
    assert migrated["runtime"]["workspace"] == "/state/workspace"
    assert migrated["agent"]["system_prompt"] == "preserve"
    assert tomllib.loads(_wake_path(workspace).read_text(encoding="utf-8")) == {
        "delivery": {
            "channel": "mobile",
            "recipient": "conversation-id",
            "session_id": "mobile:conversation-id",
        }
    }
    backup = _latest_backup(workspace)
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup / "manifest.json").stat().st_mode) == 0o600
    config_source = manifest["sources"][str(config)]
    assert (backup / config_source["backup"]).read_bytes() == original
    wake_source = manifest["sources"][str(_wake_path(workspace))]
    assert wake_source["kind"] == "absent"
    assert wake_source["backup"] is None


def test_disabled_legacy_proactive_is_removed_without_enabling_wake(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_bytes(_legacy_config(enabled=False))

    _run(module, config, workspace)

    assert "proactive" not in tomllib.loads(config.read_text(encoding="utf-8"))
    assert not _wake_path(workspace).exists()


def test_matching_wake_target_is_preserved_and_direct_retry_is_noop(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_bytes(_legacy_config())
    wake = _wake_path(workspace)
    wake.parent.mkdir(parents=True)
    original_wake = (
        "# operator target\n"
        "[delivery]\n"
        'channel = "mobile"\n'
        'recipient = "conversation-id"\n'
        'session_id = "mobile:conversation-id"\n'
    ).encode("utf-8")
    wake.write_bytes(original_wake)

    _run(module, config, workspace)
    _run(module, config, workspace)

    assert wake.read_bytes() == original_wake
    roots = sorted((workspace / "backups/migrate-proactive-delivery-target").iterdir())
    assert len(roots) == 1


def test_conflicting_wake_target_fails_before_any_write(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original_config = _legacy_config()
    config.write_bytes(original_config)
    wake = _wake_path(workspace)
    wake.parent.mkdir(parents=True)
    original_wake = (
        "[delivery]\n"
        'channel = "telegram"\n'
        'recipient = "other"\n'
        'session_id = "telegram:other"\n'
    ).encode("utf-8")
    wake.write_bytes(original_wake)

    with pytest.raises(RuntimeError, match="冲突"):
        _run(module, config, workspace)

    assert config.read_bytes() == original_config
    assert wake.read_bytes() == original_wake
    assert not (workspace / "backups").exists()


def test_enabled_target_requires_complete_identity_before_any_write(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    original = b"[proactive]\nenabled = true\n"
    config.write_bytes(original)

    with pytest.raises(ValueError, match="proactive.target"):
        _run(module, config, workspace)

    assert config.read_bytes() == original
    assert not (workspace / "backups").exists()


def test_failed_second_publication_restores_both_files_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_target = tmp_path / "config-target.toml"
    original_config = _legacy_config()
    config_target.write_bytes(original_config)
    config = tmp_path / "config.toml"
    config.symlink_to(config_target)
    original_link = os.readlink(config)
    wake = _wake_path(workspace)

    real_publish = module._publish

    def fail_config_publish(snapshot, payload):
        if snapshot.path == config:
            raise RuntimeError("forced config publication failure")
        real_publish(snapshot, payload)

    monkeypatch.setattr(module, "_publish", fail_config_publish)
    with pytest.raises(RuntimeError, match="forced config publication failure"):
        _run(module, config, workspace)

    assert config.is_symlink() and os.readlink(config) == original_link
    assert config_target.read_bytes() == original_config
    assert not wake.exists()
