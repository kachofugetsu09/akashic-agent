from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest
import tomllib
import yoyo

from agent.migrations.context import bind_migration_context


_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT
    / "migrations"
    / "yoyo"
    / "20260808_06_retire_legacy_context_state.py"
)


def _load_migration():
    """Load the retirement callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "retire_legacy_context_state_migration_under_test",
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


def _create_sessions(path: Path) -> bytes:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE sessions (key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
        )
        connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        connection.execute("INSERT INTO sessions VALUES ('chat', 4)")
        connection.execute("INSERT INTO messages VALUES ('m1', 'preserve')")
        connection.commit()
    finally:
        connection.close()
    return path.read_bytes()


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.retire_legacy_context_state(None)


def _legacy_config(*, keep_recent_tokens: int | None = None) -> bytes:
    keep_line = "" if keep_recent_tokens is None else f"keep_recent_tokens = {keep_recent_tokens}\n"
    return (
        "memory_window = 12\n"
        "[llm]\n"
        "effective_context_percent = 0.9\n"
        "compaction_trigger_percent = 0.7\n"
        "[llm.main]\n"
        "effective_context_percent = 0.8\n"
        "[llm.runtimes.alpha]\n"
        "effective_context_percent = 0.75\n"
        "compaction_trigger_percent = 0.65\n"
        "[agent.context]\n"
        "memory_window = 4\n"
        "[agent.context.compaction]\n"
        "trigger_percent = 0.74\n"
        f"{keep_line}"
    ).encode()


def _latest_backup(workspace: Path) -> Path:
    roots = sorted((workspace / "backups/retire-legacy-context-state").iterdir())
    assert len(roots) == 1
    return roots[0]


def test_retirement_archives_and_removes_legacy_state_without_session_db_write(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config = tmp_path / "config.toml"
    original_config = _legacy_config()
    config.write_bytes(original_config)
    recent = workspace / "memory/RECENT_CONTEXT.md"
    original_recent = b"legacy projection\n"
    recent.write_bytes(original_recent)
    sessions = workspace / "sessions.db"
    original_sessions = _create_sessions(sessions)

    _run(module, config, workspace)

    loaded = tomllib.loads(config.read_text(encoding="utf-8"))
    assert loaded["agent"]["context"]["compaction"] == {"keep_recent_tokens": 20_000}
    assert not recent.exists()
    assert sessions.read_bytes() == original_sessions
    backup = _latest_backup(workspace)
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup / "manifest.json").stat().st_mode) == 0o600
    assert (
        backup / manifest["sources"]["config"]["backup"]
    ).read_bytes() == original_config
    assert (
        backup / manifest["sources"]["recent_context"]["backup"]
    ).read_bytes() == original_recent
    assert (
        stat.S_IMODE(
            (backup / manifest["sources"]["config"]["backup"]).stat().st_mode
        )
        == 0o600
    )


def test_retirement_preserves_valid_tail_above_watermark(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_bytes(_legacy_config(keep_recent_tokens=21_000))
    _run(module, config, workspace)

    loaded = tomllib.loads(config.read_text(encoding="utf-8"))
    assert loaded["agent"]["context"]["compaction"]["keep_recent_tokens"] == 21_000


def test_failed_publish_restores_config_recent_symlink_identity_and_session_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config_target = tmp_path / "config-target.toml"
    original_config = _legacy_config()
    config_target.write_bytes(original_config)
    config = tmp_path / "config.toml"
    config.symlink_to(config_target)
    recent_target = tmp_path / "recent-target.md"
    original_recent = b"legacy\n"
    recent_target.write_bytes(original_recent)
    recent = workspace / "memory/RECENT_CONTEXT.md"
    recent.symlink_to(recent_target)
    sessions = workspace / "sessions.db"
    original_sessions = _create_sessions(sessions)
    links = (os.readlink(config), os.readlink(recent))

    real_publish = module._publish_config

    def fail_after_publish(snapshot, rendered):
        real_publish(snapshot, rendered)
        raise RuntimeError("forced retirement failure")

    monkeypatch.setattr(module, "_publish_config", fail_after_publish)
    with pytest.raises(RuntimeError, match="forced retirement failure"):
        _run(module, config, workspace)

    assert config.is_symlink() and os.readlink(config) == links[0]
    assert recent.is_symlink() and os.readlink(recent) == links[1]
    assert config_target.read_bytes() == original_config
    assert recent_target.read_bytes() == original_recent
    assert sessions.read_bytes() == original_sessions


def test_malformed_config_does_not_delete_recent(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config = tmp_path / "config.toml"
    config.write_bytes(b"[broken\n")
    recent = workspace / "memory/RECENT_CONTEXT.md"
    recent.write_bytes(b"keep on failure")

    with pytest.raises(tomllib.TOMLDecodeError):
        _run(module, config, workspace)

    assert config.read_bytes() == b"[broken\n"
    assert recent.read_bytes() == b"keep on failure"


def test_second_direct_run_is_idempotent_without_new_backup(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config = tmp_path / "config.toml"
    config.write_bytes(_legacy_config())
    recent = workspace / "memory/RECENT_CONTEXT.md"
    recent.write_bytes(b"legacy")

    _run(module, config, workspace)
    _run(module, config, workspace)

    roots = sorted((workspace / "backups/retire-legacy-context-state").iterdir())
    assert len(roots) == 1
