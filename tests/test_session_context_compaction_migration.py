from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
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
    / "20260807_01_session_context_compaction_ledger.py"
)


def _load_migration():
    """Load the Yoyo callback so failure injection targets the real migration owner."""

    module_name = "session_context_compaction_migration_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _MIGRATION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载迁移: {_MIGRATION_PATH}")
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _create_sessions(path: Path, *, wal: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    if wal:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    connection.execute(
        "CREATE TABLE sessions (key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
    )
    connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
    connection.execute("INSERT INTO sessions VALUES ('chat', 9)")
    connection.execute("INSERT INTO messages VALUES ('m1', 'preserve me')")
    connection.commit()
    return connection


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.migrate_session_context_compaction_ledger(None)


def _latest_backup(workspace: Path) -> Path:
    backups = sorted((workspace / "backups/session-context-compaction-ledger").iterdir())
    assert len(backups) == 1
    return backups[0]


def test_success_publishes_ledger_and_verified_backups(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config = tmp_path / "config.toml"
    original_config = (
        "memory_window = 12\n"
        "[llm]\n"
        "effective_context_percent = 0.9\n"
        "[agent.context]\n"
        "memory_window = 4\n"
    ).encode()
    config.write_bytes(original_config)
    sessions = workspace / "sessions.db"
    connection = _create_sessions(sessions)
    connection.close()
    recent = workspace / "memory/RECENT_CONTEXT.md"
    original_recent = b"retired projection\n"
    recent.write_bytes(original_recent)

    _run(module, config, workspace)

    loaded_config = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "memory_window" not in loaded_config
    assert loaded_config["agent"]["context"]["compaction"] == {
        "trigger_percent": 0.74,
        "keep_recent_tokens": 20_000,
    }
    assert not recent.exists()
    migrated = sqlite3.connect(sessions)
    try:
        assert migrated.execute(
            "SELECT last_consolidated FROM sessions WHERE key = 'chat'"
        ).fetchone() == (0,)
        assert migrated.execute("SELECT body FROM messages").fetchall() == [
            ("preserve me",)
        ]
        assert migrated.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'session_compactions'"
        ).fetchone() == ("session_compactions",)
    finally:
        migrated.close()

    backup = _latest_backup(workspace)
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert (backup / manifest["sources"]["config"]["backup"]).read_bytes() == original_config
    assert (
        backup / manifest["sources"]["recent_context"]["backup"]
    ).read_bytes() == original_recent
    archived = sqlite3.connect(backup / manifest["sources"]["sessions"]["sqlite_backup"])
    try:
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert archived.execute("SELECT body FROM messages").fetchall() == [
            ("preserve me",)
        ]
    finally:
        archived.close()
    assert not (backup / "staging").exists()


def test_failed_migration_restores_symlink_identity_and_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config_target = tmp_path / "config-target.toml"
    config_target.write_bytes(b"memory_window = 3\n")
    config = tmp_path / "config.toml"
    config.symlink_to(config_target)
    sessions_target = workspace / "sessions-target.db"
    connection = _create_sessions(sessions_target)
    connection.close()
    sessions = workspace / "sessions.db"
    sessions.symlink_to(sessions_target.name)
    recent_target = tmp_path / "recent-target.md"
    recent_target.write_bytes(b"legacy\n")
    recent = workspace / "memory/RECENT_CONTEXT.md"
    recent.symlink_to(recent_target)
    original_links = {
        path: os.readlink(path) for path in (config, sessions, recent)
    }
    original_config = config_target.read_bytes()
    original_recent = recent_target.read_bytes()

    real_publish = module._publish_staged_sqlite

    def fail_after_database_publish(snapshot, staged):
        real_publish(snapshot, staged)
        raise RuntimeError("forced publish failure")

    monkeypatch.setattr(module, "_publish_staged_sqlite", fail_after_database_publish)
    with pytest.raises(RuntimeError, match="forced publish failure"):
        _run(module, config, workspace)

    assert config.is_symlink() and os.readlink(config) == original_links[config]
    assert sessions.is_symlink() and os.readlink(sessions) == original_links[sessions]
    assert recent.is_symlink() and os.readlink(recent) == original_links[recent]
    assert config_target.read_bytes() == original_config
    assert recent_target.read_bytes() == original_recent
    restored = sqlite3.connect(sessions)
    try:
        assert restored.execute("SELECT last_consolidated FROM sessions").fetchall() == [(9,)]
        assert restored.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'session_compactions'"
        ).fetchone() is None
        assert restored.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        restored.close()


def test_failed_wal_migration_restores_committed_rows_from_online_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config = tmp_path / "config.toml"
    config.write_text("[llm]\nmodel = 'test'\n", encoding="utf-8")
    sessions = workspace / "sessions.db"
    connection = _create_sessions(sessions, wal=True)
    connection.execute("INSERT INTO messages VALUES ('wal', 'wal row')")
    connection.commit()
    recent = workspace / "memory/RECENT_CONTEXT.md"
    recent.write_text("legacy", encoding="utf-8")

    real_publish = module._publish_staged_sqlite

    def fail_after_database_publish(snapshot, staged):
        real_publish(snapshot, staged)
        raise RuntimeError("forced WAL failure")

    monkeypatch.setattr(module, "_publish_staged_sqlite", fail_after_database_publish)
    with pytest.raises(RuntimeError, match="forced WAL failure"):
        _run(module, config, workspace)
    connection.close()

    restored = sqlite3.connect(sessions)
    try:
        assert restored.execute("SELECT body FROM messages ORDER BY id").fetchall() == [
            ("preserve me",),
            ("wal row",),
        ]
        assert restored.execute("SELECT last_consolidated FROM sessions").fetchall() == [
            (9,)
        ]
        assert restored.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        restored.close()
