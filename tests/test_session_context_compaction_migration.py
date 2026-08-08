from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest
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
    """Load the additive migration callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "session_context_compaction_migration_under_test",
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


def _create_sessions(path: Path, *, cursor: object = 9) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE sessions ("
            "key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO sessions VALUES ('chat', ?)", (cursor,))
        connection.execute("INSERT INTO messages VALUES ('m1', 'preserve me')")
        connection.commit()
    finally:
        connection.close()


def _run(module, config: Path, workspace: Path) -> None:
    with bind_migration_context(config_path=config, workspace=workspace):
        module.add_session_compaction_ledger(None)


def _latest_backup(workspace: Path) -> Path:
    roots = sorted((workspace / "backups/session-context-compaction-ledger").iterdir())
    assert len(roots) == 1
    return roots[0]


def test_additive_ledger_preserves_config_recent_cursor_and_messages(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    (workspace / "memory").mkdir(parents=True)
    config = tmp_path / "config.toml"
    original_config = b"memory_window = 12\n[llm]\neffective_context_percent = 0.9\n"
    config.write_bytes(original_config)
    recent = workspace / "memory/RECENT_CONTEXT.md"
    original_recent = b"recent projection\n"
    recent.write_bytes(original_recent)
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)

    _run(module, config, workspace)

    assert config.read_bytes() == original_config
    assert recent.read_bytes() == original_recent
    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute(
            "SELECT last_consolidated FROM sessions WHERE key = 'chat'"
        ).fetchone() == (9,)
        assert connection.execute("SELECT body FROM messages").fetchall() == [
            ("preserve me",)
        ]
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'session_compactions'"
        ).fetchone() == ("session_compactions",)
        assert connection.execute(
            "SELECT COUNT(*) FROM session_compactions"
        ).fetchone() == (0,)
    finally:
        connection.close()

    backup = _latest_backup(workspace)
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sqlite_integrity"] == "ok"
    archived = sqlite3.connect(backup / manifest["backup"])
    try:
        assert archived.execute(
            "SELECT last_consolidated FROM sessions"
        ).fetchone() == (9,)
        assert archived.execute("SELECT body FROM messages").fetchall() == [
            ("preserve me",)
        ]
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        archived.close()


def test_existing_ledger_is_only_validated_and_not_rewritten(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text("current = true\n", encoding="utf-8")
    sessions = workspace / "sessions.db"
    _create_sessions(sessions, cursor=3)
    connection = sqlite3.connect(sessions)
    try:
        connection.execute("""
            CREATE TABLE session_compactions (
                session_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_generation INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                trigger TEXT NOT NULL,
                summary_format_version INTEGER NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                source_from_seq INTEGER NOT NULL,
                consolidated_through_seq INTEGER NOT NULL,
                source_message_ids_json TEXT NOT NULL,
                retained_tail_json TEXT NOT NULL,
                model_runtime_id TEXT NOT NULL,
                model TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                threshold_tokens INTEGER NOT NULL,
                hard_input_tokens INTEGER NOT NULL,
                keep_recent_tokens INTEGER NOT NULL,
                tokens_before INTEGER NOT NULL,
                tokens_after INTEGER NOT NULL,
                summary_usage_json TEXT NOT NULL,
                invalidated_at TEXT,
                invalidated_reason TEXT,
                PRIMARY KEY (session_key, generation),
                UNIQUE (session_key, source_ref)
            )
            """)
        connection.execute(
            "INSERT INTO session_compactions "
            "(session_key, generation, created_at, trigger, summary_format_version, "
            "summary, source_ref, source_from_seq, consolidated_through_seq, "
            "source_message_ids_json, retained_tail_json, model_runtime_id, model, "
            "context_window, threshold_tokens, hard_input_tokens, keep_recent_tokens, "
            "tokens_before, tokens_after, summary_usage_json) "
            "VALUES ('chat', 1, 'now', 'manual', 1, 'summary', 'ref', 0, 0, '[]', '[]', "
            "'runtime', 'model', 1, 1, 1, 1, 1, 1, '{}')"
        )
        connection.commit()
    finally:
        connection.close()
    _run(module, config, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute(
            "SELECT last_consolidated FROM sessions"
        ).fetchone() == (3,)
        assert connection.execute(
            "SELECT source_ref, summary FROM session_compactions"
        ).fetchall() == [("ref", "summary")]
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_session_compactions_active'"
        ).fetchone() == ("idx_session_compactions_active",)
    finally:
        connection.close()


def test_invalid_existing_ledger_fails_before_partial_publish(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text("current = true\n", encoding="utf-8")
    sessions = workspace / "sessions.db"
    _create_sessions(sessions)
    connection = sqlite3.connect(sessions)
    try:
        connection.execute(
            "CREATE TABLE session_compactions (session_key TEXT NOT NULL)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="缺少列"):
        _run(module, config, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute(
            "PRAGMA table_info(session_compactions)"
        ).fetchall() == [(0, "session_key", "TEXT", 1, None, 0)]
        assert connection.execute(
            "SELECT last_consolidated FROM sessions"
        ).fetchone() == (9,)
    finally:
        connection.close()
