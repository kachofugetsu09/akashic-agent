from __future__ import annotations

import importlib.util
import json
import sqlite3
import stat
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
    / "20260808_05_activate_session_compaction_cursor.py"
)


def _load_migration():
    """Load the cursor activation callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "activate_session_compaction_cursor_migration_under_test",
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


def _create_db(
    path: Path,
    *,
    cursor: object = 9,
    ledger_rows: int = 0,
    prepare_rows: int = 0,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE sessions ("
            "key TEXT PRIMARY KEY, last_consolidated INTEGER NOT NULL)"
        )
        connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        connection.execute("INSERT INTO sessions VALUES ('chat', ?)", (cursor,))
        connection.execute("INSERT INTO messages VALUES ('m1', 'preserve')")
        connection.execute(
            """
            CREATE TABLE session_compactions (
                session_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_generation INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                trigger TEXT NOT NULL,
                summary_format_version INTEGER NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                source_plan_digest TEXT NOT NULL CHECK (
                    length(source_plan_digest) = 64
                    AND source_plan_digest NOT GLOB '*[^0-9a-f]*'
                ),
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
            """
        )
        connection.execute(
            "CREATE INDEX idx_session_compactions_active ON "
            "session_compactions(session_key, invalidated_at, generation)"
        )
        connection.execute(
            """
            CREATE TABLE session_compaction_prepares (
                session_key TEXT NOT NULL,
                session_created_at TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_generation INTEGER NOT NULL,
                source_ref TEXT NOT NULL,
                source_from_seq INTEGER NOT NULL,
                consolidated_through_seq INTEGER NOT NULL,
                source_message_ids_json TEXT NOT NULL,
                retained_tail_json TEXT NOT NULL,
                prepared_at TEXT NOT NULL,
                PRIMARY KEY (session_key, generation),
                UNIQUE (session_key, source_ref)
            )
            """
        )
        connection.execute(
            "CREATE INDEX idx_session_compaction_prepares_ref ON "
            "session_compaction_prepares(session_key, source_ref)"
        )
        for index in range(ledger_rows):
            connection.execute(
                """
                INSERT INTO session_compactions(
                    session_key, generation, created_at, trigger,
                    summary_format_version, summary, source_ref, source_plan_digest,
                    source_from_seq, consolidated_through_seq,
                    source_message_ids_json, retained_tail_json, model_runtime_id,
                    model, context_window, threshold_tokens, hard_input_tokens,
                    keep_recent_tokens, tokens_before, tokens_after, summary_usage_json
                ) VALUES (?, ?, 'now', 'manual', 1, 'summary', ?, ?, 0, 0,
                    '[]', '[]', 'runtime', 'model', 100, 74, 90, 20000, 90, 40, '{}')
                """,
                (
                    "chat",
                    index + 1,
                    f"ref:{index + 1}",
                    "a" * 64,
                ),
            )
        for index in range(prepare_rows):
            connection.execute(
                """
                INSERT INTO session_compaction_prepares(
                    session_key, session_created_at, generation, parent_generation,
                    source_ref, source_from_seq, consolidated_through_seq,
                    source_message_ids_json, retained_tail_json, prepared_at
                ) VALUES ('chat', 'now', ?, 0, ?, 0, 0, '[]', '[]', 'now')
                """,
                (index + 1, f"prepare:{index + 1}"),
            )
        connection.commit()
    finally:
        connection.close()


def _run(module, workspace: Path) -> None:
    config = workspace.parent / "config.toml"
    config.write_text("current = true\n", encoding="utf-8")
    with bind_migration_context(config_path=config, workspace=workspace):
        module.activate_session_compaction_cursor(None)


def test_missing_sessions_db_is_an_exact_noop(tmp_path: Path) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _run(module, workspace)

    assert not (workspace / "backups").exists()
    assert not (workspace / "sessions.db").exists()


def test_empty_fences_reset_cursor_and_leave_messages_untouched(
    tmp_path: Path,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_db(sessions)

    _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute("SELECT last_consolidated FROM sessions").fetchall() == [(0,)]
        assert connection.execute("SELECT body FROM messages").fetchall() == [("preserve",)]
        assert connection.execute("SELECT COUNT(*) FROM session_compactions").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM session_compaction_prepares"
        ).fetchone() == (0,)
    finally:
        connection.close()

    roots = sorted((workspace / "backups/activate-session-compaction-cursor").iterdir())
    assert len(roots) == 1
    backup_root = roots[0]
    backup = backup_root / "sessions.db"
    manifest = json.loads((backup_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sqlite_integrity"] == "ok"
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE((backup_root / "manifest.json").stat().st_mode) == 0o600
    archived = sqlite3.connect(backup)
    try:
        assert archived.execute("SELECT last_consolidated FROM sessions").fetchone() == (9,)
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        archived.close()


@pytest.mark.parametrize("kwargs, message", [
    ({"ledger_rows": 1}, "session_compactions 非空"),
    ({"prepare_rows": 1}, "session_compaction_prepares 非空"),
])
def test_nonempty_fence_fails_before_backup_and_reset(
    tmp_path: Path,
    kwargs: dict[str, int],
    message: str,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_db(sessions, **kwargs)

    with pytest.raises(RuntimeError, match=message):
        _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute("SELECT last_consolidated FROM sessions").fetchone() == (9,)
    finally:
        connection.close()
    assert not (workspace / "backups").exists()


@pytest.mark.parametrize("cursor", [-1, "not-an-integer"])
def test_invalid_cursor_fails_before_backup_and_reset(tmp_path: Path, cursor: object) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_db(sessions, cursor=cursor)

    with pytest.raises(RuntimeError, match="last_consolidated 必须是非负整数"):
        _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute("SELECT last_consolidated FROM sessions").fetchone() == (cursor,)
    finally:
        connection.close()
    assert not (workspace / "backups").exists()


def test_second_preflight_failure_rolls_back_after_verified_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_db(sessions)
    real_preflight = module._preflight
    calls = 0

    def fail_second(connection):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced second preflight failure")
        return real_preflight(connection)

    monkeypatch.setattr(module, "_preflight", fail_second)
    with pytest.raises(RuntimeError, match="forced second preflight failure"):
        _run(module, workspace)

    connection = sqlite3.connect(sessions)
    try:
        assert connection.execute("SELECT last_consolidated FROM sessions").fetchone() == (9,)
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        connection.close()
    roots = sorted((workspace / "backups/activate-session-compaction-cursor").iterdir())
    assert len(roots) == 1
    archived = sqlite3.connect(roots[0] / "sessions.db")
    try:
        assert archived.execute("SELECT last_consolidated FROM sessions").fetchone() == (9,)
        assert archived.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        archived.close()
