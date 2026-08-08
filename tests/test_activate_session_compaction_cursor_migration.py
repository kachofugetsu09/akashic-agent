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
    final_ledger: bool = False,
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
        digest_column = ", source_plan_digest TEXT NOT NULL DEFAULT ''" if final_ledger else ""
        connection.execute(
            "CREATE TABLE session_compactions ("
            "session_key TEXT NOT NULL, generation INTEGER NOT NULL"
            f"{digest_column}, PRIMARY KEY (session_key, generation))"
        )
        connection.execute(
            "CREATE TABLE session_compaction_prepares ("
            "session_key TEXT NOT NULL, generation INTEGER NOT NULL, "
            "PRIMARY KEY (session_key, generation))"
        )
        for index in range(ledger_rows):
            connection.execute(
                "INSERT INTO session_compactions(session_key, generation) VALUES (?, ?)",
                ("chat", index + 1),
            )
        for index in range(prepare_rows):
            connection.execute(
                "INSERT INTO session_compaction_prepares(session_key, generation) VALUES (?, ?)",
                ("chat", index + 1),
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


@pytest.mark.parametrize("final_ledger", [False, True])
def test_empty_fences_reset_cursor_and_leave_messages_untouched(
    tmp_path: Path,
    final_ledger: bool,
) -> None:
    module = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    _create_db(sessions, final_ledger=final_ledger)

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
