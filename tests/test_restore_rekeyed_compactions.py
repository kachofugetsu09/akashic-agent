from __future__ import annotations

import importlib.util
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
import yoyo

from agent.migrations.context import bind_migration_context


_MIGRATION = (
    Path(__file__).parents[1]
    / "migrations/yoyo/20260827_03_restore_rekeyed_compactions.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("restore_rekeyed_compactions", _MIGRATION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _database(path: Path, *, missing_message: bool = False) -> None:
    session = "akashic:018f0000000070008000000000000000"
    message = f"{session}:1"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions(key TEXT PRIMARY KEY);
            CREATE TABLE messages(id TEXT PRIMARY KEY, session_key TEXT NOT NULL);
            CREATE TABLE session_compactions(
                session_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                source_ref TEXT NOT NULL,
                source_message_ids_json TEXT NOT NULL,
                retained_tail_json TEXT NOT NULL,
                invalidated_at TEXT,
                invalidated_reason TEXT,
                PRIMARY KEY(session_key, generation)
            );
            """
        )
        connection.execute("INSERT INTO sessions(key) VALUES (?)", (session,))
        if not missing_message:
            connection.execute(
                "INSERT INTO messages(id, session_key) VALUES (?, ?)",
                (message, session),
            )
        connection.execute(
            "INSERT INTO session_compactions VALUES(?,?,?,?,?,?,?)",
            (
                session,
                2,
                f"context-compaction:{session}@scope:2:digest",
                json.dumps([message]),
                json.dumps([{"id": message, "message": {"role": "user"}}]),
                "2026-08-27T00:00:00+00:00",
                "akashic_identity_rekey",
            ),
        )
        connection.commit()


def test_restores_only_identity_rekey_invalidation_after_backup(tmp_path: Path) -> None:
    database = tmp_path / "sessions.db"
    _database(database)
    module = _load_migration()

    with bind_migration_context(config_path=tmp_path / "config.toml", workspace=tmp_path):
        module.restore_rekeyed_compactions(object())

    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT invalidated_at, invalidated_reason FROM session_compactions"
        ).fetchone()
        assert row == (None, None)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    backups = list((tmp_path / "backups/restore-rekeyed-compactions").glob("*/sessions.db"))
    assert len(backups) == 1


def test_missing_rekeyed_source_fails_before_backup_or_write(tmp_path: Path) -> None:
    database = tmp_path / "sessions.db"
    _database(database, missing_message=True)
    module = _load_migration()

    with bind_migration_context(config_path=tmp_path / "config.toml", workspace=tmp_path):
        with pytest.raises(RuntimeError, match="source message is missing"):
            module.restore_rekeyed_compactions(object())

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute(
            "SELECT invalidated_reason FROM session_compactions"
        ).fetchone()[0] == "akashic_identity_rekey"
    assert not (tmp_path / "backups").exists()
