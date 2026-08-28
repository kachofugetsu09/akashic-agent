from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260828_01_migrate_eventmail_state"}
__transactional__ = False

_MIGRATION = "add-wake-content-scores"
_WAKE_DB = Path("plugin-data/wake-builtin/wake.sqlite3")
_V7_TABLES = {
    "admission_state",
    "seen_content",
    "wake_runs",
    "wake_attempts",
}
_SCORE_TABLE_SQL = """
    CREATE TABLE content_scores(
        source_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        revision TEXT NOT NULL,
        initial_score REAL NOT NULL CHECK(initial_score >= 0 AND initial_score <= 7.0),
        semantic_interest REAL NOT NULL CHECK(semantic_interest >= 0 AND semantic_interest <= 0.999),
        scored_at TEXT NOT NULL,
        PRIMARY KEY(source_id, item_id, revision)
    )
"""
_V7_ADMISSION_SQL = """
    CREATE TABLE admission_state(
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        content_high_watermark INTEGER NOT NULL,
        last_content_attempt_at TEXT
    )
"""
_V8_ADMISSION_SQL = """
    CREATE TABLE admission_state(
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        content_high_watermark INTEGER NOT NULL
    )
"""
_SEEN_SQL = "CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY)"
_RUN_SQL = """
    CREATE TABLE wake_runs(
        run_id TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        started_at TEXT NOT NULL,
        candidates_seen INTEGER NOT NULL,
        candidates_selected INTEGER NOT NULL,
        screening_json TEXT NOT NULL,
        decision TEXT,
        decision_detail TEXT,
        completed_at TEXT
    )
"""
_ATTEMPT_SQL = """
    CREATE TABLE wake_attempts(
        attempt_id TEXT PRIMARY KEY,
        timer_id TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        fired_at TEXT NOT NULL,
        mail_watermark INTEGER,
        outcome TEXT NOT NULL CHECK(outcome IN (
            'checking', 'no_due', 'content_insufficient', 'admission_rejected',
            'shared', 'model_skip', 'deferred', 'cancelled_after_fire',
            'delivery_unknown', 'failed'
        )),
        owner TEXT CHECK(owner IN ('alert', 'content', 'drift')),
        detail TEXT,
        completed_at TEXT
    )
"""


def _normalize_sql(sql: str) -> str:
    return "".join(sql.split()).lower()


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _check_integrity(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("Wake score migration integrity_check 失败")


def _owned_schema(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(name): _normalize_sql(str(sql))
        for name, sql in connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _validate_v7(connection: sqlite3.Connection) -> None:
    expected = {
        "admission_state": _V7_ADMISSION_SQL,
        "seen_content": _SEEN_SQL,
        "wake_runs": _RUN_SQL,
        "wake_attempts": _ATTEMPT_SQL,
    }
    if _owned_schema(connection) != {
        name: _normalize_sql(sql) for name, sql in expected.items()
    }:
        raise RuntimeError("Wake v7 schema identity 不匹配")
    _check_integrity(connection)


def _validate_v8(connection: sqlite3.Connection) -> None:
    if _tables(connection) != {*_V7_TABLES, "content_scores"}:
        raise RuntimeError("Wake v8 schema lineage 不兼容")
    expected = {
        "admission_state": _V8_ADMISSION_SQL,
        "seen_content": _SEEN_SQL,
        "content_scores": _SCORE_TABLE_SQL,
        "wake_runs": _RUN_SQL,
        "wake_attempts": _ATTEMPT_SQL,
    }
    if _owned_schema(connection) != {
        name: _normalize_sql(sql) for name, sql in expected.items()
    }:
        raise RuntimeError("Wake v8 schema identity 不匹配")
    _check_integrity(connection)


def add_wake_content_scores(_connection: object) -> None:
    """Back up Wake v7, then add the immutable one-time Content score ledger."""

    _ = _connection
    workspace = current_migration_context().workspace
    wake_db = workspace / _WAKE_DB
    if not wake_db.exists():
        return
    with closing(sqlite3.connect(wake_db)) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = _tables(connection)
        if version == 0 and not tables:
            return
        if version == 8:
            _validate_v8(connection)
            return
        if version != 7 or tables != _V7_TABLES:
            raise RuntimeError(
                f"Wake score migration 不支持 schema version {version}: {sorted(tables)}"
            )
        _validate_v7(connection)

    backup_root = (
        workspace
        / "backups"
        / _MIGRATION
        / uuid4().hex
        / "wake-db"
    )
    backup_sqlite_database(wake_db, backup_root, migration=_MIGRATION)

    with closing(sqlite3.connect(wake_db)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("ALTER TABLE admission_state RENAME TO admission_state_v7")
            connection.execute(_V8_ADMISSION_SQL)
            connection.execute(
                "INSERT INTO admission_state(singleton, content_high_watermark) "
                "SELECT singleton, content_high_watermark FROM admission_state_v7"
            )
            connection.execute("DROP TABLE admission_state_v7")
            connection.execute(_SCORE_TABLE_SQL)
            connection.execute("PRAGMA user_version = 8")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        _validate_v8(connection)


steps = [step(add_wake_content_scores)]
