from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)


__depends__ = {"20260808_04_session_compaction_source_plan_digest"}
__transactional__ = False

_MIGRATION_NAME = "activate-session-compaction-cursor"
_FINAL_LEDGER_SCHEMA = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("trigger", "TEXT", 1, 0),
        ("summary_format_version", "INTEGER", 1, 0),
        ("summary", "TEXT", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_plan_digest", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("model_runtime_id", "TEXT", 1, 0),
        ("model", "TEXT", 1, 0),
        ("context_window", "INTEGER", 1, 0),
        ("threshold_tokens", "INTEGER", 1, 0),
        ("hard_input_tokens", "INTEGER", 1, 0),
        ("keep_recent_tokens", "INTEGER", 1, 0),
        ("tokens_before", "INTEGER", 1, 0),
        ("tokens_after", "INTEGER", 1, 0),
        ("summary_usage_json", "TEXT", 1, 0),
        ("invalidated_at", "TEXT", 0, 0),
        ("invalidated_reason", "TEXT", 0, 0),
    ),
    "named_indexes": {
        "idx_session_compactions_active": (
            ("session_key", "invalidated_at", "generation"),
            0,
        ),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (
        "CHECK (length(source_plan_digest) = 64 AND "
        "source_plan_digest NOT GLOB '*[^0-9a-f]*')",
    ),
}
_PREPARE_SCHEMA = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("session_created_at", "TEXT", 1, 0),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("prepared_at", "TEXT", 1, 0),
    ),
    "named_indexes": {
        "idx_session_compaction_prepares_ref": (("session_key", "source_ref"), 0),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (),
}


def _validate_schema(connection: sqlite3.Connection) -> None:
    """Require the SessionDB tables owned by the preceding migration stages."""

    # 1. The cursor owner must be a keyed, non-null integer field.
    session_rows = connection.execute("PRAGMA table_info(sessions)").fetchall()
    if not session_rows:
        raise RuntimeError("sessions schema lineage 不兼容，表定义缺失")
    session_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in session_rows
    }
    for name, expected in (("key", ("TEXT", 1)), ("last_consolidated", ("INTEGER", 0))):
        column = session_columns.get(name)
        if column is None:
            raise RuntimeError(f"sessions schema lineage 不兼容，缺少列: {name}")
        expected_type, expected_pk = expected
        if column[0] != expected_type or column[2] != expected_pk:
            raise RuntimeError(f"sessions schema lineage 不兼容，列定义不匹配: {name}")
    if session_columns["last_consolidated"][1] != 1:
        raise RuntimeError("sessions schema lineage 不兼容，last_consolidated 必须 NOT NULL")

    # 2. Preceding ledger/prepare migrations own exact schema identity.
    for table, schema in (
        ("session_compactions", _FINAL_LEDGER_SCHEMA),
        ("session_compaction_prepares", _PREPARE_SCHEMA),
    ):
        validate_table_schema(
            connection,
            table=table,
            columns=schema["columns"],  # type: ignore[arg-type]
            named_indexes=schema["named_indexes"],  # type: ignore[arg-type]
            auto_indexes=schema["auto_indexes"],  # type: ignore[arg-type]
            sql_fragments=schema["sql_fragments"],  # type: ignore[arg-type]
        )


def _validate_empty_fences(connection: sqlite3.Connection) -> None:
    """Reject any durable ledger or prepare rows before cursor activation."""

    # 2. Cursor activation is safe only before either compaction fence has data.
    for table in ("session_compactions", "session_compaction_prepares"):
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        if row is None or not isinstance(row[0], int):
            raise RuntimeError(f"{table} 行数读取失败")
        if row[0] != 0:
            raise RuntimeError(f"{table} 非空，拒绝激活 last_consolidated")


def _validate_cursor_values(connection: sqlite3.Connection) -> None:
    """Require every legacy cursor to be a non-negative integer."""

    # 3. SQLite may return text/floats for malformed legacy rows; do not coerce them.
    rows = connection.execute("SELECT key, last_consolidated FROM sessions").fetchall()
    for key, cursor in rows:
        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise RuntimeError(
                "sessions.last_consolidated 必须是非负整数: "
                f"{key!r}={cursor!r}"
            )


def _preflight(connection: sqlite3.Connection) -> None:
    """Run all read-only checks used both before backup and inside the write lock."""

    _validate_schema(connection)
    _validate_empty_fences(connection)
    _validate_cursor_values(connection)


def activate_session_compaction_cursor(_connection: object) -> None:
    """Back up SessionDB and reset legacy cursors only at the empty-ledger fence."""

    _ = _connection
    current = current_migration_context()
    sessions_db = current.workspace / "sessions.db"
    if not sessions_db.exists():
        return

    # 1. Fail before backup on schema/data violations, preserving a retryable state.
    preflight_connection = sqlite3.connect(sessions_db)
    try:
        _preflight(preflight_connection)
    finally:
        preflight_connection.close()

    # 2. Persist a verified online backup before the first cursor write.
    backup_sqlite_database(
        sessions_db,
        current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex,
        migration=_MIGRATION_NAME,
    )

    # 3. Revalidate under the write lock, then activate all sessions together.
    connection = sqlite3.connect(sessions_db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _preflight(connection)
            connection.execute("UPDATE sessions SET last_consolidated = 0")
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


steps = [step(activate_session_compaction_cursor)]
