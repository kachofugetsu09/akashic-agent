from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database


__depends__ = {"20260808_04_session_compaction_source_plan_digest"}
__transactional__ = False

_MIGRATION_NAME = "activate-session-compaction-cursor"
_REQUIRED_COLUMNS = {
    "sessions": {"key": "TEXT", "last_consolidated": "INTEGER"},
    "session_compactions": {"session_key": "TEXT", "generation": "INTEGER"},
    "session_compaction_prepares": {"session_key": "TEXT", "generation": "INTEGER"},
}


def _table_columns(connection: sqlite3.Connection, table: str) -> dict[str, tuple[str, int]]:
    """Read one table's declared columns at the migration trust boundary."""

    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if not rows:
        raise RuntimeError(f"{table} schema lineage 不兼容，表定义缺失")
    return {str(row[1]): (str(row[2]).upper(), int(row[3])) for row in rows}


def _validate_schema(connection: sqlite3.Connection) -> None:
    """Require the SessionDB tables owned by the preceding migration stages."""

    # 1. Every required table and its cursor/identity columns must exist.
    for table, required in _REQUIRED_COLUMNS.items():
        columns = _table_columns(connection, table)
        missing = sorted(set(required) - set(columns))
        if missing:
            raise RuntimeError(
                f"{table} schema lineage 不兼容，缺少列: {', '.join(missing)}"
            )
        for name, expected_type in required.items():
            actual_type = columns[name][0]
            if actual_type != expected_type:
                raise RuntimeError(
                    f"{table} schema lineage 不兼容，列类型不匹配: {name}"
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
