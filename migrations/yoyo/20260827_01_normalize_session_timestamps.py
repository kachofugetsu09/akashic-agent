from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260826_03_unify_akashic_channel_identity"}
__transactional__ = False

_MIGRATION = "normalize-session-timestamps"
_LEGACY_TIMEZONE = ZoneInfo("Asia/Shanghai")
_REQUIRED_TIMESTAMP_COLUMNS = (
    ("sessions", "key", "created_at"),
    ("sessions", "key", "updated_at"),
    ("messages", "id", "ts"),
)
_OPTIONAL_TIMESTAMP_COLUMNS = (
    ("sessions", "key", "last_user_at"),
    ("sessions", "key", "last_proactive_at"),
)


def _normalized_timestamp(value: object, *, field: str) -> str | None:
    """把旧上海墙上时间转换为明确的 UTC 时间。"""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{field} 不是有效时间文本")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{field} 不是有效 ISO 时间: {value!r}") from error
    if parsed.tzinfo is not None:
        return value
    return parsed.replace(tzinfo=_LEGACY_TIMEZONE).astimezone(UTC).isoformat()


def _pending_updates(
    connection: sqlite3.Connection,
) -> list[tuple[str, str, str, str, str]]:
    """扫描固定 SessionDB 时间列，并返回需要补时区的单元格。"""

    available = {
        table: {
            str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for table in {item[0] for item in _REQUIRED_TIMESTAMP_COLUMNS}
    }
    present_required = [
        column in available[table]
        for table, _identity, column in _REQUIRED_TIMESTAMP_COLUMNS
    ]
    if not any(present_required):
        return []
    if not all(present_required):
        raise RuntimeError("SessionDB 时间 schema lineage 不完整")
    columns = (*_REQUIRED_TIMESTAMP_COLUMNS, *_OPTIONAL_TIMESTAMP_COLUMNS)
    updates: list[tuple[str, str, str, str, str]] = []
    for table, identity_column, timestamp_column in columns:
        if timestamp_column not in available[table]:
            continue
        rows = connection.execute(
            f"SELECT {identity_column}, {timestamp_column} FROM {table} "
            f"WHERE {timestamp_column} IS NOT NULL"
        ).fetchall()
        for identity, value in rows:
            field = f"{table}.{timestamp_column}:{identity}"
            normalized = _normalized_timestamp(value, field=field)
            if normalized != value:
                updates.append(
                    (
                        table,
                        identity_column,
                        timestamp_column,
                        str(identity),
                        str(normalized),
                    )
                )
    return updates


def _apply_updates(
    connection: sqlite3.Connection,
    updates: list[tuple[str, str, str, str, str]],
) -> None:
    """在一个 SQLite 事务中提交并验证全部时间修复。"""

    connection.execute("BEGIN IMMEDIATE")
    try:
        for table, identity_column, timestamp_column, identity, value in updates:
            cursor = connection.execute(
                f"UPDATE {table} SET {timestamp_column} = ? "
                f"WHERE {identity_column} = ?",
                (value, identity),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"{table}.{timestamp_column} 更新时间冲突: {identity}")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("SessionDB 时间迁移留下 foreign key 错误")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("SessionDB 时间迁移 integrity_check 失败")
    if _pending_updates(connection):
        raise RuntimeError("SessionDB 时间迁移仍有无时区时间")


def _check_compaction_identities(
    connection: sqlite3.Connection,
    updates: list[tuple[str, str, str, str, str]],
) -> None:
    """拒绝改变仍被 compaction durable identity 引用的创建时间。"""

    changed_sessions = {
        identity
        for table, _identity_column, timestamp_column, identity, _value in updates
        if table == "sessions" and timestamp_column == "created_at"
    }
    if not changed_sessions:
        return
    for table in ("session_compactions", "session_compaction_prepares"):
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if exists is None:
            continue
        dependent_sessions = changed_sessions.intersection(
            str(row[0])
            for row in connection.execute(f"SELECT DISTINCT session_key FROM {table}")
        )
        if dependent_sessions:
            session_key = sorted(dependent_sessions)[0]
            raise RuntimeError(
                f"Session 创建时间仍被 {table} durable identity 引用: {session_key}"
            )


def normalize_session_timestamps(_connection: object) -> None:
    """备份后修复早期 SessionDB 写入的无时区上海时间。"""

    _ = _connection
    current = current_migration_context()
    sessions = current.workspace / "sessions.db"
    if not sessions.is_file():
        return
    with closing(sqlite3.connect(sessions)) as connection:
        updates = _pending_updates(connection)
        _check_compaction_identities(connection, updates)
    if not updates:
        return

    backup_root = current.workspace / "backups" / _MIGRATION / uuid4().hex
    _ = backup_sqlite_database(
        sessions,
        backup_root,
        migration=_MIGRATION,
    )
    with closing(sqlite3.connect(sessions)) as connection:
        _apply_updates(connection, updates)


steps = [step(normalize_session_timestamps)]
