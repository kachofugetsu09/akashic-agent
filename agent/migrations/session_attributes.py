"""为既有 Session 增加固定属性，保留原 metadata 和全部消息原文。"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

from agent.migrations.session_db_backup import backup_sqlite_database
from session.log import (
    SessionAttributes, _session_schemas, _SESSION_ATTRIBUTES_COLUMN, _sql,
    decode_attributes, encode_attributes,
)


def _schema(connection: sqlite3.Connection) -> bool:
    """核对已知表身份；返回是否已经有当前属性列。"""
    row = connection.execute("SELECT sql FROM sqlite_master WHERE name='sessions'").fetchone()
    if row is None:
        raise ValueError("Session 属性迁移缺少 sessions 表")
    statement = _sql(row[0])
    schemas = _session_schemas()
    if statement not in schemas:
        raise ValueError("Session 属性迁移遇到未知 schema")
    if not schemas[statement]:
        return False
    for (raw,) in connection.execute("SELECT attributes FROM sessions"):
        _ = decode_attributes(raw)
    return True


def migrate(path: Path, backup_root: Path) -> None:
    """一项 SQLite 事务接纳新属性；提交后中断时重复检查，不覆盖后续会话。"""
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection:
        if _schema(connection):
            return
    _ = backup_sqlite_database(path, backup_root, migration="session-attributes")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            if not _schema(connection):
                # 1. 旧行没有全 Session 的隐藏/排除事实；逐消息的历史 effects 保持原样。
                rows = connection.execute("SELECT * FROM sessions ORDER BY key").fetchall()
                messages = connection.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall()
                attributes = encode_attributes(SessionAttributes())
                connection.execute("ALTER TABLE sessions ADD COLUMN " + _SESSION_ATTRIBUTES_COLUMN)
                current = connection.execute("SELECT * FROM sessions ORDER BY key").fetchall()
                if [row[:-1] for row in current] != rows or any(row[-1] != attributes for row in current):
                    raise ValueError("Session 属性迁移改变了既有字段")
                if connection.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall() != messages:
                    raise ValueError("Session 属性迁移改变了消息")
            # 2. schema、全部属性、外键和完整性均在唯一提交点之前核对。
            if not _schema(connection):
                raise ValueError("Session 属性未完成迁移")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Session 属性迁移完整性检查失败")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Session 属性迁移外键检查失败")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
