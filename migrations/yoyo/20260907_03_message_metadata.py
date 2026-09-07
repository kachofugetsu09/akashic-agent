"""为消息追加普通 metadata 列，不改写任何历史正文或插件内容。"""
from contextlib import closing
from pathlib import Path
import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database
from session.log import _sql
from session.message import freeze_metadata
from session.message_codec import _unique_fields
import json

__depends__ = {"20260907_02_retire_legacy_agent_config"}
__transactional__ = False

_OLD_SCHEMA = """CREATE TABLE messages (
    id TEXT PRIMARY KEY, session_key TEXT NOT NULL, seq INTEGER NOT NULL,
    ts TEXT NOT NULL, author TEXT NOT NULL, source TEXT NOT NULL,
    body TEXT NOT NULL, UNIQUE(session_key, seq))"""
_COLUMN = "metadata TEXT NOT NULL DEFAULT '{}'"
_NEW_SCHEMA = _OLD_SCHEMA.replace("UNIQUE(session_key, seq)", _COLUMN + ", UNIQUE(session_key, seq)")


def _schema(connection: sqlite3.Connection) -> bool:
    """只接受栈内已知的消息 schema；已有扩展必须仍是合法 JSON 对象。"""
    row = connection.execute("SELECT sql FROM sqlite_master WHERE name='messages'").fetchone()
    if row is None:
        raise ValueError("Message metadata 迁移缺少 messages 表")
    actual = _sql(row[0])
    if actual == _sql(_OLD_SCHEMA):
        return False
    if actual != _sql(_NEW_SCHEMA):
        raise ValueError("Message metadata 迁移遇到未知 schema")
    for (raw,) in connection.execute("SELECT metadata FROM messages"):
        freeze_metadata(json.loads(raw, object_pairs_hook=_unique_fields))
    return True


def migrate(path: Path, backup_root: Path) -> None:
    """先备份，再在一项事务内加列并核对完整旧行；重放不覆盖已有扩展。"""
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection:
        if _schema(connection):
            return
    backup = backup_sqlite_database(path, backup_root, migration="message-metadata")
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        try:
            if not _schema(connection):
                # 1. 备份与实际改动前缀必须一致，防止恢复点落后于迁移输入。
                before = connection.execute("SELECT rowid,* FROM messages ORDER BY rowid").fetchall()
                with closing(sqlite3.connect(backup)) as saved:
                    if saved.execute("SELECT rowid,* FROM messages ORDER BY rowid").fetchall() != before:
                        raise ValueError("Message metadata 备份与来源不一致")
                connection.execute("ALTER TABLE messages ADD COLUMN " + _COLUMN)
                after = connection.execute("SELECT rowid,* FROM messages ORDER BY rowid").fetchall()
                if [row[:-1] for row in after] != before or any(row[-1] != "{}" for row in after):
                    raise ValueError("Message metadata 迁移改变了既有消息")
            # 2. 唯一提交点前检查 schema、完整性和全部外键；没有减少协议。
            if not _schema(connection):
                raise ValueError("Message metadata 未完成迁移")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Message metadata 完整性检查失败")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Message metadata 外键检查失败")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


def migrate_message_metadata(_ledger):
    workspace = current_migration_context().workspace
    migrate(workspace / "sessions.db", workspace / "backups/message-metadata" / uuid4().hex)


steps = [step(migrate_message_metadata)]
