"""增加 owner 自有记录空间，与消息提交共用事务；不改旧消息或学习状态。"""

import sqlite3
from contextlib import closing
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260905_01_message_log"}
__transactional__ = False

_SCHEMA = """CREATE TABLE owner_records (
    owner TEXT NOT NULL, key TEXT NOT NULL, version INTEGER NOT NULL,
    value TEXT NOT NULL, PRIMARY KEY(owner, key)
)"""


def migrate_owner_records(_ledger):
    """核对目标名称并备份，再原子增加空表；重复运行保留已有 owner 记录。"""
    workspace = current_migration_context().workspace
    path = workspace / "sessions.db"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='owner_records'"
        ).fetchone()
        if row is not None:
            if "".join(row[0].lower().split()) != "".join(_SCHEMA.lower().split()):
                raise RuntimeError("owner_records schema 不匹配")
            return
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("owner_records 迁移发现数据库完整性错误")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("owner_records 迁移发现既有外键错误")
        backup_sqlite_database(
            path,
            workspace / "backups/owner-records-v1" / uuid4().hex,
            migration="20260905_02_owner_records",
        )
        connection.execute(_SCHEMA)


steps = [step(migrate_owner_records)]
