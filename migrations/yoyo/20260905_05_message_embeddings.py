"""让消息向量的已有持久格式由新日志 owner 接管；不修改任何既有向量。"""
import sqlite3
from contextlib import closing
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260905_04_akasha_consumption"}
__transactional__ = False

_TABLE = """CREATE TABLE message_embeddings (
    message_id TEXT NOT NULL, content_hash TEXT NOT NULL,
    model TEXT NOT NULL, embedding BLOB NOT NULL, dim INTEGER NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (message_id, model)
)"""
_INDEX = "CREATE INDEX ix_message_embeddings_hash ON message_embeddings (content_hash, model)"


def migrate_message_embeddings(_ledger):
    """既有库只校验和补齐缺失 schema；新增能力须有备份和 Yoyo 记录。"""
    workspace = current_migration_context().workspace
    path = workspace / "sessions.db"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        missing = []
        for name, wanted in (("message_embeddings", _TABLE), ("ix_message_embeddings_hash", _INDEX)):
            row = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
            if row is None:
                missing.append(wanted)
            else:
                actual = "".join(row[0].lower().split()).replace("ifnotexists", "").rstrip(";")
                if actual != "".join(wanted.lower().split()):
                    raise RuntimeError(f"{name} schema lineage 不匹配")
        if not missing:
            return
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("消息向量迁移发现数据库完整性错误")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("消息向量迁移发现既有外键错误")
        backup_sqlite_database(path, workspace / "backups/message-embeddings-owner-v1" / uuid4().hex,
                               migration="20260905_05_message_embeddings")
        for statement in missing:
            connection.execute(statement)


steps = [step(migrate_message_embeddings)]
