"""复制可重算数据到独立库，保留原表作为旧 Core 的恢复来源。"""
from __future__ import annotations

from contextlib import closing
import sqlite3
from yoyo import step
from agent.migrations.context import current_migration_context

__depends__ = {"20261007_01_message_prefix_revision"}

_TABLES = {
    "message_embeddings": """CREATE TABLE message_embeddings (
        message_id TEXT NOT NULL, content_hash TEXT NOT NULL,
        model TEXT NOT NULL, embedding BLOB NOT NULL, dim INTEGER NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (message_id, model)
    )""",
    "message_embedding_migrations": """CREATE TABLE message_embedding_migrations (
        source_id TEXT PRIMARY KEY, completed_at TEXT NOT NULL, imported_count INTEGER NOT NULL
    )""",
}


def upgrade(_connection: sqlite3.Connection) -> None:
    """目标事务复制并核对完整原行；冲突不能覆盖，源库仅只读连接。"""
    workspace = current_migration_context().workspace
    source = workspace / "sessions.db"
    if not source.exists():
        return
    target = workspace / "sessions-derived.db"
    with closing(sqlite3.connect(target, uri=True)) as database:
        database.execute("ATTACH DATABASE ? AS original", (source.as_uri() + "?mode=ro",))
        database.execute("BEGIN IMMEDIATE")
        with database:
            for table, schema in _TABLES.items():
                if database.execute("SELECT 1 FROM original.sqlite_master WHERE name=?", (table,)).fetchone() is None:
                    continue
                if database.execute("SELECT 1 FROM main.sqlite_master WHERE name=?", (table,)).fetchone() is None:
                    database.execute(schema)
                # 名称来自固定表清单；SQLite 保证逐列类型、NULL 和 BLOB 的原值复制。
                database.execute(f"INSERT OR IGNORE INTO main.{table} SELECT * FROM original.{table}")
                missing = database.execute(
                    f"SELECT * FROM original.{table} EXCEPT SELECT * FROM main.{table} LIMIT 1"
                ).fetchone()
                if missing is not None:
                    raise ValueError(f"派生库已有冲突，保留两端数据: {table}")


steps = [step(upgrade, None)]
