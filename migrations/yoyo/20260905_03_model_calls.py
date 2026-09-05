"""为已有模型配置库增加调用事实；不改配置 revision、凭据或会话。"""

import sqlite3
from contextlib import closing
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260905_02_owner_records"}
__transactional__ = False

_SCHEMA = """CREATE TABLE model_calls (
    id TEXT PRIMARY KEY NOT NULL,
    binding_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started','success','unknown')),
    usage_json TEXT,
    failure TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
)"""


def migrate_model_calls(_ledger):
    """在独立备份后原子增加调用表；重复执行不改已发生的调用。"""
    workspace = current_migration_context().workspace
    path = workspace / "model-registry.sqlite3"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='model_calls'"
        ).fetchone()
        if row is not None:
            if "".join(row[0].lower().split()) != "".join(_SCHEMA.lower().split()):
                raise RuntimeError("model_calls schema 不匹配")
            return
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {
            "model_registry_meta",
            "model_connections",
            "model_definitions",
            "embedding_models",
            "model_role_bindings",
        }.issubset(tables):
            raise RuntimeError("model registry 缺少原有配置表")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("Model calls 迁移发现数据库完整性错误")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Model calls 迁移发现既有外键错误")
        backup_sqlite_database(
            path,
            workspace / "backups/model-calls-v1" / uuid4().hex,
            migration="20260905_03_model_calls",
        )
        connection.execute(_SCHEMA)


steps = [step(migrate_model_calls)]
