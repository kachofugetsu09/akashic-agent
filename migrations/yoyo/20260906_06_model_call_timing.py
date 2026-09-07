"""调用账增加可空耗时；旧记录保持未知，不从秒级时间戳推算。"""
from contextlib import closing
import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260906_05_mobile_input_rejections"}
__transactional__ = False

_OLD_SCHEMA = """CREATE TABLE model_calls (
    id TEXT PRIMARY KEY NOT NULL,
    binding_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started','success','unknown')),
    usage_json TEXT,
    failure TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT
)"""
_SCHEMA = """CREATE TABLE model_calls (
    id TEXT PRIMARY KEY NOT NULL,
    binding_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('started','success','unknown')),
    usage_json TEXT,
    failure TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    first_token_ms REAL CHECK (first_token_ms >= 0),
    duration_ms REAL CHECK (duration_ms >= 0 AND (first_token_ms IS NULL OR duration_ms >= first_token_ms))
)"""
_COLUMNS = "id,binding_json,request_digest,state,usage_json,failure,started_at,finished_at"


def migrate_model_call_timing(_ledger):
    """锁内核对、备份并原子加列；未知 schema 不允许改写。"""
    workspace = current_migration_context().workspace
    path = workspace / "model-registry.sqlite3"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as connection, connection:
        # 1. 锁住原调用账，完整保留旧字段与其他配置表。
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_calls'"
        ).fetchone()
        if row is None:
            raise RuntimeError("model_calls schema 缺失")
        sql = "".join(row[0].lower().split())
        if connection.execute(
            "SELECT name FROM sqlite_master WHERE tbl_name='model_calls' "
            "AND type IN ('index','trigger') AND sql IS NOT NULL"
        ).fetchall():
            raise RuntimeError("model_calls 有未知索引或触发器")
        if sql == "".join(_SCHEMA.lower().split()):
            return
        if sql != "".join(_OLD_SCHEMA.lower().split()):
            raise RuntimeError("model_calls schema 不匹配")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("Model 调用库完整性错误")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Model 调用库外键错误")
        before = connection.execute(f"SELECT {_COLUMNS} FROM model_calls ORDER BY rowid").fetchall()
        backup_sqlite_database(
            path, workspace / "backups/model-call-timing" / uuid4().hex,
            migration="20260906_06_model_call_timing",
        )
        # 2. 可空加列不减少旧数据；提交前核对最终 schema 与旧字段。
        connection.execute(
            "ALTER TABLE model_calls ADD COLUMN first_token_ms REAL CHECK (first_token_ms >= 0)"
        )
        connection.execute(
            "ALTER TABLE model_calls ADD COLUMN duration_ms REAL "
            "CHECK (duration_ms >= 0 AND (first_token_ms IS NULL OR duration_ms >= first_token_ms))"
        )
        after_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_calls'"
        ).fetchone()[0]
        if "".join(after_sql.lower().split()) != "".join(_SCHEMA.lower().split()):
            raise RuntimeError("Model 调用迁移后 schema 不匹配")
        after = connection.execute(f"SELECT {_COLUMNS} FROM model_calls ORDER BY rowid").fetchall()
        if before != after:
            raise RuntimeError("Model 调用旧字段发生变化")
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RuntimeError("Model 调用迁移后完整性错误")


steps = [step(migrate_model_call_timing)]
