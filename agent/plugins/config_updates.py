"""配置请求使用现有 reload journal，只暂存未完成请求的配置，完成后保留请求摘要。"""
from __future__ import annotations

import sqlite3

SCHEMA = """CREATE TABLE config_updates (
    request_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    previous_input TEXT NOT NULL,
    input_ref TEXT NOT NULL,
    config_revision TEXT NOT NULL,
    pending_config TEXT,
    state TEXT NOT NULL CHECK(state IN ('accepted','selected','active','failed')),
    error TEXT NOT NULL DEFAULT ''
)"""


# 已发布的 20260928 migration 引用此入口；它必须识别尚未升级的旧表。
_MIGRATION_SCHEMA = SCHEMA.replace("    config_revision TEXT NOT NULL,\n    pending_config TEXT,\n", "")


def check_schema(conn: sqlite3.Connection) -> None:
    """仅供已发布迁移核对已知旧/新表，运行期使用 check_current_schema。"""
    _check(conn, (SCHEMA, _MIGRATION_SCHEMA))


def check_current_schema(conn: sqlite3.Connection) -> None:
    """正常运行只接受当前结构，旧表必须显式完成 Core 迁移。"""
    _check(conn, (SCHEMA,))


def _check(conn: sqlite3.Connection, schemas: tuple[str, ...]) -> None:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name='config_updates' AND type='table'").fetchone()
    if row is None or ' '.join(str(row[0]).split()) not in {' '.join(schema.split()) for schema in schemas}:
        raise RuntimeError("config_updates schema 不符；请执行 Core migration")
