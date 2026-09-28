"""配置请求使用现有 reload journal，固定输入保留完整恢复依据。"""
from __future__ import annotations

import sqlite3

SCHEMA = """CREATE TABLE config_updates (
    request_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    previous_input TEXT NOT NULL,
    input_ref TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('accepted','selected','active','failed')),
    error TEXT NOT NULL DEFAULT ''
)"""


def check_schema(conn: sqlite3.Connection) -> None:
    """未知形状必须显式迁移，不在运行时自动补表。"""
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name='config_updates' AND type='table'").fetchone()
    if row is None or ' '.join(str(row[0]).split()) != ' '.join(SCHEMA.split()):
        raise RuntimeError("config_updates schema 不符；请执行 Core migration")
