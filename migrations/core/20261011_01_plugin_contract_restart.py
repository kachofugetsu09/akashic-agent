"""为安装回执增加明确的合同重启事实表，不改写旧字段或业务数据。"""
from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.plugins.update_rollback import CONTRACT_RESTART_SCHEMA, check_contract_restart_schema, plugin_update_schema_state

__depends__ = {"20261006_01_config_update_receipts"}


def upgrade(connection: sqlite3.Connection) -> None:
    """备份真实 journal 后只增重启事实表，并核对原记录完整保留。"""
    path = current_migration_context().workspace / "runtime/plugin-reloads.sqlite3"
    if not path.exists():
        return
    with sqlite3.connect(path) as source:
        shape = plugin_update_schema_state(source)
        if shape != "new":
            raise RuntimeError("合同重启迁移需要已知 input_ref schema")
        if source.execute("SELECT 1 FROM sqlite_master WHERE name='plugin_contract_restarts'").fetchone():
            check_contract_restart_schema(source)
            return
        # 1. SQLite 原生备份包含 WAL；保留命名恢复点，不自动降级。
        backup = path.with_name(f"{path.name}.before-contract-restart.{uuid4().hex}.bak")
        with sqlite3.connect(backup) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RuntimeError("合同重启备份完整性检查失败")
        columns = tuple(str(row[1]) for row in source.execute("PRAGMA table_info(plugin_updates)"))
        query = "SELECT rowid," + ",".join(columns) + " FROM plugin_updates ORDER BY rowid"
        rows = source.execute(query).fetchall()
        # 2. 不改已有表和行；重启事实只按 update_id 追加。
        source.execute("BEGIN IMMEDIATE")
        source.execute(CONTRACT_RESTART_SCHEMA)
        check_contract_restart_schema(source)
        if source.execute(query).fetchall() != rows:
            raise RuntimeError("合同重启迁移改变了原记录或得到未知 schema")
        if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError("合同重启迁移完整性检查失败")


steps = [step(upgrade, None)]
