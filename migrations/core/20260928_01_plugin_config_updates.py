"""为插件自身配置应用增加宿主回执；不修改业务配置。"""
from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step
from agent.migrations.context import current_migration_context
from agent.plugins.config_updates import SCHEMA, check_schema

__depends__ = {"20260921_01_plugin_update_input_ref"}


def upgrade(connection: sqlite3.Connection) -> None:
    """已有 journal 先备份，新增回执表后检查完整性。"""
    path = current_migration_context().workspace / "runtime" / "plugin-reloads.sqlite3"
    if not path.exists():
        return
    with sqlite3.connect(path) as source:
        exists = source.execute("SELECT 1 FROM sqlite_master WHERE name='config_updates'").fetchone()
        if exists:
            check_schema(source)
            return
        backup = path.with_name(f"{path.name}.before-config-updates.{uuid4().hex}.bak")
        with sqlite3.connect(backup) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RuntimeError("配置回执迁移备份检查失败")
        source.execute(SCHEMA)
        check_schema(source)
        if source.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError("配置回执迁移检查失败")


steps = [step(upgrade, None)]
