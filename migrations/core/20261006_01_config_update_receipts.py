"""配置请求以摘要核对重放，只暂存尚未发布的正文。"""
from __future__ import annotations

import json
import re
import sqlite3
from uuid import uuid4

from yoyo import step
from agent.migrations.context import current_migration_context
from agent.plugins.config_updates import SCHEMA, check_schema

__depends__ = {"20261004_04_session_title"}


def upgrade(connection: sqlite3.Connection) -> None:
    """只转换宿主配置回执表；原字段、行序和历史归档全部保留。"""
    workspace = current_migration_context().workspace
    path = workspace / "runtime/plugin-reloads.sqlite3"
    if not path.exists():
        return
    with sqlite3.connect(path) as source:
        source.row_factory = sqlite3.Row
        columns = tuple(row[1] for row in source.execute("PRAGMA table_info(config_updates)"))
        if not columns:
            return
        if "config_revision" in columns:
            check_schema(source)
            return
        if columns != ("request_id", "plugin_id", "previous_input", "input_ref", "state", "error"):
            raise RuntimeError("旧配置回执结构未知，停止升级")
        # 1. 备份实际账本；旧输入只在本次显式迁移读取，不参与正常运行。
        backup = path.with_name(f"{path.name}.before-config-receipts.{uuid4().hex}.bak")
        with sqlite3.connect(backup) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RuntimeError("配置回执备份完整性检查失败")
        rows = [dict(row) for row in source.execute("SELECT rowid,* FROM config_updates ORDER BY rowid")]
        candidates: list[tuple[dict[str, object], str, str | None]] = []
        for row in rows:
            ref = row["input_ref"]
            if not isinstance(ref, str) or re.fullmatch(r"[0-9a-f]{64}", ref) is None:
                raise ValueError("旧配置输入身份无效")
            archived = workspace / "runtime/plugin-archives" / f"{ref}.json"
            if archived.is_symlink():
                raise ValueError("旧配置记录不能是符号链接")
            record = json.loads(archived.read_text())
            revision = record["config_revision"]
            if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{64}", revision) is None:
                raise ValueError("旧配置请求摘要无效")
            pending = json.dumps(record["config"], ensure_ascii=False) if row["state"] != "active" else None
            candidates.append((row, revision, pending))
        # 2. 同一个事务替换表形状；不改变消息库、插件配置文件或旧归档。
        source.execute("BEGIN IMMEDIATE")
        source.execute("ALTER TABLE config_updates RENAME TO config_updates_before_upgrade")
        source.execute(SCHEMA)
        for row, revision, pending in candidates:
            source.execute("INSERT INTO config_updates(rowid,request_id,plugin_id,previous_input,input_ref,"
                           "config_revision,pending_config,state,error) VALUES (?,?,?,?,?,?,?,?,?)",
                           (row["rowid"], row["request_id"], row["plugin_id"], row["previous_input"],
                            row["input_ref"], revision, pending, row["state"], row["error"]))
        retained = [dict(row) for row in source.execute(
            "SELECT rowid,request_id,plugin_id,previous_input,input_ref,state,error FROM config_updates ORDER BY rowid")]
        if retained != rows:
            raise RuntimeError("配置回执原字段或行序变化，停止升级")
        source.execute("DROP TABLE config_updates_before_upgrade")
        check_schema(source)
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("配置回执升级完整性检查失败")


steps = [step(upgrade, None)]
