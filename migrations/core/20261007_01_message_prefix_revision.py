"""为消息增量读取增加事务内的前缀失效标记。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.context import current_migration_context
from session.log import create_message_prefix_revision

__depends__ = {"20261004_02_message_body_kind_index"}


def upgrade(connection: sqlite3.Connection) -> None:
    """只新增派生标记和触发器；旧消息、身份、顺序和正文全部保留。"""
    path = current_migration_context().workspace / "sessions.db"
    if not path.exists():
        return
    with sqlite3.connect(path) as messages:
        messages.row_factory = sqlite3.Row
        if messages.execute("SELECT 1 FROM sqlite_master WHERE name='messages'").fetchone() is None:
            return
        messages.execute("BEGIN IMMEDIATE")
        create_message_prefix_revision(messages)
        if messages.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("消息前缀标记迁移完整性检查失败")


steps = [step(upgrade, None)]
