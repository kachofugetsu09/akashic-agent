"""Session 软删除：只增加 deleted_at 列，不改写任何已有数据。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.context import current_migration_context
from utils.timing import measure

__depends__ = {"20261004_02_message_body_kind_index"}


def upgrade(connection: sqlite3.Connection) -> None:
    """已有消息库增加可空 deleted_at；新库由 MessageLog 建表自带该列。"""
    path = current_migration_context().workspace / "sessions.db"
    if not path.exists():
        return
    with sqlite3.connect(path) as messages:
        if messages.execute(
            "SELECT 1 FROM sqlite_master WHERE name='sessions'"
        ).fetchone() is None:
            return
        columns = {row[1] for row in messages.execute("PRAGMA table_info(sessions)")}
        if "deleted_at" in columns:
            return
        with measure("migration.session_soft_delete"):
            messages.execute("ALTER TABLE sessions ADD COLUMN deleted_at TEXT")


steps = [step(upgrade, None)]
