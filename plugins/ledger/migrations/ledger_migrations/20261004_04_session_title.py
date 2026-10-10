"""Session 标题覆盖：只增加 title 列，不改写任何已有数据。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.context import current_migration_context
from ledger_migrations.helpers.timing import measure

__depends__ = {"20261004_03_session_soft_delete"}


def upgrade(connection: sqlite3.Connection) -> None:
    """已有消息库增加可空 title；新库由 MessageLog 建表自带该列。"""
    path = current_migration_context().workspace / "sessions.db"
    if not path.exists():
        return
    with sqlite3.connect(path) as messages:
        if messages.execute(
            "SELECT 1 FROM sqlite_master WHERE name='sessions'"
        ).fetchone() is None:
            return
        columns = {row[1] for row in messages.execute("PRAGMA table_info(sessions)")}
        if "title" in columns:
            return
        with measure("migration.session_title"):
            messages.execute("ALTER TABLE sessions ADD COLUMN title TEXT")


steps = [step(upgrade, None)]
