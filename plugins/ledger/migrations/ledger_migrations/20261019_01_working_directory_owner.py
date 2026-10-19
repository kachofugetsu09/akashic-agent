"""standard_tools 改名为 working_directory：会话目录状态随插件身份改归新 owner，不改写内容。"""
from __future__ import annotations

import sqlite3
from contextlib import closing

from yoyo import step

from agent.migrations.context import current_migration_context
from ledger_migrations.helpers.timing import measure

__depends__ = {"20261011_01_derived_storage"}

_PREDECESSOR = "plugin:standard_tools"
_SUCCESSOR = "plugin:working_directory"


# 1. 只改 plugin:standard_tools 与其市场后缀；2. 新 owner 已有同键记录时保留新的，旧记录删除。
def upgrade(connection: sqlite3.Connection) -> None:
    path = current_migration_context().workspace / "sessions.db"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as messages, messages:
        if messages.execute("SELECT 1 FROM sqlite_master WHERE name='owner_records'").fetchone() is None:
            return
        with measure("migration.working_directory_owner"):
            match = "owner = ? OR owner LIKE ?"
            arguments = (_PREDECESSOR, _PREDECESSOR + "@%")
            successor = f"? || substr(owner, {len(_PREDECESSOR) + 1})"
            messages.execute(
                f"UPDATE OR IGNORE owner_records SET owner = {successor} WHERE {match}",
                (_SUCCESSOR, *arguments),
            )
            messages.execute(f"DELETE FROM owner_records WHERE {match}", arguments)


steps = [step(upgrade, None)]
