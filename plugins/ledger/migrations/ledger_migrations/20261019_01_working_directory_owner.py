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


# 只认 plugin:standard_tools 本身与 "@市场" 后缀；用 substr 精确比较，插件名里的 "_" 不能当 LIKE 通配符。
_ARGUMENTS = (_PREDECESSOR, len(_PREDECESSOR) + 1, _PREDECESSOR + "@")


def _match(column: str) -> str:
    return f"({column} = ? OR substr({column}, 1, ?) = ?)"


def _renamed(column: str) -> str:
    return f"? || substr({column}, {len(_PREDECESSOR) + 1})"


# 1. 只改前任 owner；2. 新 owner 已有同键记录时，内容相同才去重，不同就失败并回滚，两边都保留。
def upgrade(connection: sqlite3.Connection) -> None:
    path = current_migration_context().workspace / "sessions.db"
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as messages, messages:
        if messages.execute("SELECT 1 FROM sqlite_master WHERE name='owner_records'").fetchone() is None:
            return
        with measure("migration.working_directory_owner"):
            conflicts = messages.execute(
                f"""SELECT old.owner, old.key FROM owner_records AS old
                    JOIN owner_records AS new ON new.owner = {_renamed("old.owner")} AND new.key = old.key
                    WHERE {_match("old.owner")}
                      AND (new.version IS NOT old.version OR new.value IS NOT old.value)""",
                (_SUCCESSOR, *_ARGUMENTS),
            ).fetchall()
            if conflicts:
                raise RuntimeError(f"working_directory 已有不同内容的同键记录，拒绝覆盖: {conflicts}")
            _ = messages.execute(
                f"UPDATE OR IGNORE owner_records SET owner = {_renamed('owner')} WHERE {_match('owner')}",
                (_SUCCESSOR, *_ARGUMENTS),
            )
            _ = messages.execute(f"DELETE FROM owner_records WHERE {_match('owner')}", _ARGUMENTS)


steps = [step(upgrade, None)]
