"""standard_tools 只剩会话工作目录，改名为 working_directory；用户对它的停用选择随 row 迁移。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import rename_row

__depends__ = {"20261019_04_files_split"}


def upgrade(connection: sqlite3.Connection) -> None:
    rename_row("standard_tools", "working_directory", "working-directory-rename")


steps = [step(upgrade, None)]
