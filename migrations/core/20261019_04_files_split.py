"""文件工具从 standard_tools 拆到 files 插件；停用过 standard_tools 的用户保持没有文件工具。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import inherit_disabled

__depends__ = {"20261019_03_shell_split"}


def upgrade(connection: sqlite3.Connection) -> None:
    inherit_disabled("standard_tools", ("files",), "files-split")


steps = [step(upgrade, None)]
