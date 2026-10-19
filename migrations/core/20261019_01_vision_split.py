"""read_image_vision 从 standard_tools 拆到 vision 插件；停用过 standard_tools 的用户保持没有该工具。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import inherit_disabled

__depends__ = {"20261011_02_bundle_choices"}


def upgrade(connection: sqlite3.Connection) -> None:
    inherit_disabled("standard_tools", ("vision",), "vision-split")


steps = [step(upgrade, None)]
