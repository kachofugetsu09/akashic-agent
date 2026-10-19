"""技能从 standard_tools 拆到 skills 插件；用户对 standard_tools 的显式启用或停用同样作用于技能。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import inherit_choice

__depends__ = {"20261019_01_vision_split"}


def upgrade(connection: sqlite3.Connection) -> None:
    inherit_choice("standard_tools", ("skills",), "skills-split")


steps = [step(upgrade, None)]
