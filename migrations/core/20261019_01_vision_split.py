"""read_image_vision 从 standard_tools 拆到 vision 插件；用户对 standard_tools 的显式启用或停用同样作用于 vision。"""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import inherit_choice

__depends__ = {"20261011_02_bundle_choices"}


def upgrade(connection: sqlite3.Connection) -> None:
    inherit_choice("standard_tools", ("vision",), "vision-split")


steps = [step(upgrade, None)]
