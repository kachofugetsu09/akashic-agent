"""Let workspace patch rows that only copied the preset follow later preset updates."""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import follow_preset

__depends__ = {"20261019_06_web_provider_split"}


def upgrade(connection: sqlite3.Connection) -> None:
    """Drop patch fields equal to the target preset; keep every real user choice."""
    follow_preset("follow-preset")


steps = [step(upgrade, None)]
