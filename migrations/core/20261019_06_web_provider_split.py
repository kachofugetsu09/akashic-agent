"""Preserve Web search and fetch when their providers move to separate plugins."""
from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.successors import inherit_choice

__depends__ = {"20261019_05_working_directory_rename"}


def upgrade(connection: sqlite3.Connection) -> None:
    """Keep explicit provider choices and inherit the old Web tools switch."""
    inherit_choice("standard_web", ("web_search_exa", "web_fetch_http"), "web-provider-split")


steps = [step(upgrade, None)]
