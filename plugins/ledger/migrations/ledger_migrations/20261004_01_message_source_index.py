"""Index Session/source prefixes without rewriting durable messages."""
from __future__ import annotations

import sqlite3

from yoyo import step
from agent.migrations.context import current_migration_context
from ledger_migrations.helpers.schema import create_message_source_index
from ledger_migrations.helpers.timing import measure

__depends__ = {"20260928_01_plugin_config_updates"}


def upgrade(connection: sqlite3.Connection) -> None:
    """Existing message stores add a rebuildable index; fresh stores own setup."""
    path = current_migration_context().workspace / "sessions.db"
    if not path.exists():
        return
    with sqlite3.connect(path) as messages:
        messages.row_factory = sqlite3.Row
        if messages.execute("SELECT 1 FROM sqlite_master WHERE name='messages'").fetchone() is None:
            return
        with measure("migration.message_source_index"):
            create_message_source_index(messages)


steps = [step(upgrade, None)]
