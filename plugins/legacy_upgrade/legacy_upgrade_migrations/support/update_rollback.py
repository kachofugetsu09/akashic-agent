"""Frozen schema check for the historical plugin update rollback migration."""
from __future__ import annotations

import sqlite3

SCHEMA = {
    "plugin_updates": """CREATE TABLE plugin_updates (
        update_id TEXT PRIMARY KEY,
        plugin_id TEXT NOT NULL,
        plugin_base TEXT NOT NULL,
        previous_pointers_json TEXT,
        candidate_pointer TEXT NOT NULL,
        previous_enabled INTEGER CHECK (previous_enabled IN (0, 1)),
        phase TEXT NOT NULL CHECK (phase IN ('armed', 'committed', 'rolled_back')),
        reload_tx_id TEXT UNIQUE REFERENCES reload_transactions(tx_id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT NOT NULL
    )""",
    "plugin_update_active": """CREATE UNIQUE INDEX plugin_update_active
        ON plugin_updates(plugin_id) WHERE phase='armed'""",
}


def check_schema(conn: sqlite3.Connection) -> bool:
    """Validate the exact old schema without opening the active plugin owner."""
    found = 0
    for name, statement in SCHEMA.items():
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if row is not None:
            if " ".join(str(row[0]).split()) != " ".join(statement.split()):
                raise ValueError(f"未知 plugin update schema: {name}")
            found += 1
    if found not in (0, len(SCHEMA)):
        raise ValueError("plugin update schema 不完整")
    return found == len(SCHEMA)
