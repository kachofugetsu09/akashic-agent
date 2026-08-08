from __future__ import annotations

import sqlite3

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260807_01_session_context_compaction_ledger"}


# This manifest is the migration contract shared by SessionStore and recovery readers.
SCHEMA_MANIFEST: dict[str, dict[str, tuple[str, ...]]] = {
    "session_delete_audits": {
        "columns": (
            "audit_id",
            "targets_json",
            "message_ids_json",
            "compactions_json",
            "action_source",
            "cascade",
            "backup_path",
            "started_at",
            "completed_at",
            "result",
            "deleted_count",
            "error",
        ),
        "indexes": ("idx_session_delete_audits_time",),
    },
    "session_source_mutation_audits": {
        "columns": (
            "audit_id",
            "operation",
            "session_key",
            "message_ids_json",
            "action_source",
            "backup_path",
            "completed_at",
        ),
        "indexes": ("idx_source_mutation_audits_lookup",),
    },
}


def _ensure_table(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    create_sql: str,
    index_sql: str,
) -> None:
    existing = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if existing is None:
        connection.execute(create_sql)
    actual = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    }
    missing = sorted(set(columns) - actual)
    if missing:
        raise RuntimeError(f"{table} schema lineage 不兼容，缺少列: {', '.join(missing)}")
    connection.execute(index_sql)


def add_session_mutation_audits(connection: object) -> None:
    """Create and validate the append-only SessionDB audit tables."""

    _ = connection
    sessions_db = current_migration_context().workspace / "sessions.db"
    if not sessions_db.exists():
        return
    sessions_connection = sqlite3.connect(sessions_db)
    try:
        _ensure_table(
            sessions_connection,
            "session_delete_audits",
            SCHEMA_MANIFEST["session_delete_audits"]["columns"],
            """
            CREATE TABLE session_delete_audits (
                audit_id TEXT PRIMARY KEY,
                targets_json TEXT NOT NULL,
                message_ids_json TEXT NOT NULL,
                compactions_json TEXT NOT NULL,
                action_source TEXT NOT NULL,
                cascade INTEGER NOT NULL CHECK (cascade IN (0, 1)),
                backup_path TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                result TEXT NOT NULL,
                deleted_count INTEGER NOT NULL,
                error TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_session_delete_audits_time
            ON session_delete_audits(completed_at, audit_id)
            """,
        )
        _ensure_table(
            sessions_connection,
            "session_source_mutation_audits",
            SCHEMA_MANIFEST["session_source_mutation_audits"]["columns"],
            """
            CREATE TABLE session_source_mutation_audits (
                audit_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                session_key TEXT NOT NULL,
                message_ids_json TEXT NOT NULL,
                action_source TEXT NOT NULL,
                backup_path TEXT,
                completed_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_source_mutation_audits_lookup
            ON session_source_mutation_audits(session_key, completed_at, audit_id)
            """,
        )
        sessions_connection.commit()
    finally:
        sessions_connection.close()


steps = [step(add_session_mutation_audits)]
