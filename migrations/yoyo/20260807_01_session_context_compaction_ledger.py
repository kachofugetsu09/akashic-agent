from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)

__depends__ = {"20260805_01_akasha_sparse_index_v9"}
__transactional__ = False

_MIGRATION_NAME = "session-context-compaction-ledger"
_LEGACY_SCHEMA = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("trigger", "TEXT", 1, 0),
        ("summary_format_version", "INTEGER", 1, 0),
        ("summary", "TEXT", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("model_runtime_id", "TEXT", 1, 0),
        ("model", "TEXT", 1, 0),
        ("context_window", "INTEGER", 1, 0),
        ("threshold_tokens", "INTEGER", 1, 0),
        ("hard_input_tokens", "INTEGER", 1, 0),
        ("keep_recent_tokens", "INTEGER", 1, 0),
        ("tokens_before", "INTEGER", 1, 0),
        ("tokens_after", "INTEGER", 1, 0),
        ("summary_usage_json", "TEXT", 1, 0),
        ("invalidated_at", "TEXT", 0, 0),
        ("invalidated_reason", "TEXT", 0, 0),
    ),
    "named_indexes": {
        "idx_session_compactions_active": (
            ("session_key", "invalidated_at", "generation"),
            0,
        ),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (),
}
_FINAL_SCHEMA = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("trigger", "TEXT", 1, 0),
        ("summary_format_version", "INTEGER", 1, 0),
        ("summary", "TEXT", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_plan_digest", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("model_runtime_id", "TEXT", 1, 0),
        ("model", "TEXT", 1, 0),
        ("context_window", "INTEGER", 1, 0),
        ("threshold_tokens", "INTEGER", 1, 0),
        ("hard_input_tokens", "INTEGER", 1, 0),
        ("keep_recent_tokens", "INTEGER", 1, 0),
        ("tokens_before", "INTEGER", 1, 0),
        ("tokens_after", "INTEGER", 1, 0),
        ("summary_usage_json", "TEXT", 1, 0),
        ("invalidated_at", "TEXT", 0, 0),
        ("invalidated_reason", "TEXT", 0, 0),
    ),
    "named_indexes": {
        "idx_session_compactions_active": (
            ("session_key", "invalidated_at", "generation"),
            0,
        ),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (
        "CHECK (length(source_plan_digest) = 64 AND "
        "source_plan_digest NOT GLOB '*[^0-9a-f]*')",
    ),
}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    """Return whether SQLite owns the expected table name."""

    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _ensure_ledger_schema(connection: sqlite3.Connection) -> None:
    """Create or validate the additive compaction ledger without touching data."""

    # 1. Create the first-generation ledger when the table is absent.
    if not _table_exists(connection, "session_compactions"):
        connection.execute("""
            CREATE TABLE session_compactions (
                session_key TEXT NOT NULL,
                generation INTEGER NOT NULL,
                parent_generation INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                trigger TEXT NOT NULL,
                summary_format_version INTEGER NOT NULL,
                summary TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                source_from_seq INTEGER NOT NULL,
                consolidated_through_seq INTEGER NOT NULL,
                source_message_ids_json TEXT NOT NULL,
                retained_tail_json TEXT NOT NULL,
                model_runtime_id TEXT NOT NULL,
                model TEXT NOT NULL,
                context_window INTEGER NOT NULL,
                threshold_tokens INTEGER NOT NULL,
                hard_input_tokens INTEGER NOT NULL,
                keep_recent_tokens INTEGER NOT NULL,
                tokens_before INTEGER NOT NULL,
                tokens_after INTEGER NOT NULL,
                summary_usage_json TEXT NOT NULL,
                invalidated_at TEXT,
                invalidated_reason TEXT,
                PRIMARY KEY (session_key, generation),
                UNIQUE (session_key, source_ref)
            )
            """)

    # 2. Accept only the known pre- and post-digest schema identities.
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(session_compactions)")
    }
    schema = _FINAL_SCHEMA if "source_plan_digest" in columns else _LEGACY_SCHEMA
    expected_columns = {column[0] for column in schema["columns"]}
    missing = sorted(expected_columns - columns)
    if missing:
        raise RuntimeError(
            "session_compactions schema lineage 不兼容，缺少列: " + ", ".join(missing)
        )
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_compactions_active
        ON session_compactions(session_key, invalidated_at, generation)
        """)
    validate_table_schema(
        connection,
        table="session_compactions",
        columns=schema["columns"],  # type: ignore[arg-type]
        named_indexes=schema["named_indexes"],  # type: ignore[arg-type]
        auto_indexes=schema["auto_indexes"],  # type: ignore[arg-type]
        sql_fragments=schema["sql_fragments"],  # type: ignore[arg-type]
    )


def add_session_compaction_ledger(_connection: object) -> None:
    """Back up SessionDB and add the immutable compaction ledger schema."""

    _ = _connection
    current = current_migration_context()
    sessions_db = current.workspace / "sessions.db"
    if not sessions_db.exists():
        return

    # 1. Preserve a verified online backup before any DDL write.
    backup_sqlite_database(
        sessions_db,
        current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex,
        migration=_MIGRATION_NAME,
    )

    # 2. Apply only additive DDL in a short transaction.
    connection = sqlite3.connect(sessions_db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_ledger_schema(connection)
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


steps = [step(add_session_compaction_ledger)]
