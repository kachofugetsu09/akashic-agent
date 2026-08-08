from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database


__depends__ = {"20260805_01_akasha_sparse_index_v9"}
__transactional__ = False

_MIGRATION_NAME = "session-context-compaction-ledger"
_REQUIRED_COLUMNS = {
    "session_key",
    "generation",
    "parent_generation",
    "created_at",
    "trigger",
    "summary_format_version",
    "summary",
    "source_ref",
    "source_from_seq",
    "consolidated_through_seq",
    "source_message_ids_json",
    "retained_tail_json",
    "model_runtime_id",
    "model",
    "context_window",
    "threshold_tokens",
    "hard_input_tokens",
    "keep_recent_tokens",
    "tokens_before",
    "tokens_after",
    "summary_usage_json",
    "invalidated_at",
    "invalidated_reason",
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
        connection.execute(
            """
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
            """
        )

    # 2. Existing tables may already be the later digest schema; require only
    #    the columns owned by this additive migration and leave every row intact.
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(session_compactions)")
    }
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        raise RuntimeError(
            "session_compactions schema lineage 不兼容，缺少列: "
            + ", ".join(missing)
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_compactions_active
        ON session_compactions(session_key, invalidated_at, generation)
        """
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

    # 2. Apply only additive DDL in a short transaction; rollback on any
    #    schema-lineage failure so no partial table/index is published.
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
