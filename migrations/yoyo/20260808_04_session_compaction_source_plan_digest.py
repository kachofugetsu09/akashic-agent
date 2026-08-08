from __future__ import annotations

import re
import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)


__depends__ = {"20260808_02_session_compaction_prepares"}
__transactional__ = False

_MIGRATION_NAME = "session-compaction-source-plan-digest"
_SOURCE_PLAN_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_PLAN_DIGEST_CHECK = (
    "CHECK (length(source_plan_digest) = 64 AND "
    "source_plan_digest NOT GLOB '*[^0-9a-f]*')"
)


# This manifest is the final SessionDB schema contract shared by Store and recovery.
SCHEMA_MANIFEST: dict[str, dict[str, tuple[str, ...]]] = {
    "session_compactions": {
        "columns": (
            "session_key",
            "generation",
            "parent_generation",
            "created_at",
            "trigger",
            "summary_format_version",
            "summary",
            "source_ref",
            "source_plan_digest",
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
        ),
        "indexes": ("idx_session_compactions_active",),
    },
}


_LEGACY_TABLE_SCHEMA = {
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

_FINAL_TABLE_SCHEMA = {
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
    "sql_fragments": (_SOURCE_PLAN_DIGEST_CHECK,),
}


def _create_final_table(connection: sqlite3.Connection) -> None:
    """Create the final ledger table and its query index."""

    connection.execute(
        f"""
        CREATE TABLE session_compactions (
            session_key TEXT NOT NULL,
            generation INTEGER NOT NULL,
            parent_generation INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            trigger TEXT NOT NULL,
            summary_format_version INTEGER NOT NULL,
            summary TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            source_plan_digest TEXT NOT NULL {_SOURCE_PLAN_DIGEST_CHECK},
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
    connection.execute(
        """
        CREATE INDEX idx_session_compactions_active
        ON session_compactions(session_key, invalidated_at, generation)
        """
    )


def _validate_schema(
    connection: sqlite3.Connection,
    schema: dict[str, object],
) -> None:
    """Validate one ledger schema identity through the shared SQLite checker."""

    validate_table_schema(
        connection,
        table="session_compactions",
        columns=schema["columns"],  # type: ignore[arg-type]
        named_indexes=schema["named_indexes"],  # type: ignore[arg-type]
        auto_indexes=schema["auto_indexes"],  # type: ignore[arg-type]
        sql_fragments=schema["sql_fragments"],  # type: ignore[arg-type]
    )


def _validate_digest_values(connection: sqlite3.Connection) -> None:
    """Reject persisted rows whose source-plan identity is absent or malformed."""

    rows = connection.execute(
        "SELECT session_key, generation, source_plan_digest FROM session_compactions"
    ).fetchall()
    for row in rows:
        digest = row[2]
        if not isinstance(digest, str) or _SOURCE_PLAN_DIGEST_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(
                "session_compactions source_plan_digest 非法: "
                f"{row[0]}:{row[1]}"
            )


def _rebuild_empty_legacy_table(connection: sqlite3.Connection) -> None:
    """Rebuild the zero-row pre-digest table without inventing source identities."""

    legacy_name = "session_compactions__source_plan_digest_legacy"
    existing = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (legacy_name,),
    ).fetchone()
    if existing is not None:
        raise RuntimeError("session_compactions schema upgrade staging table 已存在")
    row = connection.execute("SELECT COUNT(1) FROM session_compactions").fetchone()
    if row is None or int(row[0]) != 0:
        raise RuntimeError("session_compactions 缺少 source_plan_digest 且已有数据")

    # 1. Keep the whole operation in the caller transaction; rollback restores the old table.
    connection.execute("DROP INDEX IF EXISTS idx_session_compactions_active")
    connection.execute(
        "ALTER TABLE session_compactions RENAME TO " + legacy_name
    )
    _create_final_table(connection)
    _validate_schema(connection, _FINAL_TABLE_SCHEMA)
    connection.execute("DROP TABLE " + legacy_name)


def _ensure_source_plan_digest_schema(connection: sqlite3.Connection) -> None:
    """Upgrade only a verifiable empty legacy ledger; reject ambiguous rows."""

    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_compactions'"
    ).fetchone()
    if table is None:
        raise RuntimeError("session_compactions schema lineage 不兼容，表定义缺失")
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(session_compactions)")
    }
    if "source_plan_digest" not in columns:
        count_row = connection.execute(
            "SELECT COUNT(1) FROM session_compactions"
        ).fetchone()
        count = int(count_row[0]) if count_row is not None else 0
        if count:
            raise RuntimeError(
                "session_compactions 缺少 source_plan_digest 且已有数据，拒绝猜测回填"
            )
        _validate_schema(connection, _LEGACY_TABLE_SCHEMA)
        _rebuild_empty_legacy_table(connection)
        return
    _validate_schema(connection, _FINAL_TABLE_SCHEMA)
    _validate_digest_values(connection)


def add_source_plan_digest(_connection: object) -> None:
    """Back up and publish the final source-plan digest ledger schema."""

    _ = _connection
    current = current_migration_context()
    sessions_db = current.workspace / "sessions.db"
    if not sessions_db.exists():
        return
    backup_sqlite_database(
        sessions_db,
        current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex,
        migration=_MIGRATION_NAME,
    )
    connection = sqlite3.connect(sessions_db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_source_plan_digest_schema(connection)
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


steps = [step(add_source_plan_digest)]
