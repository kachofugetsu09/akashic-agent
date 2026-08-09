from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)


__depends__ = {"20260808_06_retire_legacy_context_state"}
__transactional__ = False

_MIGRATION_NAME = "tool-result-artifacts"

SCHEMA_MANIFEST: dict[str, dict[str, tuple[str, ...]]] = {
    "tool_result_artifacts": {
        "columns": (
            "id",
            "session_key",
            "turn_id",
            "call_id",
            "tool_name",
            "content",
            "char_count",
            "created_at",
        ),
        "indexes": ("idx_tool_result_artifacts_session",),
    },
    "tool_result_reads": {
        "columns": (
            "id",
            "artifact_id",
            "session_key",
            "turn_id",
            "offset",
            "requested_limit",
            "returned_chars",
            "created_at",
        ),
        "indexes": (
            "idx_tool_result_reads_artifact",
            "idx_tool_result_reads_session",
        ),
    },
}

_SCHEMAS = {
    "tool_result_artifacts": {
        "columns": (
            ("id", "TEXT", 0, 1),
            ("session_key", "TEXT", 1, 0),
            ("turn_id", "TEXT", 1, 0),
            ("call_id", "TEXT", 1, 0),
            ("tool_name", "TEXT", 1, 0),
            ("content", "TEXT", 1, 0),
            ("char_count", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
        ),
        "named_indexes": {
            "idx_tool_result_artifacts_session": (
                ("session_key", "created_at", "id"),
                0,
            ),
        },
        "auto_indexes": (
            ("pk", ("id",)),
            ("u", ("session_key", "call_id")),
        ),
        "sql_fragments": ("CHECK (char_count >= 0)",),
    },
    "tool_result_reads": {
        "columns": (
            ("id", "TEXT", 0, 1),
            ("artifact_id", "TEXT", 1, 0),
            ("session_key", "TEXT", 1, 0),
            ("turn_id", "TEXT", 1, 0),
            ("offset", "INTEGER", 1, 0),
            ("requested_limit", "INTEGER", 1, 0),
            ("returned_chars", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
        ),
        "named_indexes": {
            "idx_tool_result_reads_artifact": (
                ("artifact_id", "created_at", "id"),
                0,
            ),
            "idx_tool_result_reads_session": (
                ("session_key", "created_at", "id"),
                0,
            ),
        },
        "auto_indexes": (("pk", ("id",)),),
        "sql_fragments": (
            "CHECK (offset >= 0)",
            "CHECK (requested_limit > 0)",
            "CHECK (returned_chars >= 0)",
        ),
    },
}


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create and validate the two append-only tool-result tables."""

    # 1. Add immutable artifact storage and its session lookup.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS tool_result_artifacts (
            id          TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            turn_id     TEXT NOT NULL,
            call_id     TEXT NOT NULL,
            tool_name   TEXT NOT NULL,
            content     TEXT NOT NULL,
            char_count  INTEGER NOT NULL CHECK (char_count >= 0),
            created_at  TEXT NOT NULL,
            UNIQUE (session_key, call_id)
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_result_artifacts_session
        ON tool_result_artifacts(session_key, created_at, id)
        """)

    # 2. Add append-only read evidence and its two query indexes.
    connection.execute("""
        CREATE TABLE IF NOT EXISTS tool_result_reads (
            id              TEXT PRIMARY KEY,
            artifact_id     TEXT NOT NULL,
            session_key     TEXT NOT NULL,
            turn_id         TEXT NOT NULL,
            offset          INTEGER NOT NULL CHECK (offset >= 0),
            requested_limit INTEGER NOT NULL CHECK (requested_limit > 0),
            returned_chars  INTEGER NOT NULL CHECK (returned_chars >= 0),
            created_at      TEXT NOT NULL
        )
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_result_reads_artifact
        ON tool_result_reads(artifact_id, created_at, id)
        """)
    connection.execute("""
        CREATE INDEX IF NOT EXISTS idx_tool_result_reads_session
        ON tool_result_reads(session_key, created_at, id)
        """)

    # 3. Refuse unknown same-name schemas instead of normalizing them.
    for table, schema in _SCHEMAS.items():
        validate_table_schema(
            connection,
            table=table,
            columns=schema["columns"],  # type: ignore[arg-type]
            named_indexes=schema["named_indexes"],  # type: ignore[arg-type]
            auto_indexes=schema["auto_indexes"],  # type: ignore[arg-type]
            sql_fragments=schema["sql_fragments"],  # type: ignore[arg-type]
        )


def add_tool_result_artifacts(_connection: object) -> None:
    """Back up SessionDB, then add only the validated artifact schema."""

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
            _ensure_schema(connection)
        except BaseException:
            connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


steps = [step(add_tool_result_artifacts)]
