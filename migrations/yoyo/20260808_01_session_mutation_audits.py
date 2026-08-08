from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)

__depends__ = {"20260807_01_session_context_compaction_ledger"}
__transactional__ = False


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

_TABLE_SCHEMAS = {
    "session_delete_audits": {
        "columns": (
            ("audit_id", "TEXT", 0, 1),
            ("targets_json", "TEXT", 1, 0),
            ("message_ids_json", "TEXT", 1, 0),
            ("compactions_json", "TEXT", 1, 0),
            ("action_source", "TEXT", 1, 0),
            ("cascade", "INTEGER", 1, 0),
            ("backup_path", "TEXT", 0, 0),
            ("started_at", "TEXT", 1, 0),
            ("completed_at", "TEXT", 1, 0),
            ("result", "TEXT", 1, 0),
            ("deleted_count", "INTEGER", 1, 0),
            ("error", "TEXT", 0, 0),
        ),
        "named_indexes": {
            "idx_session_delete_audits_time": (("completed_at", "audit_id"), 0),
        },
        "auto_indexes": (("pk", ("audit_id",)),),
        "sql_fragments": ("CHECK (cascade IN (0, 1))",),
    },
    "session_source_mutation_audits": {
        "columns": (
            ("audit_id", "TEXT", 0, 1),
            ("operation", "TEXT", 1, 0),
            ("session_key", "TEXT", 1, 0),
            ("message_ids_json", "TEXT", 1, 0),
            ("action_source", "TEXT", 1, 0),
            ("backup_path", "TEXT", 0, 0),
            ("completed_at", "TEXT", 1, 0),
        ),
        "named_indexes": {
            "idx_source_mutation_audits_lookup": (
                ("session_key", "completed_at", "audit_id"),
                0,
            ),
        },
        "auto_indexes": (("pk", ("audit_id",)),),
        "sql_fragments": (),
    },
}


def _ensure_table(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    create_sql: str,
    index_sql: str,
) -> None:
    """创建一张审计表，并校验表与索引的 schema identity。"""

    # 1. 只创建缺失的表；已有表必须通过 manifest 校验。
    existing = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if existing is None:
        connection.execute(create_sql)

    # 2. 添加命名索引前，先校验列和内联约束。
    schema = _TABLE_SCHEMAS[table]
    validate_table_schema(
        connection,
        table=table,
        columns=schema["columns"],
        named_indexes=schema["named_indexes"],
        auto_indexes=schema["auto_indexes"],
        sql_fragments=schema["sql_fragments"],
        validate_named_indexes=False,
    )

    # 3. 创建并校验本迁移持有的查询索引。
    connection.execute(index_sql)
    validate_table_schema(
        connection,
        table=table,
        columns=schema["columns"],
        named_indexes=schema["named_indexes"],
        auto_indexes=schema["auto_indexes"],
        sql_fragments=schema["sql_fragments"],
    )


def add_session_mutation_audits(connection: object) -> None:
    """Create and validate the append-only SessionDB audit tables."""

    _ = connection
    current = current_migration_context()
    sessions_db = current.workspace / "sessions.db"
    if not sessions_db.exists():
        return
    backup_sqlite_database(
        sessions_db,
        current.workspace / "backups" / "session-mutation-audits" / uuid4().hex,
        migration="session-mutation-audits",
    )
    sessions_connection = sqlite3.connect(sessions_db)
    try:
        sessions_connection.execute("BEGIN IMMEDIATE")
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
        except BaseException:
            if sessions_connection.in_transaction:
                sessions_connection.rollback()
            raise
        sessions_connection.commit()
    finally:
        sessions_connection.close()


steps = [step(add_session_mutation_audits)]
