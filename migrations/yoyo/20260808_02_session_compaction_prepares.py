from __future__ import annotations

import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)


__depends__ = {"20260808_01_session_mutation_audits"}
__transactional__ = False


# This manifest is the durable fence contract shared by SessionStore and recovery.
SCHEMA_MANIFEST: dict[str, dict[str, tuple[str, ...]]] = {
    "session_compaction_prepares": {
        "columns": (
            "session_key",
            "session_created_at",
            "generation",
            "parent_generation",
            "source_ref",
            "source_from_seq",
            "consolidated_through_seq",
            "source_message_ids_json",
            "retained_tail_json",
            "prepared_at",
        ),
        "indexes": ("idx_session_compaction_prepares_ref",),
    },
}

_TABLE_SCHEMA = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("session_created_at", "TEXT", 1, 0),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("prepared_at", "TEXT", 1, 0),
    ),
    "named_indexes": {
        "idx_session_compaction_prepares_ref": (("session_key", "source_ref"), 0),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (),
}


def _ensure_table(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    create_sql: str,
    index_sql: str,
) -> None:
    """创建或校验一张 durable prepare 表及其查询索引。"""

    # 1. 只创建缺失的表；已有表必须通过 manifest 校验。
    existing = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    if existing is None:
        connection.execute(create_sql)

    # 2. 添加命名索引前，先校验列和内联约束。
    validate_table_schema(
        connection,
        table=table,
        columns=_TABLE_SCHEMA["columns"],
        named_indexes=_TABLE_SCHEMA["named_indexes"],
        auto_indexes=_TABLE_SCHEMA["auto_indexes"],
        sql_fragments=_TABLE_SCHEMA["sql_fragments"],
        validate_named_indexes=False,
    )

    # 3. 创建并校验本迁移持有的查询索引。
    connection.execute(index_sql)
    validate_table_schema(
        connection,
        table=table,
        columns=_TABLE_SCHEMA["columns"],
        named_indexes=_TABLE_SCHEMA["named_indexes"],
        auto_indexes=_TABLE_SCHEMA["auto_indexes"],
        sql_fragments=_TABLE_SCHEMA["sql_fragments"],
    )


def add_session_compaction_prepares(connection: object) -> None:
    """Create and validate the durable receipt-before-ledger fence table."""

    _ = connection
    current = current_migration_context()
    sessions_db = current.workspace / "sessions.db"
    if not sessions_db.exists():
        return
    backup_sqlite_database(
        sessions_db,
        current.workspace / "backups" / "session-compaction-prepares" / uuid4().hex,
        migration="session-compaction-prepares",
    )
    sessions_connection = sqlite3.connect(sessions_db)
    try:
        sessions_connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_table(
                sessions_connection,
                "session_compaction_prepares",
                SCHEMA_MANIFEST["session_compaction_prepares"]["columns"],
                """
                CREATE TABLE session_compaction_prepares (
                    session_key TEXT NOT NULL,
                    session_created_at TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    parent_generation INTEGER NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_from_seq INTEGER NOT NULL,
                    consolidated_through_seq INTEGER NOT NULL,
                    source_message_ids_json TEXT NOT NULL,
                    retained_tail_json TEXT NOT NULL,
                    prepared_at TEXT NOT NULL,
                    PRIMARY KEY (session_key, generation),
                    UNIQUE (session_key, source_ref)
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_session_compaction_prepares_ref
                ON session_compaction_prepares(session_key, source_ref)
                """,
            )
        except BaseException:
            if sessions_connection.in_transaction:
                sessions_connection.rollback()
            raise
        sessions_connection.commit()
    finally:
        sessions_connection.close()


steps = [step(add_session_compaction_prepares)]
