from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database


__depends__ = {"20260827_02_migrate_legacy_mobile_client_ids"}
__transactional__ = False

_MIGRATION = "restore-rekeyed-compactions"
_REASON = "akashic_identity_rekey"


def _affected(connection: sqlite3.Connection) -> list[tuple[str, int]]:
    """Return only compactions invalidated by the Akashic identity migration."""

    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "session_compactions" not in tables:
        return []
    return [
        (str(row[0]), int(row[1]))
        for row in connection.execute(
            "SELECT session_key, generation FROM session_compactions "
            "WHERE invalidated_reason = ? ORDER BY session_key, generation",
            (_REASON,),
        )
    ]


def _check_row(
    connection: sqlite3.Connection,
    session_key: str,
    generation: int,
) -> None:
    """Prove the prior migration rewrote every compaction-owned identity."""

    row = connection.execute(
        "SELECT source_ref, source_message_ids_json, retained_tail_json, "
        "invalidated_at, invalidated_reason FROM session_compactions "
        "WHERE session_key = ? AND generation = ?",
        (session_key, generation),
    ).fetchone()
    if row is None or row[3] is None or str(row[4]) != _REASON:
        raise RuntimeError(f"Compaction repair plan changed: {session_key}@{generation}")
    if not session_key.startswith("akashic:"):
        raise RuntimeError(f"Compaction repair found a non-Akashic session: {session_key}")
    if not str(row[0]).startswith(f"context-compaction:{session_key}@"):
        raise RuntimeError(f"Compaction source_ref was not rekeyed: {session_key}@{generation}")

    message_ids = json.loads(str(row[1]))
    retained = json.loads(str(row[2]))
    if not isinstance(message_ids, list) or not isinstance(retained, list):
        raise RuntimeError(f"Compaction identity lists are invalid: {session_key}@{generation}")
    owned_ids = [item for item in message_ids if isinstance(item, str)]
    retained_ids: list[str] = []
    for item in retained:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise RuntimeError(
                f"Compaction message identities are invalid: {session_key}@{generation}"
            )
        retained_ids.append(item["id"])
    if len(owned_ids) != len(message_ids):
        raise RuntimeError(f"Compaction message identities are invalid: {session_key}@{generation}")
    all_ids = [*owned_ids, *retained_ids]
    if any(not item.startswith(f"{session_key}:") for item in all_ids):
        raise RuntimeError(f"Compaction message identity was not rekeyed: {session_key}@{generation}")
    missing = connection.execute(
        "SELECT id FROM messages WHERE id IN ({}) AND session_key != ? LIMIT 1".format(
            ",".join("?" for _ in all_ids)
        ),
        (*all_ids, session_key),
    ).fetchone() if all_ids else None
    if missing is not None:
        raise RuntimeError(f"Compaction message belongs to another session: {missing[0]}")
    unique_ids = sorted(set(all_ids))
    found = connection.execute(
        "SELECT COUNT(*) FROM messages WHERE id IN ({})".format(
            ",".join("?" for _ in unique_ids)
        ),
        unique_ids,
    ).fetchone()[0] if unique_ids else 0
    if int(found) != len(unique_ids):
        raise RuntimeError(f"Compaction source message is missing: {session_key}@{generation}")


def restore_rekeyed_compactions(_connection: object) -> None:
    """Restore summaries whose source content survived the identity-only rekey.

    The original source-plan digest remains immutable evidence of the summarized
    content. Runtime projection reads the verified source IDs and retained tail;
    it does not replay the old cross-file receipt.
    """

    _ = _connection
    current = current_migration_context()
    database = current.workspace / "sessions.db"
    if not database.is_file():
        return
    with closing(sqlite3.connect(database)) as connection:
        affected = _affected(connection)
        for session_key, generation in affected:
            _check_row(connection, session_key, generation)
    if not affected:
        return

    _ = backup_sqlite_database(
        database,
        current.workspace / "backups" / _MIGRATION / uuid4().hex,
        migration=_MIGRATION,
    )
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for session_key, generation in affected:
                _check_row(connection, session_key, generation)
                updated = connection.execute(
                    "UPDATE session_compactions SET invalidated_at = NULL, "
                    "invalidated_reason = NULL WHERE session_key = ? AND generation = ? "
                    "AND invalidated_reason = ?",
                    (session_key, generation, _REASON),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(
                        f"Compaction repair conflict: {session_key}@{generation}"
                    )
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise RuntimeError("Compaction repair left foreign key errors")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Compaction repair integrity_check failed")


steps = [step(restore_rekeyed_compactions)]
