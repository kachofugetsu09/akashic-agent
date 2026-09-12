from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import cast
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from .support.session_db_backup import backup_sqlite_database

__depends__ = {"20260828_02_add_wake_content_scores"}
__transactional__ = False

_MIGRATION = "backfill-plugin-programmatic-effects"
_SUPPRESS = "suppress"
_ALLOW = "allow"
_PROVENANCE_FIELDS = ("plugin_id", "job_name", "generation_id", "snapshot_id")


class _Rewrite:
    """Describe one JSON replacement owned by this migration."""

    def __init__(self, table: str, key: str, column: str, payload: str) -> None:
        self.table = table
        self.key = key
        self.column = column
        self.payload = payload


def _decode_object(raw: object, *, field: str) -> dict[str, object]:
    """Decode one persisted JSON object at the SessionDB boundary."""

    if not isinstance(raw, str):
        raise ValueError(f"{field} 必须是 JSON 文本")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON object")
    return cast(dict[str, object], value)


def _is_plugin_programmatic(metadata: dict[str, object], *, field: str) -> bool:
    """Recognize only Core-owned plugin job provenance."""

    if metadata.get("programmatic") is not True:
        return False
    present = [name in metadata for name in _PROVENANCE_FIELDS]
    if not any(present):
        return False
    if not all(present):
        raise ValueError(f"{field} 的 plugin programmatic provenance 不完整")
    for name in _PROVENANCE_FIELDS:
        value = metadata.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field}.{name} 必须是非空字符串")
    return True


def _set_suppress(metadata: dict[str, object], *, field: str) -> bool:
    """Set the generic effect and reject contradictory persisted facts."""

    raw_effects = metadata.get("effects")
    if raw_effects is None:
        effects: dict[str, object] = {}
    elif isinstance(raw_effects, dict):
        effects = dict(raw_effects)
    else:
        raise ValueError(f"{field}.effects 必须是 object")
    current = effects.get("post_commit")
    if current == _SUPPRESS:
        return False
    if current == _ALLOW:
        raise RuntimeError(f"{field} 与 plugin programmatic suppress 合同冲突")
    if current is not None:
        raise ValueError(f"{field}.effects.post_commit 值无效")
    effects["post_commit"] = _SUPPRESS
    metadata["effects"] = effects
    return True


def _render(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _has_columns(
    connection: sqlite3.Connection,
    table: str,
    required: set[str],
) -> bool:
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    return required.issubset(columns)


def _plan_rewrites(connection: sqlite3.Connection) -> list[_Rewrite]:
    """Build the complete repair plan before taking the write lock."""

    if not (
        _has_columns(connection, "sessions", {"key", "metadata"})
        and _has_columns(
            connection,
            "messages",
            {"id", "session_key", "seq", "extra"},
        )
        and _has_columns(connection, "turns", {"id", "session_key", "input_json"})
    ):
        return []
    sessions: set[str] = set()
    for session_key, raw_metadata in connection.execute(
        "SELECT key, metadata FROM sessions ORDER BY key"
    ):
        metadata = _decode_object(
            raw_metadata,
            field=f"sessions.metadata[{session_key}]",
        )
        if _is_plugin_programmatic(
            metadata,
            field=f"sessions.metadata[{session_key}]",
        ):
            sessions.add(str(session_key))
    if not sessions:
        return []

    rewrites: list[_Rewrite] = []
    placeholders = ",".join("?" for _ in sessions)
    ordered_sessions = tuple(sorted(sessions))
    for message_id, raw_extra in connection.execute(
        f"SELECT id, extra FROM messages WHERE session_key IN ({placeholders}) "
        "ORDER BY session_key, seq, id",
        ordered_sessions,
    ):
        extra = _decode_object(raw_extra, field=f"messages.extra[{message_id}]")
        if _set_suppress(extra, field=f"messages.extra[{message_id}]"):
            rewrites.append(
                _Rewrite("messages", str(message_id), "extra", _render(extra))
            )

    for turn_id, raw_input in connection.execute(
        f"SELECT id, input_json FROM turns WHERE session_key IN ({placeholders}) "
        "ORDER BY id",
        ordered_sessions,
    ):
        turn_input = _decode_object(raw_input, field=f"turns.input_json[{turn_id}]")
        raw_metadata = turn_input.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"turns.input_json[{turn_id}].metadata 必须是 object")
        metadata = dict(cast(dict[str, object], raw_metadata))
        raw_inbound = metadata.get("inboundMetadata", {})
        if not isinstance(raw_inbound, dict):
            raise ValueError(
                f"turns.input_json[{turn_id}].metadata.inboundMetadata 必须是 object"
            )
        inbound = dict(cast(dict[str, object], raw_inbound))
        if _set_suppress(
            inbound,
            field=f"turns.input_json[{turn_id}].metadata.inboundMetadata",
        ):
            metadata["inboundMetadata"] = inbound
            turn_input["metadata"] = metadata
            rewrites.append(
                _Rewrite("turns", str(turn_id), "input_json", _render(turn_input))
            )
    return rewrites


def _apply_rewrites(
    connection: sqlite3.Connection,
    rewrites: list[_Rewrite],
) -> None:
    for rewrite in rewrites:
        cursor = connection.execute(
            f"UPDATE {rewrite.table} SET {rewrite.column} = ? WHERE id = ?",
            (rewrite.payload, rewrite.key),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Programmatic effect migration target changed: "
                f"{rewrite.table}:{rewrite.key}"
            )


def backfill_plugin_programmatic_effects(_connection: object) -> None:
    """Repair effects lost before plugin programmatic inputs were locked."""

    _ = _connection
    workspace = current_migration_context().workspace
    sessions_db = workspace / "sessions.db"
    if not sessions_db.exists():
        return

    with closing(sqlite3.connect(sessions_db)) as connection:
        rewrites = _plan_rewrites(connection)
    if not rewrites:
        return

    _ = backup_sqlite_database(
        sessions_db,
        workspace / "backups" / _MIGRATION / uuid4().hex,
        migration=_MIGRATION,
    )
    with closing(sqlite3.connect(sessions_db)) as connection:
        _ = connection.execute("BEGIN IMMEDIATE")
        try:
            locked_rewrites = _plan_rewrites(connection)
            _apply_rewrites(connection, locked_rewrites)
            if _plan_rewrites(connection):
                raise RuntimeError(
                    "Programmatic effect migration left rewrites pending"
                )
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("Programmatic effect migration integrity_check 失败")
        except BaseException:
            connection.rollback()
            raise
        connection.commit()


steps = [step(backfill_plugin_programmatic_effects)]
