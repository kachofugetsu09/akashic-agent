from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import cast
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from .support.session_db_backup import backup_sqlite_database

__depends__ = {"20260829_01_backfill_plugin_programmatic_effects"}
__transactional__ = False

_MIGRATION = "backfill-explicit-programmatic-effects"
_SUPPRESS = "suppress"
_ALLOW = "allow"


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


def _declared_effect(metadata: dict[str, object], *, field: str) -> str | None:
    raw_effects = metadata.get("effects")
    if raw_effects is None:
        return None
    if not isinstance(raw_effects, dict):
        raise ValueError(f"{field}.effects 必须是 object")
    value = cast(dict[str, object], raw_effects).get("post_commit")
    if value is None:
        return None
    if value not in {_ALLOW, _SUPPRESS}:
        raise ValueError(f"{field}.effects.post_commit 值无效")
    return cast(str, value)


def _set_suppress(metadata: dict[str, object], *, field: str) -> bool:
    """Set suppress without replacing unrelated effect fields."""

    current = _declared_effect(metadata, field=field)
    if current == _SUPPRESS:
        return False
    if current == _ALLOW:
        raise RuntimeError(f"{field} 与明确的 suppress 事实冲突")
    raw_effects = metadata.get("effects")
    effects = (
        dict(cast(dict[str, object], raw_effects))
        if isinstance(raw_effects, dict)
        else {}
    )
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
    """Repair only rows whose persisted session or Turn explicitly suppresses."""

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

    session_suppresses: dict[str, bool] = {}
    for session_key, raw_metadata in connection.execute(
        "SELECT key, metadata FROM sessions "
        "WHERE key LIKE 'programmatic:%' ORDER BY key"
    ):
        metadata = _decode_object(
            raw_metadata,
            field=f"sessions.metadata[{session_key}]",
        )
        session_suppresses[str(session_key)] = _declared_effect(
            metadata,
            field=f"sessions.metadata[{session_key}]",
        ) == _SUPPRESS
    if not session_suppresses:
        return []

    rewrites: list[_Rewrite] = []
    turn_suppresses: dict[str, bool] = {}
    placeholders = ",".join("?" for _ in session_suppresses)
    keys = tuple(sorted(session_suppresses))
    for turn_id, session_key, raw_input in connection.execute(
        f"SELECT id, session_key, input_json FROM turns "
        f"WHERE session_key IN ({placeholders}) ORDER BY session_key, id",
        keys,
    ):
        field = f"turns.input_json[{turn_id}]"
        turn_input = _decode_object(raw_input, field=field)
        raw_metadata = turn_input.get("metadata")
        if not isinstance(raw_metadata, dict):
            raise ValueError(f"{field}.metadata 必须是 object")
        metadata = dict(cast(dict[str, object], raw_metadata))
        outer = _declared_effect(metadata, field=f"{field}.metadata")
        raw_inbound = metadata.get("inboundMetadata", {})
        if not isinstance(raw_inbound, dict):
            raise ValueError(f"{field}.metadata.inboundMetadata 必须是 object")
        inbound = dict(cast(dict[str, object], raw_inbound))
        nested = _declared_effect(
            inbound,
            field=f"{field}.metadata.inboundMetadata",
        )
        explicit_suppress = _SUPPRESS in {outer, nested}
        suppresses = session_suppresses[str(session_key)] or explicit_suppress
        if suppresses and _ALLOW in {outer, nested}:
            raise RuntimeError(f"{field} 同时声明 allow 与 suppress")
        turn_suppresses[str(turn_id)] = suppresses
        if suppresses and _set_suppress(
            inbound,
            field=f"{field}.metadata.inboundMetadata",
        ):
            metadata["inboundMetadata"] = inbound
            turn_input["metadata"] = metadata
            rewrites.append(
                _Rewrite("turns", str(turn_id), "input_json", _render(turn_input))
            )

    for message_id, session_key, raw_extra in connection.execute(
        f"SELECT id, session_key, extra FROM messages "
        f"WHERE session_key IN ({placeholders}) ORDER BY session_key, seq, id",
        keys,
    ):
        extra = _decode_object(raw_extra, field=f"messages.extra[{message_id}]")
        suppresses = session_suppresses[str(session_key)]
        turn_id = extra.get("control_turn_id")
        if isinstance(turn_id, str):
            suppresses = turn_suppresses.get(turn_id, suppresses)
        if suppresses and _set_suppress(
            extra,
            field=f"messages.extra[{message_id}]",
        ):
            rewrites.append(
                _Rewrite("messages", str(message_id), "extra", _render(extra))
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
                f"Explicit effect migration target changed: "
                f"{rewrite.table}:{rewrite.key}"
            )


def backfill_explicit_programmatic_effects(_connection: object) -> None:
    """Project historical explicit suppress facts into their durable messages."""

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
                raise RuntimeError("Explicit effect migration left rewrites pending")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("Explicit effect migration integrity_check 失败")
        except BaseException:
            connection.rollback()
            raise
        connection.commit()


steps = [step(backfill_explicit_programmatic_effects)]
