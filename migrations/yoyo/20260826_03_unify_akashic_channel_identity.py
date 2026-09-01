from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tomllib
from collections.abc import MutableMapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid5

import tomlkit
from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {
    "20260826_01_migrate_turn_effects",
    "20260826_02_backfill_akasha_message_embeddings",
}
__transactional__ = False

_MIGRATION = "unify-akashic-channel-identity"
_NAMESPACE = UUID("9f1bca7e-4c8d-5f1e-8c30-b9a3a8f29c17")
_SESSION_FIELDS = frozenset(
    {
        "session_id",
        "session_key",
        "thread_id",
        "accepted_session_id",
        "busySessionId",
        "chatId",
        "projection_session_id",
        "session_key_override",
        "target_session_id",
    }
)
_MESSAGE_FIELDS = frozenset(
    {
        "message_id",
        "session_message_id",
        "reply_to_message_id",
        "persisted_user_message_id",
        "persisted_assistant_message_id",
        "projection_message_id",
        "anchor_message_id",
        "sessionMessageId",
    }
)
_MESSAGE_LIST_FIELDS = frozenset(
    {
        "message_ids",
        "persisted_user_message_ids",
        "source_message_ids",
        "target_message_ids",
    }
)
_TURN_LIST_FIELDS = frozenset({"target_turn_ids"})
_MIXED_MEMORY_LIST_FIELDS = frozenset({"cited_memory_ids"})
_SOURCE_REF_LIST_FIELDS = frozenset({"source_refs"})


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _new_session_key(old_session_key: str) -> str:
    return f"akashic:{uuid5(_NAMESPACE, old_session_key).hex}"


def _chat_id(session_key: str) -> str:
    channel, separator, chat_id = session_key.partition(":")
    if not separator or not channel or not chat_id:
        raise ValueError(f"session key 格式无效: {session_key!r}")
    return chat_id


def _is_old_session(value: str, old_channels: frozenset[str]) -> bool:
    return value.partition(":")[0] in old_channels


def _decode_json(raw: object, *, field: str) -> object:
    if not isinstance(raw, str):
        raise ValueError(f"{field} 必须是 JSON 文本")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} JSON 无效") from error


def _encode_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _rewrite_turn_reference(value: str, message_map: dict[str, str]) -> str:
    """Rewrite one Akasha user-message::assistant-message Turn identity."""

    parts = value.split("::")
    if len(parts) != 2:
        return value
    mapped = [message_map.get(part, part) for part in parts]
    changed = [old != new for old, new in zip(parts, mapped, strict=True)]
    if any(changed) and not all(changed):
        raise RuntimeError(f"Akasha Turn identity 只迁移了一半: {value}")
    return "::".join(mapped)


def _rewrite_memory_reference(value: str, message_map: dict[str, str]) -> str:
    """Rewrite a cited Message or Turn while preserving opaque memory IDs."""

    return message_map.get(value, _rewrite_turn_reference(value, message_map))


def _rewrite_identity_fields(
    value: object,
    *,
    session_map: dict[str, str],
    message_map: dict[str, str],
    source_ref_map: dict[str, str],
    field: str | None = None,
) -> object:
    """Rewrite only fields whose schema names declare an identity."""

    if isinstance(value, dict):
        return {
            str(key): _rewrite_identity_fields(
                item,
                session_map=session_map,
                message_map=message_map,
                source_ref_map=source_ref_map,
                field=str(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        if field in _MESSAGE_LIST_FIELDS:
            return [
                message_map.get(item, item) if isinstance(item, str) else item
                for item in value
            ]
        if field in _TURN_LIST_FIELDS:
            return [
                (
                    _rewrite_turn_reference(item, message_map)
                    if isinstance(item, str)
                    else item
                )
                for item in value
            ]
        if field in _MIXED_MEMORY_LIST_FIELDS:
            return [
                (
                    _rewrite_memory_reference(item, message_map)
                    if isinstance(item, str)
                    else item
                )
                for item in value
            ]
        if field in _SOURCE_REF_LIST_FIELDS:
            return [
                (
                    source_ref_map.get(item, message_map.get(item, item))
                    if isinstance(item, str)
                    else _rewrite_identity_fields(
                        item,
                        session_map=session_map,
                        message_map=message_map,
                        source_ref_map=source_ref_map,
                        field=field,
                    )
                )
                for item in value
            ]
        return [
            _rewrite_identity_fields(
                item,
                session_map=session_map,
                message_map=message_map,
                source_ref_map=source_ref_map,
                field=field,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    if field == "channel" and any(old.startswith(f"{value}:") for old in session_map):
        return "akashic"
    if field in _SESSION_FIELDS:
        return session_map.get(value, value)
    if field in _MESSAGE_FIELDS or (
        field is not None and field.endswith("_message_id")
    ):
        return message_map.get(value, value)
    if field == "source_ref":
        return source_ref_map.get(value, message_map.get(value, value))
    return value


def _rewrite_message_id_list(
    raw: object, message_map: dict[str, str], *, field: str
) -> str:
    payload = _decode_json(raw, field=field)
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(f"{field} 必须是字符串数组")
    return _encode_json([message_map.get(item, item) for item in payload])


def _rewrite_retained_tail(
    raw: object,
    *,
    session_map: dict[str, str],
    message_map: dict[str, str],
    source_ref_map: dict[str, str],
    field: str,
) -> str:
    payload = _decode_json(raw, field=field)
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError(f"{field} 必须是 object 数组")
    rewritten: list[dict[str, object]] = []
    for item in payload:
        current = dict(item)
        identity = current.get("id")
        if not isinstance(identity, str):
            raise ValueError(f"{field}[].id 必须是字符串")
        current["id"] = message_map.get(identity, identity)
        message = current.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"{field}[].message 必须是 object")
        current["message"] = _rewrite_identity_fields(
            message,
            session_map=session_map,
            message_map=message_map,
            source_ref_map=source_ref_map,
        )
        rewritten.append(current)
    return _encode_json(rewritten)


def _rewrite_json_column(
    raw: object,
    *,
    session_map: dict[str, str],
    message_map: dict[str, str],
    source_ref_map: dict[str, str],
    field: str,
) -> str:
    payload = _decode_json(raw, field=field)
    rewritten = _rewrite_identity_fields(
        payload,
        session_map=session_map,
        message_map=message_map,
        source_ref_map=source_ref_map,
    )
    return raw if rewritten == payload else _encode_json(rewritten)


def _preflight_sessions(connection: sqlite3.Connection) -> None:
    for table in (
        "session_admissions",
        "inbound_handoffs",
        "session_compaction_prepares",
    ):
        if _table_exists(connection, table):
            count = int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            if count:
                raise RuntimeError(
                    f"Akashic identity 迁移要求 {table} 为空，当前 {count}"
                )
    if _table_exists(connection, "turns"):
        active = int(
            connection.execute(
                "SELECT COUNT(*) FROM turns WHERE status NOT IN "
                "('completed', 'failed', 'cancelled', 'interrupted')"
            ).fetchone()[0]
        )
        if active:
            raise RuntimeError(f"Akashic identity 迁移发现 {active} 个未终态 Turn")


def _build_maps(
    connection: sqlite3.Connection,
    old_channels: frozenset[str],
) -> tuple[dict[str, str], dict[str, str]]:
    sessions = [
        str(row[0])
        for row in connection.execute("SELECT key FROM sessions ORDER BY key")
        if str(row[0]).partition(":")[0] in old_channels
    ]
    if not sessions:
        return {}, {}
    session_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)")
    }
    message_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")
    }
    if "next_seq" not in session_columns and not {
        "session_key",
        "seq",
    }.issubset(message_columns):
        raise RuntimeError(
            "旧 Session schema 缺少 next_seq，且 Message schema 缺少 session_key/seq"
        )
    session_query = (
        "SELECT key, next_seq FROM sessions ORDER BY key"
        if "next_seq" in session_columns
        else "SELECT sessions.key, COALESCE(MAX(messages.seq) + 1, 0) "
        "FROM sessions LEFT JOIN messages ON messages.session_key = sessions.key "
        "GROUP BY sessions.key ORDER BY sessions.key"
    )
    session_rows = [
        (str(row[0]), int(row[1]))
        for row in connection.execute(session_query)
        if str(row[0]).partition(":")[0] in old_channels
    ]
    session_map = {old: _new_session_key(old) for old in sessions}
    if len(set(session_map.values())) != len(session_map):
        raise RuntimeError("Akashic Session UUIDv5 mapping 发生碰撞")
    occupied = {
        str(row[0])
        for row in connection.execute(
            "SELECT key FROM sessions WHERE key LIKE 'akashic:%'"
        )
    }
    collisions = occupied.intersection(session_map.values())
    if collisions:
        raise RuntimeError(
            f"Akashic Session mapping 与现有身份冲突: {sorted(collisions)[:3]}"
        )
    message_map: dict[str, str] = {}
    for old_session, new_session in session_map.items():
        for row in connection.execute(
            "SELECT id, seq FROM messages WHERE session_key = ? ORDER BY seq",
            (old_session,),
        ):
            old_message = str(row[0])
            new_message = f"{new_session}:{int(row[1])}"
            if old_message in message_map or new_message in message_map.values():
                raise RuntimeError("Akashic Message mapping 不唯一")
            message_map[old_message] = new_message
    for old_session, next_seq in session_rows:
        new_session = session_map[old_session]
        occupied = set(message_map.values())
        for seq in range(next_seq):
            old_message = f"{old_session}:{seq}"
            new_message = f"{new_session}:{seq}"
            if old_message not in message_map and new_message not in occupied:
                message_map[old_message] = new_message
                occupied.add(new_message)
    return session_map, message_map


def _migrate_sessions(
    path: Path,
    *,
    old_channels: frozenset[str],
) -> tuple[dict[str, str], dict[str, str]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        _preflight_sessions(connection)
        session_map, message_map = _build_maps(connection, old_channels)
        if not session_map:
            return {}, {}
        source_ref_map: dict[str, str] = {}
        if _table_exists(connection, "session_compactions"):
            from agent.model_runtime.context_compaction import (
                compaction_scope_id,
                compaction_source_ref,
            )

            for row in connection.execute(
                "SELECT c.session_key, c.generation, c.source_ref, s.created_at "
                "FROM session_compactions c JOIN sessions s ON s.key = c.session_key"
            ):
                old_session = str(row["session_key"])
                if old_session not in session_map:
                    continue
                new_scope = compaction_scope_id(
                    session_map[old_session],
                    str(row["created_at"]),
                )
                source_ref_map[str(row["source_ref"])] = compaction_source_ref(
                    new_scope,
                    int(row["generation"]),
                )

        connection.execute("BEGIN IMMEDIATE")
        connection.execute("PRAGMA defer_foreign_keys = ON")
        now = datetime.now(UTC).isoformat()

        for row in connection.execute("SELECT id, extra FROM messages ORDER BY rowid"):
            rewritten = (
                _rewrite_json_column(
                    row["extra"],
                    session_map=session_map,
                    message_map=message_map,
                    source_ref_map=source_ref_map,
                    field=f"messages.extra:{row['id']}",
                )
                if row["extra"] not in (None, "")
                else row["extra"]
            )
            if rewritten != row["extra"]:
                connection.execute(
                    "UPDATE messages SET extra = ? WHERE id = ?",
                    (rewritten, row["id"]),
                )

        for table in ("session_compactions",):
            if not _table_exists(connection, table):
                continue
            rows = connection.execute(
                f"SELECT session_key, generation, source_ref, source_message_ids_json, retained_tail_json FROM {table}"
            ).fetchall()
            for row in rows:
                old_session = str(row["session_key"])
                if old_session not in session_map:
                    continue
                connection.execute(
                    f"UPDATE {table} SET session_key = ?, source_ref = ?, "
                    "source_message_ids_json = ?, retained_tail_json = ?, "
                    "invalidated_at = COALESCE(invalidated_at, ?), "
                    "invalidated_reason = COALESCE(invalidated_reason, 'akashic_identity_rekey') "
                    "WHERE session_key = ? AND generation = ?",
                    (
                        session_map[old_session],
                        source_ref_map.get(
                            str(row["source_ref"]), str(row["source_ref"])
                        ),
                        _rewrite_message_id_list(
                            row["source_message_ids_json"],
                            message_map,
                            field=f"{table}.source_message_ids_json",
                        ),
                        _rewrite_retained_tail(
                            row["retained_tail_json"],
                            session_map=session_map,
                            message_map=message_map,
                            source_ref_map=source_ref_map,
                            field=f"{table}.retained_tail_json",
                        ),
                        now,
                        old_session,
                        int(row["generation"]),
                    ),
                )

        if _table_exists(connection, "session_delete_audits"):
            for row in connection.execute(
                "SELECT audit_id, targets_json, message_ids_json, compactions_json FROM session_delete_audits"
            ).fetchall():
                targets = _decode_json(
                    row["targets_json"], field="session_delete_audits.targets_json"
                )
                if isinstance(targets, list):
                    targets = [
                        (
                            session_map.get(item, item)
                            if isinstance(item, str)
                            else _rewrite_identity_fields(
                                item,
                                session_map=session_map,
                                message_map=message_map,
                                source_ref_map=source_ref_map,
                            )
                        )
                        for item in targets
                    ]
                else:
                    targets = _rewrite_identity_fields(
                        targets,
                        session_map=session_map,
                        message_map=message_map,
                        source_ref_map=source_ref_map,
                    )
                connection.execute(
                    "UPDATE session_delete_audits SET targets_json = ?, message_ids_json = ?, compactions_json = ? WHERE audit_id = ?",
                    (
                        _encode_json(targets),
                        _rewrite_message_id_list(
                            row["message_ids_json"],
                            message_map,
                            field="session_delete_audits.message_ids_json",
                        ),
                        _rewrite_json_column(
                            row["compactions_json"],
                            session_map=session_map,
                            message_map=message_map,
                            source_ref_map=source_ref_map,
                            field="session_delete_audits.compactions_json",
                        ),
                        row["audit_id"],
                    ),
                )

        if _table_exists(connection, "session_source_mutation_audits"):
            for row in connection.execute(
                "SELECT audit_id, session_key, message_ids_json FROM session_source_mutation_audits"
            ).fetchall():
                old_session = str(row["session_key"])
                if old_session not in session_map:
                    continue
                connection.execute(
                    "UPDATE session_source_mutation_audits SET session_key = ?, message_ids_json = ? WHERE audit_id = ?",
                    (
                        session_map[old_session],
                        _rewrite_message_id_list(
                            row["message_ids_json"],
                            message_map,
                            field="session_source_mutation_audits.message_ids_json",
                        ),
                        row["audit_id"],
                    ),
                )

        if _table_exists(connection, "turns"):
            for row in connection.execute(
                "SELECT id, session_key, input_json, items_json FROM turns"
            ).fetchall():
                old_session = str(row["session_key"])
                new_session = session_map.get(old_session, old_session)
                input_json = _rewrite_json_column(
                    row["input_json"],
                    session_map=session_map,
                    message_map=message_map,
                    source_ref_map=source_ref_map,
                    field="turns.input_json",
                )
                items_json = _rewrite_json_column(
                    row["items_json"],
                    session_map=session_map,
                    message_map=message_map,
                    source_ref_map=source_ref_map,
                    field="turns.items_json",
                )
                if (
                    new_session == old_session
                    and input_json == row["input_json"]
                    and items_json == row["items_json"]
                ):
                    continue
                connection.execute(
                    "UPDATE turns SET session_key = ?, input_json = ?, items_json = ? WHERE id = ?",
                    (
                        new_session,
                        input_json,
                        items_json,
                        row["id"],
                    ),
                )

        if _table_exists(connection, "message_attachments"):
            for old, new in message_map.items():
                connection.execute(
                    "UPDATE message_attachments SET message_id = ? WHERE message_id = ?",
                    (new, old),
                )
        if _table_exists(connection, "message_embeddings"):
            for old, new in message_map.items():
                connection.execute(
                    "UPDATE message_embeddings SET message_id = ? WHERE message_id = ?",
                    (new, old),
                )
        for old, new in message_map.items():
            connection.execute("UPDATE messages SET id = ? WHERE id = ?", (new, old))
        for old, new in session_map.items():
            connection.execute(
                "UPDATE messages SET session_key = ? WHERE session_key = ?", (new, old)
            )
            connection.execute(
                "UPDATE sessions SET key = ?, last_consolidated = 0 WHERE key = ?",
                (new, old),
            )

        old_channel_values = tuple(sorted(old_channels))
        placeholders = ",".join("?" for _ in old_channel_values)
        connection.execute(
            f"DELETE FROM channel_identities WHERE channel IN ({placeholders})",
            old_channel_values,
        )
        connection.execute(
            f"DELETE FROM channel_identity_migrations WHERE channel IN ({placeholders})",
            old_channel_values,
        )
        connection.executemany(
            "INSERT INTO channel_identities(channel, identity, chat_id, updated_at) "
            "VALUES ('akashic', ?, ?, ?) ON CONFLICT(channel, identity) DO UPDATE SET "
            "chat_id = excluded.chat_id, updated_at = excluded.updated_at",
            ((_chat_id(new), _chat_id(new), now) for new in session_map.values()),
        )
        connection.execute(
            "INSERT INTO channel_identity_migrations(channel, migrated_at) VALUES ('akashic', ?) "
            "ON CONFLICT(channel) DO NOTHING",
            (now,),
        )
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("Akashic SessionDB migration 留下 foreign key 错误")
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Akashic SessionDB integrity_check 失败")
        return session_map, message_map
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _rewrite_target_object(
    value: object,
    *,
    old_channels: frozenset[str],
    session_map: dict[str, str],
    message_map: dict[str, str],
) -> object:
    if isinstance(value, list):
        return [
            _rewrite_target_object(
                item,
                old_channels=old_channels,
                session_map=session_map,
                message_map=message_map,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    result = {
        str(key): _rewrite_target_object(
            item,
            old_channels=old_channels,
            session_map=session_map,
            message_map=message_map,
        )
        for key, item in value.items()
    }
    channel = result.get("channel")
    recipient_key = (
        "recipient"
        if "recipient" in result
        else "chat_id" if "chat_id" in result else None
    )
    if isinstance(channel, str) and channel in old_channels:
        if recipient_key is None:
            raise RuntimeError("调度 Akashic 目标缺少 recipient/chat_id")
        recipient = result.get(recipient_key)
        if not isinstance(recipient, str):
            raise RuntimeError("调度 Akashic 目标 recipient/chat_id 无效")
        new_session = session_map.get(f"{channel}:{recipient}")
        if new_session is None:
            raise RuntimeError("调度目标引用不存在的旧 Akashic Session")
        result["channel"] = "akashic"
        result[recipient_key] = _chat_id(new_session)
    return _rewrite_identity_fields(
        result,
        session_map=session_map,
        message_map=message_map,
        source_ref_map={},
    )


def _write_atomic(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        candidate.write_bytes(payload)
        candidate.chmod(mode)
        with candidate.open("rb") as stream:
            os.fsync(stream.fileno())
        candidate.replace(path)
        _fsync_directory(path.parent)
    finally:
        candidate.unlink(missing_ok=True)


def _migrate_json_file(
    path: Path,
    *,
    old_channels: frozenset[str],
    session_map: dict[str, str],
    message_map: dict[str, str],
) -> None:
    if not path.is_file():
        return
    payload = _decode_json(path.read_text(encoding="utf-8"), field=str(path))
    migrated = _rewrite_target_object(
        payload,
        old_channels=old_channels,
        session_map=session_map,
        message_map=message_map,
    )
    if migrated != payload:
        _write_atomic(
            path,
            (_encode_json(migrated) + "\n").encode("utf-8"),
            path.stat().st_mode & 0o777,
        )


def _migrate_wake_config(
    path: Path,
    *,
    old_channels: frozenset[str],
    session_map: dict[str, str],
) -> None:
    if not path.is_file():
        return
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    delivery = document.get("delivery")
    if not isinstance(delivery, MutableMapping):
        return
    channel = delivery.get("channel")
    recipient = delivery.get("recipient")
    session_id = delivery.get("session_id")
    if isinstance(session_id, str) and _is_old_session(session_id, old_channels):
        new_context_session = session_map.get(session_id)
        if new_context_session is None:
            raise RuntimeError("Wake context 指向不存在的旧 Akashic Session")
        delivery["session_id"] = new_context_session
    if isinstance(channel, str) and channel in old_channels:
        if not isinstance(recipient, str):
            raise RuntimeError("Wake Akashic delivery 缺少 recipient")
        new_delivery_session = session_map.get(f"{channel}:{recipient}")
        if new_delivery_session is None:
            raise RuntimeError("Wake delivery 指向不存在的旧 Akashic Session")
        delivery["channel"] = "akashic"
        delivery["recipient"] = _chat_id(new_delivery_session)
    _write_atomic(
        path, tomlkit.dumps(document).encode("utf-8"), path.stat().st_mode & 0o777
    )


def _migrated_config(raw: bytes) -> bytes:
    """Remove the retired Web identity selector before loading current config."""

    document = tomlkit.parse(raw.decode("utf-8"))
    document_channels = document.get("channels")
    if isinstance(document_channels, MutableMapping):
        document_chat = document_channels.get("chat")
        if isinstance(document_chat, MutableMapping):
            document_chat.pop("channel_name", None)
    return tomlkit.dumps(document).encode("utf-8")


def _rekey_sqlite_column(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    session_map: dict[str, str],
    old_channels: frozenset[str],
) -> None:
    rows = connection.execute(
        f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"
    ).fetchall()
    for row in rows:
        old = str(row[0])
        if not _is_old_session(old, old_channels):
            continue
        new = session_map.get(old)
        if new is None:
            raise RuntimeError(f"{table}.{column} 引用不存在的旧 Session: {old}")
        connection.execute(
            f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
            (new, old),
        )


def _migrate_plugin_state(
    path: Path,
    *,
    columns: tuple[tuple[str, str], ...],
    session_map: dict[str, str],
    old_channels: frozenset[str],
) -> None:
    if not path.is_file():
        return
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for table, column in columns:
            _rekey_sqlite_column(
                connection,
                table=table,
                column=column,
                session_map=session_map,
                old_channels=old_channels,
            )
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"插件状态 integrity_check 失败: {path}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _load_mobile_message_effects(
    path: Path,
    *,
    migrated_sessions: frozenset[str],
) -> dict[str, str]:
    """Map each migrated Mobile command ID to its persisted user Session."""

    if not path.is_file() or not migrated_sessions:
        return {}
    effects: dict[str, str] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        for message_id, session_id, role, raw_extra in connection.execute(
            "SELECT id, session_key, role, extra FROM messages WHERE extra IS NOT NULL"
        ):
            session_key = str(session_id)
            if session_key not in migrated_sessions:
                continue
            extra = _decode_json(raw_extra, field=f"messages.extra:{message_id}")
            if not isinstance(extra, dict):
                raise ValueError(f"messages.extra:{message_id} 必须是 object")
            client_message_id = extra.get("client_message_id")
            if client_message_id is None:
                continue
            if not isinstance(client_message_id, str) or not client_message_id:
                raise ValueError(f"messages.extra:{message_id}.client_message_id 无效")
            if str(role) != "user":
                raise RuntimeError(
                    f"Mobile client_message_id 绑定了非用户消息: {client_message_id}"
                )
            if client_message_id in effects:
                raise RuntimeError(
                    f"Mobile client_message_id 重复: {client_message_id}"
                )
            effects[client_message_id] = session_key
    return effects


def _settle_legacy_gateway_receipts(
    connection: sqlite3.Connection,
    *,
    message_effects: dict[str, str],
    completed_at: str,
) -> None:
    """Close old-client receipts without replaying any command side effect."""

    unknown = int(
        connection.execute(
            "SELECT COUNT(*) FROM mobile_command_receipts "
            "WHERE status = 'outcome_unknown'"
        ).fetchone()[0]
    )
    if unknown:
        raise RuntimeError(f"Mobile Gateway 存在 {unknown} 个 outcome_unknown receipt")

    rows = connection.execute(
        "SELECT device_id, command_id, command_type "
        "FROM mobile_command_receipts WHERE status = 'processing' "
        "ORDER BY device_id, command_id"
    ).fetchall()
    for row in rows:
        device_id = str(row["device_id"])
        command_id = str(row["command_id"])
        command_type = str(row["command_type"])
        if command_type in {"session.list", "plugin.ui.call"}:
            connection.execute(
                "DELETE FROM mobile_command_receipts "
                "WHERE device_id = ? AND command_id = ? AND status = 'processing'",
                (device_id, command_id),
            )
            continue
        if command_type != "message.send":
            raise RuntimeError(
                "Mobile Gateway 存在无法收束的 processing receipt: "
                f"{device_id}/{command_id}/{command_type}"
            )

        session_id = message_effects.get(command_id)
        if session_id is None:
            reply_type = "message.send.error"
            reply_payload: dict[str, object] = {
                "code": "command_interrupted",
                "message": "上次发送在服务重启时中断，可以安全重试",
            }
        else:
            reply_type = "message.send.ok"
            reply_payload = {
                "accepted": True,
                "client_message_id": command_id,
            }
        updated = connection.execute(
            "UPDATE mobile_command_receipts SET status = 'completed', "
            "reply_type = ?, reply_payload_json = ?, session_id = ?, "
            "turn_id = NULL, completed_at = ? "
            "WHERE device_id = ? AND command_id = ? AND status = 'processing'",
            (
                reply_type,
                _encode_json(reply_payload),
                session_id,
                completed_at,
                device_id,
                command_id,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("Mobile Gateway processing receipt 收束冲突")


def _migrate_gateway(
    path: Path,
    *,
    session_map: dict[str, str],
    message_map: dict[str, str],
    message_effects: dict[str, str],
) -> None:
    if not path.is_file():
        return
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        from infra.mobile_realtime.gateway import _encode_stored_event, _new_ulid

        pending_imports = int(
            connection.execute(
                "SELECT COUNT(*) FROM mobile_attachment_imports WHERE phase != 'message_bound'"
            ).fetchone()[0]
        )
        if pending_imports:
            raise RuntimeError(
                f"Mobile Gateway 存在 {pending_imports} 个未完成 attachment import"
            )
        connection.execute("BEGIN IMMEDIATE")
        _settle_legacy_gateway_receipts(
            connection,
            message_effects=message_effects,
            completed_at=datetime.now(UTC).isoformat(),
        )
        for table in (
            "mobile_device_sessions",
            "mobile_attachments",
            "mobile_attachment_imports",
        ):
            for old, new in session_map.items():
                connection.execute(
                    f"UPDATE {table} SET session_id = ? WHERE session_id = ?",
                    (new, old),
                )
        for old, new in message_map.items():
            connection.execute(
                "UPDATE mobile_message_attachments SET message_id = ? WHERE message_id = ?",
                (new, old),
            )
        for row in connection.execute(
            "SELECT device_id, command_id, session_id, reply_payload_json FROM mobile_command_receipts"
        ).fetchall():
            session_id = row["session_id"]
            payload = row["reply_payload_json"]
            connection.execute(
                "UPDATE mobile_command_receipts SET session_id = ?, reply_payload_json = ? WHERE device_id = ? AND command_id = ?",
                (
                    (
                        session_map.get(str(session_id), str(session_id))
                        if session_id is not None
                        else None
                    ),
                    (
                        _rewrite_json_column(
                            payload,
                            session_map=session_map,
                            message_map=message_map,
                            source_ref_map={},
                            field="mobile_command_receipts.reply_payload_json",
                        )
                        if payload is not None
                        else None
                    ),
                    row["device_id"],
                    row["command_id"],
                ),
            )
        # Inbox 是可重建投影。丢弃所有旧身份事件，并让每台有效设备
        # 在自己的下一个 durable 序号收到唯一的全量重建边界。
        connection.execute("DELETE FROM mobile_device_inbox")
        reset_at = datetime.now(UTC).isoformat()
        for row in connection.execute(
            "SELECT c.device_id, c.next_event_seq "
            "FROM mobile_device_cursors c "
            "JOIN mobile_devices d ON d.device_id = c.device_id "
            "WHERE d.revoked_at IS NULL ORDER BY c.device_id"
        ).fetchall():
            event_id = _new_ulid()
            event_seq = int(row["next_event_seq"])
            connection.execute(
                "INSERT INTO mobile_device_inbox("
                "device_id, event_seq, event_id, priority, envelope_json, created_at"
                ") VALUES (?, ?, ?, 'P0', ?, ?)",
                (
                    row["device_id"],
                    event_seq,
                    event_id,
                    _encode_stored_event(
                        event_id=event_id,
                        event_type="sync.reset_required",
                        payload={"reason": "akashic_identity_rekey"},
                    ),
                    reset_at,
                ),
            )
            updated = connection.execute(
                "UPDATE mobile_device_cursors SET next_event_seq = ? "
                "WHERE device_id = ? AND next_event_seq = ?",
                (event_seq + 1, row["device_id"], event_seq),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Mobile Gateway reset event 序号分配冲突")
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Mobile Gateway integrity_check 失败")
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _migrate_delivery_ledger(
    path: Path,
    *,
    old_channels: frozenset[str],
    session_map: dict[str, str],
    message_map: dict[str, str],
) -> None:
    if not path.is_file():
        return
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        forward = int(
            connection.execute(
                "SELECT COUNT(*) FROM deliveries WHERE state NOT IN ('settled', 'rejected', 'uncertain')"
            ).fetchone()[0]
        )
        if forward:
            raise RuntimeError(f"durable delivery ledger 存在 {forward} 条未终态记录")
        connection.execute("BEGIN IMMEDIATE")
        for row in connection.execute("SELECT * FROM deliveries").fetchall():
            channel = str(row["channel"])
            recipient = str(row["recipient"])
            new_channel = channel
            new_recipient = recipient
            if channel in old_channels:
                new_session = session_map.get(f"{channel}:{recipient}")
                if new_session is None:
                    raise RuntimeError("delivery ledger 引用不存在的旧 Akashic Session")
                new_channel = "akashic"
                new_recipient = _chat_id(new_session)
            connection.execute(
                "UPDATE deliveries SET accepted_session_id = ?, channel = ?, recipient = ?, projection_session_id = ?, projection_message_id = ?, metadata_json = ? WHERE logical_delivery_id = ?",
                (
                    session_map.get(
                        str(row["accepted_session_id"]), str(row["accepted_session_id"])
                    ),
                    new_channel,
                    new_recipient,
                    session_map.get(
                        str(row["projection_session_id"]),
                        str(row["projection_session_id"]),
                    ),
                    (
                        message_map.get(
                            str(row["projection_message_id"]),
                            str(row["projection_message_id"]),
                        )
                        if row["projection_message_id"] is not None
                        else None
                    ),
                    _rewrite_json_column(
                        row["metadata_json"],
                        session_map=session_map,
                        message_map=message_map,
                        source_ref_map={},
                        field="deliveries.metadata_json",
                    ),
                    row["logical_delivery_id"],
                ),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _backup_file(path: Path, backup_root: Path, name: str) -> None:
    if not path.is_file():
        return
    payload = path.read_bytes()
    destination = backup_root / name
    _write_atomic(destination, payload, 0o600)
    if (
        hashlib.sha256(destination.read_bytes()).digest()
        != hashlib.sha256(payload).digest()
    ):
        raise RuntimeError(f"Akashic migration backup 校验失败: {path}")


def _restore_sqlite(backup: Path, target: Path) -> None:
    """Restore one verified SQLite backup through a private candidate."""

    candidate = target.with_name(f".{target.name}.{uuid4().hex}.restore")
    try:
        with (
            closing(sqlite3.connect(f"file:{backup}?mode=ro", uri=True)) as source,
            closing(sqlite3.connect(candidate)) as destination,
        ):
            source.backup(destination)
        with closing(sqlite3.connect(candidate)) as restored:
            if restored.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError(
                    f"Akashic migration restore integrity_check 失败: {target}"
                )
        candidate.chmod(0o600)
        _remove_sqlite_target(target)
        os.replace(candidate, target)
        _fsync_directory(target.parent)
    finally:
        candidate.unlink(missing_ok=True)


def _remove_sqlite_target(target: Path) -> None:
    """Remove a SQLite database and journal sidecars before restore."""

    for path in (target, Path(f"{target}-wal"), Path(f"{target}-shm")):
        path.unlink(missing_ok=True)
    if target.parent.is_dir():
        _fsync_directory(target.parent)


def _restore_targets(
    records: list[dict[str, object]],
    *,
    workspace: Path,
    config_path: Path,
    backup_root: Path,
) -> None:
    """Restore every authority changed by one incomplete migration run."""

    workspace_root = workspace.resolve()
    config_target = config_path.resolve()
    backup_boundary = backup_root.resolve()
    for record in reversed(records):
        target = Path(str(record["target"])).resolve()
        if target != config_target and not target.is_relative_to(workspace_root):
            raise RuntimeError(f"Akashic migration restore target 越界: {target}")
        if record.get("existed") is not True:
            if record.get("kind") == "sqlite":
                _remove_sqlite_target(target)
            else:
                target.unlink(missing_ok=True)
                if target.parent.is_dir():
                    _fsync_directory(target.parent)
            continue
        backup_value = record.get("backup")
        if not isinstance(backup_value, str):
            raise RuntimeError(f"Akashic migration restore 缺少 backup: {target}")
        backup = Path(backup_value).resolve()
        if not backup.is_relative_to(backup_boundary) or not backup.is_file():
            raise RuntimeError(f"Akashic migration restore backup 无效: {backup}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if record.get("kind") == "sqlite":
            _restore_sqlite(backup, target)
        elif record.get("kind") == "file":
            mode = record.get("mode", 0o600)
            if not isinstance(mode, int):
                raise RuntimeError(f"Akashic migration restore mode 无效: {target}")
            _write_atomic(
                target,
                backup.read_bytes(),
                mode,
            )
        else:
            raise RuntimeError(
                f"Akashic migration restore kind 无效: {record.get('kind')}"
            )


def _recover_incomplete_migration(
    marker: Path,
    *,
    workspace: Path,
    config_path: Path,
) -> None:
    if not marker.is_file():
        return
    payload = _decode_json(marker.read_text(encoding="utf-8"), field=str(marker))
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list):
        raise RuntimeError("Akashic migration recovery marker 无效")
    _restore_targets(
        payload["targets"],
        workspace=workspace,
        config_path=config_path,
        backup_root=Path(str(payload.get("backup_root", ""))),
    )
    marker.unlink()
    _fsync_directory(marker.parent)


def _backup_target(
    target: Path,
    *,
    backup_root: Path,
    name: str,
    kind: str,
) -> dict[str, object]:
    exists = target.is_file()
    record: dict[str, object] = {
        "target": str(target),
        "kind": kind,
        "existed": exists,
        "backup": None,
        "mode": target.stat().st_mode & 0o777 if exists else 0o600,
    }
    if not exists:
        return record
    if kind == "sqlite":
        backup = backup_sqlite_database(
            target,
            backup_root / name,
            migration=_MIGRATION,
        )
    elif kind == "file":
        backup = backup_root / f"{name}.before"
        _backup_file(target, backup_root, backup.name)
    else:
        raise ValueError(f"backup kind 无效: {kind}")
    record["backup"] = str(backup)
    return record


def _akasha_targets(workspace: Path) -> tuple[Path, Path]:
    from agent.plugins.manifest import builtin_plugin_data_dir
    from plugins.akasha.config import load_akasha_config, resolve_memory_path

    plugin = load_akasha_config(
        builtin_plugin_data_dir("akasha", workspace) / "config.local.toml"
    )
    return (
        resolve_memory_path(workspace / "memory", plugin.index_path),
        resolve_memory_path(workspace / "memory", plugin.db_path),
    )


def _has_old_session_identity(path: Path, old_channels: frozenset[str]) -> bool:
    """Check whether SessionDB contains an identity owned by an old channel."""

    if not path.is_file():
        return False
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        if table is None:
            return False
        return any(
            connection.execute(
                "SELECT 1 FROM sessions WHERE key LIKE ? LIMIT 1",
                (f"{channel}:%",),
            ).fetchone()
            is not None
            for channel in old_channels
        )


def unify_akashic_identity(_connection: object) -> None:
    """Rekey one stopped workspace through a single reviewed migration plan."""

    _ = _connection
    current = current_migration_context()
    marker = current.workspace / "backups" / _MIGRATION / "in-progress.json"
    _recover_incomplete_migration(
        marker,
        workspace=current.workspace,
        config_path=current.config_path,
    )
    sessions = current.workspace / "sessions.db"
    if not current.config_path.is_file():
        if not _has_old_session_identity(
            sessions,
            frozenset({"web", "mobile"}),
        ):
            return
        raise RuntimeError("Akashic identity 迁移发现旧 Session，但缺少 config.toml")
    raw_config = current.config_path.read_bytes()
    migrated_config = _migrated_config(raw_config)
    config = tomllib.loads(raw_config.decode("utf-8"))
    channels = config.get("channels", {})
    if not isinstance(channels, dict):
        raise ValueError("channels 必须是 table")
    chat = channels.get("chat", {})
    if not isinstance(chat, dict):
        raise ValueError("channels.chat 必须是 table")
    old_web_channel = str(chat.get("channel_name", "web") or "web")
    old_channels = frozenset({old_web_channel, "mobile"} - {"akashic"})

    mobile_config = config.get("mobile_realtime", {})
    if not isinstance(mobile_config, dict):
        raise ValueError("mobile_realtime 必须是 table")
    mobile_db = current.workspace / str(
        mobile_config.get("database", "data/mobile_realtime.db")
    )
    delivery_db = current.workspace / "runtime" / "deliveries" / "settlements.sqlite"
    schedules = current.workspace / "schedules.json"
    wake = current.workspace / "plugin-data" / "wake-builtin" / "config.local.toml"
    content_db = (
        current.workspace / "plugin-data" / "content-builtin" / "content.sqlite3"
    )
    drift_db = current.workspace / "plugin-data" / "drift-builtin" / "drift.sqlite3"
    has_old_sessions = _has_old_session_identity(sessions, old_channels)

    backup_root = current.workspace / "backups" / _MIGRATION / uuid4().hex
    backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    records = [
        _backup_target(
            sessions,
            backup_root=backup_root,
            name="sessions",
            kind="sqlite",
        ),
        _backup_target(
            mobile_db,
            backup_root=backup_root,
            name="mobile",
            kind="sqlite",
        ),
        _backup_target(
            delivery_db,
            backup_root=backup_root,
            name="deliveries",
            kind="sqlite",
        ),
        _backup_target(
            current.config_path,
            backup_root=backup_root,
            name="config.toml",
            kind="file",
        ),
        _backup_target(
            schedules,
            backup_root=backup_root,
            name="schedules.json",
            kind="file",
        ),
        _backup_target(
            wake,
            backup_root=backup_root,
            name="wake-config.local.toml",
            kind="file",
        ),
        _backup_target(
            content_db,
            backup_root=backup_root,
            name="content",
            kind="sqlite",
        ),
        _backup_target(
            drift_db,
            backup_root=backup_root,
            name="drift",
            kind="sqlite",
        ),
    ]
    if has_old_sessions:
        akasha_index, akasha_memory = _akasha_targets(current.workspace)
        records.extend(
            (
                _backup_target(
                    akasha_index,
                    backup_root=backup_root,
                    name="akasha-index",
                    kind="sqlite",
                ),
                _backup_target(
                    akasha_memory,
                    backup_root=backup_root,
                    name="akasha-memory",
                    kind="sqlite",
                ),
            )
        )
    marker_payload = {
        "migration": _MIGRATION,
        "backup_root": str(backup_root),
        "targets": records,
    }
    marker.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    _write_atomic(
        marker,
        (_encode_json(marker_payload) + "\n").encode("utf-8"),
        0o600,
    )

    try:
        if has_old_sessions:
            from agent.migrations.akasha_embedding_backfill import (
                backfill_akasha_message_embeddings,
            )

            _ = backfill_akasha_message_embeddings(
                config_path=current.config_path,
                migrated_config=migrated_config,
                workspace=current.workspace,
            )
        if sessions.is_file():
            session_map, message_map = _migrate_sessions(
                sessions,
                old_channels=old_channels,
            )
        else:
            session_map, message_map = {}, {}
        plan = {
            "migration": _MIGRATION,
            "namespace": str(_NAMESPACE),
            "old_channels": sorted(old_channels),
            "sessions": session_map,
            "messages": message_map,
        }
        _write_atomic(
            backup_root / "plan.json",
            (_encode_json(plan) + "\n").encode("utf-8"),
            0o600,
        )
        _migrate_gateway(
            mobile_db,
            session_map=session_map,
            message_map=message_map,
            message_effects=_load_mobile_message_effects(
                sessions,
                migrated_sessions=frozenset(
                    new for old, new in session_map.items() if old.startswith("mobile:")
                ),
            ),
        )
        _migrate_delivery_ledger(
            delivery_db,
            old_channels=old_channels,
            session_map=session_map,
            message_map=message_map,
        )
        _migrate_json_file(
            schedules,
            old_channels=old_channels,
            session_map=session_map,
            message_map=message_map,
        )
        _migrate_wake_config(
            wake,
            old_channels=old_channels,
            session_map=session_map,
        )
        _migrate_plugin_state(
            content_db,
            columns=(
                ("items", "selected_session_id"),
                ("content_selections", "accepted_session_id"),
            ),
            session_map=session_map,
            old_channels=old_channels,
        )
        _migrate_plugin_state(
            drift_db,
            columns=(("proposals", "selected_session_id"),),
            session_map=session_map,
            old_channels=old_channels,
        )

        _write_atomic(
            current.config_path,
            migrated_config,
            current.config_path.stat().st_mode & 0o777,
        )

        if session_map:
            from agent.migrations.akasha_sidecar import rebuild_akasha_sidecars

            _ = rebuild_akasha_sidecars(
                config_path=current.config_path,
                workspace=current.workspace,
                backup_dir=backup_root / "akasha-rebuild",
                accepted_versions=set(),
                force=True,
            )
    except BaseException as migration_error:
        try:
            _restore_targets(
                records,
                workspace=current.workspace,
                config_path=current.config_path,
                backup_root=backup_root,
            )
            marker.unlink()
            _fsync_directory(marker.parent)
        except BaseException as restore_error:
            raise BaseExceptionGroup(
                "Akashic identity 迁移失败且恢复失败",
                [migration_error, restore_error],
            ) from migration_error
        raise
    marker.unlink()
    _fsync_directory(marker.parent)


steps = [step(unify_akashic_identity)]
