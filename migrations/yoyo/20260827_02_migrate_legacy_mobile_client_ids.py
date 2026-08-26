from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tomllib
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260827_01_normalize_session_timestamps"}
__transactional__ = False

_MIGRATION = "migrate-legacy-mobile-client-ids"
_FRAME_ID = re.compile(
    r"^(?:[0-9A-HJKMNP-TV-Z]{26}|"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-7[0-9A-Fa-f]{3}-"
    r"[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12})$"
)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _replacement_id(old_id: str, timestamp: str) -> str:
    """Build one stable UUIDv7 from the original command and message time."""

    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise RuntimeError(f"Mobile 历史消息时间无效: {timestamp!r}") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"Mobile 历史消息时间缺少时区: {timestamp!r}")
    milliseconds = int(parsed.timestamp() * 1000)
    if not 0 <= milliseconds < 1 << 48:
        raise RuntimeError(f"Mobile 历史消息时间超出 UUIDv7 范围: {timestamp!r}")
    random_bits = int.from_bytes(
        hashlib.sha256(old_id.encode("utf-8")).digest()[:10], "big"
    ) & ((1 << 74) - 1)
    random_a = random_bits >> 62
    random_b = random_bits & ((1 << 62) - 1)
    value = (milliseconds << 80) | (7 << 76) | (random_a << 64) | (2 << 62) | random_b
    return str(UUID(int=value))


def _legacy_client_ids(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, str, str]]:
    """Return old UUIDv4 ids and reject unknown historical id formats."""

    columns = _table_columns(connection, "messages")
    if not columns:
        return {}
    if not {"id", "session_key", "role", "extra", "ts"}.issubset(columns):
        return {}
    legacy: dict[str, tuple[str, str, str]] = {}
    current_ids: set[str] = set()
    rows = connection.execute(
        "SELECT id, role, extra, ts FROM messages "
        "WHERE session_key LIKE 'akashic:%' AND extra IS NOT NULL"
    ).fetchall()
    for message_id, role, raw_extra, timestamp in rows:
        try:
            extra = json.loads(str(raw_extra))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"messages.extra JSON 无效: {message_id}") from error
        if not isinstance(extra, dict):
            raise TypeError(f"messages.extra 不是 object: {message_id}")
        client_id = extra.get("client_message_id")
        if client_id is None:
            continue
        if not isinstance(client_id, str) or not client_id:
            raise RuntimeError(f"client_message_id 无效: {message_id}")
        if _FRAME_ID.fullmatch(client_id):
            current_ids.add(client_id)
            continue
        try:
            parsed_id = UUID(client_id)
        except ValueError as error:
            raise RuntimeError(
                f"client_message_id 不是 UUIDv4、UUIDv7 或 ULID: {message_id}"
            ) from error
        if parsed_id.version != 4 or str(parsed_id) != client_id.lower():
            raise RuntimeError(f"client_message_id 不是可迁移 UUIDv4: {message_id}")
        if str(role) != "user":
            raise RuntimeError(f"旧 client_message_id 绑定了非用户消息: {message_id}")
        if client_id in legacy:
            raise RuntimeError(f"旧 client_message_id 重复: {client_id}")
        replacement = _replacement_id(client_id, str(timestamp))
        legacy[client_id] = (replacement, str(message_id), str(raw_extra))
    collisions = current_ids.intersection(item[0] for item in legacy.values())
    if collisions:
        raise RuntimeError(f"client_message_id 迁移发生碰撞: {min(collisions)}")
    return legacy


def _gateway_path(config_path: Path, workspace: Path) -> Path:
    if not config_path.is_file():
        return workspace / "data/mobile_realtime.db"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    mobile = config.get("mobile_realtime", {})
    if not isinstance(mobile, dict):
        raise TypeError("mobile_realtime 配置不是 table")
    configured = Path(str(mobile.get("database", "data/mobile_realtime.db")))
    return configured if configured.is_absolute() else workspace / configured


def _check_gateway_references(
    connection: sqlite3.Connection,
    legacy_ids: frozenset[str],
) -> None:
    """Fail before writes if an old command still owns durable gateway state."""

    for table, column in (
        ("mobile_command_receipts", "command_id"),
        ("mobile_attachment_imports", "client_message_id"),
    ):
        if column not in _table_columns(connection, table):
            continue
        placeholders = ",".join("?" for _ in legacy_ids)
        row = connection.execute(
            f"SELECT {column} FROM {table} WHERE {column} IN ({placeholders}) LIMIT 1",
            tuple(sorted(legacy_ids)),
        ).fetchone()
        if row is not None:
            raise RuntimeError(f"旧 client_message_id 仍被 {table} 引用: {row[0]}")


def _update_sessions(
    connection: sqlite3.Connection,
    legacy: dict[str, tuple[str, str, str]],
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        for old_id, (new_id, message_id, raw_extra) in legacy.items():
            extra = json.loads(raw_extra)
            if extra.get("client_message_id") != old_id:
                raise RuntimeError(f"client_message_id 迁移计划失效: {message_id}")
            extra["client_message_id"] = new_id
            updated = connection.execute(
                "UPDATE messages SET extra = ? WHERE id = ? AND extra = ?",
                (
                    json.dumps(
                        extra,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    message_id,
                    raw_extra,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"client_message_id 更新时间冲突: {message_id}")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("SessionDB client_message_id 迁移留下 foreign key 错误")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _reset_gateway_inbox(connection: sqlite3.Connection) -> None:
    """Replace derived history events with one rebuild boundary per device."""

    required = {
        "mobile_devices": {"device_id", "revoked_at"},
        "mobile_device_cursors": {"device_id", "next_event_seq"},
        "mobile_device_inbox": {
            "device_id",
            "event_seq",
            "event_id",
            "priority",
            "envelope_json",
            "created_at",
        },
    }
    if not any(_table_columns(connection, table) for table in required):
        return
    for table, columns in required.items():
        if not columns.issubset(_table_columns(connection, table)):
            raise RuntimeError(f"Mobile Gateway schema lineage 不完整: {table}")

    from infra.mobile_realtime.gateway import _encode_stored_event, _new_ulid, _utc_now

    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("DELETE FROM mobile_device_inbox")
        created_at = _utc_now().isoformat()
        rows = connection.execute(
            "SELECT c.device_id, c.next_event_seq FROM mobile_device_cursors c "
            "JOIN mobile_devices d ON d.device_id = c.device_id "
            "WHERE d.revoked_at IS NULL ORDER BY c.device_id"
        ).fetchall()
        for device_id, event_seq in rows:
            event_id = _new_ulid()
            connection.execute(
                "INSERT INTO mobile_device_inbox("
                "device_id,event_seq,event_id,priority,envelope_json,created_at"
                ") VALUES(?,?,?,'P0',?,?)",
                (
                    device_id,
                    event_seq,
                    event_id,
                    _encode_stored_event(
                        event_id=event_id,
                        event_type="sync.reset_required",
                        payload={"reason": "legacy_client_message_id_rekey"},
                    ),
                    created_at,
                ),
            )
            updated = connection.execute(
                "UPDATE mobile_device_cursors SET next_event_seq = ? "
                "WHERE device_id = ? AND next_event_seq = ?",
                (int(event_seq) + 1, device_id, event_seq),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Mobile Gateway reset 序号冲突: {device_id}")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("Mobile Gateway reset 留下 foreign key 错误")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def migrate_legacy_mobile_client_ids(_connection: object) -> None:
    """Back up and rekey the last pre-UUIDv7 mobile message correlations."""

    _ = _connection
    current = current_migration_context()
    sessions = current.workspace / "sessions.db"
    if not sessions.is_file():
        return
    gateway = _gateway_path(current.config_path, current.workspace)
    with closing(sqlite3.connect(sessions)) as connection:
        legacy = _legacy_client_ids(connection)
    if not legacy:
        return
    if gateway.is_file():
        with closing(sqlite3.connect(gateway)) as connection:
            _check_gateway_references(connection, frozenset(legacy))

    backup_root = current.workspace / "backups" / _MIGRATION / uuid4().hex
    _ = backup_sqlite_database(
        sessions,
        backup_root / "sessions",
        migration=_MIGRATION,
    )
    if gateway.is_file():
        _ = backup_sqlite_database(
            gateway,
            backup_root / "gateway",
            migration=_MIGRATION,
        )

    if gateway.is_file():
        with closing(sqlite3.connect(gateway)) as connection:
            _reset_gateway_inbox(connection)
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Mobile Gateway reset integrity_check 失败")
    with closing(sqlite3.connect(sessions)) as connection:
        _update_sessions(connection, legacy)
        if _legacy_client_ids(connection):
            raise RuntimeError("SessionDB 仍有旧 client_message_id")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SessionDB client_message_id 迁移 integrity_check 失败")


steps = [step(migrate_legacy_mobile_client_ids)]
