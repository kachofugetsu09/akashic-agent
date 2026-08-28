from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.payloads.eventmail_v3 import EventMailV3MigrationStore
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260827_02_migrate_legacy_mobile_client_ids"}
__transactional__ = False

_MIGRATION = "migrate-eventmail-state"
_CONTENT_ROOT = Path("plugin-data/content-builtin")
_EVENTMAIL_ROOT = Path("plugin-data/eventmail-builtin")
_WAKE_DB = Path("plugin-data/wake-builtin/wake.sqlite3")
_CONTENT_DB_FILES = {
    "content.sqlite3",
    "content.sqlite3-shm",
    "content.sqlite3-wal",
}
_WAKE_RUN_SQL = """
    CREATE TABLE wake_runs(
        run_id TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        started_at TEXT NOT NULL,
        candidates_seen INTEGER NOT NULL,
        candidates_selected INTEGER NOT NULL,
        screening_json TEXT NOT NULL,
        decision TEXT,
        decision_detail TEXT,
        completed_at TEXT
    )
"""
_WAKE_ATTEMPT_SQL = """
    CREATE TABLE wake_attempts(
        attempt_id TEXT PRIMARY KEY,
        timer_id TEXT NOT NULL,
        scheduled_for TEXT NOT NULL,
        fired_at TEXT NOT NULL,
        mail_watermark INTEGER,
        outcome TEXT NOT NULL CHECK(outcome IN (
            'checking', 'no_due', 'content_insufficient', 'admission_rejected',
            'shared', 'model_skip', 'deferred', 'cancelled_after_fire',
            'delivery_unknown', 'failed'
        )),
        owner TEXT CHECK(owner IN ('alert', 'content', 'drift')),
        detail TEXT,
        completed_at TEXT
    )
"""


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _integrity(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchall()
    if result != [("ok",)]:
        raise RuntimeError(f"EventMail migration integrity_check failed: {path}")


def _logical_digest(path: Path) -> str:
    """Hash every owned logical row without depending on SQLite file layout."""

    with closing(sqlite3.connect(path)) as connection:
        tables = sorted(_tables(connection))
        snapshot: list[object] = []
        for table in tables:
            escaped = table.replace('"', '""')
            columns = tuple(
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{escaped}")')
            )
            rows = connection.execute(
                f'SELECT * FROM "{escaped}" ORDER BY rowid'
            ).fetchall()
            snapshot.append((table, columns, rows))
    encoded = json.dumps(
        snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_content_root(
    source: Path, target: Path, database_backup: Path | None
) -> None:
    """Copy opaque plugin data while replacing SQLite with an online backup."""

    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        target.mkdir(parents=True)
    for name in _CONTENT_DB_FILES:
        path = target / name
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                raise RuntimeError(f"旧 Content SQLite sidecar 不是文件: {path}")
            path.unlink()
    if database_backup is not None:
        shutil.copy2(database_backup, target / "eventmail.sqlite3")


def _legacy_wake_rows(
    wake_db: Path,
) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
    if not wake_db.is_file():
        return [], []
    with closing(sqlite3.connect(wake_db)) as connection:
        connection.row_factory = sqlite3.Row
        names = _tables(connection)
        alerts: list[sqlite3.Row] = []
        contexts: list[sqlite3.Row] = []
        if "alert_events" in names:
            expiry_join = (
                "LEFT JOIN alert_expiry AS expiry "
                "ON expiry.source_id=alert.source_id "
                "AND expiry.event_id=alert.event_id"
                if "alert_expiry" in names
                else ""
            )
            expiry_column = "expiry.expires_at" if "alert_expiry" in names else "NULL"
            alerts = connection.execute(
                "SELECT alert.source_id, alert.event_id, alert.payload_json, "
                "alert.observed_at, alert.not_before, alert.status, "
                "alert.accepted_session, alert.accepted_turn, "
                f"{expiry_column} AS expires_at FROM alert_events AS alert "
                f"{expiry_join} ORDER BY alert.source_id, alert.event_id"
            ).fetchall()
        if "context_events" in names:
            contexts = connection.execute(
                "SELECT source_id, event_id, payload_json, observed_at, expires_at "
                "FROM context_events ORDER BY source_id, event_id"
            ).fetchall()
        return alerts, contexts


def _datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"旧 {field} 不是 ISO 时间")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise RuntimeError(f"旧 {field} 缺少时区")
    return parsed


def _payload(value: object, field: str) -> dict[str, object]:
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"旧 {field} payload JSON 无效") from error
    if not isinstance(decoded, dict) or not decoded:
        raise RuntimeError(f"旧 {field} payload 不是非空 object")
    return decoded


def _import_wake_mail(
    eventmail_db: Path,
    alerts: list[sqlite3.Row],
    contexts: list[sqlite3.Row],
) -> None:
    """Append legacy Wake mail and preserve its exact terminal projection."""

    store = EventMailV3MigrationStore(eventmail_db)
    store.initialize()
    for row in alerts:
        status = str(row["status"])
        if status not in {"pending", "selected", "delivered", "skipped"}:
            raise RuntimeError(f"旧 Alert status 无效: {status}")
        observed = _datetime(row["observed_at"], "Alert observed_at")
        not_before = _datetime(row["not_before"], "Alert not_before")
        expires = (
            None
            if row["expires_at"] is None
            else _datetime(row["expires_at"], "Alert expires_at")
        )
        receipt = store.report_alert(
            source_id=str(row["source_id"]),
            event_id=str(row["event_id"]),
            payload=_payload(row["payload_json"], "Alert"),
            observed_at=observed,
            expires_at=expires,
        )
        mail_id = str(receipt["mail_id"])
        accepted_session = row["accepted_session"]
        accepted_turn = row["accepted_turn"]
        if status == "selected" and (
            not isinstance(accepted_session, str)
            or not accepted_session
            or not isinstance(accepted_turn, str)
            or not accepted_turn
        ):
            raise RuntimeError("旧 selected Alert 缺少 accepted Turn")
        if status != "selected" and (
            accepted_session is not None or accepted_turn is not None
        ):
            raise RuntimeError("旧非 selected Alert 持有 accepted Turn")
        with closing(sqlite3.connect(eventmail_db)) as connection, connection:
            connection.row_factory = sqlite3.Row
            connection.execute(
                "UPDATE alert_projection SET status=?, not_before=?, "
                "accepted_session=?, accepted_turn=? "
                "WHERE source_id=? AND event_id=? AND mail_id=?",
                (
                    status,
                    not_before.isoformat(),
                    accepted_session,
                    accepted_turn,
                    str(row["source_id"]),
                    str(row["event_id"]),
                    mail_id,
                ),
            )
            if status != "pending":
                detail = (
                    {
                        "accepted_turn": {
                            "session_id": str(accepted_session),
                            "turn_id": str(accepted_turn),
                        }
                    }
                    if status == "selected"
                    else {}
                )
                EventMailV3MigrationStore._append_transition(  # pyright: ignore[reportPrivateUsage]
                    connection,
                    mail_id,
                    "alert",
                    status,
                    detail,
                )
    for row in contexts:
        store.report_context(
            source_id=str(row["source_id"]),
            event_id=str(row["event_id"]),
            payload=_payload(row["payload_json"], "Context"),
            observed_at=_datetime(row["observed_at"], "Context observed_at"),
            expires_at=(
                None
                if row["expires_at"] is None
                else _datetime(row["expires_at"], "Context expires_at")
            ),
        )


def _verify_import(
    eventmail_db: Path,
    alerts: list[sqlite3.Row],
    contexts: list[sqlite3.Row],
) -> None:
    store = EventMailV3MigrationStore(eventmail_db)
    store.initialize()
    with closing(sqlite3.connect(eventmail_db)) as connection:
        connection.row_factory = sqlite3.Row
        for row in alerts:
            actual = connection.execute(
                "SELECT envelope.payload_json, envelope.observed_at, "
                "projection.not_before, projection.expires_at, projection.status, "
                "projection.accepted_session, projection.accepted_turn "
                "FROM alert_projection AS projection "
                "JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id "
                "WHERE projection.source_id=? AND projection.event_id=?",
                (row["source_id"], row["event_id"]),
            ).fetchone()
            expected = (
                _payload(row["payload_json"], "Alert"),
                _datetime(row["observed_at"], "Alert observed_at"),
                _datetime(row["not_before"], "Alert not_before"),
                (
                    None
                    if row["expires_at"] is None
                    else _datetime(row["expires_at"], "Alert expires_at")
                ),
                str(row["status"]),
                row["accepted_session"],
                row["accepted_turn"],
            )
            observed = None
            if actual is not None:
                observed = (
                    _payload(actual["payload_json"], "migrated Alert"),
                    _datetime(actual["observed_at"], "migrated Alert observed_at"),
                    _datetime(actual["not_before"], "migrated Alert not_before"),
                    (
                        None
                        if actual["expires_at"] is None
                        else _datetime(
                            actual["expires_at"], "migrated Alert expires_at"
                        )
                    ),
                    str(actual["status"]),
                    actual["accepted_session"],
                    actual["accepted_turn"],
                )
            if observed != expected:
                raise RuntimeError(
                    "旧 Alert 迁移核对失败: " f"{row['source_id']}/{row['event_id']}"
                )
        for row in contexts:
            actual = connection.execute(
                "SELECT envelope.payload_json, envelope.observed_at, projection.expires_at "
                "FROM context_projection AS projection "
                "JOIN mail_envelopes AS envelope ON envelope.mail_id=projection.mail_id "
                "WHERE projection.source_id=? AND projection.event_id=?",
                (row["source_id"], row["event_id"]),
            ).fetchone()
            expected = (
                _payload(row["payload_json"], "Context"),
                _datetime(row["observed_at"], "Context observed_at"),
                (
                    None
                    if row["expires_at"] is None
                    else _datetime(row["expires_at"], "Context expires_at")
                ),
            )
            observed = None
            if actual is not None:
                observed = (
                    _payload(actual["payload_json"], "migrated Context"),
                    _datetime(actual["observed_at"], "migrated Context observed_at"),
                    (
                        None
                        if actual["expires_at"] is None
                        else _datetime(
                            actual["expires_at"], "migrated Context expires_at"
                        )
                    ),
                )
            if observed != expected:
                raise RuntimeError(
                    "旧 Context 迁移核对失败: " f"{row['source_id']}/{row['event_id']}"
                )
    _integrity(eventmail_db)


def _retire_wake_mail(wake_db: Path) -> None:
    """Remove migrated mail tables and publish the exact Wake v7 schema."""

    if not wake_db.is_file():
        return
    with closing(sqlite3.connect(wake_db)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("BEGIN IMMEDIATE")
        names = _tables(connection)
        if "admission_state" not in names:
            raise RuntimeError("旧 Wake DB 缺少 admission_state")
        if "seen_content" not in names:
            connection.execute(
                "CREATE TABLE seen_content(item_identity TEXT PRIMARY KEY)"
            )
        if "wake_runs" not in names:
            connection.execute(_WAKE_RUN_SQL)
        if "wake_attempts" in names:
            connection.execute("ALTER TABLE wake_attempts RENAME TO old_wake_attempts")
            connection.execute(_WAKE_ATTEMPT_SQL)
            connection.execute(
                "INSERT INTO wake_attempts("
                "attempt_id, timer_id, scheduled_for, fired_at, mail_watermark, "
                "outcome, owner, detail, completed_at"
                ") SELECT attempt_id, timer_id, scheduled_for, fired_at, "
                "mail_watermark, CASE outcome WHEN 'completed' THEN 'delivery_unknown' "
                "ELSE outcome END, owner, detail, completed_at FROM old_wake_attempts"
            )
            connection.execute("DROP TABLE old_wake_attempts")
        else:
            connection.execute(_WAKE_ATTEMPT_SQL)
        connection.execute("DROP TABLE IF EXISTS alert_expiry")
        connection.execute("DROP TABLE IF EXISTS alert_events")
        connection.execute("DROP TABLE IF EXISTS context_events")
        connection.execute("PRAGMA user_version=7")
        remaining = _tables(connection)
        expected = {
            "admission_state",
            "seen_content",
            "wake_runs",
            "wake_attempts",
        }
        if remaining != expected:
            raise RuntimeError(f"Wake v7 schema tables 不匹配: {sorted(remaining)}")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Wake v7 integrity_check 失败")


def _write_receipt(
    target_root: Path,
    *,
    backup_root: Path,
    alerts: int,
    contexts: int,
) -> None:
    receipt = {
        "schema_version": 1,
        "migration": _MIGRATION,
        "backup_root": str(backup_root),
        "legacy_alerts": alerts,
        "legacy_contexts": contexts,
        "eventmail_integrity": "ok",
        "eventmail_logical_digest": _logical_digest(target_root / "eventmail.sqlite3"),
        "wake_schema": 7,
    }
    path = target_root / "migration-receipt.json"
    path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _verify_receipt(target_root: Path) -> None:
    """Bind hard-crash recovery to the exact published EventMail state."""

    path = target_root / "migration-receipt.json"
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("EventMail migration receipt 必须是 object")
    if decoded.get("schema_version") != 1 or decoded.get("migration") != _MIGRATION:
        raise RuntimeError("EventMail migration receipt identity 无效")
    expected = decoded.get("eventmail_logical_digest")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("EventMail migration receipt 缺少 logical digest")
    actual = _logical_digest(target_root / "eventmail.sqlite3")
    if actual != expected:
        raise RuntimeError("EventMail target 与 migration receipt digest 不一致")


def migrate_eventmail_state(_connection: object) -> None:
    """Move all mail authority to EventMail, then retire both old sources."""

    _ = _connection
    workspace = current_migration_context().workspace
    old_root = workspace / _CONTENT_ROOT
    target_root = workspace / _EVENTMAIL_ROOT
    wake_db = workspace / _WAKE_DB
    alerts, contexts = _legacy_wake_rows(wake_db)
    content_source = old_root
    recovered_backup_root: Path | None = None
    if not old_root.exists() and not target_root.exists():
        retired_sources = sorted(
            (workspace / "backups" / _MIGRATION).glob("*/retired-content-builtin")
        )
        if len(retired_sources) > 1:
            raise RuntimeError("EventMail migration 发现多个未完成的 Content 退休源")
        if retired_sources:
            content_source = retired_sources[0]
            recovered_backup_root = content_source.parent
    has_old_content = content_source.is_dir()
    if target_root.exists():
        receipt = target_root / "migration-receipt.json"
        if not receipt.is_file():
            if has_old_content or alerts or contexts:
                raise RuntimeError("EventMail target 已存在但缺少迁移 receipt")
            return
        _verify_receipt(target_root)
        _verify_import(target_root / "eventmail.sqlite3", alerts, contexts)
        _retire_wake_mail(wake_db)
        return
    if not has_old_content and not alerts and not contexts:
        return

    backup_root = recovered_backup_root or (
        workspace / "backups" / _MIGRATION / uuid4().hex
    )
    if recovered_backup_root is None:
        backup_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(backup_root, 0o700)
    content_db = content_source / "content.sqlite3"
    content_backup = None
    if content_db.is_file():
        existing_backup = backup_root / "content-db" / content_db.name
        content_backup = (
            existing_backup
            if existing_backup.is_file()
            else backup_sqlite_database(
                content_db,
                backup_root / "content-db",
                migration=_MIGRATION,
            )
        )
    if wake_db.is_file():
        wake_backup = backup_root / "wake-db" / wake_db.name
        if not wake_backup.is_file():
            _ = backup_sqlite_database(
                wake_db,
                backup_root / "wake-db",
                migration=_MIGRATION,
            )

    target_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_root.with_name(f".{target_root.name}.{uuid4().hex}.tmp")
    retired = backup_root / "retired-content-builtin"
    old_retired = False
    target_published = False
    try:
        _copy_content_root(content_source, temporary, content_backup)
        eventmail_db = temporary / "eventmail.sqlite3"
        EventMailV3MigrationStore(eventmail_db).initialize()
        _import_wake_mail(eventmail_db, alerts, contexts)
        _verify_import(eventmail_db, alerts, contexts)
        _write_receipt(
            temporary,
            backup_root=backup_root,
            alerts=len(alerts),
            contexts=len(contexts),
        )
        _verify_receipt(temporary)

        if has_old_content and recovered_backup_root is None:
            old_root.rename(retired)
            old_retired = True
        temporary.rename(target_root)
        target_published = True
        _retire_wake_mail(wake_db)
    except BaseException:
        if target_published and target_root.exists():
            target_root.rename(temporary)
            target_published = False
        if old_retired and retired.exists():
            retired.rename(old_root)
            old_retired = False
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


steps = [step(migrate_eventmail_state)]
