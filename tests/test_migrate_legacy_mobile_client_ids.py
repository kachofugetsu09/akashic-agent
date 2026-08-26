from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from uuid import UUID

import pytest
import yoyo

from agent.migrations.context import bind_migration_context

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT / "migrations/yoyo/20260827_02_migrate_legacy_mobile_client_ids.py"
)
_OLD_ID = "7067a51e-40ef-41fb-9b58-f669752c1729"


def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "migrate_legacy_mobile_client_ids_under_test",
        _MIGRATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载迁移: {_MIGRATION_PATH}")
    original_step = yoyo.step
    yoyo.step = lambda callback: callback  # type: ignore[assignment]
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        yoyo.step = original_step
    return module


def _sessions(path: Path, *, client_id: str = _OLD_ID) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE messages("
            "id TEXT PRIMARY KEY, session_key TEXT NOT NULL, role TEXT NOT NULL, "
            "extra TEXT, ts TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO messages VALUES('akashic:test:1','akashic:test','user',?,?)",
            (
                json.dumps({"client_message_id": client_id, "kept": True}),
                "2026-08-01T04:05:06.789000+00:00",
            ),
        )


def _add_telegram_message(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO messages VALUES("
            "'telegram:1:1','telegram:1','user',?,'2026-08-01T04:00:00+00:00')",
            (json.dumps({"client_message_id": "22098"}),),
        )


def _gateway(path: Path, *, referenced: bool = False) -> None:
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE mobile_devices(
                device_id TEXT PRIMARY KEY, revoked_at TEXT
            );
            CREATE TABLE mobile_device_cursors(
                device_id TEXT PRIMARY KEY,
                next_event_seq INTEGER NOT NULL,
                sent_event_seq INTEGER NOT NULL,
                acknowledged_event_seq INTEGER NOT NULL,
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
            );
            CREATE TABLE mobile_device_inbox(
                device_id TEXT NOT NULL,
                event_seq INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                priority TEXT NOT NULL,
                envelope_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(device_id,event_seq),
                FOREIGN KEY(device_id) REFERENCES mobile_devices(device_id)
            );
            CREATE TABLE mobile_command_receipts(command_id TEXT NOT NULL);
            CREATE TABLE mobile_attachment_imports(client_message_id TEXT NOT NULL);
            INSERT INTO mobile_devices VALUES('pixel',NULL);
            INSERT INTO mobile_device_cursors VALUES('pixel',1,0,0);
            """
        )
        if referenced:
            connection.execute(
                "INSERT INTO mobile_command_receipts VALUES(?)", (_OLD_ID,)
            )


def test_rekeys_uuid4_and_resets_derived_gateway_history(tmp_path: Path) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    gateway = workspace / "data/mobile_realtime.db"
    _sessions(sessions)
    _add_telegram_message(sessions)
    _gateway(gateway)
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    with bind_migration_context(config_path=config, workspace=workspace):
        migration.migrate_legacy_mobile_client_ids(object())

    with closing(sqlite3.connect(sessions)) as connection:
        extra = json.loads(
            connection.execute("SELECT extra FROM messages").fetchone()[0]
        )
    replacement = extra["client_message_id"]
    assert UUID(replacement).version == 7
    assert extra["kept"] is True
    with closing(sqlite3.connect(sessions)) as connection:
        telegram_extra = json.loads(
            connection.execute(
                "SELECT extra FROM messages WHERE session_key='telegram:1'"
            ).fetchone()[0]
        )
    assert telegram_extra["client_message_id"] == "22098"
    with closing(sqlite3.connect(gateway)) as connection:
        cursor = connection.execute(
            "SELECT next_event_seq,sent_event_seq,acknowledged_event_seq "
            "FROM mobile_device_cursors"
        ).fetchone()
        envelope = json.loads(
            connection.execute(
                "SELECT envelope_json FROM mobile_device_inbox"
            ).fetchone()[0]
        )
    assert cursor == (2, 0, 0)
    assert envelope["type"] == "sync.reset_required"
    backups = workspace / "backups" / migration._MIGRATION
    assert len(list(backups.glob("*/sessions/sessions.db"))) == 1
    assert len(list(backups.glob("*/gateway/mobile_realtime.db"))) == 1

    with bind_migration_context(config_path=config, workspace=workspace):
        migration.migrate_legacy_mobile_client_ids(object())
    assert len(list(backups.glob("*/sessions/sessions.db"))) == 1


def test_refuses_live_gateway_reference_before_backup_or_write(tmp_path: Path) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = workspace / "sessions.db"
    gateway = workspace / "data/mobile_realtime.db"
    _sessions(sessions)
    _gateway(gateway, referenced=True)
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    with (
        bind_migration_context(config_path=config, workspace=workspace),
        pytest.raises(RuntimeError, match="仍被 mobile_command_receipts 引用"),
    ):
        migration.migrate_legacy_mobile_client_ids(object())

    assert not (workspace / "backups" / migration._MIGRATION).exists()
    with closing(sqlite3.connect(sessions)) as connection:
        extra = json.loads(
            connection.execute("SELECT extra FROM messages").fetchone()[0]
        )
    assert extra["client_message_id"] == _OLD_ID


def test_rejects_unknown_client_id_format_before_backup(tmp_path: Path) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _sessions(workspace / "sessions.db", client_id="legacy-command")
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    with (
        bind_migration_context(config_path=config, workspace=workspace),
        pytest.raises(RuntimeError, match="不是 UUIDv4、UUIDv7 或 ULID"),
    ):
        migration.migrate_legacy_mobile_client_ids(object())
    assert not (workspace / "backups" / migration._MIGRATION).exists()
