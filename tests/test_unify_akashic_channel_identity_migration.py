import importlib.util
import json
import hashlib
import shutil
import sqlite3
import struct
import subprocess
import sys
import tomllib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yoyo

from session.store import SessionStore
from infra.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage
from memory2.embedder import Embedder
from agent.migrations.runner import MigrationRunner
from agent.migrations.context import bind_migration_context
from plugins.akasha.infrastructure.loader import load_turns

_PROJECT_ROOT = Path(__file__).parents[1]
_MIGRATION_PATH = (
    _PROJECT_ROOT / "migrations/yoyo/20260826_03_unify_akashic_channel_identity.py"
)


def _load_migration():
    """Load the migration callback without wrapping it in Yoyo."""

    spec = importlib.util.spec_from_file_location(
        "unify_akashic_channel_identity_under_test",
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


def _create_session_database(path: Path) -> None:
    store = SessionStore(path)
    store.close()
    old_session = "web:family"
    old_message = "web:family:0"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions(key, created_at, updated_at, last_consolidated, "
            "metadata, next_seq) VALUES (?, ?, ?, ?, ?, ?)",
            (
                old_session,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                1,
                "{}",
                2,
            ),
        )
        connection.execute(
            "INSERT INTO messages(id, session_key, seq, role, content, extra, ts) "
            "VALUES (?, ?, 0, 'user', ?, ?, ?)",
            (
                old_message,
                old_session,
                "hello",
                json.dumps(
                    {
                        "session_id": old_session,
                        "reply_to_message_id": old_message,
                        "akasha_reinforce": {
                            "target_message_ids": [old_message],
                            "target_turn_ids": [f"{old_message}::{old_message}"],
                        },
                        "cited_memory_ids": [
                            old_message,
                            f"{old_message}::{old_message}",
                            "opaque-memory-id",
                        ],
                        "note": old_message,
                    }
                ),
                "2026-01-01",
            ),
        )
        connection.execute(
            "INSERT INTO turns(id, session_key, status, input_json, items_json, "
            "created_at) VALUES (?, ?, 'completed', ?, ?, ?)",
            (
                "turn-1",
                old_session,
                json.dumps(
                    {
                        "session_id": old_session,
                        "message_id": old_message,
                        "metadata": {
                            "channel": "web",
                            "chatId": old_session,
                            "busySessionId": old_session,
                            "session_key_override": old_session,
                        },
                    }
                ),
                json.dumps(
                    [
                        {
                            "persisted_user_message_id": old_message,
                            "sessionMessageId": old_message,
                        }
                    ]
                ),
                "2026-01-01",
            ),
        )
        connection.execute(
            "INSERT INTO attachments(artifact_id, storage_key, kind, size_bytes, "
            "sha256, state, created_at) VALUES "
            "('artifact-1', 'one', 'file', 1, ?, 'ready', '2026-01-01')",
            ("a" * 64,),
        )
        connection.execute(
            "INSERT INTO message_attachments(message_id, ordinal, artifact_id, "
            "direction) VALUES (?, 0, 'artifact-1', 'inbound')",
            (old_message,),
        )
        connection.execute(
            "CREATE TABLE message_embeddings(message_id TEXT PRIMARY KEY, value TEXT)"
        )
        connection.execute(
            "INSERT INTO message_embeddings(message_id, value) VALUES (?, 'vector')",
            (old_message,),
        )
        connection.execute(
            "INSERT INTO session_compactions("
            "session_key, generation, parent_generation, created_at, trigger, "
            "summary_format_version, summary, source_ref, source_plan_digest, "
            "source_from_seq, consolidated_through_seq, source_message_ids_json, "
            "retained_tail_json, model_runtime_id, model, context_window, "
            "threshold_tokens, hard_input_tokens, keep_recent_tokens, tokens_before, "
            "tokens_after, summary_usage_json) VALUES "
            "(?, 1, 0, '2026-01-01', 'threshold', 1, 'summary', 'old-source', ?, "
            "0, 0, ?, ?, 'runtime', 'model', 100, 80, 90, 20, 50, 10, '{}')",
            (
                old_session,
                "b" * 64,
                json.dumps([old_message]),
                json.dumps(
                    [
                        {
                            "id": old_message,
                            "message": {
                                "role": "user",
                                "content": old_message,
                                "reply_to_message_id": old_message,
                            },
                        }
                    ]
                ),
            ),
        )
        connection.execute(
            "INSERT INTO session_source_mutation_audits("
            "audit_id, operation, session_key, message_ids_json, action_source, "
            "completed_at) VALUES ('source-audit', 'test', ?, ?, 'test', '2026-01-01')",
            (old_session, json.dumps([old_message])),
        )
        connection.execute(
            "INSERT INTO session_delete_audits("
            "audit_id, targets_json, message_ids_json, compactions_json, "
            "action_source, cascade, started_at, completed_at, result, deleted_count) "
            "VALUES ('delete-audit', ?, ?, ?, 'test', 0, '2026-01-01', "
            "'2026-01-01', 'ok', 0)",
            (
                json.dumps([old_session]),
                json.dumps([old_message]),
                json.dumps([{"session_key": old_session, "message_id": old_message}]),
            ),
        )
        connection.execute(
            "INSERT INTO channel_identities(channel, identity, chat_id, updated_at) "
            "VALUES ('web', 'browser', 'family', '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO channel_identity_migrations(channel, migrated_at) "
            "VALUES ('web', '2026-01-01')"
        )
        connection.execute(
            "INSERT INTO sessions(key, created_at, updated_at, last_consolidated, "
            "metadata, next_seq) VALUES "
            "('telegram:owner', '2026-01-01', '2026-01-01', 0, '{}', 1)"
        )
        connection.execute(
            "INSERT INTO messages(id, session_key, seq, role, content, extra, ts) "
            "VALUES ('telegram:owner:0', 'telegram:owner', 0, 'assistant', "
            "'cross reference', ?, '2026-01-01')",
            (
                json.dumps(
                    {
                        "cited_memory_ids": [
                            old_message,
                            f"{old_message}::{old_message}",
                        ]
                    }
                ),
            ),
        )
        connection.execute(
            "INSERT INTO turns(id, session_key, status, input_json, items_json, "
            "created_at) VALUES ('turn-cross', 'telegram:owner', 'completed', ?, ?, "
            "'2026-01-01')",
            (
                json.dumps({"metadata": {"busySessionId": old_session}}),
                json.dumps(
                    [
                        {
                            "persisted_user_message_ids": [f"{old_session}:1"],
                            "sessionMessageId": f"{old_session}:1",
                            "source_ref": f"{old_session}:1",
                            "source_refs": [f"{old_session}:1", "opaque-source"],
                        }
                    ]
                ),
            ),
        )


def _create_public_runner_fixture(path: Path) -> None:
    """Create two old client sessions with complete frozen embeddings."""

    store = SessionStore(path)
    store.close()
    now = "2026-08-26T00:00:00+00:00"
    rows = (
        ("web:family", "Web remembered this", "Web answer"),
        ("mobile:family", "Mobile remembered this", "Mobile answer"),
    )
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE message_embeddings ("
            "message_id TEXT NOT NULL, content_hash TEXT NOT NULL, model TEXT NOT NULL, "
            "embedding BLOB NOT NULL, dim INTEGER NOT NULL, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, PRIMARY KEY(message_id, model))"
        )
        for session_key, user, assistant in rows:
            connection.execute(
                "INSERT INTO sessions(key, created_at, updated_at, last_consolidated, "
                "metadata, next_seq) VALUES (?, ?, ?, 0, '{}', 2)",
                (session_key, now, now),
            )
            for seq, (role, content) in enumerate(
                (("user", user), ("assistant", assistant))
            ):
                message_id = f"{session_key}:{seq}"
                connection.execute(
                    "INSERT INTO messages(id, session_key, seq, role, content, extra, ts) "
                    "VALUES (?, ?, ?, ?, ?, NULL, ?)",
                    (message_id, session_key, seq, role, content, now),
                )
                if message_id != "mobile:family:1":
                    connection.execute(
                        "INSERT INTO message_embeddings VALUES (?, ?, "
                        "'embedding-model', ?, 2, ?, ?)",
                        (
                            message_id,
                            hashlib.sha256(content.encode()).hexdigest(),
                            sqlite3.Binary(struct.pack("<2f", 1.0, float(seq))),
                            now,
                            now,
                        ),
                    )


def test_public_runner_rekeys_both_clients_and_rebuilds_queryable_akasha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the public runner, real Akasha rebuild, and idempotent replay."""

    async def embed_missing(
        _embedder: Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        assert texts == ["Mobile answer"]
        return [[1.0, 1.0]]

    monkeypatch.setattr(Embedder, "embed_batch", embed_missing)

    root = tmp_path / "installation"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    config = root / "config.toml"
    config.write_text(
        """
[llm]
main = "test_main"

[llm.runtimes.test_main]
provider = "openai"
model = "chat-model"
api_key = "chat-key"
base_url = "https://chat.invalid/v1"
input_modalities = ["text"]

[memory]
enabled = true

[memory.embedding]
model = "embedding-model"
api_key = "embedding-key"
base_url = "https://embedding.invalid/v1"
output_dimensionality = 2

[channels.chat]
channel_name = "web"
""".strip() + "\n",
        encoding="utf-8",
    )
    sessions = workspace / "sessions.db"
    _create_public_runner_fixture(sessions)
    index = workspace / "memory" / "akasha-v2-index.db"
    index.parent.mkdir(parents=True)
    with closing(sqlite3.connect(index)) as connection, connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO metadata VALUES('index_version', '10')")

    # Keep the public runner while isolating this cutover and its declared parents.
    repo = root / "repo"
    catalog = repo / "migrations" / "yoyo"
    catalog.mkdir(parents=True)
    (catalog / "20260826_01_migrate_turn_effects.py").write_text(
        "from yoyo import step\nsteps = [step(lambda connection: None)]\n",
        encoding="utf-8",
    )
    (catalog / "20260826_02_backfill_akasha_message_embeddings.py").write_text(
        "from yoyo import step\n"
        "__depends__ = {'20260826_01_migrate_turn_effects'}\n"
        "steps = [step(lambda connection: None)]\n",
        encoding="utf-8",
    )
    shutil.copy2(_MIGRATION_PATH, catalog / _MIGRATION_PATH.name)
    runner = MigrationRunner(
        repo_root=repo,
        config_path=config,
        workspace=workspace,
    )

    first = runner.run()

    assert first.state == "migrated"
    turns = load_turns(index)
    assert len(turns) == 2
    assert {turn.user_text for turn in turns} == {
        "Web remembered this",
        "Mobile remembered this",
    }
    assert all(turn.session_key.startswith("akashic:") for turn in turns)
    assert not any(turn.session_key.startswith(("web:", "mobile:")) for turn in turns)
    with closing(sqlite3.connect(sessions)) as connection:
        migrated_sessions = {
            row[0] for row in connection.execute("SELECT key FROM sessions")
        }
    assert migrated_sessions == {turn.session_key for turn in turns}
    assert "channel_name" not in config.read_text(encoding="utf-8")

    second = runner.run()
    assert second.state == "current"
    assert second.migrations == ()
    assert [turn.turn_id for turn in load_turns(index)] == [
        turn.turn_id for turn in turns
    ]


def test_empty_workspace_does_not_require_akasha_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not load Akasha when there is no Session graph to rebuild."""

    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    SessionStore(workspace / "sessions.db").close()
    config = tmp_path / "config.toml"
    config.write_text("[channels.chat]\nchannel_name = 'web'\n", encoding="utf-8")
    monkeypatch.setattr(
        migration,
        "_akasha_targets",
        lambda _workspace: pytest.fail("empty workspace must not load Akasha"),
    )

    with bind_migration_context(config_path=config, workspace=workspace):
        migration.unify_akashic_identity(object())

    assert "channel_name" not in config.read_text(encoding="utf-8")


def test_migrates_historical_session_message_and_reference_identity(
    tmp_path: Path,
) -> None:
    migration = _load_migration()
    database = tmp_path / "sessions.db"
    _create_session_database(database)

    session_map, message_map = migration._migrate_sessions(
        database,
        old_channels=frozenset({"web", "mobile"}),
    )

    new_session = session_map["web:family"]
    new_message = message_map["web:family:0"]
    assert new_session.startswith("akashic:")
    assert new_message == f"{new_session}:0"
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        session = connection.execute(
            "SELECT * FROM sessions WHERE key = ?", (new_session,)
        ).fetchone()
        message = connection.execute(
            "SELECT * FROM messages WHERE id = ?", (new_message,)
        ).fetchone()
        assert session["key"] == new_session
        assert session["last_consolidated"] == 0
        assert message["id"] == new_message
        assert message["session_key"] == new_session
        extra = json.loads(message["extra"])
        assert extra["session_id"] == new_session
        assert extra["reply_to_message_id"] == new_message
        assert extra["akasha_reinforce"] == {
            "target_message_ids": [new_message],
            "target_turn_ids": [f"{new_message}::{new_message}"],
        }
        assert extra["cited_memory_ids"] == [
            new_message,
            f"{new_message}::{new_message}",
            "opaque-memory-id",
        ]
        assert extra["note"] == "web:family:0"
        cross_reference = json.loads(
            connection.execute(
                "SELECT extra FROM messages WHERE id = 'telegram:owner:0'"
            ).fetchone()[0]
        )
        assert cross_reference["cited_memory_ids"] == [
            new_message,
            f"{new_message}::{new_message}",
        ]
        assert (
            connection.execute("SELECT message_id FROM message_attachments").fetchone()[
                0
            ]
            == new_message
        )
        assert (
            connection.execute("SELECT message_id FROM message_embeddings").fetchone()[
                0
            ]
            == new_message
        )
        turn = connection.execute("SELECT * FROM turns").fetchone()
        assert turn["session_key"] == new_session
        turn_input = json.loads(turn["input_json"])
        assert turn_input["message_id"] == new_message
        assert turn_input["metadata"] == {
            "channel": "akashic",
            "chatId": new_session,
            "busySessionId": new_session,
            "session_key_override": new_session,
        }
        turn_items = json.loads(turn["items_json"])
        assert turn_items == [
            {
                "persisted_user_message_id": new_message,
                "sessionMessageId": new_message,
            }
        ]
        cross_turn = connection.execute(
            "SELECT * FROM turns WHERE id = 'turn-cross'"
        ).fetchone()
        assert cross_turn["session_key"] == "telegram:owner"
        assert json.loads(cross_turn["input_json"]) == {
            "metadata": {"busySessionId": new_session}
        }
        assert json.loads(cross_turn["items_json"]) == [
            {
                "persisted_user_message_ids": [f"{new_session}:1"],
                "sessionMessageId": f"{new_session}:1",
                "source_ref": f"{new_session}:1",
                "source_refs": [f"{new_session}:1", "opaque-source"],
            }
        ]
        compaction = connection.execute("SELECT * FROM session_compactions").fetchone()
        assert compaction["session_key"] == new_session
        assert compaction["invalidated_reason"] == "akashic_identity_rekey"
        retained = json.loads(compaction["retained_tail_json"])[0]
        assert retained["id"] == new_message
        assert retained["message"]["reply_to_message_id"] == new_message
        assert retained["message"]["content"] == "web:family:0"
        identity = connection.execute(
            "SELECT channel, identity, chat_id FROM channel_identities"
        ).fetchone()
        assert tuple(identity) == (
            "akashic",
            new_session.removeprefix("akashic:"),
            new_session.removeprefix("akashic:"),
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_fails_before_write_when_a_turn_is_active(tmp_path: Path) -> None:
    migration = _load_migration()
    database = tmp_path / "sessions.db"
    _create_session_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("UPDATE turns SET status = 'running'")

    with pytest.raises(RuntimeError, match="未终态 Turn"):
        migration._migrate_sessions(
            database,
            old_channels=frozenset({"web", "mobile"}),
        )

    with closing(sqlite3.connect(database)) as connection:
        assert (
            connection.execute(
                "SELECT key FROM sessions WHERE key = 'web:family'"
            ).fetchone()[0]
            == "web:family"
        )


def test_migrates_legacy_sessions_schema_without_next_seq(tmp_path: Path) -> None:
    migration = _load_migration()
    database = tmp_path / "sessions.db"
    _create_session_database(database)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("ALTER TABLE sessions DROP COLUMN next_seq")

    session_map, message_map = migration._migrate_sessions(
        database,
        old_channels=frozenset({"web", "mobile"}),
    )

    assert session_map["web:family"].startswith("akashic:")
    assert message_map["web:family:0"] == f"{session_map['web:family']}:0"


def test_restore_absent_file_allows_absent_parent(tmp_path: Path) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    backup_root = workspace / "backups" / "run"
    backup_root.mkdir(parents=True)
    target = workspace / "plugin-data" / "wake-builtin" / "config.local.toml"

    migration._restore_targets(
        [
            {
                "target": str(target),
                "kind": "file",
                "existed": False,
            }
        ],
        workspace=workspace,
        config_path=config,
        backup_root=backup_root,
    )

    assert not target.exists()


def test_recovers_every_recorded_target_after_an_interrupted_run(
    tmp_path: Path,
) -> None:
    migration = _load_migration()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "config.toml"
    config.write_text("[channels.chat]\nchannel_name = 'web'\n", encoding="utf-8")
    database = workspace / "sessions.db"
    _create_session_database(database)
    backup_root = workspace / "backups" / "run"
    backup_root.mkdir(parents=True)
    records = [
        migration._backup_target(
            database,
            backup_root=backup_root,
            name="sessions",
            kind="sqlite",
        ),
        migration._backup_target(
            config,
            backup_root=backup_root,
            name="config",
            kind="file",
        ),
    ]
    marker = workspace / "backups" / migration._MIGRATION / "in-progress.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps({"backup_root": str(backup_root), "targets": records}),
        encoding="utf-8",
    )
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("DELETE FROM messages")
    config.write_text("[channels.chat]\n", encoding="utf-8")

    migration._recover_incomplete_migration(
        marker,
        workspace=workspace,
        config_path=config,
    )

    assert not marker.exists()
    assert "channel_name" in config.read_text(encoding="utf-8")
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_restore_discards_wal_frames_left_by_a_hard_crash(tmp_path: Path) -> None:
    migration = _load_migration()
    database = tmp_path / "state.db"
    backup = tmp_path / "state.backup.db"
    script = """
import os
import sqlite3
import sys

database, backup = sys.argv[1:]
connection = sqlite3.connect(database)
connection.execute("PRAGMA journal_mode=WAL")
connection.execute("CREATE TABLE facts(value TEXT)")
connection.execute("INSERT INTO facts VALUES ('old')")
connection.commit()
connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
with sqlite3.connect(backup) as destination:
    connection.backup(destination)
connection.execute("INSERT INTO facts VALUES ('half-migrated')")
connection.commit()
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(database), str(backup)],
        check=True,
    )
    assert Path(f"{database}-wal").is_file()

    migration._restore_sqlite(backup, database)

    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT value FROM facts").fetchall() == [("old",)]


def test_gateway_discards_old_inbox_and_appends_one_reset_boundary(
    tmp_path: Path,
) -> None:
    migration = _load_migration()
    database = tmp_path / "mobile.db"
    storage = MobileRealtimeStorage(database)
    storage.register_device(
        DeviceRecord(
            device_id="device-1",
            public_key="public-key",
            display_name="Phone",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            revoked_at=None,
            capabilities=("chat",),
        )
    )
    storage.close()
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "INSERT INTO mobile_device_inbox("
            "device_id, event_seq, event_id, priority, envelope_json, created_at"
            ") VALUES ('device-1', 1, 'old-event', 'P0', ?, '2026-01-01')",
            (json.dumps({"type": "message.final", "session_id": "mobile:old"}),),
        )
        connection.execute(
            "UPDATE mobile_device_cursors SET next_event_seq = 2 "
            "WHERE device_id = 'device-1'"
        )

    migration._migrate_gateway(
        database,
        session_map={},
        message_map={},
        message_effects={},
    )

    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        event = connection.execute("SELECT * FROM mobile_device_inbox").fetchone()
        assert event["event_seq"] == 2
        assert json.loads(event["envelope_json"])["type"] == "sync.reset_required"
        assert json.loads(event["envelope_json"])["payload"] == {
            "reason": "akashic_identity_rekey"
        }
        cursor = connection.execute(
            "SELECT * FROM mobile_device_cursors WHERE device_id = 'device-1'"
        ).fetchone()
        assert cursor["next_event_seq"] == 3
        assert cursor["acknowledged_event_seq"] == 0


def test_gateway_settles_old_processing_receipts_without_replaying(
    tmp_path: Path,
) -> None:
    migration = _load_migration()
    database = tmp_path / "mobile.db"
    storage = MobileRealtimeStorage(database)
    storage.register_device(
        DeviceRecord(
            device_id="device-1",
            public_key="public-key",
            display_name="Phone",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            revoked_at=None,
            capabilities=("chat",),
        )
    )
    storage.close()
    with closing(sqlite3.connect(database)) as connection, connection:
        for command_id, command_type in (
            ("query-1", "session.list"),
            ("query-2", "plugin.ui.call"),
            ("sent-1", "message.send"),
            ("not-sent-1", "message.send"),
        ):
            connection.execute(
                "INSERT INTO mobile_command_receipts("
                "device_id, command_id, command_type, request_hash, status, created_at"
                ") VALUES ('device-1', ?, ?, 'hash', 'processing', '2026-01-01')",
                (command_id, command_type),
            )

    migration._migrate_gateway(
        database,
        session_map={},
        message_map={},
        message_effects={"sent-1": "akashic:session-1"},
    )

    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM mobile_command_receipts ORDER BY command_id"
        ).fetchall()
        assert [row["command_id"] for row in rows] == ["not-sent-1", "sent-1"]
        interrupted, accepted = rows
        assert interrupted["status"] == "completed"
        assert interrupted["reply_type"] == "message.send.error"
        assert json.loads(interrupted["reply_payload_json"])["code"] == (
            "command_interrupted"
        )
        assert interrupted["session_id"] is None
        assert accepted["status"] == "completed"
        assert accepted["reply_type"] == "message.send.ok"
        assert json.loads(accepted["reply_payload_json"]) == {
            "accepted": True,
            "client_message_id": "sent-1",
        }
        assert accepted["session_id"] == "akashic:session-1"


def test_gateway_fails_loud_for_unknown_processing_command(tmp_path: Path) -> None:
    migration = _load_migration()
    database = tmp_path / "mobile.db"
    storage = MobileRealtimeStorage(database)
    storage.register_device(
        DeviceRecord(
            device_id="device-1",
            public_key="public-key",
            display_name="Phone",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            revoked_at=None,
            capabilities=("chat",),
        )
    )
    storage.close()
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "INSERT INTO mobile_command_receipts("
            "device_id, command_id, command_type, request_hash, status, created_at"
            ") VALUES ('device-1', 'unknown-1', 'device.update', 'hash', "
            "'processing', '2026-01-01')"
        )

    with pytest.raises(RuntimeError, match="无法收束"):
        migration._migrate_gateway(
            database,
            session_map={},
            message_map={},
            message_effects={},
        )

    with closing(sqlite3.connect(database)) as connection:
        assert (
            connection.execute("SELECT status FROM mobile_command_receipts").fetchone()[
                0
            ]
            == "processing"
        )


def test_rekeys_plugin_state_and_external_wake_context(tmp_path: Path) -> None:
    migration = _load_migration()
    old_session = "web:family"
    new_session = migration._new_session_key(old_session)
    content = tmp_path / "content.sqlite3"
    with closing(sqlite3.connect(content)) as connection, connection:
        connection.executescript(
            "CREATE TABLE items(selected_session_id TEXT);"
            "CREATE TABLE content_selections(accepted_session_id TEXT);"
        )
        connection.execute("INSERT INTO items VALUES (?)", (old_session,))
        connection.execute("INSERT INTO content_selections VALUES (?)", (old_session,))
    migration._migrate_plugin_state(
        content,
        columns=(
            ("items", "selected_session_id"),
            ("content_selections", "accepted_session_id"),
        ),
        session_map={old_session: new_session},
        old_channels=frozenset({"web", "mobile"}),
    )
    with closing(sqlite3.connect(content)) as connection:
        assert (
            connection.execute("SELECT selected_session_id FROM items").fetchone()[0]
            == new_session
        )
        assert (
            connection.execute(
                "SELECT accepted_session_id FROM content_selections"
            ).fetchone()[0]
            == new_session
        )

    wake = tmp_path / "wake.toml"
    wake.write_text(
        "[delivery]\nchannel = 'telegram'\nrecipient = 'owner'\n"
        "session_id = 'web:family'\n",
        encoding="utf-8",
    )
    migration._migrate_wake_config(
        wake,
        old_channels=frozenset({"web", "mobile"}),
        session_map={old_session: new_session},
    )
    migrated_wake = tomllib.loads(wake.read_text(encoding="utf-8"))["delivery"]
    assert migrated_wake["session_id"] == new_session
    assert migrated_wake["channel"] == "telegram"
