import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from agent.migrations.context import bind_migration_context
from session.log import MessageLog
from session.message import Input, Output, ToolCall
from session.store import SessionStore


@pytest.fixture
def migration(monkeypatch):
    import yoyo

    monkeypatch.setattr(yoyo, "step", lambda callback: callback)
    path = Path(__file__).parents[1] / "migrations/yoyo/20260905_01_message_log.py"
    spec = importlib.util.spec_from_file_location("message_log_migration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def old_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "sessions.db"
    store = SessionStore(path)
    store.close()
    stamp = "2026-09-05T10:00:00+08:00"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO sessions (key,created_at,updated_at,metadata,next_seq) VALUES (?,?,?,?,?)",
            ("s", stamp, stamp, '{"value": "保留"}', 100),
        )
        connection.executemany(
            "INSERT INTO messages (id,session_key,seq,role,content,tool_chain,extra,ts) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    "user",
                    "s",
                    4,
                    "user",
                    "不要改这段正文\n",
                    None,
                    '{ "client_message_id": "client", "media": ["old.png"] }',
                    stamp,
                ),
                (
                    "reply",
                    "s",
                    8,
                    "assistant",
                    "",
                    '[ { "calls": [{"name": "shell", "arguments": {}, "result": "old result"}], "text": "reasoning" } ]',
                    '{"proactive": true}',
                    stamp,
                ),
                ("nullable", "s", 9, "assistant", None, None, None, stamp),
            ],
        )
        connection.execute(
            "INSERT INTO turns (id,session_key,status,input_json,items_json,created_at) VALUES (?,?,?,?,?,?)",
            (
                "open",
                "s",
                "interrupted",
                '{"input":"尚未落消息", "metadata":{}}',
                "[]",
                stamp,
            ),
        )
        # 用真实外键表验证 DROP/rename 不级联丢失原 message 引用。
        connection.execute(
            "CREATE TABLE fixture_refs (message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE)"
        )
        connection.execute("INSERT INTO fixture_refs VALUES ('reply')")
    return workspace


def run(migration, workspace):
    with bind_migration_context(
        workspace=workspace, config_path=workspace / "config.toml"
    ):
        migration.migrate_message_log(None)


def snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        return "\n".join(connection.iterdump())


def test_upgrade_preserves_raw_facts_references_sequences_and_backup(
    migration, old_workspace
):
    path = old_workspace / "sessions.db"
    before = snapshot(path)
    run(migration, old_workspace)
    backups = list((old_workspace / "backups/message-log-v1").glob("*/sessions.db"))
    assert len(backups) == 1
    assert snapshot(backups[0]) == before
    manifest = json.loads(backups[0].with_name("manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(backups[0].read_bytes()).hexdigest()
    log = MessageLog(path)
    try:
        messages = log.reader("s").read()
        assert [(m.message_id, m.seq) for m in messages] == [
            ("user", 4),
            ("reply", 8),
            ("nullable", 9),
        ]
        assert isinstance(messages[0].body, Input)
        assert messages[0].body.parts[0].value == "不要改这段正文\n"
        assert (
            messages[0].body.parts[1].value["extra"]
            == '{ "client_message_id": "client", "media": ["old.png"] }'
        )
        assert all(m.source == "legacy-unattributed" for m in messages)
        assert isinstance(messages[1].body, Output)
        assert not any(isinstance(part, ToolCall) for part in messages[1].body.parts)
        transcript = messages[1].body.parts[2].value
        assert (
            hashlib.sha256(transcript["raw"].encode()).hexdigest()
            == transcript["sha256"]
        )
        assert transcript["completeness"] == "unknown"
        assert messages[2].body.parts[0].value["content_was_null"] is True
        writer = log.writer(
            "s", author="user", source="conversation", body_types=(Input,), content={}
        )
        assert writer.append("new", Input(())).seq == 100
    finally:
        log.close()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT message_id FROM fixture_refs").fetchall() == [
            ("reply",)
        ]
        assert (
            connection.execute("SELECT input_json FROM turns").fetchone()[0]
            == '{"input":"尚未落消息", "metadata":{}}'
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    # 模拟转换已提交但 yoyo ledger ACK 丢失，后续新写入不使原 receipt 失效。
    after = snapshot(path)
    run(migration, old_workspace)
    assert snapshot(path) == after
    assert (
        len(list((old_workspace / "backups/message-log-v1").glob("*/sessions.db"))) == 1
    )


@pytest.mark.parametrize(
    "sql",
    [
        "ALTER TABLE messages ADD COLUMN mystery TEXT",
        "CREATE INDEX unowned ON messages(role)",
        "DROP TRIGGER messages_ai; CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN DELETE FROM fixture_refs; END",
        "DROP TRIGGER messages_ai; DROP TRIGGER messages_ad; DROP TRIGGER messages_au; DROP TABLE messages_fts; CREATE TABLE messages_fts (content TEXT)",
        "UPDATE sessions SET next_seq=8",
        "DELETE FROM sessions",
        "CREATE VIRTUAL TABLE custom_fts USING fts5(content, content='messages', content_rowid='rowid')",
    ],
)
def test_unknown_schema_or_broken_identity_stops_before_backup_or_write(
    migration, old_workspace, sql
):
    path = old_workspace / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(sql)
    before = snapshot(path)
    with pytest.raises(RuntimeError):
        run(migration, old_workspace)
    assert snapshot(path) == before
    assert not (old_workspace / "backups").exists()


def test_failure_after_table_replacement_rolls_back_and_retry_is_complete(
    migration, old_workspace, monkeypatch
):
    path = old_workspace / "sessions.db"
    before = snapshot(path)
    check = migration._check

    def fail_after_replace(connection):
        if "body" in {
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        }:
            raise RuntimeError("fixture crash before commit")
        check(connection)

    monkeypatch.setattr(migration, "_check", fail_after_replace)
    with pytest.raises(RuntimeError, match="fixture crash"):
        run(migration, old_workspace)
    assert snapshot(path) == before
    monkeypatch.setattr(migration, "_check", check)
    run(migration, old_workspace)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3


def test_committed_receipt_detects_changed_old_message(migration, old_workspace):
    path = old_workspace / "sessions.db"
    run(migration, old_workspace)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE messages SET author='changed' WHERE id='user'")
    before = snapshot(path)
    with pytest.raises(RuntimeError, match="原消息发生变化"):
        run(migration, old_workspace)
    assert snapshot(path) == before


def test_fresh_workspace_does_not_create_legacy_database(migration, tmp_path):
    run(migration, tmp_path)
    assert not (tmp_path / "sessions.db").exists()


@pytest.mark.parametrize("suffix", ["CHECK(seq >= 0)", "WITHOUT ROWID"])
def test_same_columns_with_unregistered_constraint_are_rejected_before_backup(
    migration, old_workspace, suffix
):
    path = old_workspace / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        rows = connection.execute("SELECT * FROM messages").fetchall()
        sql = (
            connection.execute("SELECT sql FROM sqlite_master WHERE name='messages'")
            .fetchone()[0]
            .rstrip()
        )
        sql = (
            sql[:-1] + ", CHECK(seq >= 0))"
            if suffix.startswith("CHECK")
            else sql + " WITHOUT ROWID"
        )
        connection.execute("DROP TABLE messages_fts")
        connection.execute("DROP TABLE messages")
        connection.execute(sql)
        connection.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", rows)
    before = snapshot(path)
    with pytest.raises(RuntimeError, match="未知"):
        run(migration, old_workspace)
    assert snapshot(path) == before
    assert not (old_workspace / "backups").exists()


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE sessions SET next_seq='bad'",
        "UPDATE sessions SET next_seq=-1",
        "UPDATE messages SET extra='{\"attachment_ids\":[\"absent\"]}' WHERE id='user'",
    ],
)
def test_invalid_values_stop_before_backup(migration, old_workspace, sql):
    path = old_workspace / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(sql)
    before = snapshot(path)
    with pytest.raises((ValueError, RuntimeError)):
        run(migration, old_workspace)
    assert snapshot(path) == before
    assert not (old_workspace / "backups").exists()


def test_foreign_key_to_retired_column_stops_before_backup(migration, old_workspace):
    path = old_workspace / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "CREATE TABLE retired_ref (content TEXT REFERENCES MESSAGES(content))"
        )
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="退休"):
        run(migration, old_workspace)
    assert path.read_bytes() == before
    assert not (old_workspace / "backups").exists()


def test_attachment_and_binary_reference_owners_keep_exact_facts(
    migration, old_workspace
):
    path = old_workspace / "sessions.db"
    artifact = old_workspace / "fixture-image.png"
    payload = b"fixture artifact bytes"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "image",
                artifact.name,
                "file",
                "image.png",
                "application/octet-stream",
                len(payload),
                digest,
                "ready",
                "2026-09-05T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO message_attachments VALUES ('reply',0,'image','outbound')"
        )
        connection.execute(
            "UPDATE messages SET extra='{ \"attachment_ids\": [\"image\"] }' WHERE id='reply'"
        )
        connection.execute(
            "CREATE TABLE vector_fixture (message_id TEXT PRIMARY KEY REFERENCES MESSAGES(id), vector BLOB NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO vector_fixture VALUES ('reply', ?)", (b"\x00\xff",)
        )
        rowids = connection.execute(
            "SELECT id,rowid FROM messages ORDER BY id"
        ).fetchall()
    run(migration, old_workspace)
    assert artifact.read_bytes() == payload
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT * FROM message_attachments").fetchall() == [
            ("reply", 0, "image", "outbound")
        ]
        assert (
            connection.execute("SELECT vector FROM vector_fixture").fetchone()[0]
            == b"\x00\xff"
        )
        assert (
            connection.execute("SELECT id,rowid FROM messages ORDER BY id").fetchall()
            == rowids
        )


def test_process_exit_during_ddl_leaves_complete_old_database(migration, old_workspace):
    path = old_workspace / "sessions.db"
    before = snapshot(path)
    source = """
import importlib.util, os, pathlib, sys, yoyo
from agent.migrations.context import bind_migration_context
yoyo.step = lambda callback: callback
spec = importlib.util.spec_from_file_location('crash_fixture', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
check = module._check
def crash(connection):
    if 'body' in {row[1] for row in connection.execute('PRAGMA table_info(messages)')}:
        os._exit(77)
    check(connection)
module._check = crash
workspace = pathlib.Path(sys.argv[2])
with bind_migration_context(workspace=workspace, config_path=workspace/'config.toml'):
    module.migrate_message_log(None)
"""
    result = subprocess.run(
        [sys.executable, "-c", source, migration.__file__, str(old_workspace)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 77, result.stderr
    assert snapshot(path) == before
    run(migration, old_workspace)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
