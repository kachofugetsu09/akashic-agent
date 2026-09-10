"""消息附加信息的提交、恢复、同步与迁移合同。"""
import sqlite3
from contextlib import aclosing, closing
from pathlib import Path
from typing import Any, cast

import pytest
from yoyo import get_backend, read_migrations

from agent.migrations.context import bind_migration_context
from infra.channels.message_view import follow_messages, message_rows
from session.log import MessageConflict, MessageLog
from session.message import ContentPart, MAX_METADATA_BYTES, Output
from agent.plugin_contracts.content import check_text


def writer(log, namespaces=frozenset({"citation", "meme"})):
    return log.writer("s", author="assistant", source="conversation", body_types=(Output,),
                      content={"text": check_text}, message_metadata_keys=namespaces)


def test_metadata_is_immutable_and_part_of_atomic_replay(tmp_path):
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        output = writer(log)
        body = Output((ContentPart("text", "answer"),), "complete")
        metadata = {"citation": {"references": [{"id": "old"}]}}
        message = output.append("reply", body, metadata=metadata, expected_source_head=-1)
        metadata["citation"]["references"][0]["id"] = "changed"
        assert cast(Any, message.metadata["citation"])["references"][0]["id"] == "old"
        with pytest.raises(TypeError):
            cast(Any, message.metadata["citation"])["references"][0]["id"] = "changed"
        assert output.append("reply", body, metadata=message.metadata, expected_source_head=-1) == message
        with pytest.raises(MessageConflict, match="metadata"):
            output.append("reply", body, metadata=metadata)
        with pytest.raises(MessageConflict, match="metadata"):
            output.append("reply", body)
        with pytest.raises(PermissionError, match="命名空间"):
            output.append("forged", body, metadata={"unrelated": {"x": 1}})

        numeric = output.append("numeric", body, metadata={"meme": {"value": 1}})
        with pytest.raises(MessageConflict, match="metadata"):
            output.append("numeric", body, metadata={"meme": {"value": True}})

        owner = log.owner("consumer")
        def fail(tx):
            tx.append(output, "new", body, metadata={"meme": {"category": "happy"}})
            tx.save("cursor", {"message_id": "new"}, expected_version=None)
            raise ValueError("consumer failed")
        with pytest.raises(ValueError, match="consumer failed"):
            owner.transact(fail)
        assert owner.read("cursor") is None
        assert log.reader("s").snapshot() == (message, numeric)


@pytest.mark.parametrize("metadata", [
    [], {"": {}}, {"citation": float("nan")}, {"citation": {1: "bad key"}},
    {"citation": "界" * (MAX_METADATA_BYTES // 3)},
])
def test_invalid_metadata_does_not_create_a_session_or_consume_a_sequence(tmp_path, metadata):
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        with pytest.raises((ValueError, TypeError)):
            writer(log).append("bad", Output((), "quiet"), metadata=metadata)
        assert log.catalog().sessions().items == ()
        assert writer(log).append("good", Output((), "quiet")).seq == 0


@pytest.mark.asyncio
async def test_unknown_metadata_survives_restart_history_and_follow_without_plugins(tmp_path):
    path = tmp_path / "sessions.db"
    extra = {"future-plugin": {"version": 27, "values": [None, True, 1, "原文", {"x": [2]}]}}
    with closing(MessageLog(path)) as log:
        saved = writer(log, frozenset(extra)).append(
            "reply", Output((ContentPart("text", "still readable"),), "complete"), metadata=extra)
    # 当前没有 Content 注册、插件模块或 schema；读取只依赖已提交的公共 Message。
    with closing(MessageLog(path)) as log:
        reader = log.reader("s")
        assert reader.get("reply") == saved
        assert log.catalog().sessions().items[0].first_message == saved
        page = reader.read_page()
        row = message_rows(page)[0]
        assert row["metadata"] == extra
        assert cast(Any, row["body"])["parts"] == [{"kind": "text", "value": "still readable"}]
        async with aclosing(follow_messages(reader, after_seq=-1)) as follower:
            assert (await anext(follower))["items"] == [row]


def migration_source():
    return Path(__file__).parents[1] / "migrations/yoyo/20260907_03_message_metadata.py"


def load_migration():
    migration = next(item for item in read_migrations(str(migration_source().parent))
                     if item.id == migration_source().stem)
    migration.load()
    return migration.module


def test_yoyo_adds_only_metadata_and_replay_keeps_new_extensions(tmp_path):
    migration = load_migration()
    path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(migration._OLD_SCHEMA)
    # 已发布中间迁移仍能通过原日志接纳无扩展消息，不能提前写入未迁移的字段。
    with closing(MessageLog(path)) as log:
        saved = writer(log).append("old", Output((ContentPart("text", "original"),), "complete"))
        with pytest.raises(RuntimeError, match="yoyo"):
            writer(log).append("too-early", saved.body, metadata={"citation": {}})
        with closing(sqlite3.connect(path)) as connection:
            before = connection.execute("SELECT rowid,* FROM messages ORDER BY rowid").fetchall()
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "20260907_02_retire_legacy_agent_config.py").write_text('from yoyo import step\nsteps = [step("SELECT 1")]\n')
    (directory / migration_source().name).write_bytes(migration_source().read_bytes())
    backend = get_backend(f"sqlite:///{tmp_path / 'ledger.db'}")
    migrations = read_migrations(str(directory))
    with backend, bind_migration_context(workspace=tmp_path, config_path=tmp_path / "config.toml"):
        backend.apply_migrations(backend.to_apply(migrations))
    with closing(sqlite3.connect(path)) as connection:
        after = connection.execute("SELECT rowid,* FROM messages ORDER BY rowid").fetchall()
        assert [row[:-1] for row in after] == before
        assert [row[-1] for row in after] == ["{}"]
        assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    backups = list((tmp_path / "backups/message-metadata").rglob("*.db"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as backup:
        assert backup.execute("SELECT rowid,* FROM messages ORDER BY rowid").fetchall() == before
    with closing(MessageLog(path)) as log:
        new = writer(log).append("new", saved.body, metadata={"meme": {"category": "happy"}})
        assert log.reader("s").get("old") == saved
    migration.migrate(path, tmp_path / "unused-backup")
    with closing(MessageLog(path)) as log:
        assert log.reader("s").get("new") == new
    assert not (tmp_path / "unused-backup").exists()


def test_metadata_migration_rejects_unknown_schema_without_writes(tmp_path):
    migration = load_migration()
    path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(migration._OLD_SCHEMA.replace("body TEXT NOT NULL", "body TEXT"))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="未知 schema"):
        migration.migrate(path, tmp_path / "backup")
    assert path.read_bytes() == before
    assert not (tmp_path / "backup").exists()


@pytest.mark.parametrize("raw", ['{"": {}}', '{"citation": NaN}', '{"citation": 1, "citation": 2}'])
def test_corrupt_metadata_names_the_persisted_message(tmp_path, raw):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        writer(log).append("broken", Output((), "quiet"))
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE messages SET metadata=? WHERE id='broken'", (raw,))
        with pytest.raises(ValueError, match="Session s Message broken metadata"):
            log.reader("s").get("broken")
        with pytest.raises(ValueError, match="Session s Message broken metadata"):
            log.catalog().sessions()
        with pytest.raises(ValueError, match="Session s Message broken metadata"):
            load_migration().migrate(path, tmp_path / "backup")
        assert not (tmp_path / "backup").exists()
