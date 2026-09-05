import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent.migrations.context import bind_migration_context
from session.log import MessageLog
from session.message import Input


@pytest.fixture
def migration(monkeypatch):
    import yoyo

    monkeypatch.setattr(yoyo, "step", lambda callback: callback)
    path = Path(__file__).parents[1] / "migrations/yoyo/20260905_02_owner_records.py"
    spec = importlib.util.spec_from_file_location("owner_records_migration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    writer = log.writer("s", author="user", source="s", body_types=(Input,), content={})
    writer.append("input", Input(()))
    log.close()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DROP TABLE owner_records")
    return tmp_path


def snapshot(path):
    with closing(sqlite3.connect(path)) as connection:
        return tuple(connection.iterdump())


def run(migration, workspace):
    with bind_migration_context(
        workspace=workspace, config_path=workspace / "config.toml"
    ):
        migration.migrate_owner_records(None)


def test_upgrade_and_lost_ledger_ack_preserve_messages_and_owner_state(
    migration, workspace
):
    path = workspace / "sessions.db"
    before = snapshot(path)
    log = MessageLog(path)
    with pytest.raises(RuntimeError, match="yoyo"):
        log.owner("consumer")
    log.close()
    assert snapshot(path) == before
    run(migration, workspace)
    backups = list((workspace / "backups/owner-records-v1").glob("*/sessions.db"))
    assert len(backups) == 1
    assert snapshot(backups[0]) == before
    log = MessageLog(path)
    try:
        assert log.reader("s").read()[0].message_id == "input"
        log.owner("consumer").transact(
            lambda tx: tx.save("cursor", {"message_id": "input"}, expected_version=None)
        )
    finally:
        log.close()
    after = snapshot(path)
    run(migration, workspace)
    assert snapshot(path) == after
    assert (
        list((workspace / "backups/owner-records-v1").glob("*/sessions.db")) == backups
    )


def test_conflicting_schema_is_rejected_without_writing(migration, workspace):
    path = workspace / "sessions.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE owner_records (value TEXT)")
    before = snapshot(path)
    with pytest.raises(RuntimeError, match="schema"):
        run(migration, workspace)
    assert snapshot(path) == before
    assert not (workspace / "backups").exists()


def test_backup_failure_prevents_schema_publication(migration, workspace, monkeypatch):
    path = workspace / "sessions.db"
    before = snapshot(path)

    def fail(*args, **kwargs):
        raise OSError("backup storage unavailable")

    monkeypatch.setattr(migration, "backup_sqlite_database", fail)
    with pytest.raises(OSError, match="backup storage"):
        run(migration, workspace)
    assert snapshot(path) == before
