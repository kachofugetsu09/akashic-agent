"""历史失败只变更已声明状态，消息、身份与恢复备份保持完整。"""
import json
import runpy
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest


def load_migration():
    with patch("yoyo.step", lambda callback: callback):
        return runpy.run_path(str(Path(__file__).parents[1] / "migrations/yoyo/20260909_02_execution_failures.py"))


@pytest.fixture
def migration():
    return load_migration()


def rows(path, table):
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")]


@pytest.mark.parametrize("table", ["model_calls", "mobile_command_receipts"])
def test_failure_migration_preserves_records_and_backup(tmp_path, migration, table):
    """真实旧表覆盖未知终态、活跃行和成功行，再验证幂等与新 CHECK。"""
    path = tmp_path / "records.db"
    mobile = table == "mobile_command_receipts"
    old = migration["_MOBILE_OLD" if mobile else "_MODEL_OLD"]
    new = migration["_MOBILE_NEW" if mobile else "_MODEL_NEW"]
    with closing(sqlite3.connect(path)) as connection, connection:
        if mobile:
            connection.execute("CREATE TABLE mobile_devices(device_id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO mobile_devices VALUES('device')")
        connection.execute(old)
        if mobile:
            for identity, status in (("lost", "outcome_unknown"), ("live", "processing"), ("done", "completed")):
                complete = status == "completed"
                connection.execute(
                    "INSERT INTO mobile_command_receipts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("device", identity, "message.send", "hash:" + identity, status,
                     "message.send.ok" if complete else None, '{"id":"original"}' if complete else None,
                     int(identity == "lost"), "s", "t", "2026-09-09T00:00:00+00:00", "2026-09-09T00:00:01+00:00" if complete else None),
                )
        else:
            for status in ("unknown", "started", "success"):
                connection.execute("INSERT INTO model_calls VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (status, '{"model":"original"}', "digest", status, '{"partial":true}', "original error", "start", "finish", 10, 20))
    before = rows(path, table)
    backups = tmp_path / "backups"
    migration["_migrate"](path, table, old, new, backups)
    backup, = backups.glob("*/records.db")
    assert rows(backup, table) == before
    after = rows(path, table)
    assert len(after) == len(before)
    assert after[1:] == before[1:]
    if mobile:
        changed = {**before[0], "status": "completed", "reply_type": "message.send.error",
                   "reply_payload_json": after[0]["reply_payload_json"], "completed_at": after[0]["completed_at"]}
        assert after[0] == changed and after[0]["completed_at"]
        assert json.loads(after[0]["reply_payload_json"])["code"] == "command_interrupted"
    else:
        assert after[0] == {**before[0], "state": "error"}
        from plugins.models.store import require_model_calls_schema
        with closing(sqlite3.connect(path)) as connection:
            require_model_calls_schema(connection)
    migration["_migrate"](path, table, old, new, backups)
    assert rows(path, table) == after and len(list(backups.iterdir())) == 1
    with closing(sqlite3.connect(path)) as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute(f"UPDATE {table} SET {'status' if mobile else 'state'}=?", ("outcome_unknown" if mobile else "unknown",))


def test_failure_migration_refuses_unowned_schema_before_backup(tmp_path, migration):
    path = tmp_path / "records.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE model_calls(unowned TEXT)")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="schema"):
        migration["_migrate"](path, "model_calls", migration["_MODEL_OLD"], migration["_MODEL_NEW"], tmp_path / "backups")
    assert path.read_bytes() == before and not (tmp_path / "backups").exists()


@pytest.mark.parametrize("kind", ["wake", "delivery"])
def test_owner_ledgers_require_migration_and_preserve_failed_work(tmp_path, migration, kind):
    """运行时拒绝旧表，正式迁移保留原领取、正文和错误详情。"""
    path = tmp_path / "ledger.db"
    if kind == "wake":
        from plugins.wake.state import WakeState
        owner = WakeState(path)
        owner.initialize()
        table, old, new, versions, index = "wake_attempts", migration["_WAKE_OLD"], migration["_WAKE_NEW"], (8, 9), None
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("DROP TABLE wake_attempts")
            connection.execute(old)
            connection.execute("INSERT INTO wake_attempts VALUES(?,?,?,?,?,?,?,?,?)",
                ("flow", "timer", "scheduled", "fired", 12, "delivery_unknown", "content", "original evidence", "finished"))
    else:
        from agent.plugin_composition.durable_delivery_store import DurableDeliveryStore
        owner = DurableDeliveryStore(path)
        table, old, new, versions, index = "deliveries", migration["_LEDGER_OLD"], migration["_LEDGER_NEW"], (1, 2), migration["_LEDGER_INDEX"]
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(old)
            connection.execute(index)
            value: dict[str, object] = {row[1]: None for row in connection.execute("PRAGMA table_info(deliveries)")}
            value.update(logical_delivery_id="effect", accepted_session_id="s", accepted_turn_id="t", target_service="wake",
                channel="telegram", recipient="room", projection_session_id="s", body="original body", metadata_json="{}",
                state="uncertain", provider_receipt_json='{"status":"unknown","provider_ids":["partial"]}', created_at="created", updated_at="updated")
            connection.execute("INSERT INTO deliveries VALUES(" + ",".join("?" for _ in value) + ")", tuple(value.values()))
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(f"PRAGMA user_version = {versions[0]}")
    before = rows(path, table)
    with pytest.raises(RuntimeError):
        owner.initialize()
    migration["_migrate"](path, table, old, new, tmp_path / "backups", versions=versions, index=index)
    owner.initialize()
    assert rows(path, table) == [{**before[0], "outcome" if kind == "wake" else "state": "failed"}]
    backup, = (tmp_path / "backups").glob("*/ledger.db")
    assert rows(backup, table) == before
    assert backup.stat().st_mode & 0o777 == 0o600
    assert backup.parent.stat().st_mode & 0o777 == 0o700
    assert (backup.parent / "manifest.json").is_file()
    migration["_migrate"](path, table, old, new, tmp_path / "backups", versions=versions, index=index)
    assert len(list((tmp_path / "backups").iterdir())) == 1


@pytest.mark.parametrize("absolute", [False, True])
def test_public_migration_uses_configured_mobile_path_and_model_owner(tmp_path, migration, absolute):
    """正式入口同时迁移配置指定的 Mobile 库和工作区的模型调用库。"""
    from agent.migrations.context import bind_migration_context
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mobile = tmp_path / "external.db" if absolute else workspace / "custom.db"
    model = workspace / "model-registry.sqlite3"
    from agent.plugins.manifest import builtin_plugin_data_dir
    from plugins.wake.state import WakeState
    wake = builtin_plugin_data_dir("wake", workspace) / "wake.sqlite3"
    WakeState(wake).initialize()
    with closing(sqlite3.connect(wake)) as connection, connection:
        connection.execute("DROP TABLE wake_attempts")
        connection.execute(migration["_WAKE_OLD"])
        connection.execute("PRAGMA user_version = 8")
    ledger = workspace / "runtime/deliveries/settlements.sqlite"
    ledger.parent.mkdir(parents=True)
    with closing(sqlite3.connect(ledger)) as connection:
        connection.execute(migration["_LEDGER_OLD"])
        connection.execute(migration["_LEDGER_INDEX"])
        connection.execute("PRAGMA user_version = 1")
    with closing(sqlite3.connect(mobile)) as connection:
        connection.execute("CREATE TABLE mobile_devices(device_id TEXT PRIMARY KEY)")
        connection.execute(migration["_MOBILE_OLD"])
    with closing(sqlite3.connect(model)) as connection:
        connection.execute(migration["_MODEL_OLD"])
    config = tmp_path / "config.toml"
    config.write_text(f'[mobile_realtime]\ndatabase = "{mobile if absolute else mobile.name}"\n')
    with bind_migration_context(workspace=workspace, config_path=config):
        migration["steps"][0](None)
    for path, table, target in ((mobile, "mobile_command_receipts", "_MOBILE_NEW"), (model, "model_calls", "_MODEL_NEW"),
                                (wake, "wake_attempts", "_WAKE_NEW"), (ledger, "deliveries", "_LEDGER_NEW")):
        with closing(sqlite3.connect(path)) as connection:
            sql = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        assert migration["_sql"](sql) == migration["_sql"](migration[target])
        backups = list((workspace / "backups").glob(f"*/*/{path.name}"))
        assert len(backups) == 1 and (backups[0].parent / "manifest.json").is_file()
