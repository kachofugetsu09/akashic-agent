import asyncio
import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelContinuation,
    ModelRequest,
    ModelRole,
    ModelUnavailableError,
    ModelUsage,
    UsageCoverage,
)
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from agent.migrations.context import bind_migration_context


@pytest.fixture
def store(tmp_path):
    store = ModelsStore(tmp_path / "model-registry.sqlite3", tmp_path / "backups")
    store.initialize()
    return store


@pytest.fixture
def descriptor():
    return BoundModelDescriptor(
        binding_id="bound",
        plugin_snapshot_id="snapshot",
        model_revision=0,
        model_id="model",
        connection_id="connection",
        driver_id="driver",
        driver_contract_version="1",
        auth_identity="identity",
        model="model",
        role=ModelRole.AGENT,
        reasoning_effort=None,
        capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(),
        capability_digest="digest",
    )


def call_ids(store):
    with closing(sqlite3.connect(store.path)) as connection:
        return [
            row[0]
            for row in connection.execute("SELECT id FROM model_calls ORDER BY rowid")
        ]


@pytest.mark.asyncio
async def test_started_is_durable_before_io_and_usage_survives_without_message(
    store, descriptor
):
    usage = ModelUsage(
        input_tokens=4,
        output_tokens=2,
        covered_request_count=1,
        coverage=UsageCoverage.EXACT,
    )

    class Driver:
        async def complete(self, request):
            (call_id,) = call_ids(store)
            assert store.read_call(call_id)["state"] == "started"
            assert store.read_call(call_id)["usage"] is None
            with pytest.raises(TypeError):
                request.messages[0]["content"] = "changed"
            return LLMResponse("uncommitted output", usage=usage)

    messages = [{"role": "user", "content": "input"}]
    request = ModelRequest(messages)
    messages[0]["content"] = "later change"
    assert request.messages[0]["content"] == "input"
    response = await _BoundChat(descriptor, Driver(), store).complete(request)
    record = ModelsStore(store.path, store.backup_dir).read_call(
        response.call_record_id
    )
    assert record["state"] == "success"
    assert record["binding"]["binding_id"] == descriptor.binding_id
    assert record["usage"]["input_tokens"] == 4
    assert "uncommitted output" not in str(record)
    assert store.read_snapshot().revision == 0
    assert not store.backup_dir.exists()
    with pytest.raises(RuntimeError, match="已经结算"):
        store.finish_call(response.call_record_id, usage=None, failure="late")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("transport"), asyncio.CancelledError()])
async def test_failure_and_cancel_record_unknown_cost_without_retry(
    store, descriptor, failure
):
    seen = []

    class Driver:
        async def complete(self, request):
            seen.append(request)
            raise failure

    with pytest.raises(type(failure)) as raised:
        await _BoundChat(descriptor, Driver(), store).complete(ModelRequest(()))
    assert raised.value is failure
    assert len(seen) == 1
    (call_id,) = call_ids(store)
    record = store.read_call(call_id)
    assert record["state"] == "unknown"
    assert record["usage"] is None
    assert record["failure"] == type(failure).__name__


@pytest.mark.asyncio
async def test_missing_migration_or_wrong_binding_stops_before_io(store, descriptor):
    class Driver:
        async def complete(self, request):
            pytest.fail("provider I/O must not start")

    model = _BoundChat(descriptor, Driver(), store)
    with pytest.raises(ModelUnavailableError):
        await model.complete(
            ModelRequest((), continuation=ModelContinuation("other", {}))
        )
    assert call_ids(store) == []
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DROP TABLE model_calls")
    store.initialize()
    with pytest.raises(RuntimeError, match="yoyo"):
        await model.complete(ModelRequest(()))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("transport"), asyncio.CancelledError()])
async def test_settlement_failure_keeps_provider_failure_and_durable_unknown(
    store, descriptor, failure, monkeypatch
):
    class Driver:
        async def complete(self, request):
            raise failure

    record_failure = sqlite3.OperationalError("disk full")

    def fail(*args, **kwargs):
        raise record_failure

    monkeypatch.setattr(store, "finish_call", fail)
    with pytest.raises(type(failure)) as raised:
        await _BoundChat(descriptor, Driver(), store).complete(ModelRequest(()))
    assert raised.value is failure
    assert raised.value.__cause__ is record_failure
    (call_id,) = call_ids(store)
    assert store.read_call(call_id)["state"] == "started"
    assert store.read_call(call_id)["usage"] is None


@pytest.fixture
def migration(monkeypatch):
    import yoyo

    monkeypatch.setattr(yoyo, "step", lambda callback: callback)
    path = Path(__file__).parents[1] / "migrations/yoyo/20260905_03_model_calls.py"
    spec = importlib.util.spec_from_file_location("model_calls_migration_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path):
    with closing(sqlite3.connect(path)) as connection:
        return tuple(connection.iterdump())


def run_migration(migration, workspace):
    with bind_migration_context(
        workspace=workspace, config_path=workspace / "config.toml"
    ):
        migration.migrate_model_calls(None)


def test_migration_preserves_registry_and_lost_ack_preserves_real_call(
    store, descriptor, migration
):
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DROP TABLE model_calls")
    before = dump(store.path)
    run_migration(migration, store.path.parent)
    backups = list(
        (store.path.parent / "backups/model-calls-v1").glob("*/model-registry.sqlite3")
    )
    assert len(backups) == 1
    assert dump(backups[0]) == before
    call_id = store.start_call(descriptor, ModelRequest(()))
    # 模拟 provider 已接到请求，进程在收到响应前崩溃；不会重放或把费用补成零。
    reopened = ModelsStore(store.path, store.backup_dir)
    reopened.initialize()
    assert reopened.read_call(call_id)["state"] == "started"
    assert reopened.read_call(call_id)["usage"] is None
    after = dump(store.path)
    run_migration(migration, store.path.parent)
    assert dump(store.path) == after
    assert store.read_snapshot().revision == 0


def test_migration_rejects_same_name_with_other_schema_without_write(store, migration):
    with closing(sqlite3.connect(store.path)) as connection, connection:
        connection.execute("DROP TABLE model_calls")
        connection.execute("CREATE TABLE model_calls (id TEXT)")
    before = dump(store.path)
    with pytest.raises(RuntimeError, match="schema"):
        run_migration(migration, store.path.parent)
    assert dump(store.path) == before
    assert not store.backup_dir.exists()
