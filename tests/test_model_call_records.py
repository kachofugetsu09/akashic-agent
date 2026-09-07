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


@pytest.mark.asyncio
async def test_message_projection_keeps_provider_ids_and_interrupted_inputs(
    store, descriptor
):
    from datetime import UTC, datetime
    from agent.plugin_composition.models import ToolCall as ModelToolCall
    from plugins.models.projection import MessageProjection, response_facts
    from session.message import (
        CallRef,
        ContentPart,
        Control,
        Input,
        Message,
        Output,
        ToolCall,
        ToolResult,
    )

    class Driver:
        async def complete(self, request):
            return LLMResponse(
                "checking",
                tool_calls=[
                    ModelToolCall("provider-call", "original_name", {"value": 1})
                ],
                thinking="private reasoning",
            )

    model = _BoundChat(descriptor, Driver(), store)
    response = await model.complete(ModelRequest(()))
    facts = response_facts(response, [1])
    assert "usage" not in facts.value
    assert "binding" not in facts.value

    def message(seq, body):
        return Message(
            str(seq), "s", seq, datetime.now(UTC), "system", "conversation", body
        )

    messages = (
        message(0, Input((ContentPart("text", "u1"),))),
        message(
            1,
            Output(
                (
                    ContentPart("text", "checking"),
                    ToolCall("old-tool-binding", {"value": 1}),
                    facts,
                ),
                "continue",
            ),
        ),
        message(2, Input((ContentPart("text", "u2 interruption"),))),
        message(3, Control("pause", 2)),
        message(
            4,
            ToolResult(
                CallRef("1", 1), "unknown", (ContentPart("text", "connection lost"),)
            ),
        ),
    )
    projection = MessageProjection(
        model,
        source="conversation",
        render_content=lambda part: ({"type": "text", "text": part.value},),
        tool_name=lambda binding: {"old-tool-binding": "original_name"}[binding],
        read_call=store.read_call,
    )
    request = projection.render(messages, after_seq=-1)
    assert [row["role"] for row in request.messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert request.messages[1]["tool_calls"][0]["id"] == "provider-call"
    assert request.messages[2]["tool_call_id"] == "provider-call"
    assert "unknown" in request.messages[2]["content"][0]["text"]
    assert request.messages[-1]["content"][0]["text"] == "u2 interruption"
    assert request.messages[1]["reasoning_content"] == "private reasoning"
    with pytest.raises(ValueError, match="未结算"):
        projection.render(messages[:-1], after_seq=-1)
    with pytest.raises(ValueError, match="缺少"):
        projection.render(messages, after_seq=1)
    assert projection.render(messages, after_seq=4).messages == ()


@pytest.mark.asyncio
async def test_message_projection_keeps_source_continuation_and_rejects_unsafe_summary(
    store, descriptor
):
    from dataclasses import replace
    from datetime import UTC, datetime
    from plugins.models.projection import MessageProjection, response_facts
    from session.message import ContentPart, Message, Output

    class Driver:
        async def complete(self, request):
            return LLMResponse(
                "text",
                continuation=ModelContinuation(
                    "bound", {"opaque": "conversation-state"}
                ),
            )

    model = _BoundChat(descriptor, Driver(), store)
    response = await model.complete(ModelRequest(()))
    message = Message(
        "a",
        "s",
        0,
        datetime.now(UTC),
        "agent",
        "conversation",
        Output((ContentPart("text", "text"), response_facts(response, [])), "complete"),
    )
    later_other_source = replace(
        message,
        message_id="wake-a",
        seq=1,
        source="wake",
        body=Output((ContentPart("text", "wake text"),), "complete"),
    )
    projection = MessageProjection(
        model,
        source="conversation",
        render_content=lambda part: ({"type": "text", "text": part.value},),
        tool_name=lambda binding: "unused",
        read_call=store.read_call,
    )
    request = projection.render((message, later_other_source), after_seq=-1)
    assert request.continuation == response.continuation
    assert len(request.messages) == 2
    with pytest.raises(ValueError, match="摘要"):
        projection.render((message, later_other_source), after_seq=0)
    changed = MessageProjection(
        _BoundChat(replace(descriptor, binding_id="new-model"), Driver(), store),
        source="conversation",
        render_content=lambda part: (),
        tool_name=lambda binding: "unused",
        read_call=store.read_call,
    )
    with pytest.raises(ValueError, match="另一 binding"):
        changed.render((message,), after_seq=-1)


def test_chat_completions_places_output_images_after_complete_tool_group():
    from copy import deepcopy
    from plugins.openai_compatible.driver import _normalize_messages
    from agent.plugin_composition.models import InvalidRequestError

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "read pictures"}],
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
                {
                    "id": "two",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "one",
            "content": [{"type": "text", "text": "first image"}, image],
        },
        {
            "role": "tool",
            "tool_call_id": "two",
            "content": [{"type": "text", "text": "second result"}],
        },
    ]
    original = deepcopy(messages)
    wire = _normalize_messages(messages)
    assert messages == original
    assert [row["role"] for row in wire] == ["assistant", "tool", "tool", "user"]
    assert [row["tool_call_id"] for row in wire[1:3]] == ["one", "two"]
    assert wire[1]["content"] == [{"type": "text", "text": "first image"}]
    assert wire[-1]["content"][-1] == image
    assert "one" in wire[-1]["content"][0]["text"]
    with pytest.raises(InvalidRequestError, match="缺少结果"):
        _normalize_messages(messages[:2])


@pytest.mark.parametrize("assistant_picture", [False, True])
def test_only_images_keep_valid_roles_and_flush_once_after_all_results(
    assistant_picture,
):
    from plugins.openai_compatible.driver import _normalize_messages

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}
    calls = [
        {
            "id": name,
            "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }
        for name in ("one", "two")
    ]
    messages = [
        {
            "role": "assistant",
            "content": [image] if assistant_picture else "read",
            "tool_calls": calls,
        },
        {"role": "tool", "tool_call_id": "two", "content": [image]},
        {"role": "tool", "tool_call_id": "one", "content": "plain result"},
    ]
    wire = _normalize_messages(messages)
    assert [row["role"] for row in wire] == ["assistant", "tool", "tool", "user"]
    assert wire[1]["content"] and all(
        block["type"] == "text" for block in wire[1]["content"]
    )
    if assistant_picture:
        assert wire[0]["content"] and all(
            block["type"] == "text" for block in wire[0]["content"]
        )
    assert (
        sum(block["type"] == "image_url" for block in wire[-1]["content"])
        == 1 + assistant_picture
    )
    plain = [
        messages[0] | {"content": "read"},
        messages[1] | {"content": "plain"},
        messages[2],
    ]
    assert _normalize_messages(plain) == plain
