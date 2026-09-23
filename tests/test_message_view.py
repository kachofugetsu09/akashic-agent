import asyncio
from contextlib import closing
from collections.abc import Mapping
import json
import pytest
import sqlite3
from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.services import MessageCatalogPort
from infra.channels.message_view import MessageDisplayProviders, message_rows
from plugins.akashic_clients.web_chat import WebChatChannel
from plugins.models.projection import check_facts, display_facts
from session.log import MessageLog, SessionAttributes
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult
from tests.test_message_artifacts import storage
from tests.sqlite_helpers import snapshot


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return cast(Mapping[str, object], value)


@pytest.mark.asyncio
async def test_live_message_projection_uses_only_page_providers_and_releases_scopes(storage):
    from agent.plugin_composition import CompositionRoot, FiberState, PluginRuntime, ServiceKey
    from agent.plugin_composition.message_view import project_message_rows

    path, log, _ = storage
    log.save_binding("tool", {"service": "tools.v1", "metadata": {"tool": {"name": "tool"}}})
    log.writer(
        "s", author="user", source="conversation", body_types=(Input, Output),
        content={"example.fact": lambda _: ContentReferences()},
    ).append("input", Input((ContentPart("example.fact", {"secret": "keep"}),)))
    log.writer(
        "s", author="assistant", source="conversation", body_types=(Input, Output),
        content={"example.fact": lambda _: ContentReferences()},
        check_call=lambda _call: None,
    ).append(
        "output",
        Output((ContentPart("example.fact", {"secret": "keep"}), ToolCall("tool", {"q": "x"})), "continue"),
    )
    log.writer(
        "s", author="assistant", source="conversation", body_types=(Output,),
        content={
            "missing.fact": lambda _: ContentReferences(),
            "loading.fact": lambda _: ContentReferences(),
        },
    ).append("missing", Output((ContentPart("missing.fact", {"x": 1}),), "complete"))
    log.writer(
        "s", author="assistant", source="conversation", body_types=(Output,),
        content={
            "missing.fact": lambda _: ContentReferences(),
            "loading.fact": lambda _: ContentReferences(),
        },
    ).append("loading", Output((ContentPart("loading.fact", {"x": 2}),), "complete"))
    page = log.reader("s").read_tail()
    root = CompositionRoot("live-message-view")
    runtime = PluginRuntime(
        plugin_id="display-owner", generation_id="display-owner-1",
        plugin_dir=path.parent, data_dir=path.parent,
        workspace=path.parent, config={},
    )
    calls: list[str] = []
    unused_calls: list[str] = []
    labels = ["live"]
    closed: list[str] = []
    fact_key = ServiceKey("message.display:example.fact")
    tool_key = ServiceKey("tools.display-name.v1")
    loading_key = ServiceKey("message.display:loading.fact")

    async def apply(ctx):
        label = labels[0]

        async def close():
            closed.append(label)

        await ctx.effect(lambda: close)

        def display_fact(part):
            ctx.require_runtime_owner(fact_key, display_fact)
            calls.append(part.kind)
            return {"label": label}

        def display_tool(binding_id):
            ctx.require_runtime_owner(tool_key, display_tool)
            calls.append(binding_id)
            return "tool-name"

        def unused(_part):
            unused_calls.append("called")
            return {"label": "wrong"}

        await ctx.provide(fact_key, display_fact)
        await ctx.provide(tool_key, display_tool)
        await ctx.provide(ServiceKey("message.display:not-on-page"), unused)

    unrelated_key = ServiceKey("message.display:unrelated")

    async def unrelated(ctx):
        await ctx.provide(unrelated_key, lambda _part: {"label": "unrelated"})

    owner = await root.mount(apply, name="display-owner", runtime=runtime)
    unrelated_fiber = await root.mount(unrelated, name="unrelated-display")
    loading_started = asyncio.Event()
    loading_release = asyncio.Event()
    loading_handles: list[object] = []
    loading_calls: list[str] = []

    async def loading(ctx):
        def display_loading(part):
            ctx.require_runtime_owner(loading_key, display_loading)
            loading_calls.append(part.kind)
            return {"label": "loading"}

        await ctx.provide(loading_key, display_loading)
        loading_handles.append(ctx.fiber)
        loading_started.set()
        await loading_release.wait()

    loading_task = asyncio.create_task(root.mount(loading, name="loading-display", runtime=runtime))
    await loading_started.wait()
    assert loading_handles[0].state is FiberState.LOADING
    assert root.service_value(loading_key) is None
    root_token = root.instance_token
    old_context = owner.context
    unrelated_context = unrelated_fiber.context
    unrelated_token = unrelated_context.fiber.activation_token
    try:
        rows = await project_message_rows(root, page, display_only=True)
        bodies = [_mapping(row["body"]) for row in rows]
        assert bodies[0]["parts"] == [{"kind": "example.fact", "value": {"label": "live"}}]
        assert bodies[1]["parts"] == [
            {"kind": "example.fact", "value": {"label": "live"}},
            {"kind": "tool_call", "binding_id": "tool", "name": "tool-name", "arguments": {"q": "x"}},
        ]
        assert bodies[2]["parts"] == [{"kind": "missing.fact", "display": "unavailable"}]
        assert bodies[3]["parts"] == [{"kind": "loading.fact", "display": "unavailable"}]
        assert calls == ["example.fact", "example.fact", "tool"]
        assert unused_calls == []
        assert loading_calls == []
        assert not owner._in_flight_calls
        assert not unrelated_fiber._in_flight_calls

        loading_release.set()
        await loading_task
        assert root.service_value(loading_key) is not None
        rows = await project_message_rows(root, page, display_only=True)
        assert _mapping(rows[3]["body"])["parts"] == [
            {"kind": "loading.fact", "value": {"label": "loading"}},
        ]
        assert loading_calls == ["loading.fact"]

        await owner.dispose()
        assert closed == ["live"]
        labels[0] = "new"
        owner = await root.mount(apply, name="display-owner", runtime=runtime)
        calls.clear()
        rows = await project_message_rows(root, page, display_only=True)
        assert _mapping(rows[0]["body"])["parts"] == [
            {"kind": "example.fact", "value": {"label": "new"}},
        ]
        assert old_context is not owner.context
        assert root.instance_token is root_token
        assert unrelated_fiber.context is unrelated_context
        assert unrelated_context.fiber.activation_token is unrelated_token
        assert closed == ["live"]
    finally:
        loading_release.set()
        if not loading_task.done():
            await loading_task
        await root.dispose()


@pytest.mark.asyncio
async def test_live_message_projection_releases_scope_on_renderer_error(storage):
    from agent.plugin_composition import CompositionRoot, PluginRuntime, ServiceKey
    from agent.plugin_composition.message_view import project_message_rows

    path, log, _ = storage
    log.writer(
        "s", author="assistant", source="conversation", body_types=(Output,),
        content={"example.fact": lambda _: ContentReferences()},
    ).append("output", Output((ContentPart("example.fact", {"x": 1}),), "complete"))
    page = log.reader("s").read_tail()
    root = CompositionRoot("live-message-view-error")
    key = ServiceKey("message.display:example.fact")

    async def apply(ctx):
        def fail(part):
            ctx.require_runtime_owner(key, fail)
            raise RuntimeError("renderer failed")

        await ctx.provide(key, fail)

    owner = await root.mount(
        apply, name="display-owner",
        runtime=PluginRuntime(
            plugin_id="display-owner", generation_id="display-owner-1",
            plugin_dir=path.parent, data_dir=path.parent,
            workspace=path.parent, config={},
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="renderer failed"):
            await project_message_rows(root, page, display_only=False)
        assert not owner._in_flight_calls
    finally:
        await root.dispose()


def test_view_keeps_independent_facts_and_hides_private_configuration(storage):
    path, log, ref = storage
    log.save_binding("tool", {"root_ref": {"private": "root-secret"}, "service": "tools.v1",
                              "metadata": {"tool": {"name": "original-name"}, "state": {"private": "tool-secret"}}})
    checks = {"text": lambda part: ContentReferences(), "model.facts": check_facts,
              "artifact_ref": lambda part: ContentReferences(artifact_ids=(part.value,)),
              "history.future": lambda part: ContentReferences()}
    def append(identity, body, call_ref=None):
        return log.writer("s", author="真实作者", source="来源", body_types=(type(body),),
                          content=checks, call_ref=call_ref, check_call=lambda call: None).append(identity, body)
    append("input", Input((ContentPart("artifact_ref", ref.artifact_id),)))
    facts = ContentPart("model.facts", {"call_record_id": "call-record", "tool_ids": {"1": "provider-call"},
                                       "thinking": "可读思考", "continuation": {"binding_id": "model", "payload": {"private": "model-secret"}}})
    output = append("output", Output((facts, ToolCall("tool", {"query": "原参数"})), "continue"))
    append("pause", Control("pause", output.seq))
    append("result", ToolResult(CallRef("output", 1), "error", (ContentPart("text", "效果待确认"),)), CallRef("output", 1))
    append("quiet", Output((ContentPart("history.future", {"private": "content-secret"}),), "quiet"))
    before = snapshot(path)
    rows = [
        _mapping(row)
        for row in message_rows(
            log.reader("s").read_tail(),
            providers=MessageDisplayProviders(
                tool_name=lambda binding_id: "original-name" if binding_id == "tool" else binding_id,
                part_display={"model.facts": display_facts},
            ),
        )
    ]
    assert [row["id"] for row in rows] == ["input", "output", "pause", "result", "quiet"]
    body = [_mapping(row["body"]) for row in rows]
    assert [item["kind"] for item in body] == ["input", "output", "control", "tool_result", "output"]
    assert all(row["author"] == "真实作者" and row["source"] == "来源" for row in rows)
    assert _mapping(cast(list[object], rows[0]["attachments"])[0])["artifact_id"] == ref.artifact_id
    assert _mapping(cast(list[object], body[1]["parts"])[0])["value"] == {"call_record_id": "call-record", "thinking": "可读思考"}
    assert _mapping(cast(list[object], body[1]["parts"])[1])["name"] == "original-name"
    assert body[3]["call_ref"] == {"message_id": "output", "part_index": 1}
    assert body[3]["outcome"] == "error"
    assert body[4]["parts"] == [{"kind": "history.future", "display": "unavailable"}]
    encoded = json.dumps(rows, ensure_ascii=False)
    assert "secret" not in encoded and "artifact.bin" not in encoded and "provider-call" not in encoded
    assert snapshot(path) == before


def test_message_view_has_no_plugin_implementation_imports_and_degrades_without_providers(storage):
    _, log, _ = storage
    log.save_binding(
        "missing-binding",
        {"service": "tools.v1", "metadata": {"tool": {"name": "private-tool"}}},
    )
    checks = {"model.facts": check_facts}
    writer = log.writer(
        "s",
        author="assistant",
        source="conversation",
        body_types=(Output,),
        content=checks,
        check_call=lambda call: None,
    )
    writer.append(
        "output",
        Output(
            (
                ContentPart(
                    "model.facts",
                    {
                        "call_record_id": "call",
                        "tool_ids": {},
                        "thinking": "private",
                        "continuation": {
                            "binding_id": "provider",
                            "payload": {"opaque": "secret"},
                        },
                    },
                ),
                ToolCall("missing-binding", {"private": "argument"}),
            ),
            "continue",
        ),
    )
    source = Path(__file__).parents[1] / "infra/channels/message_view.py"
    text = source.read_text(encoding="utf-8")
    assert "plugins.models" not in text and "plugins.tools" not in text
    parts = _mapping(_mapping(message_rows(log.reader("s").read_tail())[0])["body"])["parts"]
    assert parts == [
        {"kind": "model.facts", "display": "unavailable"},
        {"kind": "tool_call", "binding_id": "missing-binding", "display": "unavailable"},
    ]
    assert "secret" not in json.dumps(parts, ensure_ascii=False)


def test_web_catalog_and_history_use_real_log_without_session_manager(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        channel = WebChatChannel()
        session = f"{channel.name}:s"
        log.ensure_session(f"{channel.name}:empty", SessionAttributes())
        log.ensure_session(f"{channel.name}:internal", SessionAttributes(visibility="internal"))
        for index in range(4):
            log.writer(session, author="user", source="conversation", body_types=(Input,), content={}).append(str(index), Input(()))
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=cast(MessageCatalogPort, log.catalog()))
        before = snapshot(path)
        with TestClient(app, raise_server_exceptions=False) as client:
            first = client.get("/api/chat/sessions", params={"page_size": 1}).json()
            assert first["total"] == 2 and len(first["items"]) == 1
            cursor = first["next_cursor"]
            second = client.get("/api/chat/sessions", params={"page_size": 1, "after_time": cursor["updated_at"], "after_key": cursor["session_id"]}).json()
            assert len({first["items"][0]["key"], second["items"][0]["key"]}) == 2
            endpoint = f"/api/chat/sessions/{session}/messages"
            page = client.get(endpoint, params={"page_size": 2}).json()
            assert page["version"] == 2 and page["through_seq"] == 3
            assert [row["seq"] for row in page["items"]] == [2, 3]
            earlier = client.get(endpoint, params={"page_size": 2, "through_seq": 3, "before_seq": page["before_seq"]}).json()
            assert [row["seq"] for row in earlier["items"]] == [0, 1] and not earlier["has_more"]
            assert client.get(endpoint, params={"through_seq": 99}).status_code == 422
            assert client.get("/api/chat/sessions", params={"after_time": "bad", "after_key": "s"}).status_code == 422
            assert client.get("/api/chat/sessions/unknown/messages").status_code == 404
            assert snapshot(path) == before
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE messages SET body=? WHERE id='3'", ('{"kind":"broken"}',))
            assert client.get(endpoint).status_code == 500
