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


@pytest.mark.asyncio
async def test_runtime_display_discovers_new_kind_and_releases_each_generation(storage):
    """新内容只需插件贡献；同一 reader 在切代后不保留旧回调。"""
    from agent.plugin_composition import CompositionRoot, ServiceKey
    from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore, get_current_runtime_snapshot
    from bootstrap.message_display import RuntimeMessageDisplay

    _, log, _ = storage
    log.writer("custom", author="test", source="test", body_types=(Input,),
        content={"example.fact": lambda _: ContentReferences()}).append(
            "fact", Input((ContentPart("example.fact", {"private": "secret"}),)))
    page = log.reader("custom").read_tail()
    store = RuntimeSnapshotStore()
    reader = RuntimeMessageDisplay(store)
    roots = []
    try:
        for label in ("first", "replacement", None):
            root = CompositionRoot("display-" + str(label))
            roots.append(root)
            if label is not None:
                def display(part):
                    assert part.kind == "example.fact"
                    assert get_current_runtime_snapshot() is selected
                    assert selected.lease_count == 1
                    return {"label": label}
                async def apply(ctx):
                    await ctx.provide(ServiceKey("message.display:example.fact"), display)
                await root.mount(apply, name="independent-display")
            selected = RuntimeSnapshotCompiler().compile({}, composition_root=root, snapshot_revision=str(label))
            if store.current is None:
                store.install(selected)
            else:
                await store.commit(store.begin_publish(selected))
            rows = await reader(page, display_only=True)
            expected = {"kind": "example.fact", "display": "unavailable"} if label is None else {
                "kind": "example.fact", "value": {"label": label}}
            body = rows[0]["body"]
            assert isinstance(body, Mapping)
            assert body["parts"] == [expected]
            assert selected.lease_count == 0
    finally:
        await store.close()
        for root in roots:
            await root.dispose()
