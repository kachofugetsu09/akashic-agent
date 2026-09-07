from contextlib import closing
import json
import sqlite3

from fastapi.testclient import TestClient

from bootstrap.chat_api import create_chat_app
from infra.channels.message_view import message_rows
from infra.channels.web_chat_channel import WebChatChannel
from plugins.models.projection import check_facts
from session.log import MessageLog, SessionAttributes
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult
from tests.test_message_artifacts import storage
from tests.test_message_log_migration import migration, old_workspace, run, snapshot


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
    append("result", ToolResult(CallRef("output", 1), "unknown", (ContentPart("text", "效果待确认"),)), CallRef("output", 1))
    append("quiet", Output((ContentPart("history.future", {"private": "content-secret"}),), "quiet"))
    before = snapshot(path)
    rows = message_rows(log.reader("s").read_tail())
    assert [row["id"] for row in rows] == ["input", "output", "pause", "result", "quiet"]
    assert [row["body"]["kind"] for row in rows] == ["input", "output", "control", "tool_result", "output"]
    assert all(row["author"] == "真实作者" and row["source"] == "来源" for row in rows)
    assert rows[0]["attachments"][0]["artifact_id"] == ref.artifact_id
    assert rows[1]["body"]["parts"][0]["value"] == {"call_record_id": "call-record", "thinking": "可读思考"}
    assert rows[1]["body"]["parts"][1]["name"] == "original-name"
    assert rows[3]["body"]["call_ref"] == {"message_id": "output", "part_index": 1}
    assert rows[3]["body"]["outcome"] == "unknown"
    assert rows[4]["body"]["parts"] == [{"kind": "history.future", "display": "unavailable"}]
    encoded = json.dumps(rows, ensure_ascii=False)
    assert "secret" not in encoded and "artifact.bin" not in encoded and "provider-call" not in encoded
    assert snapshot(path) == before


def test_migrated_history_stays_inside_original_message(migration, old_workspace):
    run(migration, old_workspace)
    path = old_workspace / "sessions.db"
    before = snapshot(path)
    with closing(MessageLog(path)) as log:
        rows = message_rows(log.reader("s").read_tail())
    assert [(row["id"], row["seq"]) for row in rows] == [("user", 4), ("reply", 8), ("nullable", 9)]
    transcript = rows[1]["body"]["parts"][2]
    assert transcript["kind"] == "history.transcript"
    assert '"result": "old result"' in transcript["archive"]["raw"]
    assert transcript["archive"]["completeness"] == "unknown"
    assert rows[2]["body"]["parts"][0]["archive"]["content_was_null"] is True
    assert all(row["source"] == "legacy-unattributed" for row in rows)
    assert snapshot(path) == before


def test_web_catalog_and_history_use_real_log_without_session_manager(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        channel = WebChatChannel()
        session = f"{channel.name}:s"
        log.ensure_session(f"{channel.name}:empty", SessionAttributes())
        log.ensure_session(f"{channel.name}:internal", SessionAttributes(visibility="internal"))
        for index in range(4):
            log.writer(session, author="user", source="conversation", body_types=(Input,), content={}).append(str(index), Input(()))
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=log.catalog())
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
