from contextlib import closing
import sqlite3

import pytest

from session.log import MessageLog, SessionAttributes
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult
from tests.test_message_artifacts import storage


def append(log, session, identity, body, call_ref=None):
    return log.writer(session, author="actor", source="conversation", body_types=(type(body),),
                      content={"text": lambda part: ContentReferences(),
                               "image": lambda part: ContentReferences(artifact_ids=(part.value,))},
                      call_ref=call_ref, check_call=lambda call: None).append(identity, body)


def test_directory_filters_counts_gaps_and_keeps_recent_live_order(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        for key, visibility in (("a_:one", "listed"), ("a_:two", "listed"), ("a_:hidden", "internal"), ("ab:other", "listed")):
            log.ensure_session(key, SessionAttributes(visibility=visibility))
            append(log, key, key, Input((ContentPart("text", key),)))
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE sessions SET updated_at='2026-09-06T09:00:00+08:00'")
            connection.execute("UPDATE sessions SET updated_at='2026-09-06T02:00:00+00:00' WHERE key='a_:two'")
            # 模拟迁移保留的 seq 空洞，数量与高水位不同。
            connection.execute("UPDATE messages SET seq=8 WHERE session_key='a_:two'")
            connection.execute("UPDATE sessions SET next_seq=9 WHERE key='a_:two'")
            before = tuple(connection.iterdump())
        first = log.catalog().sessions(prefix="a_:", visibility="listed", limit=1)
        assert first.total == 2
        assert [row.session_id for row in first.items] == ["a_:two"]
        assert (first.items[0].head_seq, first.items[0].message_count) == (8, 1)
        assert first.items[0].first_message.body == Input((ContentPart("text", "a_:two"),))
        second = log.catalog().sessions(prefix="a_:", visibility="listed", limit=1, after=first.next_cursor)
        assert [row.session_id for row in second.items] == ["a_:one"]
        assert second.next_cursor is None
        with closing(sqlite3.connect(path)) as connection:
            assert tuple(connection.iterdump()) == before
        # live 目录发生更新会移动到 cursor 之前；刷新首屏取得新顺序。
        append(log, "a_:one", "new", Input(()))
        assert log.catalog().sessions(prefix="a_:", visibility="listed", after=first.next_cursor).items == ()
        assert log.catalog().sessions(prefix="a_:", visibility="listed").items[0].session_id == "a_:one"



def test_directory_counts_only_selected_sessions(tmp_path):
    """单页目录的计数必须按 Session 查索引，不能全表扫描消息。"""
    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        for index in range(30):
            append(log, f"s:{index}", str(index), Input(()))
        queries = []
        log._connection.set_trace_callback(queries.append)
        page = log.catalog().sessions(limit=1)
        log._connection.set_trace_callback(None)
        assert len(page.items) == 1 and page.total == 30
        sql = next(sql for sql in queries if "WITH page AS MATERIALIZED" in sql)
        plan = [row[3] for row in log._connection.execute("EXPLAIN QUERY PLAN " + sql)]
        assert not any("SCAN m " in step for step in plan), plan
        assert any("SEARCH m " in step and "session_key=?" in step for step in plan), plan


def test_forward_and_tail_pages_freeze_head_under_another_connection(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log, closing(MessageLog(path)) as other:
        for index in range(5):
            append(log, "s", str(index), Input(()))
        reader = log.reader("s")
        page = reader.read_page(limit=2)
        tail = reader.read_tail(limit=2)
        append(other, "s", "late", Input(()))
        assert page.through_seq == tail.through_seq == 4
        forward = reader.read_page(after_seq=page.messages[-1].seq, through_seq=page.through_seq, limit=2)
        end = reader.read_page(after_seq=forward.messages[-1].seq, through_seq=page.through_seq, limit=2)
        assert [m.message_id for m in (*page.messages, *forward.messages, *end.messages)] == ["0", "1", "2", "3", "4"]
        earlier = reader.read_tail(before_seq=tail.messages[0].seq, through_seq=tail.through_seq, limit=2)
        start = reader.read_tail(before_seq=earlier.messages[0].seq, through_seq=tail.through_seq, limit=2)
        assert [m.message_id for m in (*start.messages, *earlier.messages, *tail.messages)] == ["0", "1", "2", "3", "4"]
        assert not start.has_more and not end.has_more
        assert reader.read_tail().messages[-1].message_id == "late"
        with pytest.raises(ValueError, match="上界"):
            reader.read_page(through_seq=100)


def test_page_keeps_late_result_control_and_batched_references(storage):
    path, log, ref = storage
    log.save_binding("tool", {"version": 1, "service": "tools.v1", "metadata": {"tool": {"name": "original"}}})
    parts = (ContentPart("image", ref.artifact_id), ContentPart("image", ref.artifact_id))
    accepted = append(log, "s", "input", Input(parts))
    call = append(log, "s", "call", Output((ToolCall("tool", {}),), "continue"))
    paused = append(log, "s", "pause", Control("pause", call.seq))
    result = append(log, "s", "result", ToolResult(CallRef("call", 0), "success", ()), CallRef("call", 0))
    # 另一 Session 的资源不能混进本页。
    append(log, "other", "unrelated", Input(parts))
    queries = []
    log._connection.set_trace_callback(queries.append)
    page = log.reader("s").read_tail()
    log._connection.set_trace_callback(None)
    assert page.messages == (accepted, call, paused, result)
    assert page.attachments == {"input": (ref, ref), "call": (), "pause": (), "result": ()}
    assert page.bindings["tool"]["metadata"]["tool"]["name"] == "original"
    assert len([sql for sql in queries if sql.startswith("SELECT")]) == 4
    assert not any("COUNT(" in sql for sql in queries)
    with pytest.raises(TypeError):
        page.bindings["tool"]["metadata"]["tool"]["name"] = "changed"


def test_unknown_empty_and_gapped_message_pages(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        with pytest.raises(KeyError):
            log.reader("unknown").read_tail()
        assert log.catalog().snapshot_heads() == {}
        log.ensure_session("empty", SessionAttributes())
        page = log.reader("empty").read_page()
        assert page.through_seq == -1 and page.messages == () and not page.has_more
        append(log, "s", "first", Input(()))
        append(log, "s", "last", Input(()))
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("UPDATE messages SET seq=50 WHERE id='last'")
            connection.execute("UPDATE sessions SET next_seq=51 WHERE key='s'")
        page = log.reader("s").read_page(after_seq=0, through_seq=49, limit=1)
        assert not page.has_more and page.messages == ()
        page = log.reader("s").read_tail(limit=2)
        assert not page.has_more and [m.seq for m in page.messages] == [0, 50]


@pytest.mark.parametrize("field,value", [("metadata", '{"x":1,"x":2}'), ("attributes", '{"visibility":"listed","learning":"broken"}'), ("created_at", "broken")])
def test_directory_rejects_corrupt_selected_record(tmp_path, field, value):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        log.ensure_session("s", SessionAttributes())
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(f"UPDATE sessions SET {field}=?", (value,))
        with pytest.raises(ValueError):
            log.catalog().sessions()


def test_page_rejects_dangling_attachment_reference(storage):
    path, log, ref = storage
    append(log, "s", "input", Input((ContentPart("image", ref.artifact_id),)))
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("DELETE FROM attachments")
    with pytest.raises(ValueError, match="附件引用损坏"):
        log.reader("s").read_page()
