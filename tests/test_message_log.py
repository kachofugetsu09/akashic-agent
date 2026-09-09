from session.message import ContentReferences
import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier

import pytest

from session.log import MessageConflict, MessageLog, WriterExpired
from session.message import (
    CallRef,
    ContentPart,
    Control,
    Input,
    Output,
    ToolCall,
    ToolResult,
)
from session.message_codec import decode_body, encode_body


def text_schema(part):
    if not isinstance(part.value, str):
        raise ValueError("text 必须是字符串")
    return ContentReferences()


def writer(
    log,
    *,
    source="conversation",
    author="user",
    bodies=(Input,),
    call_ref=None,
    check_call=None,
):
    return log.writer(
        "s",
        author=author,
        source=source,
        body_types=bodies,
        content={"text": text_schema},
        call_ref=call_ref,
        check_call=check_call,
    )


@pytest.fixture
def log(tmp_path):
    result = MessageLog(tmp_path / "sessions.db")
    try:
        yield result
    finally:
        result.close()


def test_ack_loss_returns_original_message_before_head_check(log):
    inputs = writer(log)
    first = inputs.append(
        "u1", Input((ContentPart("text", "first"),)), expected_source_head=-1
    )
    inputs.append("u2", Input(()))
    assert inputs.append("u1", first.body, expected_source_head=-1) == first
    with pytest.raises(MessageConflict):
        inputs.append("u1", Input((ContentPart("text", "different"),)))
    assert len(log.reader("s").read()) == 2


def test_same_source_conflicts_but_independent_source_does_not(log):
    writer(log).append("u1", Input(()))
    response = writer(log, author="agent", bodies=(Output,))
    writer(log, source="wake", author="app", bodies=(Output,)).append(
        "p", Output((), "complete")
    )
    answer = response.append("a1", Output((), "complete"), expected_source_head=0)
    assert answer.seq == 2
    writer(log).append("u2", Input(()))
    with pytest.raises(MessageConflict):
        response.append(
            "stale", Output((), "complete"), expected_source_head=answer.seq
        )


def test_expired_writer_cannot_commit_even_if_source_head_is_unchanged(log):
    output = writer(log, author="agent", bodies=(Output,))
    output.expire()
    with pytest.raises(WriterExpired):
        output.append("late", Output((), "complete"), expected_source_head=-1)
    assert log.reader("s").read() == ()


def test_concurrent_writers_allocate_one_sequence_per_fact(log):
    barrier = Barrier(2)

    def append(source):
        bound = writer(log, source=source)
        barrier.wait()
        return bound.append(source, Input(())).seq

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(append, ("one", "two"))) == [0, 1]
    assert len(log.reader("s").read()) == 2


def test_missing_resource_rolls_back_message_and_sequence(log):
    outputs = writer(
        log, author="agent", bodies=(Output,), check_call=lambda call: None
    )
    body = Output((ToolCall("missing", {}),), "continue")
    with pytest.raises(sqlite3.IntegrityError):
        outputs.append("call", body)
    assert log.reader("s").read() == ()
    log.save_binding("missing", {"artifact": "immutable-revision"})
    assert outputs.append("call", body).seq == 0
    with pytest.raises(MessageConflict):
        log.save_binding("missing", {"artifact": "new-revision"})


def test_result_writer_is_bound_to_one_real_call_and_result(log):
    log.save_binding("b", {"artifact": "v1"})
    outputs = writer(
        log, author="agent", bodies=(Output,), check_call=lambda call: None
    )
    outputs.append(
        "call", Output((ContentPart("text", "thinking"), ToolCall("b", {})), "continue")
    )
    ref = CallRef("call", 1)
    results = writer(log, author="tool", bodies=(ToolResult,), call_ref=ref)
    with pytest.raises(PermissionError):
        results.append("wrong", ToolResult(CallRef("call", 0), "success", ()))
    result = results.append("result", ToolResult(ref, "unknown", ()))
    assert results.append("result", result.body) == result
    with pytest.raises(MessageConflict):
        results.append("second-result", result.body)
    assert log.reader("s").head() == 1


def test_content_owner_validates_before_any_message_is_committed(log):
    inputs = writer(log)
    with pytest.raises(ValueError, match="text"):
        inputs.append("bad", Input((ContentPart("text", {}),)))
    with pytest.raises(PermissionError):
        inputs.append("forged", Input((ContentPart("history.record", {}),)))
    with pytest.raises(PermissionError):
        inputs.append("output", Output((), "quiet"))
    assert log.reader("s").head() == -1


@pytest.mark.asyncio
async def test_follow_catches_up_and_reconnects_using_only_sequence(log, monkeypatch):
    inputs = writer(log)
    inputs.append("before-subscribe", Input(()))
    reader = log.reader("s")
    empty_read = asyncio.Event()
    read = reader.read

    def read_and_signal(**kwargs):
        messages = read(**kwargs)
        if not messages:
            empty_read.set()
        return messages

    monkeypatch.setattr(reader, "read", read_and_signal)
    feed = reader.follow()
    first = await anext(feed)
    pending = asyncio.create_task(anext(feed))
    await asyncio.wait_for(empty_read.wait(), 1)
    inputs.append("during-subscribe", Input(()))
    second = await asyncio.wait_for(pending, 1)
    assert (first.seq, second.seq) == (0, 1)
    await feed.aclose()
    inputs.append("after-disconnect", Input(()))
    recovered = log.reader("s").follow(after_seq=second.seq)
    assert (await anext(recovered)).message_id == "after-disconnect"
    await recovered.aclose()


def test_opening_old_schema_requires_migration_without_changing_it(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, content TEXT)")
        before = connection.execute("SELECT name,sql FROM sqlite_master").fetchall()
    with pytest.raises(RuntimeError, match="yoyo"):
        MessageLog(path)
    with closing(sqlite3.connect(path)) as connection:
        assert (
            connection.execute("SELECT name,sql FROM sqlite_master").fetchall()
            == before
        )


@pytest.mark.parametrize(
    "body",
    [
        Input((ContentPart("text", "输入"),)),
        Output(
            (
                ToolCall("b", {"nested": [1, {"x": True}]}),
                ContentPart("model.facts", {"opaque": "state"}),
            ),
            "continue",
        ),
        ToolResult(CallRef("m", 2), "error", (ContentPart("text", "error"),)),
        Control("pause", 0, "reason"),
    ],
)
def test_persisted_body_roundtrip_keeps_replay_and_nested_content(body):
    assert decode_body(encode_body(body)) == body


def test_corrupt_persisted_schema_is_not_silently_normalized():
    with pytest.raises(ValueError, match="重复"):
        decode_body('{"kind":"output","kind":"input","parts":[]}')
    with pytest.raises(ValueError, match="字段"):
        decode_body(json.dumps({"kind": "input", "parts": [], "unexpected": True}))


def test_ordinary_output_cannot_claim_model_facts_or_propose_tools(log):
    replies = writer(log, author="command", bodies=(Output,))
    with pytest.raises(PermissionError, match="model.facts"):
        replies.append(
            "fake-model",
            Output((ContentPart("model.facts", {"usage": 1}),), "complete"),
        )
    log.save_binding("b", {"artifact": "v1"})
    with pytest.raises(PermissionError, match="提出权"):
        replies.append("fake-call", Output((ToolCall("b", {}),), "continue"))
    assert log.reader("s").read() == ()


def test_call_receipt_replay_survives_revoked_proposal_right(log):
    log.save_binding("b", {"artifact": "v1"})
    allowed = True

    def check_call(call):
        if not allowed or call.binding_id != "b":
            raise PermissionError("binding no longer visible")

    outputs = writer(log, author="agent", bodies=(Output,), check_call=check_call)
    original = outputs.append("call", Output((ToolCall("b", {}),), "continue"))
    allowed = False
    assert outputs.append("call", original.body, expected_source_head=-1) == original
    with pytest.raises(PermissionError):
        outputs.append("different", original.body)


def test_new_model_output_uses_the_same_content_schema_grants(log):
    def check_facts(part):
        if part.value["binding"] != "selected-model":
            raise ValueError("Model facts belong to another binding")
        return ContentReferences(binding_ids=("selected-model",))

    outputs = log.writer(
        "s",
        author="model",
        source="conversation",
        body_types=(Output,),
        content={"model.facts": check_facts},
    )
    output = Output(
        (ContentPart("model.facts", {"binding": "selected-model", "opaque": [1]}),),
        "complete",
    )
    with pytest.raises(sqlite3.IntegrityError):
        outputs.append("model", output)
    assert log.reader("s").read() == ()
    log.save_binding("selected-model", {"artifact": "provider-v1"})
    assert outputs.append("model", output).body == output
    with pytest.raises(ValueError):
        outputs.append(
            "wrong",
            Output((ContentPart("model.facts", {"binding": "other"}),), "complete"),
        )


def test_stale_output_does_not_run_content_or_call_checks(log):
    checks = []

    def check_content(part):
        checks.append(part)
        return ContentReferences()

    def check_call(call):
        checks.append(call)

    outputs = log.writer(
        "s",
        author="agent",
        source="conversation",
        body_types=(Output,),
        content={"text": check_content},
        check_call=check_call,
    )
    writer(log).append("new-input", Input(()))
    with pytest.raises(MessageConflict):
        outputs.append(
            "stale",
            Output(
                (ContentPart("text", "draft"), ToolCall("unresolved", {})), "continue"
            ),
            expected_source_head=-1,
        )
    assert checks == []


def test_content_binding_pins_commit_with_the_message(tmp_path):
    path = tmp_path / "messages.db"
    log = MessageLog(path)
    try:
        log.save_binding("model", {"artifact": "provider-v1"})
        output = log.writer(
            "s",
            author="agent",
            source="s",
            body_types=(Output,),
            content={"model.facts": lambda part: ContentReferences(binding_ids=("model",))},
        )
        output.append(
            "m", Output((ContentPart("model.facts", {"binding": "model"}),), "complete")
        )
    finally:
        log.close()
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute(
            "SELECT message_id,binding_id FROM message_bindings"
        ).fetchall() == [("m", "model")]


def test_catalog_heads_fix_cross_session_prefixes_without_creating_sessions(log):
    catalog = log.catalog()
    assert catalog.snapshot_heads() == {}
    assert catalog.reader("absent").read() == ()
    assert catalog.snapshot_heads() == {}
    writer(log).append("u1", Input(()))
    other = log.writer(
        "other", author="user", source="conversation", body_types=(Input,), content={}
    )
    other.append("other1", Input(()))
    heads = catalog.snapshot_heads()
    writer(log).append("u2", Input(()))
    other.append("other2", Input(()))
    assert {
        session: tuple(m.message_id for m in catalog.reader(session).read(through_seq=head))
        for session, head in heads.items()
    } == {"s": ("u1",), "other": ("other1",)}
    with pytest.raises(TypeError):
        heads["s"] = 100


@pytest.mark.asyncio
async def test_catalog_follow_discovers_new_sessions_and_closes(log, monkeypatch):
    catalog = log.catalog()
    feed = catalog.follow()
    assert await anext(feed) == {}
    observed = asyncio.Event()
    snapshot = catalog.snapshot_heads

    def read_and_signal():
        result = snapshot()
        observed.set()
        return result

    monkeypatch.setattr(catalog, "snapshot_heads", read_and_signal)
    pending = asyncio.create_task(anext(feed))
    await asyncio.wait_for(observed.wait(), 1)
    writer(log).append("first", Input(()))
    assert await asyncio.wait_for(pending, 1) == {"s": 0}
    # 通知在消费者处理上一份快照时到达，重读仍发现新会话。
    log.writer("new", author="app", source="timer", body_types=(Input,), content={}).append(
        "new-input", Input(())
    )
    assert await anext(feed) == {"new": 0, "s": 0}
    observed.clear()
    pending = asyncio.create_task(anext(feed))
    await asyncio.wait_for(observed.wait(), 1)
    log.close()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, 1)


def test_reader_ranges_keep_prefix_source_and_page_boundaries(log):
    inputs = writer(log)
    first = inputs.append("first", Input(()))
    other = writer(log, source="wake").append("wake", Input(()))
    output = writer(log, author="agent", bodies=(Output,))
    for index in range(1002):
        output.append(f"output-{index}", Output((), "complete"))
    reader = log.reader("s")
    head = reader.head()
    later = inputs.append("later", Input(()))
    before = tuple(log._connection.iterdump())
    assert reader.source_names() == frozenset({"conversation", "wake"})
    assert reader.latest_input("conversation", through_seq=head) == first
    assert reader.latest_input("conversation", through_seq=later.seq) == later
    assert reader.latest_input("wake", through_seq=head) == other
    assert reader.latest_input("missing", through_seq=head) is None
    assert reader.latest_input("conversation", through_seq=-1) is None
    bounded = reader.snapshot(after_seq=other.seq, through_seq=head)
    assert len(bounded) == 1002
    assert [message.seq for message in bounded] == list(range(other.seq + 1, head + 1))
    assert reader.snapshot(after_seq=head) == (later,)
    assert reader.snapshot(after_seq=head, through_seq=head) == ()
    assert tuple(log._connection.iterdump()) == before


def test_incremental_reader_decodes_only_new_messages(log, monkeypatch):
    import session.log as message_log

    inputs = writer(log)
    first = inputs.append("first", Input((ContentPart("text", "first"),)))
    reader = log.reader("s").incremental()
    decoded = []
    decode = message_log._message

    def read_message(row):
        decoded.append(row["id"])
        return decode(row)

    monkeypatch.setattr(message_log, "_message", read_message)
    assert reader.snapshot() == (first,)
    assert decoded == ["first"]
    decoded.clear()
    assert reader.snapshot(through_seq=first.seq) == (first,)
    assert reader.snapshot(after_seq=first.seq) == ()
    assert decoded == []
    second = inputs.append("second", Input(()))
    decoded.clear()
    assert reader.snapshot(after_seq=first.seq) == (second,)
    assert decoded == ["second"]
    decoded.clear()
    assert reader.snapshot() == (first, second)
    assert reader.snapshot(through_seq=first.seq) == (first,)
    assert decoded == []
    assert not log._connection.in_transaction


@pytest.mark.parametrize("operation", ["edit", "delete", "replace"])
def test_incremental_reader_reloads_external_changes_with_same_head(log, tmp_path, operation):
    inputs = writer(log)
    first = inputs.append("first", Input((ContentPart("text", "old"),)))
    last = inputs.append("last", Input(()))
    reader = log.reader("s").incremental()
    original = reader.snapshot()
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection, connection:
        if operation == "edit":
            connection.execute("UPDATE messages SET body=? WHERE id='first'",
                               (encode_body(Input((ContentPart("text", "edited"),))),))
        elif operation == "delete":
            connection.execute("DELETE FROM messages WHERE id='first'")
        else:
            connection.execute("UPDATE messages SET id='replacement' WHERE id='first'")
    assert reader.head() == last.seq
    # 即使先只请求尾部，后续完整读取也不能复用已失效的旧正文。
    assert reader.snapshot(after_seq=first.seq) == (last,)
    assert reader.snapshot() == log.reader("s").snapshot()
    assert reader.snapshot() != original
    assert original == (first, last)


def test_incremental_reader_does_not_keep_rolled_back_rows(log):
    inputs = writer(log)
    first = inputs.append("first", Input(()))
    reader = log.reader("s").incremental()
    assert reader.snapshot() == (first,)
    with pytest.raises(RuntimeError, match="rollback"):
        with log._connection:
            log._connection.execute("BEGIN")
            log._connection.execute(
                "INSERT INTO messages SELECT 'uncommitted',session_key,seq+1,ts,author,source,body,metadata "
                "FROM messages WHERE id='first'"
            )
            assert [message.message_id for message in reader.snapshot()] == ["first", "uncommitted"]
            raise RuntimeError("rollback")
    assert reader.snapshot() == (first,)
    second = inputs.append("committed", Input(()))
    assert reader.snapshot() == (first, second)


def test_incremental_reader_keeps_one_snapshot_during_external_edit(log, tmp_path, monkeypatch):
    log._connection.execute("PRAGMA journal_mode=WAL")
    inputs = writer(log)
    for index in range(1001):
        inputs.append(f"input-{index}", Input((ContentPart("text", "old"),)))
    reader = log.reader("s").incremental()
    read = reader.read
    edited = False

    def read_and_edit(**kwargs):
        nonlocal edited
        page = read(**kwargs)
        if not edited:
            edited = True
            with closing(sqlite3.connect(tmp_path / "sessions.db")) as connection, connection:
                connection.execute("UPDATE messages SET body=? WHERE id IN ('input-0','input-1000')",
                                   (encode_body(Input((ContentPart("text", "new"),))),))
        return page

    monkeypatch.setattr(reader, "read", read_and_edit)
    original = reader.snapshot()
    assert len(original) == 1001
    assert original[0].body.parts[0].value == original[-1].body.parts[0].value == "old"
    updated = reader.snapshot()
    assert updated[0].body.parts[0].value == updated[-1].body.parts[0].value == "new"
    assert updated == log.reader("s").snapshot()
    assert not log._connection.in_transaction


def test_live_message_reuse_checks_entire_row_and_does_not_keep_history(log):
    import gc
    import weakref
    import sqlite3
    from contextlib import closing
    inputs = log.writer("s", author="user", source="chat", body_types=(Input,), content={})
    inputs.append("reused", Input(()))
    first = log.reader("s").get("reused")
    second = log.reader("s").get("reused")
    assert second is first
    held = weakref.ref(first)
    with closing(sqlite3.connect(log._connection.execute("PRAGMA database_list").fetchone()[2])) as connection, connection:
        connection.execute('UPDATE messages SET author=? WHERE id=?', ("changed", "reused"))
    changed = log.reader("s").get("reused")
    assert changed.author == "changed" and first.author == "user"
    assert changed is not first
    del first, second
    gc.collect()
    assert held() is None
