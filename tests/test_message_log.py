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
