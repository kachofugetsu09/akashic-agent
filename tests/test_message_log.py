from plugins.ledger.contract import ContentReferences
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from plugins.ledger.log import MessageConflict, MessageLog, SessionAttributes
from plugins.ledger.contract import CallRef, ContentPart, Input, Output, ToolCall, ToolResult

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

def test_session_scope_is_fixed_at_admission_and_absent_for_old_sessions(log, tmp_path):
    _ = writer(log).append("m1", Input(()))
    scoped = SessionAttributes.scoped({"project": "p_1"})
    assert log.ensure_session("web:a", scoped) == scoped
    assert log.ensure_session("web:a", scoped) == scoped
    with pytest.raises(MessageConflict):
        _ = log.ensure_session("web:a", SessionAttributes.scoped({"project": "p_2"}))
    with pytest.raises(MessageConflict):
        _ = log.ensure_session("s", scoped)
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as raw:
        rows = dict(raw.execute("SELECT key, attributes FROM sessions").fetchall())
    assert rows["s"] == '{"learning": "eligible", "visibility": "listed"}'
    assert SessionAttributes().dimension("project") == "default"


def _call_message(log, message_id="call", source="conversation", parts=None):
    log.save_binding("tool", {"artifact": "immutable-revision"})
    outputs = writer(
        log, source=source, author="agent", bodies=(Output,),
        check_call=lambda call: None,
    )
    return outputs.append(
        message_id, Output(parts or (ToolCall("tool", {}),), "continue")
    )


def _result_writer(log, ref, source="conversation"):
    return writer(
        log, source=source, author="tool", bodies=(ToolResult,), call_ref=ref,
    )


def test_tool_result_commits_against_real_tool_call(log):
    _ = _call_message(log)
    ref = CallRef("call", 0)
    result = _result_writer(log, ref).append("result", ToolResult(ref, "success", ()))
    assert result.seq == 1
    with pytest.raises(MessageConflict, match="已经有结果消息"):
        _result_writer(log, ref).append("result2", ToolResult(ref, "success", ()))


def test_tool_result_rejects_unknown_or_foreign_call(log):
    _ = _call_message(log)
    missing = CallRef("missing", 0)
    with pytest.raises(ValueError, match="调用不在 writer 获授的 Session/source 内"):
        _result_writer(log, missing).append("r1", ToolResult(missing, "success", ()))
    foreign = CallRef("foreign-call", 0)
    with pytest.raises(ValueError, match="调用不在 writer 获授的 Session/source 内"):
        _result_writer(log, foreign, source="tools").append(
            "r2", ToolResult(foreign, "success", ())
        )
    # 调用真实存在但属于另一来源：结果 writer 不得跨来源写入。
    _ = _call_message(log, "other-call", source="other")
    other = CallRef("other-call", 0)
    with pytest.raises(ValueError, match="调用不在 writer 获授的 Session/source 内"):
        _result_writer(log, other).append("r3", ToolResult(other, "success", ()))


def test_tool_result_rejects_non_call_target(log):
    _ = writer(log).append("input", Input(()))
    _ = _call_message(
        log, "mixed",
        parts=(ToolCall("tool", {}), ContentPart("text", "说明")),
    )
    input_ref = CallRef("input", 0)
    with pytest.raises(ValueError, match="call_ref 未指向真实工具调用"):
        _result_writer(log, input_ref).append("r1", ToolResult(input_ref, "success", ()))
    text_ref = CallRef("mixed", 1)
    with pytest.raises(ValueError, match="call_ref 未指向真实工具调用"):
        _result_writer(log, text_ref).append("r2", ToolResult(text_ref, "success", ()))
    out_of_range = CallRef("mixed", 2)
    with pytest.raises(ValueError, match="call_ref 未指向真实工具调用"):
        _result_writer(log, out_of_range).append(
            "r3", ToolResult(out_of_range, "success", ())
        )
