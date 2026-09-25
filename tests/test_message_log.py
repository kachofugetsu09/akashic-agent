from session.message import ContentReferences
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from session.log import MessageConflict, MessageLog
from session.message import Input, Output, ToolCall

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
