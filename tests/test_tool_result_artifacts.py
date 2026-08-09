from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from agent.control.models import TurnRecord, TurnStatus
from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextCompactor,
    ContextPayloadSegments,
)
from agent.model_runtime.tool_result_projection import tool_result_placeholder
from agent.tools.base import ToolExecutionContext, tool_execution_context_scope
from agent.tools.tool_result import ReadToolResultTool
from session.store import SessionStore


class _Provider:
    runtime_id = "test"
    context_window = 100_000
    max_output_tokens = 1_000

    def resolve_model(self, model: str) -> str:
        return model

    def estimate_context_tokens(self, messages, tools) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    def estimate_appended_message_tokens(self, messages) -> int:
        return self.estimate_context_tokens(messages, [])


def _turn(turn_id: str, session_key: str) -> TurnRecord:
    return TurnRecord(
        id=turn_id,
        thread_id=session_key,
        status=TurnStatus.QUEUED,
        input="test",
        metadata={},
        items=[],
        usage=None,
        error=None,
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


def _tool_batch(call_id: str, content: str) -> tuple[dict, dict]:
    return (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "read_file",
            "content": content,
        },
    )


def test_projection_keeps_latest_batch_raw_and_masks_older_batches() -> None:
    first = _tool_batch("call:first", "A" * 9_000)
    second = _tool_batch("call:second", "B" * 9_000)
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(),
        current_anchor=({"role": "user", "content": "start"},),
        active_batches=(first, second),
    )
    compactor = ContextCompactor(
        provider=_Provider(),  # type: ignore[arg-type]
        model="model",
        scope_id="session",
        payload_segments=segments,
        tool_result_artifacts={
            "call:first": "artifact:first",
            "call:second": "artifact:second",
        },
    )

    projected = compactor.project_tool_results(segments.flatten())

    assert projected.messages[2]["content"] == tool_result_placeholder(
        "artifact:first"
    )
    assert projected.messages[4]["content"] == "B" * 9_000
    assert projected.masked_result_count == 1
    assert segments.active_batches[0][1]["content"] == "A" * 9_000


def test_projection_masks_committed_results_but_preserves_attempt_replay() -> None:
    archived = _tool_batch("call:old", "old-result")
    unit = CommittedContextUnit(
        source_from_seq=1,
        consolidated_through_seq=1,
        source_message_ids=("message:1",),
        messages=archived,
    )
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(unit,),
        current_anchor=({"role": "user", "content": "next"},),
    )
    masked = ContextCompactor(
        provider=_Provider(),  # type: ignore[arg-type]
        model="model",
        scope_id="session",
        payload_segments=segments,
        tool_result_artifacts={"call:old": "artifact:old"},
    ).project_tool_results(segments.flatten())
    replayed = ContextCompactor(
        provider=_Provider(),  # type: ignore[arg-type]
        model="model",
        scope_id="session",
        payload_segments=segments,
        tool_result_artifacts={"call:old": "artifact:old"},
        protected_tool_result_calls={"call:old"},
    ).project_tool_results(segments.flatten())

    assert masked.messages[1]["content"] == tool_result_placeholder("artifact:old")
    assert replayed.messages[1]["content"] == "old-result"


def test_store_archives_idempotently_and_read_tool_records_success(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.create_session(key="session:one")
    store.create_turn(_turn("turn:one", "session:one"))
    artifact = store.archive_tool_result(
        session_key="session:one",
        turn_id="turn:one",
        call_id="call:one",
        tool_name="web_fetch",
        content="abcdefghij",
    )
    repeated = store.archive_tool_result(
        session_key="session:one",
        turn_id="turn:one",
        call_id="call:one",
        tool_name="web_fetch",
        content="abcdefghij",
    )

    with tool_execution_context_scope(
        ToolExecutionContext(
            origin_session_key="session:one",
            turn_id="turn:one",
        )
    ):
        payload = json.loads(
            asyncio.run(ReadToolResultTool(store).execute(artifact.id, offset=2, limit=4))
        )

    assert repeated == artifact
    assert payload == {
        "artifact_id": artifact.id,
        "offset": 2,
        "next_offset": 6,
        "total_chars": 10,
        "eof": False,
        "content": "cdef",
    }
    database = sqlite3.connect(db_path)
    try:
        reads = database.execute(
            "SELECT artifact_id, session_key, turn_id, offset, requested_limit, "
            "returned_chars FROM tool_result_reads"
        ).fetchall()
    finally:
        database.close()
    assert reads == [(artifact.id, "session:one", "turn:one", 2, 4, 4)]
    store.close()


def test_cross_session_read_fails_without_audit_event(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    for suffix in ("one", "two"):
        store.create_session(key=f"session:{suffix}")
        store.create_turn(_turn(f"turn:{suffix}", f"session:{suffix}"))
    artifact = store.archive_tool_result(
        session_key="session:one",
        turn_id="turn:one",
        call_id="call:one",
        tool_name="web_fetch",
        content="secret",
    )

    with tool_execution_context_scope(
        ToolExecutionContext(
            origin_session_key="session:two",
            turn_id="turn:two",
        )
    ):
        with pytest.raises(PermissionError, match="不属于当前 session"):
            asyncio.run(ReadToolResultTool(store).execute(artifact.id))

    database = sqlite3.connect(db_path)
    try:
        count = database.execute("SELECT COUNT(1) FROM tool_result_reads").fetchone()[0]
    finally:
        database.close()
    assert count == 0
    store.close()


def test_session_cascade_removes_artifacts_and_reads_after_backup(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.create_session(key="session:one")
    store.create_turn(_turn("turn:one", "session:one"))
    artifact = store.archive_tool_result(
        session_key="session:one",
        turn_id="turn:one",
        call_id="call:one",
        tool_name="read_file",
        content="payload",
    )
    store.read_tool_result(
        session_key="session:one",
        reader_turn_id="turn:one",
        artifact_id=artifact.id,
        offset=0,
        limit=4,
    )

    audit = store.delete_session_with_audit("session:one", cascade=True)

    assert audit.backup_path is not None
    backup = sqlite3.connect(audit.backup_path)
    try:
        assert backup.execute("SELECT COUNT(1) FROM tool_result_artifacts").fetchone() == (
            1,
        )
        assert backup.execute("SELECT COUNT(1) FROM tool_result_reads").fetchone() == (
            1,
        )
    finally:
        backup.close()
    database = sqlite3.connect(db_path)
    try:
        assert database.execute("SELECT COUNT(1) FROM tool_result_artifacts").fetchone() == (
            0,
        )
        assert database.execute("SELECT COUNT(1) FROM tool_result_reads").fetchone() == (
            0,
        )
    finally:
        database.close()
    store.close()
