from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugin_composition.models import LLMResponse
from agent.plugin_contracts.content import check_text
from plugins.models.projection import check_facts
from session.embedding_store import MessageEmbeddingStore
from session.log import MessageLog
from session.message import ContentPart, Input, Output
from tests.test_message_react import runtime
from tests_scenarios.contracts.oracles import assert_no_forbidden_writes, assert_rows_unchanged


def _seed_history(log: MessageLog) -> tuple:
    """通过真实 MessageLog writer 建立已结算历史。"""
    input_writer = log.writer(
        "s", author="user", source="conversation", body_types=(Input,),
        content={"text": check_text},
        metadata_keys=frozenset({"semantic_marker"}),
        update_metadata=lambda _body: {"semantic_marker": "history"},
    )
    output_writer = log.writer(
        "s", author="assistant", source="conversation", body_types=(Output,),
        content={"text": check_text, "model.facts": check_facts},
    )
    for index in range(3):
        input_writer.append(
            f"old-input-{index}",
            Input((ContentPart("text", f"old input {index}"),)),
        )
        output_writer.append(
            f"old-output-{index}",
            Output((ContentPart("text", f"old output {index}"),), "complete"),
        )
    return log.reader("s").snapshot()


def _text(message) -> str:
    part = message.body.parts[0]
    assert isinstance(part, ContentPart)
    assert isinstance(part.value, str)
    return part.value


def _vector_snapshot(store: MessageEmbeddingStore, messages) -> dict[str, list[float]]:
    """通过 embedding owner 读取指定历史消息的向量。"""
    values: dict[str, list[float]] = {}
    for message in messages:
        vector = store.get(
            message_id=message.message_id,
            content=_text(message),
            model="gate-model",
        )
        assert vector is not None
        values[message.message_id] = vector
    return values


def _message_rows(messages) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (message.message_id, message.session_id, message.seq, message.author,
         message.source, message.body)
        for message in messages
    )


def _embedding_rows(vectors: dict[str, list[float]]) -> tuple[tuple[object, ...], ...]:
    return tuple((message_id, tuple(vector)) for message_id, vector in sorted(vectors.items()))


@pytest.mark.asyncio
async def test_real_message_reply_preserves_history_embeddings_and_restart_seq(tmp_path: Path) -> None:
    """真实回复只投影历史，不改写原消息或向量。"""
    requests: list[object] = []

    async def complete(request):
        requests.append(request)
        return LLMResponse("ok")

    async def invoke(_key, _arguments):
        pytest.fail("this history oracle has no tool effects")

    db_path = tmp_path / "sessions.db"
    with_runtime = runtime(tmp_path, complete, invoke)
    async with with_runtime as (conversation, log, _models, _run):
        before = _seed_history(log)
        before_metadata = log.reader("s").metadata()
        embeddings = MessageEmbeddingStore(db_path)
        for message in before:
            embeddings.upsert(
                message_id=message.message_id,
                content=_text(message),
                model="gate-model",
                embedding=[float(message.seq), 1.0],
            )
        before_vectors = _vector_snapshot(embeddings, before)
        before_message_rows = _message_rows(before)
        before_embedding_rows = _embedding_rows(before_vectors)
        statements: list[str] = []
        log._connection.set_trace_callback(statements.append)  # pyright: ignore[reportPrivateUsage]
        await conversation.accept("current", Input((ContentPart("text", "continue"),)))
        task = await conversation.start(_run)
        assert task is not None
        await task.join()
        after = log.reader("s").snapshot()
        log._connection.set_trace_callback(None)  # pyright: ignore[reportPrivateUsage]

        assert_rows_unchanged(
            before_message_rows,
            _message_rows(after[: len(before)]),
            state_name="sessions.db/messages",
        )
        assert len(after) == len(before) + 2
        assert log.reader("s").metadata() == before_metadata
        assert len(requests) == 1
        assert "old input 0" in str(requests[0])
        assert "old output 2" in str(requests[0])
        assert_no_forbidden_writes(
            statements,
            tables=("messages", "message_embeddings"),
        )
        after_vectors = _vector_snapshot(embeddings, before)
        assert_rows_unchanged(
            before_embedding_rows,
            _embedding_rows(after_vectors),
            state_name="message_embeddings",
        )
        embeddings.close()
        highwater = after[-1].seq

    reopened = MessageLog(db_path)
    try:
        assert reopened.reader("s").snapshot() == after
        writer = reopened.writer(
            "s", author="user", source="conversation", body_types=(Input,),
            content={"text": check_text},
        )
        appended = writer.append("after-reopen", Input((ContentPart("text", "after restart"),)))
        assert appended.seq == highwater + 1
        assert reopened.reader("s").head() == appended.seq
    finally:
        reopened.close()


def test_history_oracle_rejects_historical_delete_mutant(tmp_path: Path) -> None:
    """追加 oracle 同时发现消息和向量被删除。"""
    log = MessageLog(tmp_path / "sessions.db")
    try:
        before = _seed_history(log)
        embeddings = MessageEmbeddingStore(tmp_path / "sessions.db")
        for message in before:
            embeddings.upsert(
                message_id=message.message_id,
                content=_text(message),
                model="gate-model",
                embedding=[float(message.seq), 1.0],
            )
        before_vectors = _vector_snapshot(embeddings, before)
        before_message_rows = _message_rows(before)
        before_embedding_rows = _embedding_rows(before_vectors)
        with log._lock:  # pyright: ignore[reportPrivateUsage]
            log._connection.execute("DELETE FROM message_embeddings WHERE message_id = ?", (before[0].message_id,))  # pyright: ignore[reportPrivateUsage]
            log._connection.execute("DELETE FROM messages WHERE id = ?", (before[0].message_id,))  # pyright: ignore[reportPrivateUsage]
            log._connection.commit()  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AssertionError, match="既有行发生删改"):
            assert_rows_unchanged(
                before_message_rows,
                _message_rows(log.reader("s").snapshot()),
                state_name="sessions.db/messages",
            )
        current_vectors = {
            message.message_id: vector
            for message in before[1:]
            for vector in [_vector_snapshot(embeddings, (message,))[message.message_id]]
        }
        with pytest.raises(AssertionError, match="既有行发生删改"):
            assert_rows_unchanged(
                before_embedding_rows,
                _embedding_rows(current_vectors),
                state_name="message_embeddings",
            )
        assert before_vectors[before[0].message_id] == [0.0, 1.0]
        embeddings.close()
    finally:
        log.close()
