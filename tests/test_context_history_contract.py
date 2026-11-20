from __future__ import annotations

from pathlib import Path

import pytest

from plugins.models.contract import LLMResponse
from plugins.content.plugin import check_text
from plugins.models.projection import check_facts
from plugins.ledger.embedding_store import MessageEmbeddings, EmbeddingRecords
from plugins.ledger.log import MessageLog
from plugins.ledger.contract import ContentPart, Input, Output
from tests.support.message_react import runtime


def assert_rows_unchanged(
    before: tuple[tuple[object, ...], ...],
    after: tuple[tuple[object, ...], ...],
    *,
    state_name: str,
) -> None:
    """断言权威状态中的既有行没有被删改。"""
    if list(after) != list(before):
        raise AssertionError(f"{state_name} 的既有行发生删改")


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


def _vector_snapshot(store: EmbeddingRecords, messages) -> dict[str, tuple[float, ...]]:
    """通过 embedding owner 读取指定历史消息的向量。"""
    values: dict[str, tuple[float, ...]] = {}
    for message in messages:
        vector = store.read(message, model="gate-model", dimension=2)
        assert vector is not None
        values[message.message_id] = vector
    return values


def _message_rows(messages) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (message.message_id, message.session_id, message.seq, message.author,
         message.source, message.body)
        for message in messages
    )


def _embedding_rows(vectors: dict[str, tuple[float, ...]]) -> tuple[tuple[object, ...], ...]:
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
        embedding_store = MessageEmbeddings(log, tmp_path / "sessions-derived.db")
        embeddings = embedding_store.bind(_text)
        for message in before:
            embeddings.save(
                message,
                model="gate-model",
                embedding=[float(message.seq), 1.0],
            )
        before_vectors = _vector_snapshot(embeddings, before)
        before_message_rows = _message_rows(before)
        before_embedding_rows = _embedding_rows(before_vectors)
        await conversation.accept("current", Input((ContentPart("text", "continue"),)))
        task = await conversation.start(_run)
        assert task is not None
        await task.join()
        after = log.reader("s").snapshot()

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
        after_vectors = _vector_snapshot(embeddings, before)
        assert_rows_unchanged(
            before_embedding_rows,
            _embedding_rows(after_vectors),
            state_name="message_embeddings",
        )
        embedding_store.close()
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
