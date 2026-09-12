from session.message import ContentReferences
from dataclasses import replace
from contextlib import closing
import sqlite3

import pytest

from session.embedding_store import MessageEmbeddingStore, MessageEmbeddings
from session.log import MessageConflict, MessageLog
from session.message import ContentPart, Input


def message_text(message):
    value = message.body.parts[0].value
    assert isinstance(value, str)
    return value


def test_message_vectors_reuse_legacy_rows_and_never_overwrite_fixed_facts(tmp_path):
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    store = MessageEmbeddingStore(path)
    try:
        writer = log.writer("s", author="user", source="chat", body_types=(Input,), content={"text": lambda part: ContentReferences()})
        message = writer.append("u", Input((ContentPart("text", "actual"),)))
        store.upsert(message_id="u", content="actual", model="frozen-model", embedding=[0.25, 0.5])
        records = MessageEmbeddings(log).bind(message_text)
        assert records.read(message, model="frozen-model", dimension=2) == (0.25, 0.5)
        records.save(message, model="frozen-model", embedding=[0.25, 0.5])
        with pytest.raises(MessageConflict):
            records.save(message, model="frozen-model", embedding=[0.5, 0.5])
        with pytest.raises(MessageConflict):
            records.read(replace(message, body=Input((ContentPart("text", "forged"),))), model="frozen-model", dimension=2)
        with pytest.raises(ValueError, match="不匹配"):
            records.read(message, model="frozen-model", dimension=3)
        other_projection = MessageEmbeddings(log).bind(lambda m: "different")
        with pytest.raises(ValueError, match="不匹配"):
            other_projection.read(message, model="frozen-model", dimension=2)
        assert records.read(message, model="new-model", dimension=2) is None
        records.save(message, model="new-model", embedding=[1.0, 0.0])
        assert records.read(message, model="frozen-model", dimension=2) == (0.25, 0.5)
        assert log.reader("s").snapshot() == (message,)
    finally:
        store.close()
        log.close()


def test_missing_schema_and_corrupt_vectors_fail_without_reembedding(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    message = log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append("u", Input(()))
    with closing(sqlite3.connect(tmp_path / "sessions.db")) as db, db:
        db.execute("DROP TABLE message_embeddings")
    records = MessageEmbeddings(log).bind(lambda m: "")
    try:
        with pytest.raises(sqlite3.OperationalError, match="message_embeddings"):
            records.read(message, model="model", dimension=2)
        with pytest.raises(ValueError):
            records.save(message, model="model", embedding=[float("nan"), 0.0])
        with pytest.raises(RuntimeError, match="candidate"):
            MessageEmbeddings(None).bind(lambda m: "")
    finally:
        log.close()
