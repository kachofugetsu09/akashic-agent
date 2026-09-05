from session.message import ContentReferences
from dataclasses import replace
from contextlib import closing
from datetime import datetime
import os
import sqlite3

import numpy as np
import pytest

from plugins.akasha.application.cycle import MemoryCycle
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.consumption import Consumption, LegacyPrefix, turns_digest
from plugins.akasha.infrastructure.persistence import (
    load_consumption, load_memory_state, logical_state_sha256, write_memory_database,
)
from plugins.akasha.projection import applied_source, dialogue_turn, project_samples
from plugins.turn_projection.plugin import TurnProjection
from session.embedding_store import MessageEmbeddingStore, MessageEmbeddings
from session.log import MessageLog
from session.message import ContentPart, Input, Output, Control


@pytest.fixture
def conversation(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    store = MessageEmbeddingStore(tmp_path / "sessions.db")
    def append(kind, identity, body, source="chat"):
        return log.writer("s", author="actor", source=source, body_types=(kind,),
                          content={"text": lambda part: ContentReferences()}).append(identity, body)
    def text(message):
        return "".join(part.value for part in message.body.parts if part.kind == "text")
    records = MessageEmbeddings(log).bind(text)
    def add(identity, value, body=Input, source="chat"):
        parts = (ContentPart("text", value),)
        message = append(body, identity, body(parts) if body is Input else body(parts, "complete"), source)
        records.save(message, model="fixed", embedding=[0.6, 0.8])
        return message
    try:
        yield log, append, add, text, records
    finally:
        store.close()
        log.close()


def publish(path, cycle, state):
    return write_memory_database(
        path, turns=cycle.turns, graph=cycle.graph, events=cycle.events,
        evidence=cycle.evidence, captures=[], context=cycle.context,
        burst_members=cycle.burst_members, config=cycle.config, metadata={},
        recalls=cycle.recalls, consumption=state,
    )


def restore(path, turns):
    graph, events, evidence, context, recalls, burst_members = load_memory_state(
        path, turns=turns, config=MemoryConfig(), source_index_sha256=None,
    )
    return MemoryCycle.restore(
        config=MemoryConfig(), turns=turns, graph=graph, context=context,
        events=events, evidence=evidence, recalls=recalls, burst_members=burst_members,
    )


def test_interrupted_inputs_learn_one_real_graph_node_and_restore_without_replay(conversation, tmp_path):
    log, append, add, text, records = conversation
    add("u1", "first input")
    append(Control, "pause1", Control("pause", 0))
    add("u2", "second input")
    add("wake", "independent observation", Output, "wake")
    append(Control, "pause2", Control("pause", 2))
    add("u3", "third input")
    add("answer", "full answer", Output)
    sample = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: source == "chat")[0]
    turn = dialogue_turn(sample, node_id=0, previous=None, text=text, embeddings=records,
                         embedding_model="fixed", dimension=2)
    assert turn.user_text == "first input\n\nsecond input\n\nthird input"
    assert turn.assistant_text == "full answer"
    np.testing.assert_allclose(turn.user_dense, [0.6, 0.8])
    state = Consumption(legacy_prefix=LegacyPrefix(count=0, index_state_sha256="0" * 64,
                                                  turns_digest=turns_digest([])), cutover_heads=())
    state = state.append(applied_source(sample, learning_binding="exact-projection"))
    cycle = MemoryCycle()
    cycle.commit(turn, None)
    path = tmp_path / "akasha.db"
    publish(path, cycle, state)
    before = logical_state_sha256(path)
    loaded = load_consumption(path)
    assert loaded == state
    restored = restore(path, [turn])
    assert restored.state_version == 1
    publish(path, restored, loaded)
    assert logical_state_sha256(path) == before
    assert [ref[1] for ref in loaded.applied[0].members] == ["u1", "u2", "u3", "answer"]
    with pytest.raises(ValueError, match="重复学习"):
        loaded.append(loaded.applied[0])
    changed = loaded.model_dump(mode="json")
    changed["applied"][0]["learning_binding"] = "other-version"
    with closing(sqlite3.connect(path)) as db:
        import json
        db.execute("UPDATE metadata SET value=? WHERE key='consumer_state_json'", (json.dumps(changed),))
        db.commit()
    assert logical_state_sha256(path) != before


def test_missing_embedding_fails_without_learning_or_mutating_cache(conversation):
    log, append, add, text, records = conversation
    add("u", "question")
    append(Output, "a", Output((ContentPart("text", "answer"),), "complete"))
    sample = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)[0]
    before = log.reader("s").snapshot()
    with pytest.raises(ValueError, match="缺少固定 embedding: a"):
        dialogue_turn(sample, node_id=0, previous=None, text=text, embeddings=records,
                      embedding_model="fixed", dimension=2)
    assert log.reader("s").snapshot() == before
    assert records.read(sample.ending, model="fixed", dimension=2) is None


def test_cutover_preserves_old_graph_and_publish_failure_keeps_old_snapshot(conversation, tmp_path, monkeypatch):
    log, append, add, text, records = conversation
    add("u", "old question")
    add("a", "old answer", Output)
    old = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)[0]
    turn = dialogue_turn(old, node_id=0, previous=None, text=text, embeddings=records,
                         embedding_model="fixed", dimension=2)
    cycle = MemoryCycle()
    cycle.commit(turn, None)
    path = tmp_path / "akasha.db"
    publish(path, cycle, None)
    old_graph = logical_state_sha256(path)
    state = Consumption(legacy_prefix=LegacyPrefix(count=1, index_state_sha256="1" * 64,
                                                  turns_digest=turns_digest([turn])),
                        cutover_heads=tuple(sorted(log.catalog().snapshot_heads().items())))
    publish(path, restore(path, [turn]), state)
    assert load_consumption(path).legacy_prefix.count == 1
    assert restore(path, [turn]).state_version == 1
    with pytest.raises(ValueError, match="旧 writer"):
        publish(path, cycle, None)
    with pytest.raises(ValueError, match="切换前"):
        state.append(applied_source(old, learning_binding="new"))
    with pytest.raises(ValueError, match="旧学习前缀"):
        restore(path, [replace(turn, user_text="changed")])
    assert old_graph != logical_state_sha256(path)  # graph plus new provenance
    add("u2", "new question")
    add("a2", "new answer", Output)
    sample = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)[1]
    next_turn = dialogue_turn(sample, node_id=1, previous=datetime.fromisoformat(turn.committed_at),
                              text=text, embeddings=records, embedding_model="fixed", dimension=2)
    next_state = state.append(applied_source(sample, learning_binding="new"))
    cycle.commit(next_turn, None)
    before = path.read_bytes()
    def fail_replace(source, target):
        raise OSError("injected publication failure")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        publish(path, cycle, next_state)
    assert path.read_bytes() == before
    assert load_consumption(path) == state
    assert restore(path, [turn]).state_version == 1
    assert list(tmp_path.glob("akasha.db.*.tmp"))  # named recovery material


def test_real_consumer_is_idempotent_after_restart_and_stops_after_uncertain_publish(conversation, tmp_path, monkeypatch):
    from plugins.akasha.application.consumer import MessageConsumer
    from plugins.akasha.projection import restore_sample
    import plugins.akasha.infrastructure.persistence as persistence
    log, append, add, text, records = conversation
    def build(sample, node=0, previous=None):
        return dialogue_turn(sample, node_id=node, previous=previous, text=text,
                             embeddings=records, embedding_model='fixed', dimension=2)
    add('u1', 'question one')
    add('a1', 'answer one', Output)
    first = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)[0]
    entry = applied_source(first, learning_binding='fixed')
    state = Consumption(legacy_prefix=LegacyPrefix(count=0, index_state_sha256='0' * 64,
                                                  turns_digest=turns_digest([])), cutover_heads=())
    path = tmp_path / 'consumer.db'
    consumer = MessageConsumer(path, turns=[], state=state, config=MemoryConfig())
    try:
        assert consumer.apply(build(first), entry)
        before = logical_state_sha256(path)
        assert not consumer.apply(build(first), entry)
        assert consumer.cycle.state_version == 1
        assert logical_state_sha256(path) == before
    finally:
        consumer.close()
    state = load_consumption(path)
    restored_sample = restore_sample(log.catalog(), TurnProjection(), state.applied[0])
    restored_turn = build(restored_sample)
    consumer = MessageConsumer(path, turns=[restored_turn], state=state, config=MemoryConfig())
    assert not consumer.apply(restored_turn, entry)
    add('u2', 'question two')
    add('a2', 'answer two', Output)
    second = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)[1]
    second_entry = applied_source(second, learning_binding='fixed')
    second_turn = build(second, 1, datetime.fromisoformat(restored_turn.committed_at))
    original_replace = persistence.os.replace
    def publish_then_fail(source, target):
        original_replace(source, target)
        raise OSError('lost publication acknowledgement')
    with monkeypatch.context() as patch:
        patch.setattr(persistence.os, 'replace', publish_then_fail)
        with pytest.raises(OSError, match='acknowledgement'):
            consumer.apply(second_turn, second_entry)
    with pytest.raises(RuntimeError, match='重新读取'):
        consumer.apply(second_turn, second_entry)
    consumer.close()
    # replace 已提交：重新读取实际文件发现第二次学习已完成，不补学一次。
    state = load_consumption(path)
    assert len(state.applied) == 2
    consumer = MessageConsumer(path, turns=[restored_turn, second_turn], state=state, config=MemoryConfig())
    try:
        assert not consumer.apply(second_turn, second_entry)
        assert consumer.cycle.state_version == 2
    finally:
        consumer.close()
    with pytest.raises(RuntimeError, match='已关闭'):
        consumer.apply(second_turn, second_entry)


def test_reprojection_rejects_changed_members_and_unknown_consumer_version(conversation):
    from plugins.akasha.infrastructure.consumption import Consumption
    from plugins.akasha.projection import restore_sample
    from pydantic import ValidationError
    log, append, add, text, records = conversation
    add('u1', 'one')
    add('u2', 'two')
    add('a', 'answer', Output)
    sample = project_samples(log.catalog(), TurnProjection(), include=lambda session, source: True)[0]
    entry = applied_source(sample, learning_binding='fixed')
    class WrongProjection:
        def project(self, messages, source):
            return tuple(replace(turn, message_ids=turn.message_ids[1:])
                         for turn in TurnProjection().project(messages, source))
    with pytest.raises(ValueError, match='出处发生改变'):
        restore_sample(log.catalog(), WrongProjection(), entry)
    state = Consumption(legacy_prefix=LegacyPrefix(count=0, index_state_sha256='0' * 64,
                                                  turns_digest=turns_digest([])), cutover_heads=())
    payload = state.model_dump_json().replace('"version":1', '"version":2')
    with pytest.raises(ValidationError):
        Consumption.model_validate_json(payload)
