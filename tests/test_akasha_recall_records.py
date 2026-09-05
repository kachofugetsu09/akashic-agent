from datetime import UTC, datetime

import pytest

from plugins.akasha.recalls import ContextSource, Hit, Recall, RecallRecords
from session.log import MessageConflict, MessageLog, OwnerTransaction
from session.message import ContentPart, ContentReferences, Input, Output


def record(head):
    return Recall(
        learning_binding="fixed-learning", graph_version=1,
        source=ContextSource(session_id="s", source="chat", through_seq=head),
        timestamp=datetime(2026, 9, 5, tzinfo=UTC), limit=10,
        hits=(Hit(node_id=0, session_id="s", message_ids=("u1", "u2", "a"), score=0.9,
                  lane="dense", sources=("direct_dense",)),),
        active_basin_count=0, pushes=1, residual_l1=0.0,
    )


def test_retrieval_evidence_survives_new_messages_and_restart_without_a_learning_graph(tmp_path):
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    try:
        inputs = log.writer("s", author="user", source="chat", body_types=(Input,),
                            content={"text": lambda part: ContentReferences()})
        outputs = log.writer("s", author="assistant", source="chat", body_types=(Output,),
                             content={"text": lambda part: ContentReferences()})
        inputs.append("u1", Input((ContentPart("text", "first input"),)))
        inputs.append("u2", Input((ContentPart("text", "correction"),)))
        outputs.append("a", Output((ContentPart("text", "answer"),), "complete"))
        inputs.append("query", Input((ContentPart("text", "recall it"),)))
        before = log.reader("s").snapshot()
        records = RecallRecords(log.owner("akasha-fixture"))
        original = record(log.reader("s").head())
        records.save("actual-query", original)
        assert log.reader("s").snapshot() == before
        inputs.append("later", Input((ContentPart("text", "later interrupt"),)))
        with pytest.raises(MessageConflict):
            records.save("actual-query", record(log.reader("s").head()))
        assert records.read("actual-query") == original
    finally:
        log.close()
    restored = MessageLog(path)
    try:
        records = RecallRecords(restored.owner("akasha-fixture"))
        actual = records.read("actual-query")
        assert actual == original
        assert [restored.reader(hit.session_id).get(identity).message_id
                for hit in actual.hits for identity in hit.message_ids] == ["u1", "u2", "a"]
        assert records.read("unknown-query") is None
    finally:
        restored.close()


def test_failed_retrieval_record_publication_leaves_no_reference(tmp_path, monkeypatch):
    log = MessageLog(tmp_path / "sessions.db")
    records = RecallRecords(log.owner("akasha-fixture"))
    save = OwnerTransaction.save
    def failed(self, *args, **kwargs):
        save(self, *args, **kwargs)
        raise OSError("storage failed after insert")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "save", failed)
            with pytest.raises(OSError, match="storage failed"):
                records.save("query", record(0))
        assert records.read("query") is None
        records.save("query", record(0))
        assert records.read("query") is not None
    finally:
        log.close()


def test_actual_retrieval_keeps_all_interrupted_members_after_graph_advances(tmp_path):
    from dataclasses import replace
    from plugins.akasha.application.consumer import MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig
    from plugins.akasha.infrastructure.consumption import Consumption, LegacyPrefix, turns_digest
    from plugins.akasha.infrastructure.persistence import logical_state_sha256
    from plugins.akasha.learning import Learning
    from plugins.akasha.projection import applied_source, dialogue_turn, project_samples
    from plugins.akasha.recalls import select_hits
    from plugins.turn_projection.plugin import TurnProjection
    from session.embedding_store import MessageEmbeddings
    from session.message import Control
    log = MessageLog(tmp_path / "sessions.db")
    consumer = None
    try:
        rule = Learning(TurnProjection(), owner="akasha")
        vectors = MessageEmbeddings(log).bind(rule.text)
        def add(identity, text, kind=Input):
            parts = (ContentPart("text", text),)
            message = log.writer("s", author="test", source="chat", body_types=(kind,),
                                 content={"text": lambda part: ContentReferences()}).append(
                                     identity, Input(parts) if kind is Input else Output(parts, "complete"))
            vectors.save(message, model="fixed", embedding=[0.6, 0.8])
            return message
        state = Consumption(legacy_prefix=LegacyPrefix(count=0, index_state_sha256="0" * 64,
                                                       turns_digest=turns_digest([])), cutover_heads=())
        consumer = MessageConsumer(tmp_path / "memory.db", turns=[], state=state, config=MemoryConfig())
        add("u1", "first input")
        log.writer("s", author="user", source="chat", body_types=(Control,), content={}).append(
            "pause", Control("pause", 0))
        add("u2", "correction")
        add("a", "complete answer", Output)
        def turn():
            sample = project_samples(log.catalog(), rule.projection, include=lambda session, source: True)[-1]
            material = dialogue_turn(sample, node_id=consumer.cycle.state_version,
                previous=None if not consumer.cycle.turns else datetime.fromisoformat(consumer.cycle.turns[-1].committed_at),
                text=rule.text, embeddings=vectors, embedding_model="fixed", dimension=2)
            return material, applied_source(sample, learning_binding="fixed-learning")
        first, entry = turn()
        consumer.apply(first, entry)
        query = add("query", "recall the correction")
        cue = replace(first, node_id=1, turn_id=query.message_id, user_message_id=query.message_id,
                      user_seq=query.seq, started_at=query.recorded_at.isoformat(),
                      committed_at=query.recorded_at.isoformat(), user_text=rule.text(query),
                      assistant_text="", assistant_message_id="", assistant_dense=None, assistant_terms=(),
                      inter_gap_seconds=(query.recorded_at - datetime.fromisoformat(first.committed_at)).total_seconds())
        graph_before = logical_state_sha256(tmp_path / "memory.db")
        ticket = consumer.cycle.retrieve(cue)
        hits = select_hits(consumer.cycle.turns, consumer.state, ticket, cue,
                           inhibited=consumer.cycle.inhibited_nodes, limit=10)
        assert len(hits) == 1
        assert hits[0].message_ids == ("u1", "u2", "a")
        recalls = RecallRecords(log.owner("akasha-fixture"))
        observed = Recall(learning_binding="fixed-learning", graph_version=ticket.state_version,
            source=ContextSource(session_id="s", source="chat", through_seq=query.seq),
            timestamp=query.recorded_at, limit=10, hits=hits,
            active_basin_count=ticket.completion.active_basin_count,
            pushes=ticket.completion.pushes, residual_l1=ticket.completion.residual_l1)
        recalls.save("query", observed)
        assert logical_state_sha256(tmp_path / "memory.db") == graph_before
        add("later-answer", "later learned answer", Output)
        second, entry = turn()
        consumer.apply(second, entry)
        assert consumer.cycle.state_version == 2
        assert recalls.read("query") == observed
        with pytest.raises(ValueError, match="图版本"):
            select_hits(consumer.cycle.turns, consumer.state, ticket, cue,
                        inhibited=consumer.cycle.inhibited_nodes, limit=10)
    finally:
        if consumer is not None:
            consumer.close()
        log.close()


def test_inspector_keeps_all_hit_members_and_marks_only_presented_messages(tmp_path):
    from contextlib import closing
    from plugins.akasha.inspector import RecallInspector

    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        for identity, text, kind in (("u1", "first input", Input), ("u2", "correction", Input),
                                     ("a", "answer", Output)):
            parts = (ContentPart("text", text),)
            log.writer("s", author="test", source="chat", body_types=(kind,),
                       content={"text": lambda part: ContentReferences()}).append(
                identity, Input(parts) if kind is Input else Output(parts, "complete"))
        records = RecallRecords(log.owner("akasha"))
        records.save("query", record(2).model_copy(update={"presented_message_ids": ("u2",)}))
    with closing(MessageLog(path)) as log:
        records = RecallRecords(log.owner("akasha"))
        inspector = RecallInspector(read=records.read, list_records=records.list, catalog=log.catalog())
        detail = inspector.mobile_detail("query")
        assert detail["hit_count"] == 1
        assert detail["presented_count"] == 1
        assert [message["preview"] for message in detail["hits"][0]["messages"]] == ["first input", "correction", "answer"]
        assert [message["presented"] for message in detail["hits"][0]["messages"]] == [False, True, False]
        assert inspector.recent(page=2, page_size=1) == {"schema": "akasha.queries.v1", "items": [], "total": 1, "page": 2, "page_size": 1}
        assert inspector.mobile_detail("unknown") is None


def test_mobile_inspector_rejects_oversized_complete_references_without_trimming(tmp_path):
    from contextlib import closing
    from agent.plugin_composition import MobileUiRpcInvalidRequest
    from plugins.akasha.inspector import RecallInspector

    with closing(MessageLog(tmp_path / "sessions.db")) as log:
        identity = "x" * (193 * 1024)
        log.writer("s", author="user", source="chat", body_types=(Input,),
                   content={"text": lambda part: ContentReferences()}).append(
            identity, Input((ContentPart("text", "small"),)))
        records = RecallRecords(log.owner("akasha"))
        original = record(0).model_copy(update={"hits": (Hit(node_id=0, session_id="s", message_ids=(identity,),
                                                             score=1.0, lane="dense", sources=("direct_dense",)),)})
        records.save("query", original)
        inspector = RecallInspector(read=records.read, list_records=records.list, catalog=log.catalog())
        with pytest.raises(MobileUiRpcInvalidRequest, match="查询记录仍完整保留"):
            inspector.mobile_detail("query")
        assert records.read("query") == original
