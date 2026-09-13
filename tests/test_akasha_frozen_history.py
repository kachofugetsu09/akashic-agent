from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agent.plugin_composition.models import EmbeddingResult, EmbeddingSpaceDescriptor
from agent.plugin_contracts import ContentPart, Input, Message, Output
from plugins.akasha.application.consumer import MessageConsumer
from plugins.akasha.application.cycle import MemoryCycle
from plugins.akasha.domain.model import MemoryConfig, Turn
from plugins.akasha.infrastructure.consumption import Applied, Consumption, LegacyPrefix, turns_digest
from plugins.akasha.infrastructure.frozen_history import (
    FrozenApplied,
    FrozenBinding,
    FrozenEmbeddingSpace,
    FrozenHistory,
    FrozenHistoryError,
    FrozenHistoryManifest,
    FrozenLearningRule,
    FrozenMaterial,
    FrozenMessageRef,
    FrozenProvenance,
    FrozenRecall,
    FrozenTurn,
    applied_entries_digest,
    applied_record_key,
    binding_provenance_reference,
    decode_manifest,
    embedding_descriptor_digest,
    embedding_provenance_reference,
    encode_manifest,
    message_digest,
    message_provenance_reference,
    recall_entries_digest,
    recall_provenance_reference,
    recall_record_digest,
    recall_record_key,
)
from plugins.akasha.infrastructure.persistence import write_memory_database
from plugins.akasha.recalls import Hit, ProgramSource, Recall
from plugins.akasha.recall_tool import PreparedRecall, RecallTool


class _Reader:
    def __init__(self, messages: dict[str, Message]):
        self._messages = messages

    def get(self, message_id: str) -> Message | None:
        return self._messages.get(message_id)

    def snapshot(self, *, through_seq: int | None = None) -> tuple[Message, ...]:
        values = tuple(sorted(self._messages.values(), key=lambda message: message.seq))
        if through_seq is None:
            return values
        return tuple(message for message in values if message.seq <= through_seq)


class _Catalog:
    def __init__(self, messages: tuple[Message, ...]):
        self._messages = {message.message_id: message for message in messages}

    def reader(self, session_id: str) -> _Reader:
        return _Reader({
            identity: message
            for identity, message in self._messages.items()
            if message.session_id == session_id
        })


class _NoBindingOpen:
    def open(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("冻结历史不应打开旧 learning binding")


def _fixture() -> tuple[
    tuple[Message, Message], Turn, Applied, FrozenHistoryManifest, Recall,
]:
    started = datetime(2026, 9, 13, 8, 0, tzinfo=UTC)
    user = Message(
        "u1", "s", 0, started, "user", "chat",
        Input((ContentPart("text", "old question"),)),
    )
    assistant = Message(
        "a1", "s", 1, started + timedelta(seconds=2), "assistant", "chat",
        Output((ContentPart("text", "old answer"),), "complete"),
    )
    descriptor = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="old-snapshot", model_revision=7, model_id="embed",
        connection_id="conn", driver_id="driver", driver_contract_version="v1",
        auth_identity="account", connection_fingerprint="fingerprint", model="embed-v7",
        dimensions=2, normalization="unit", capability_digest="a" * 64,
    )
    space = FrozenEmbeddingSpace.from_descriptor(descriptor)
    turn = Turn(
        node_id=0, turn_id="turn-1", session_key="s", user_seq=0,
        user_message_id="u1", assistant_message_id="a1",
        started_at=started.isoformat(), committed_at=assistant.recorded_at.isoformat(),
        user_text="old question", assistant_text="old answer",
        user_dense=np.asarray([0.6, 0.8], dtype=np.float32),
        assistant_dense=np.asarray([0.8, 0.6], dtype=np.float32),
        user_terms=(("old", 1), ("question", 1)),
        assistant_terms=(("old", 1), ("answer", 1)), inter_gap_seconds=None,
    )
    entry = Applied(
        learning_binding="legacy-learning", session_id="s", ending=(1, "a1"),
        members=((0, "u1"), (1, "a1")), observations=(), source_digest="b" * 64,
    )
    algorithm_digest = "c" * 64
    refs = tuple(
        FrozenMessageRef(session_id=message.session_id, seq=message.seq,
                         message_id=message.message_id, digest=message_digest(message))
        for message in (user, assistant)
    )
    frozen_applied = FrozenApplied(
        record_key=applied_record_key(entry, algorithm_digest),
        algorithm_digest=algorithm_digest, entry=entry,
        rule=FrozenLearningRule(embedding_model=space.identity, dimension=2, sources=("chat",)),
        embedding=space, messages=refs, turn=FrozenTurn.from_value(turn),
    )
    recall = Recall(
        learning_binding="legacy-learning", graph_version=1,
        source=ProgramSource(key="old-query", query="old question"), timestamp=started,
        limit=10, hits=(Hit(node_id=0, session_id="s", message_ids=("u1", "a1"),
                             score=0.9, lane="dense", sources=("direct_dense",)),),
        presented_message_ids=("u1", "a1"), active_basin_count=0, pushes=1,
        residual_l1=0.0,
    )
    recall_digest = recall_record_digest(recall)
    frozen_recall = FrozenRecall(
        record_key=recall_record_key("tool:old-query", recall_digest, algorithm_digest),
        identity="tool:old-query", record_digest=recall_digest,
        algorithm_digest=algorithm_digest, recall=recall.model_dump(mode="json"),
        material=FrozenMaterial(
            reminders=({"name": "recall", "text": "old answer", "priority": 300},),
            references=(
                {"ref": "u1", "resolved_ref": "u1", "retrieval_ref": "tool:old-query"},
                {"ref": "a1", "resolved_ref": "a1", "retrieval_ref": "tool:old-query"},
            ),
        ), messages=refs,
    )
    provenance = (
        FrozenProvenance(kind="binding", reference=binding_provenance_reference("legacy-learning"), digest="f" * 64),
        FrozenProvenance(kind="embedding", reference=embedding_provenance_reference(space), digest=embedding_descriptor_digest(space)),
        *(FrozenProvenance(kind="message", reference=message_provenance_reference(ref), digest=ref.digest) for ref in refs),
        FrozenProvenance(kind="recall", reference=recall_provenance_reference("tool:old-query"), digest=recall_digest),
    )
    prefix = LegacyPrefix(count=0, index_state_sha256="0" * 64, turns_digest=turns_digest([]))
    manifest = FrozenHistoryManifest(
        source_core_commit="0f2a76e", source_plugin_snapshot_id="old-snapshot",
        source_python_tag="cpython-3.12", source_algorithm_digest=algorithm_digest,
        legacy_prefix=prefix, consumer_state_sha256="d" * 64, graph_state_sha256="e" * 64,
        bindings=(FrozenBinding(
            binding_id="legacy-learning", service_key="akasha.learning.v1", binding_api=1,
            descriptor_digest="f" * 64, plugin_snapshot_id="old-snapshot",
        ),),
        applied_count=1, applied_digest=applied_entries_digest((frozen_applied,)),
        applied=(frozen_applied,), recall_count=1,
        recall_digest=recall_entries_digest((frozen_recall,)), recalls=(frozen_recall,),
        reference_count=len(provenance), provenance=provenance,
        embedding_spaces=(space,),
    )
    return (user, assistant), turn, entry, manifest, recall


def test_manifest_round_trip_rejects_duplicate_fields_and_count_drift() -> None:
    _messages, _turn, _entry, manifest, _recall = _fixture()
    assert decode_manifest(encode_manifest(manifest)) == manifest
    with pytest.raises(FrozenHistoryError, match="duplicate"):
        decode_manifest(b'{"format":"akasha.frozen-history.v1","format":"akasha.frozen-history.v1"}')
    drifted = manifest.model_copy(update={"applied_count": 2})
    with pytest.raises(FrozenHistoryError, match="applied_count"):
        decode_manifest(encode_manifest(drifted))


def test_manifest_rejects_algorithm_binding_embedding_and_provenance_drift() -> None:
    _messages, _turn, entry, manifest, _recall = _fixture()
    with pytest.raises(FrozenHistoryError, match="algorithm_digest"):
        decode_manifest(encode_manifest(manifest.model_copy(update={"source_algorithm_digest": "0" * 64})))

    missing_binding_entry = entry.model_copy(update={"learning_binding": "missing-learning"})
    original = manifest.applied[0]
    missing_binding = FrozenApplied(
        record_key=applied_record_key(missing_binding_entry, original.algorithm_digest),
        algorithm_digest=original.algorithm_digest, entry=missing_binding_entry,
        rule=original.rule, embedding=original.embedding, messages=original.messages,
        turn=original.turn,
    )
    with pytest.raises(FrozenHistoryError, match="未声明"):
        decode_manifest(encode_manifest(manifest.model_copy(update={
            "applied": (missing_binding,),
            "applied_digest": applied_entries_digest((missing_binding,)),
        })))

    other_space = original.embedding.model_copy(update={"model_revision": 8})
    unlisted_space = original.model_copy(update={
        "rule": FrozenLearningRule(
            embedding_model=other_space.identity, dimension=other_space.dimensions,
            sources=original.rule.sources,
        ),
        "embedding": other_space,
    })
    with pytest.raises(FrozenHistoryError, match="未声明的 embedding"):
        decode_manifest(encode_manifest(manifest.model_copy(update={
            "applied": (unlisted_space,),
            "applied_digest": applied_entries_digest((unlisted_space,)),
        })))

    with pytest.raises(FrozenHistoryError, match="reference_count"):
        decode_manifest(encode_manifest(manifest.model_copy(update={
            "reference_count": manifest.reference_count - 1,
        })))


@pytest.mark.asyncio
async def test_consumer_restores_old_applied_without_opening_binding(tmp_path: Path) -> None:
    messages, turn, entry, manifest, _recall = _fixture()
    history = FrozenHistory(manifest)
    config = MemoryConfig()
    state = Consumption(legacy_prefix=manifest.legacy_prefix, cutover_heads=(), applied=(entry,))
    cycle = MemoryCycle(config)
    cycle.commit(turn, None)
    memory_path = tmp_path / "akasha.db"
    _ = write_memory_database(
        memory_path, turns=[turn], graph=cycle.graph, events=cycle.events,
        evidence=cycle.evidence, captures=[], context=cycle.context,
        burst_members=cycle.burst_members, config=config, metadata={}, consumption=state,
    )
    consumer = await MessageConsumer.load(
        memory_path, legacy_index=None, catalog=_Catalog(messages), embeddings=object(),
        bindings=_NoBindingOpen(), config=config, frozen_history=history,
    )
    try:
        restored = consumer.cycle.turns[0]
        assert restored.turn_id == turn.turn_id
        assert restored.user_message_id == turn.user_message_id
        assert restored.assistant_message_id == turn.assistant_message_id
        np.testing.assert_array_equal(restored.user_dense, turn.user_dense)
        np.testing.assert_array_equal(restored.assistant_dense, turn.assistant_dense)
        assert consumer.state.applied == (entry,)
    finally:
        consumer.close()


@pytest.mark.asyncio
async def test_consumer_reopens_frozen_prefix_and_current_api2_suffix_without_old_binding(
    tmp_path: Path,
) -> None:
    messages, turn, entry, manifest, _recall = _fixture()
    history = FrozenHistory(manifest)
    started = datetime(2026, 9, 13, 8, 0, tzinfo=UTC)
    user = Message(
        "u2", "s", 2, started + timedelta(seconds=4), "user", "chat",
        Input((ContentPart("text", "new question"),)),
    )
    assistant = Message(
        "a2", "s", 3, started + timedelta(seconds=6), "assistant", "chat",
        Output((ContentPart("text", "new answer"),), "complete"),
    )
    new_turn = Turn(
        node_id=1, turn_id="turn-2", session_key="s", user_seq=2,
        user_message_id="u2", assistant_message_id="a2",
        started_at=user.recorded_at.isoformat(), committed_at=assistant.recorded_at.isoformat(),
        user_text="new question", assistant_text="new answer",
        user_dense=np.asarray([0.6, 0.8], dtype=np.float32),
        assistant_dense=np.asarray([0.8, 0.6], dtype=np.float32),
        user_terms=(("new", 1), ("question", 1)),
        assistant_terms=(("new", 1), ("answer", 1)), inter_gap_seconds=2.0,
    )
    suffix = Applied(
        learning_binding="current-api2", session_id="s", ending=(3, "a2"),
        members=((2, "u2"), (3, "a2")), observations=(), source_digest="1" * 64,
    )
    state = Consumption(legacy_prefix=manifest.legacy_prefix, cutover_heads=(), applied=(entry, suffix))
    config = MemoryConfig()
    cycle = MemoryCycle(config)
    cycle.commit(turn, None)
    cycle.commit(new_turn, None)
    memory_path = tmp_path / "akasha.db"
    _ = write_memory_database(
        memory_path, turns=[turn, new_turn], graph=cycle.graph, events=cycle.events,
        evidence=cycle.evidence, captures=[], context=cycle.context,
        burst_members=cycle.burst_members, config=config, metadata={}, consumption=state,
    )

    class CurrentLearning:
        def restore(self, *_args: object, **_kwargs: object) -> Turn:
            return new_turn

    class CurrentBindings:
        def __init__(self) -> None:
            self.calls: list[str] = []

        @asynccontextmanager
        async def open(self, identity: str, _service: object):
            self.calls.append(identity)
            if identity != "current-api2":
                raise AssertionError("reopen 不应打开历史 API1 binding")
            yield CurrentLearning(), {
                "embedding_model": manifest.embedding_spaces[0].identity,
                "dimension": 2,
                "sources": ("chat",),
            }

    bindings = CurrentBindings()
    catalog = _Catalog((*messages, user, assistant))
    for _ in range(2):
        consumer = await MessageConsumer.load(
            memory_path, legacy_index=None, catalog=catalog, embeddings=object(),
            bindings=bindings, config=config, frozen_history=history,
        )
        try:
            assert tuple(item.turn_id for item in consumer.cycle.turns) == ("turn-1", "turn-2")
        finally:
            consumer.close()
    assert bindings.calls == ["current-api2", "current-api2"]


def test_frozen_recall_validates_live_message_identity() -> None:
    messages, _turn, _entry, manifest, recall = _fixture()
    history = FrozenHistory(manifest)
    assert history.material_for("tool:old-query", recall, _Catalog(messages))["references"]
    changed = Message(
        "u1", "s", 0, messages[0].recorded_at, "user", "chat",
        Input((ContentPart("text", "tampered question"),)),
    )
    with pytest.raises(FrozenHistoryError, match="内容发生变化"):
        history.material_for("tool:old-query", recall, _Catalog((changed, messages[1])))
    with pytest.raises(FrozenHistoryError, match="出处缺失"):
        history.material_for("tool:old-query", recall, _Catalog((messages[0],)))


def test_embedding_match_allows_new_snapshot_but_not_new_revision_or_model() -> None:
    _messages, _turn, _entry, manifest, _recall = _fixture()
    frozen = manifest.embedding_spaces[0]
    current = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="new-snapshot", model_revision=frozen.model_revision,
        model_id=frozen.model_id, connection_id=frozen.connection_id,
        driver_id=frozen.driver_id, driver_contract_version=frozen.driver_contract_version,
        auth_identity=frozen.auth_identity, connection_fingerprint=frozen.connection_fingerprint,
        model=frozen.model, dimensions=frozen.dimensions, normalization=frozen.normalization,
        capability_digest=frozen.capability_digest, schema_version=frozen.schema_version,
    )
    assert frozen.matches_provider(current)
    assert not frozen.matches_provider(replace(current, model_revision=current.model_revision + 1))
    assert not frozen.matches_provider(replace(current, model="same-name-but-different-model"))


@pytest.mark.asyncio
async def test_recall_invoke_passes_frozen_history_to_read_memory_and_opens_only_current_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _messages, _turn, _entry, manifest, _old_recall = _fixture()
    history = FrozenHistory(manifest)
    calls: list[str] = []
    seen: dict[str, object] = {}
    current_descriptor = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="new-snapshot", model_revision=8, model_id="current",
        connection_id="current-conn", driver_id="driver", driver_contract_version="v1",
        auth_identity="account", connection_fingerprint="current-fingerprint", model="current-v8",
        dimensions=2, normalization="unit", capability_digest="1" * 64,
    )

    class CurrentBindings:
        @asynccontextmanager
        async def open(self, identity: str, _service: object):
            calls.append(identity)
            if identity != "current-api2":
                raise AssertionError("历史 API1 binding 不能在新 Recall 中打开")
            yield object(), {
                "embedding_model": current_descriptor.identity,
                "dimension": 2,
                "sources": ("chat",),
            }

    class Model:
        descriptor = current_descriptor

        async def embed(self, _texts: list[str]) -> EmbeddingResult:
            return EmbeddingResult(((0.6, 0.8),))

    @asynccontextmanager
    async def open_embedding(_identity: str):
        yield Model()

    class Records:
        def __init__(self) -> None:
            self.saved: Recall | None = None

        def read(self, _identity: str) -> Recall | None:
            return None

        def save(self, _identity: str, recall: Recall) -> str:
            self.saved = recall
            return "tool:new"

    records = Records()
    new_recall = Recall(
        learning_binding="current-api2", graph_version=0,
        source=ProgramSource(key="new", query="new question"),
        timestamp=datetime(2026, 9, 13, 9, 0, tzinfo=UTC), limit=1, hits=(),
        active_basin_count=0, pushes=0, residual_l1=0.0,
    )

    @asynccontextmanager
    async def fake_read_memory(*_args: object, **kwargs: object):
        seen["frozen_history"] = kwargs.get("frozen_history")
        yield object(), object()

    monkeypatch.setattr("plugins.akasha.recall_tool.read_memory", fake_read_memory)
    monkeypatch.setattr("plugins.akasha.recall_tool.query_memory", lambda *_args, **_kwargs: new_recall)
    monkeypatch.setattr(
        "plugins.akasha.recall_tool.render_materials",
        lambda *_args, **_kwargs: {
            "reminders": ({"name": "recall", "text": "new", "priority": 300},),
            "references": (),
        },
    )
    tool = RecallTool(
        memory=Path("/unused"), legacy_index=None, config=MemoryConfig(), catalog=object(),
        embeddings=object(), bindings=CurrentBindings(),
        select_learning=lambda: ("current-api2", "current-model"), records=records,
        open_embedding=open_embedding, frozen_history=history,
    )
    arguments = PreparedRecall(
        query="new question", limit=1, source=None, learning_binding="current-api2",
        embedding_binding="current-model", max_chars=12000,
    ).model_dump(mode="json")
    result = await tool.invoke("new", arguments)
    assert result.outcome == "success"
    assert calls == ["current-api2"]
    assert seen["frozen_history"] is history
    assert records.saved is not None


@pytest.mark.asyncio
async def test_context_reuses_saved_frozen_material_without_opening_old_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plugins.akasha import runtime as runtime_module
    from plugins.akasha.learning import LearningConfig

    messages, _turn, _entry, manifest, recall = _fixture()
    history = FrozenHistory(manifest)

    class Projection:
        def project(self, _snapshot: tuple[Message, ...], _source: str):
            return (type("ProjectionResult", (), {"status": "open", "message_ids": ("u1",)})(),)

    class Learning:
        projection = Projection()

        @staticmethod
        def text(_message: Message) -> str:
            return "old question"

    class Records:
        def read(self, _identity: str) -> Recall:
            return recall

        def list(self) -> tuple[tuple[str, Recall], ...]:
            return ()

    monkeypatch.setattr(runtime_module, "tool_references", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(history, "has_recall", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        history, "material_for", lambda *_args, **_kwargs: manifest.recalls[0].material.as_material(),
    )
    material = await runtime_module.prepare_materials(
        (messages[0],), "chat", cycle=MemoryCycle(), state=object(), catalog=_Catalog(messages),
        embeddings=object(), bindings=_NoBindingOpen(), learning_binding="legacy-learning",
        learning=Learning(), rule=LearningConfig(embedding_model="current", dimension=2, sources=("chat",)),
        records=Records(), embed_batch=lambda _texts: None, limit=1, max_chars=12000,
        frozen_history=history,
    )
    assert tuple(row["ref"] for row in material["references"]) == ("u1", "a1")
