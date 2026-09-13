from __future__ import annotations

from datetime import UTC, datetime, timedelta
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agent.plugin_composition.models import EmbeddingSpaceDescriptor
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
    decode_manifest,
    encode_manifest,
    message_digest,
    recall_entries_digest,
    recall_record_digest,
    recall_record_key,
)
from plugins.akasha.infrastructure.persistence import write_memory_database
from plugins.akasha.recalls import Hit, ProgramSource, Recall


class _Reader:
    def __init__(self, messages: dict[str, Message]):
        self._messages = messages

    def get(self, message_id: str) -> Message | None:
        return self._messages.get(message_id)


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
        reference_count=1,
        provenance=(FrozenProvenance(kind="binding", reference="legacy-learning", digest="f" * 64),),
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
