"""Akasha frozen-history v1: read-only records exported by the old runtime.

The exporter lives in a separate migration artifact.  This module only owns the
versioned JSON value contract and the ordinary-plugin reader.  It never opens a
binding, calls a model, writes a database, or mutates a Message.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from agent.plugin_composition.messages import MessageCatalog
from agent.plugin_composition.models import EmbeddingSpaceDescriptor
from agent.plugin_contracts import Message, body_to_dict, json_value

from ..domain.model import Turn, TurnFeedback
from ..recalls import Recall
from .consumption import Applied, Consumption, LegacyPrefix


FROZEN_HISTORY_FORMAT = "akasha.frozen-history.v1"
_DIGEST = r"^[0-9a-f]{64}$"
_CAPABILITY_DIGEST = r"^(?:[0-9a-f]{20}|[0-9a-f]{64})$"
Digest = Annotated[str, Field(pattern=_DIGEST)]
CapabilityDigest = Annotated[str, Field(pattern=_CAPABILITY_DIGEST)]
Text = Annotated[str, Field(min_length=1)]


class FrozenHistoryError(ValueError):
    """The immutable history is absent, incomplete, or does not match input."""


def _canonical(value: object) -> bytes:
    """Encode one JSON value with the stable bytes used by all history keys."""
    return json.dumps(
        json_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenHistoryError(f"frozen history JSON contains duplicate field: {key}")
        result[key] = value
    return result


class FrozenVector(BaseModel):
    """Preserve one learned vector byte-for-byte without a provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    dtype: Text
    shape: tuple[Annotated[int, Field(ge=0)], ...]
    bytes_hex: Annotated[str, Field(min_length=2)]

    @model_validator(mode="after")
    def check_storage(self) -> Self:
        try:
            dtype = np.dtype(self.dtype)
            raw = bytes.fromhex(self.bytes_hex)
        except (TypeError, ValueError) as error:
            raise FrozenHistoryError("冻结向量的 dtype 或 bytes 无效") from error
        if dtype.kind != "f":
            raise FrozenHistoryError("冻结向量必须是浮点 dtype")
        size = dtype.itemsize
        for extent in self.shape:
            size *= extent
        if size != len(raw):
            raise FrozenHistoryError("冻结向量 shape 与 bytes 长度不一致")
        return self

    @classmethod
    def from_array(cls, value: np.ndarray) -> FrozenVector:
        array = np.asarray(value)
        if array.dtype.kind != "f":
            raise FrozenHistoryError("学习向量必须是浮点数组")
        return cls(dtype=array.dtype.str, shape=tuple(array.shape), bytes_hex=array.tobytes().hex())

    def to_array(self) -> np.ndarray:
        array = np.frombuffer(bytes.fromhex(self.bytes_hex), dtype=np.dtype(self.dtype))
        array = array.reshape(self.shape).copy()
        array.setflags(write=False)
        return array


class FrozenTurnFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    remember_nodes: tuple[Annotated[int, Field(ge=0)], ...] = ()
    forget_nodes: tuple[Annotated[int, Field(ge=0)], ...] = ()
    remember_boost: FiniteFloat = 1.0

    @classmethod
    def from_value(cls, value: TurnFeedback) -> FrozenTurnFeedback:
        return cls(
            remember_nodes=value.remember_nodes,
            forget_nodes=value.forget_nodes,
            remember_boost=value.remember_boost,
        )

    def to_value(self) -> TurnFeedback:
        return TurnFeedback(
            remember_nodes=self.remember_nodes,
            forget_nodes=self.forget_nodes,
            remember_boost=self.remember_boost,
        )


class FrozenTurn(BaseModel):
    """One derived Turn; original Message and graph stores remain authoritative."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    node_id: Annotated[int, Field(ge=0)]
    turn_id: Text
    session_key: Text
    user_seq: Annotated[int, Field(ge=0)]
    user_message_id: Text
    assistant_message_id: Text
    started_at: Text
    committed_at: Text
    user_text: str
    assistant_text: str
    user_dense: FrozenVector | None
    assistant_dense: FrozenVector | None
    user_terms: tuple[tuple[Text, int], ...]
    assistant_terms: tuple[tuple[Text, int], ...]
    inter_gap_seconds: FiniteFloat | None
    feedback: FrozenTurnFeedback = FrozenTurnFeedback()

    @classmethod
    def from_value(cls, value: Turn) -> FrozenTurn:
        return cls(
            node_id=value.node_id,
            turn_id=value.turn_id,
            session_key=value.session_key,
            user_seq=value.user_seq,
            user_message_id=value.user_message_id,
            assistant_message_id=value.assistant_message_id,
            started_at=value.started_at,
            committed_at=value.committed_at,
            user_text=value.user_text,
            assistant_text=value.assistant_text,
            user_dense=None if value.user_dense is None else FrozenVector.from_array(value.user_dense),
            assistant_dense=None if value.assistant_dense is None else FrozenVector.from_array(value.assistant_dense),
            user_terms=value.user_terms,
            assistant_terms=value.assistant_terms,
            inter_gap_seconds=value.inter_gap_seconds,
            feedback=FrozenTurnFeedback.from_value(value.feedback),
        )

    def to_value(self) -> Turn:
        return Turn(
            node_id=self.node_id,
            turn_id=self.turn_id,
            session_key=self.session_key,
            user_seq=self.user_seq,
            user_message_id=self.user_message_id,
            assistant_message_id=self.assistant_message_id,
            started_at=self.started_at,
            committed_at=self.committed_at,
            user_text=self.user_text,
            assistant_text=self.assistant_text,
            user_dense=None if self.user_dense is None else self.user_dense.to_array(),
            assistant_dense=None if self.assistant_dense is None else self.assistant_dense.to_array(),
            user_terms=self.user_terms,
            assistant_terms=self.assistant_terms,
            inter_gap_seconds=self.inter_gap_seconds,
            feedback=self.feedback.to_value(),
        )


class FrozenMessageRef(BaseModel):
    """A source message identity plus its exporter-time immutable digest."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: Text
    seq: Annotated[int, Field(ge=0)]
    message_id: Text
    digest: Digest


class FrozenEmbeddingSpace(BaseModel):
    """Full provider identity plus the exact identity stored by the old rule."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    plugin_snapshot_id: Text
    model_revision: Annotated[int, Field(ge=0)]
    model_id: Text
    connection_id: Text
    driver_id: Text
    driver_contract_version: Text
    auth_identity: Text
    connection_fingerprint: Text
    model: Text
    dimensions: Annotated[int, Field(gt=0)]
    normalization: Text
    capability_digest: CapabilityDigest
    schema_version: Annotated[int, Field(gt=0)] = 1
    source_identity: Text | None = None

    @classmethod
    def from_descriptor(cls, value: EmbeddingSpaceDescriptor) -> FrozenEmbeddingSpace:
        return cls(**asdict(value))

    @property
    def semantic_identity(self) -> str:
        return ":".join((
            self.driver_id, self.driver_contract_version, self.connection_id,
            self.auth_identity, self.connection_fingerprint, self.model_id,
            str(self.dimensions), self.normalization, self.capability_digest,
            str(self.schema_version),
        ))

    @property
    def identity(self) -> str:
        """Keep the existing name for the verified API2 semantic identity."""
        return self.semantic_identity

    @property
    def rule_identity(self) -> str:
        """Return the exact identity used by the exported Learning rule."""
        return self.source_identity or self.semantic_identity

    @model_validator(mode="after")
    def check_source_identity(self) -> Self:
        if self.source_identity is None or self.source_identity == self.semantic_identity:
            return self
        parts = self.source_identity.split(":")
        if len(parts) != 11:
            raise FrozenHistoryError("冻结 embedding source identity 格式无效")
        expected_prefix = (
            self.driver_id,
            self.driver_contract_version,
            self.connection_id,
            self.auth_identity,
        )
        if tuple(parts[:4]) != expected_prefix:
            raise FrozenHistoryError("冻结 embedding source identity 前缀不匹配")
        expected = (
            self.model_id,
            self.connection_fingerprint,
            self.model_id,
            str(self.dimensions),
            self.normalization,
            self.capability_digest,
            str(self.schema_version),
        )
        if tuple(parts[4:]) != expected:
            raise FrozenHistoryError("冻结 embedding source identity 与 descriptor 不匹配")
        return self

    def matches_provider(self, value: EmbeddingSpaceDescriptor) -> bool:
        """Compare behavior and credentials exactly; permit a new snapshot ID only."""
        candidate = FrozenEmbeddingSpace.from_descriptor(value)
        return candidate.model_copy(update={
            "plugin_snapshot_id": self.plugin_snapshot_id,
            "source_identity": self.source_identity,
        }) == self


class FrozenLearningRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    embedding_model: Text
    dimension: Annotated[int, Field(gt=0)]
    sources: tuple[Text, ...]


class FrozenBinding(BaseModel):
    """Provenance for one old binding closure, without source code or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    binding_id: Text
    service_key: Text
    binding_api: Literal[1] = 1
    descriptor_digest: Digest
    plugin_snapshot_id: Text
    component_digests: tuple[tuple[Text, Digest], ...] = ()


class FrozenApplied(BaseModel):
    """One old Applied record and its deterministic Turn result."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    record_key: Digest
    algorithm_digest: Digest
    entry: Applied
    rule: FrozenLearningRule
    embedding: FrozenEmbeddingSpace
    messages: tuple[FrozenMessageRef, ...]
    turn: FrozenTurn

    @model_validator(mode="after")
    def check_record(self) -> Self:
        expected_key = applied_record_key(self.entry, self.algorithm_digest)
        if expected_key != self.record_key:
            raise FrozenHistoryError("冻结 Applied key 与内容不一致")
        expected_refs = self.entry.members + self.entry.observations
        actual_refs = tuple((item.seq, item.message_id) for item in self.messages)
        if actual_refs != expected_refs:
            raise FrozenHistoryError("冻结 Applied 消息出处与原记录不一致")
        if any(item.session_id != self.entry.session_id for item in self.messages):
            raise FrozenHistoryError("冻结 Applied 消息跨 Session")
        if self.rule.dimension != self.embedding.dimensions:
            raise FrozenHistoryError("冻结 Learning rule 与 embedding 维度不一致")
        if self.rule.embedding_model != self.embedding.rule_identity:
            raise FrozenHistoryError("冻结 Learning rule 与 embedding identity 不一致")
        for vector in (self.turn.user_dense, self.turn.assistant_dense):
            if vector is None:
                continue
            array = vector.to_array()
            if array.ndim != 1 or array.shape != (self.embedding.dimensions,):
                raise FrozenHistoryError("冻结 Turn 向量必须是一维且匹配 embedding 维度")
            if not np.isfinite(array).all():
                raise FrozenHistoryError("冻结 Turn 向量必须全部有限")
        if self.turn.session_key != self.entry.session_id:
            raise FrozenHistoryError("冻结 Turn Session 与 Applied 不一致")
        if self.turn.assistant_message_id != self.entry.ending[1]:
            raise FrozenHistoryError("冻结 Turn ending 与 Applied 不一致")
        if (self.turn.user_seq, self.turn.user_message_id) not in self.entry.members:
            raise FrozenHistoryError("冻结 Turn 输入不属于 Applied members")
        return self


class FrozenMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    reminders: tuple[dict[str, object], ...] = ()
    references: tuple[dict[str, object], ...] = ()

    def as_material(self) -> dict[str, object]:
        return {"reminders": self.reminders, "references": self.references}


class FrozenRecall(BaseModel):
    """An old saved Recall plus its exact rendered result."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    record_key: Digest
    identity: Text
    record_digest: Digest
    algorithm_digest: Digest
    recall: dict[str, object]
    material: FrozenMaterial
    messages: tuple[FrozenMessageRef, ...]

    @model_validator(mode="after")
    def check_record(self) -> Self:
        try:
            parsed = Recall.model_validate_json(json.dumps(json_value(self.recall), ensure_ascii=False))
        except (TypeError, ValueError) as error:
            raise FrozenHistoryError("冻结 Recall 不是当前 Recall schema") from error
        if _digest(parsed.model_dump(mode="json")) != self.record_digest:
            raise FrozenHistoryError("冻结 Recall record digest 与内容不一致")
        expected = recall_record_key(self.identity, self.record_digest, self.algorithm_digest)
        if expected != self.record_key:
            raise FrozenHistoryError("冻结 Recall key 与内容不一致")
        source_ids = tuple(item.message_id for item in self.messages)
        if not set(parsed.presented_message_ids).issubset(source_ids):
            raise FrozenHistoryError("冻结 Recall 缺少呈现消息的出处")
        material_ids: list[str] = []
        for row in self.material.references:
            ref = row.get("ref")
            if not isinstance(ref, str) or not ref:
                raise FrozenHistoryError("冻结 Recall reference 缺少 ref")
            material_ids.append(ref)
        if tuple(dict.fromkeys(material_ids)) != parsed.presented_message_ids:
            raise FrozenHistoryError("冻结 Recall material 与呈现消息顺序不一致")
        return self

    def parsed(self) -> Recall:
        return Recall.model_validate_json(json.dumps(json_value(self.recall), ensure_ascii=False))


class FrozenProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["binding", "message", "recall", "embedding"]
    reference: Text
    digest: Digest


class FrozenHistoryManifest(BaseModel):
    """Versioned exporter output; counts are validated from the included input."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    format: Literal["akasha.frozen-history.v1"] = FROZEN_HISTORY_FORMAT
    source_core_commit: Text
    source_plugin_snapshot_id: Text
    source_python_tag: Text
    legacy_prefix: LegacyPrefix
    consumer_state_sha256: Digest
    graph_state_sha256: Digest
    bindings: tuple[FrozenBinding, ...]
    applied_count: Annotated[int, Field(ge=0)]
    applied_digest: Digest
    applied: tuple[FrozenApplied, ...]
    recall_count: Annotated[int, Field(ge=0)]
    recall_digest: Digest
    recalls: tuple[FrozenRecall, ...]
    reference_count: Annotated[int, Field(ge=0)]
    provenance: tuple[FrozenProvenance, ...]
    embedding_spaces: tuple[FrozenEmbeddingSpace, ...]
    algorithm_closure_set_digest: Digest

    @model_validator(mode="after")
    def check_counts(self) -> Self:
        if self.applied_count != len(self.applied):
            raise FrozenHistoryError("applied_count 不能脱离 exporter 输入")
        if self.recall_count != len(self.recalls):
            raise FrozenHistoryError("recall_count 不能脱离 exporter 输入")
        if self.reference_count != len(self.provenance):
            raise FrozenHistoryError("reference_count 不能脱离 exporter 输入")
        if len({item.binding_id for item in self.bindings}) != len(self.bindings):
            raise FrozenHistoryError("冻结 binding provenance 不能重复")
        if len({item.record_key for item in self.applied}) != len(self.applied):
            raise FrozenHistoryError("冻结 Applied record key 不能重复")
        if len({item.record_key for item in self.recalls}) != len(self.recalls):
            raise FrozenHistoryError("冻结 Recall record key 不能重复")
        bindings = {item.binding_id: item for item in self.bindings}
        if self.algorithm_closure_set_digest != algorithm_closure_set_digest(self.bindings):
            raise FrozenHistoryError("algorithm_closure_set_digest 与 binding closure 不一致")
        for item in self.applied:
            binding = bindings.get(item.entry.learning_binding)
            if binding is None:
                raise FrozenHistoryError("Applied 使用了未声明的 learning binding")
            if item.algorithm_digest != binding_closure_digest(binding):
                raise FrozenHistoryError("Applied algorithm_digest 与引用 binding closure 不一致")
            if item.embedding not in self.embedding_spaces:
                raise FrozenHistoryError("Applied 使用了未声明的 embedding space")
        for item in self.recalls:
            binding = bindings.get(item.parsed().learning_binding)
            if binding is None:
                raise FrozenHistoryError("Recall 使用了未声明的 learning binding")
            if item.algorithm_digest != binding_closure_digest(binding):
                raise FrozenHistoryError("Recall algorithm_digest 与引用 binding closure 不一致")
        if any(
            len({name for name, _digest in item.component_digests}) != len(item.component_digests)
            for item in self.bindings
        ):
            raise FrozenHistoryError("binding component digest 不能重复")
        if applied_entries_digest(self.applied) != self.applied_digest:
            raise FrozenHistoryError("applied_digest 与输入记录不一致")
        if recall_entries_digest(self.recalls) != self.recall_digest:
            raise FrozenHistoryError("recall_digest 与输入记录不一致")
        expected_provenance: dict[tuple[str, str], str] = {}

        def add_provenance(kind: str, reference: str, digest: str) -> None:
            key = (kind, reference)
            previous = expected_provenance.get(key)
            if previous is not None and previous != digest:
                raise FrozenHistoryError("同一 provenance reference 的 digest 不一致")
            expected_provenance[key] = digest

        for item in self.bindings:
            add_provenance("binding", binding_provenance_reference(item.binding_id), item.descriptor_digest)
        for item in self.embedding_spaces:
            add_provenance("embedding", embedding_provenance_reference(item), embedding_descriptor_digest(item))
        for item in self.applied:
            for ref in item.messages:
                add_provenance("message", message_provenance_reference(ref), ref.digest)
        for item in self.recalls:
            add_provenance("recall", recall_provenance_reference(item.identity), item.record_digest)
            for ref in item.messages:
                add_provenance("message", message_provenance_reference(ref), ref.digest)
        actual_provenance = {
            (item.kind, item.reference): item.digest
            for item in self.provenance
        }
        if len(actual_provenance) != len(self.provenance):
            raise FrozenHistoryError("provenance reference 不能重复")
        if self.reference_count != len(expected_provenance) or actual_provenance != expected_provenance:
            raise FrozenHistoryError("reference_count 或 provenance 闭包与输入引用不一致")
        return self


def applied_identity(entry: Applied) -> tuple[str, str, tuple[int, str], Digest]:
    return entry.learning_binding, entry.session_id, entry.ending, entry.source_digest


def applied_record_key(entry: Applied, algorithm_digest: str) -> str:
    return _digest({
        "learning_binding": entry.learning_binding,
        "session_id": entry.session_id,
        "ending": entry.ending,
        "source_digest": entry.source_digest,
        "algorithm_digest": algorithm_digest,
    })


def applied_entries_digest(values: Sequence[FrozenApplied]) -> str:
    """Digest exporter Applied rows in their committed order."""
    rows: list[dict[str, object]] = []
    for item in values:
        row = item.model_dump(mode="json")
        embedding = row.get("embedding")
        if isinstance(embedding, dict) and embedding.get("source_identity") is None:
            del embedding["source_identity"]
        rows.append(row)
    return _digest(rows)


def recall_record_digest(recall: Recall) -> str:
    return _digest(recall.model_dump(mode="json"))


def recall_record_key(identity: str, record_digest: str, algorithm_digest: str) -> str:
    return _digest({
        "identity": identity,
        "record_digest": record_digest,
        "algorithm_digest": algorithm_digest,
    })


def recall_entries_digest(values: Sequence[FrozenRecall]) -> str:
    """Digest exporter Recall rows in their source order."""
    return _digest([item.model_dump(mode="json") for item in values])


def binding_provenance_reference(binding_id: str) -> str:
    """Return the stable provenance key for one binding descriptor."""
    return binding_id


def embedding_descriptor_digest(space: FrozenEmbeddingSpace) -> str:
    """Digest the full exported embedding descriptor, including its snapshot."""
    row = space.model_dump(mode="json")
    if row.get("source_identity") is None:
        del row["source_identity"]
    return _digest(row)


def embedding_provenance_reference(space: FrozenEmbeddingSpace) -> str:
    """Use the full descriptor digest so snapshots cannot collapse into one key."""
    return embedding_descriptor_digest(space)


def binding_closure_digest(binding: FrozenBinding) -> str:
    """Digest one binding descriptor and its complete named component closure."""
    return _digest({
        "binding_id": binding.binding_id,
        "service_key": binding.service_key,
        "binding_api": binding.binding_api,
        "descriptor_digest": binding.descriptor_digest,
        "plugin_snapshot_id": binding.plugin_snapshot_id,
        "component_digests": {
            name: digest for name, digest in sorted(binding.component_digests)
        },
    })


def algorithm_closure_set_digest(bindings: Sequence[FrozenBinding]) -> str:
    """Digest the binding-id-sorted map of complete algorithm closures."""
    return _digest({
        binding.binding_id: binding_closure_digest(binding)
        for binding in sorted(bindings, key=lambda item: item.binding_id)
    })


def message_provenance_reference(ref: FrozenMessageRef) -> str:
    """Return a collision-safe provenance key for one Message reference."""
    return _digest({"session_id": ref.session_id, "seq": ref.seq, "message_id": ref.message_id})


def recall_provenance_reference(identity: str) -> str:
    """Return the stable provenance key for one saved Recall identity."""
    return identity


def message_digest(message: Message) -> str:
    """Digest the exact current Message representation without copying its authority."""
    return _digest({
        "message_id": message.message_id,
        "session_id": message.session_id,
        "seq": message.seq,
        "recorded_at": message.recorded_at.isoformat(),
        "author": message.author,
        "source": message.source,
        "body": body_to_dict(message.body),
        "metadata": json_value(message.metadata),
    })


def encode_manifest(manifest: FrozenHistoryManifest) -> bytes:
    """Return canonical JSON for the offline exporter to publish atomically."""
    return _canonical(manifest.model_dump(mode="json")) + b"\n"


def decode_manifest(payload: bytes | str) -> FrozenHistoryManifest:
    """Decode strict JSON and reject duplicate fields before Pydantic validation."""
    try:
        value = json.loads(payload, object_pairs_hook=_unique_pairs)
        # JSON arrays are intentionally accepted for tuple fields; Python-mode
        # validation would reject the exporter’s normal JSON representation.
        return FrozenHistoryManifest.model_validate_json(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, FrozenHistoryError):
            raise
        raise FrozenHistoryError(f"冻结历史 manifest 无效: {error}") from error


class FrozenHistory:
    """Read-only lookup for historical suffix and saved Recall materials."""

    def __init__(self, manifest: FrozenHistoryManifest):
        self.manifest = manifest
        self._applied = {applied_identity(item.entry): item for item in manifest.applied}
        self._legacy_bindings = {item.binding_id for item in manifest.bindings}
        self._recalls = {item.identity: item for item in manifest.recalls}

    @classmethod
    def load(cls, path: Path) -> FrozenHistory:
        try:
            return cls(decode_manifest(path.read_bytes()))
        except OSError as error:
            raise FrozenHistoryError(f"无法读取冻结历史: {path}") from error

    @classmethod
    def load_optional(cls, path: Path) -> FrozenHistory | None:
        if not path.exists():
            return None
        return cls.load(path)

    def validate_consumption(self, state: Consumption) -> None:
        """Check the original prefix and frozen applied prefix before reading any Turn."""
        if state.legacy_prefix != self.manifest.legacy_prefix:
            raise FrozenHistoryError("冻结历史与当前 legacy prefix 不一致")
        original = tuple(item.entry for item in self.manifest.applied)
        if len(state.applied) < len(original) or tuple(state.applied[:len(original)]) != original:
            raise FrozenHistoryError("冻结历史与当前 applied lineage 不一致")

    def uses_binding(self, binding_id: str) -> bool:
        return binding_id in self._legacy_bindings

    def has_applied(self, entry: Applied) -> bool:
        return applied_identity(entry) in self._applied

    def _require_applied(self, entry: Applied) -> FrozenApplied:
        frozen = self._applied.get(applied_identity(entry))
        if frozen is None or frozen.entry != entry:
            raise FrozenHistoryError("缺少精确冻结 Applied 记录，禁止回退到当前算法")
        return frozen

    def embedding_for(self, entry: Applied) -> FrozenEmbeddingSpace:
        """Return the exact exported space for one old Applied entry."""
        return self._require_applied(entry).embedding

    def restore_turn(self, entry: Applied, catalog: MessageCatalog) -> Turn:
        """Restore one exact old Applied result and validate current message bytes."""
        frozen = self._require_applied(entry)
        _check_messages(catalog, frozen.messages)
        return frozen.turn.to_value()

    def has_recall(self, identity: str, recall: Recall) -> bool:
        frozen = self._recalls.get(identity)
        return frozen is not None and frozen.record_digest == recall_record_digest(recall)

    def material_for(self, identity: str, recall: Recall, catalog: MessageCatalog) -> dict[str, object]:
        """Return the old rendered result after validating its exact source messages."""
        frozen = self._recalls.get(identity)
        if frozen is None or frozen.record_digest != recall_record_digest(recall):
            raise FrozenHistoryError("缺少精确冻结 Recall 记录，禁止当前代码重渲染旧结果")
        _check_messages(catalog, frozen.messages)
        return frozen.material.as_material()

    def requires_frozen_binding(self, binding_id: str) -> bool:
        """Return whether an old binding must be served by this sidecar."""
        return self.uses_binding(binding_id)

    def matches_embedding(self, descriptor: EmbeddingSpaceDescriptor) -> bool:
        return any(item.matches_provider(descriptor) for item in self.manifest.embedding_spaces)

    def require_embedding(self, descriptor: EmbeddingSpaceDescriptor) -> None:
        if not self.matches_embedding(descriptor):
            raise FrozenHistoryError(
                "当前 embedding provider 与冻结空间不完全一致；禁止同名或默认模型替代"
            )


def _check_messages(catalog: MessageCatalog, refs: tuple[FrozenMessageRef, ...]) -> None:
    for ref in refs:
        message = catalog.reader(ref.session_id).get(ref.message_id)
        if message is None or message.session_id != ref.session_id or message.seq != ref.seq:
            raise FrozenHistoryError(f"冻结历史消息出处缺失或序号变化: {ref.message_id}")
        if message_digest(message) != ref.digest:
            raise FrozenHistoryError(f"冻结历史消息内容发生变化: {ref.message_id}")
