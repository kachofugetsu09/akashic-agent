from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent.control.context import running_turn_id
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.request_projection import RequestHistoryUnit
from agent.turn_effects import suppresses_post_commit

if TYPE_CHECKING:
    from session.manager import SessionManager
    from session.store import CompactionHead, CompactionPrepare, SessionCompaction


@dataclass(frozen=True, slots=True)
class SessionCompactionCommit:
    """Source-neutral ledger values accepted by the Session storage atom."""

    summary: str
    trigger: str
    source_ref: str
    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    retained_tail: tuple[dict[str, Any], ...]
    model_runtime_id: str
    model: str
    context_window: int
    soft_limit_tokens: int
    hard_input_tokens: int
    keep_recent_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int


@dataclass(frozen=True, slots=True)
class SessionProjectionGrant:
    """Opaque authority for one Session incarnation during one active turn."""

    _issuer: object
    _session_key: str
    _session_created_at: str
    _turn_id: str
    _nonce: object

    @classmethod
    def issue(
        cls,
        issuer: object,
        *,
        session_key: str,
        session_created_at: str,
        turn_id: str,
    ) -> SessionProjectionGrant:
        if not session_key.strip() or not session_created_at.strip():
            raise ValueError("projection grant 缺少 Session identity")
        if not turn_id.strip() or running_turn_id.get() != turn_id:
            raise RuntimeError("projection grant 只能绑定当前 active turn")
        return cls(issuer, session_key, session_created_at, turn_id, object())

    def allows(
        self,
        issuer: object,
        *,
        session_key: str,
        session_created_at: str,
    ) -> bool:
        """Validate owner identity, Session incarnation, and active turn."""

        return (
            self._issuer is issuer
            and self._session_key == session_key
            and self._session_created_at == session_created_at
            and bool(self._turn_id)
            and running_turn_id.get() == self._turn_id
        )


class SessionCompactionStorage:
    """Expose only the Session writes needed by a context projection plugin."""

    def __init__(
        self,
        manager: SessionManager | None,
        grant: SessionProjectionGrant | None = None,
    ) -> None:
        self._manager = manager
        self._grant = grant

    @classmethod
    def candidate_validation(cls) -> SessionCompactionStorage:
        """Reject formal Session access while preserving candidate topology."""

        return cls(None)

    @property
    def formal(self) -> bool:
        return self._manager is not None

    def scope(self, grant: object | None) -> SessionCompactionStorage:
        """Bind the storage atom to one owner-issued current-turn grant."""

        if not isinstance(grant, SessionProjectionGrant):
            raise PermissionError("context projection 缺少有效 Session grant")
        return SessionCompactionStorage(self._manager, grant)

    def _store(self, session_key: str):
        manager = self._manager
        if manager is None:
            raise RuntimeError("candidate 验证期禁止读写正式 Session compaction")
        grant = self._grant
        if grant is None:
            raise PermissionError("Session compaction storage 尚未绑定 grant")
        meta = manager.control_store.get_session_meta(session_key)
        if meta is None:
            raise KeyError(f"context projection session 不存在: {session_key}")
        if not manager.validate_projection_grant(
            grant,
            session_key=session_key,
            session_created_at=str(meta["created_at"]),
        ):
            raise PermissionError("Session compaction grant scope 不匹配")
        return manager.control_store

    def history_units(self, session_key: str) -> tuple[RequestHistoryUnit, ...]:
        """Return a detached immutable view scoped to one Session."""

        store = self._store(session_key)
        manager = self._manager
        assert manager is not None
        session = manager.get_existing(session_key)
        return tuple(
            RequestHistoryUnit(
                source_from_seq=unit.source_from_seq,
                consolidated_through_seq=unit.consolidated_through_seq,
                source_message_ids=unit.source_message_ids,
                messages_json=json.dumps(
                    unit.messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                message_refs=unit.message_refs,
            )
            for unit in session.history_units(after_seq=-1)
        )

    def get_head(self, session_key: str) -> CompactionHead:
        return self._store(session_key).get_compaction_head(session_key)

    def get_active(self, session_key: str) -> SessionCompaction | None:
        return self._store(session_key).get_active_compaction(session_key)

    def get_prepare(
        self,
        session_key: str,
        *,
        source_ref: str,
    ) -> CompactionPrepare | None:
        return self._store(session_key).get_compaction_prepare(
            session_key,
            source_ref=source_ref,
        )

    def clear_orphan_prepare(self, prepare: CompactionPrepare) -> None:
        _ = self._store(prepare.session_key).release_orphan_compaction_prepare(prepare)

    def session_created_at(self, session_key: str) -> str:
        meta = self._store(session_key).get_session_meta(session_key)
        if meta is None:
            raise RuntimeError(f"context projection session 不存在: {session_key}")
        return str(meta["created_at"])

    def prepare(
        self,
        *,
        session_key: str,
        session_created_at: str,
        generation: int,
        parent_generation: int,
        source_ref: str,
        source_from_seq: int,
        consolidated_through_seq: int,
        source_message_ids: tuple[str, ...],
        retained_tail: tuple[dict[str, Any], ...],
        source_mutation_digest: str,
    ) -> CompactionPrepare:
        if self.session_created_at(session_key) != session_created_at:
            raise PermissionError("Session compaction incarnation 不匹配")
        return self._store(session_key).prepare_compaction(
            session_key=session_key,
            session_created_at=session_created_at,
            generation=generation,
            parent_generation=parent_generation,
            source_ref=source_ref,
            source_from_seq=source_from_seq,
            consolidated_through_seq=consolidated_through_seq,
            source_message_ids=source_message_ids,
            retained_tail=retained_tail,
            source_mutation_digest=source_mutation_digest,
        )

    def validate_provenance(
        self,
        session_key: str,
        commit: SessionCompactionCommit,
    ) -> None:
        self._store(session_key).validate_compaction_provenance(
            session_key,
            source_message_ids=commit.source_message_ids,
            retained_tail=commit.retained_tail,
            source_from_seq=commit.source_from_seq,
            consolidated_through_seq=commit.consolidated_through_seq,
        )

    def source_mutation_digest(
        self,
        session_key: str,
        source_ids: tuple[str, ...],
    ) -> str:
        return self._store(session_key).source_mutation_digest(session_key, source_ids)

    def suppresses_post_commit(
        self,
        session_key: str,
        message_ids: tuple[str, ...],
    ) -> bool:
        messages = self._store(session_key).fetch_by_ids(list(message_ids))
        if any(
            str(message.get("session_key", "")) != session_key for message in messages
        ):
            raise RuntimeError("context projection source message 越过 Session scope")
        actual_ids = tuple(str(message["id"]) for message in messages)
        if actual_ids != message_ids:
            raise RuntimeError("context projection source messages 不完整或顺序不一致")
        return any(suppresses_post_commit(message) for message in messages)

    def get_committed(
        self,
        session_key: str,
        source_ref: str,
    ) -> SessionCompaction | None:
        return self._store(session_key).get_compaction_by_source_ref(
            session_key, source_ref
        )

    def persist(
        self,
        *,
        commit: SessionCompactionCommit,
        head: CompactionHead,
        source_plan_digest: str,
        summary_usage: dict[str, object],
        prepare: CompactionPrepare | None,
        source_mutation_digest: str | None,
    ) -> SessionCompaction:
        if commit.source_ref.strip() == "":
            raise ValueError("compaction checkpoint source_ref 不能为空")
        return self._store(head.session_key).persist_compaction(
            session_key=head.session_key,
            trigger=commit.trigger,
            summary=commit.summary,
            source_ref=commit.source_ref,
            source_plan_digest=source_plan_digest,
            source_from_seq=commit.source_from_seq,
            consolidated_through_seq=commit.consolidated_through_seq,
            source_message_ids=commit.source_message_ids,
            retained_tail=commit.retained_tail,
            model_runtime_id=commit.model_runtime_id,
            model=commit.model,
            context_window=commit.context_window,
            threshold_tokens=commit.soft_limit_tokens,
            hard_input_tokens=commit.hard_input_tokens,
            keep_recent_tokens=commit.keep_recent_tokens,
            tokens_before=commit.estimated_tokens_before,
            tokens_after=commit.estimated_tokens_after,
            summary_usage=summary_usage,
            parent_generation=head.parent_generation,
            generation=head.next_generation,
            summary_format_version=1,
            prepare=prepare,
            source_mutation_digest=source_mutation_digest,
        )


SESSION_COMPACTION_STORAGE = ServiceKey[SessionCompactionStorage](
    "core.session_compaction_storage"
)
