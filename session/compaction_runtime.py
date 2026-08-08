from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from agent.model_runtime.context_compaction import (
    ActiveCompaction,
    CommittedContextUnit,
    ContextCompaction,
    ContextPayloadSegments,
    build_compaction_messages,
    canonical_source_plan,
    compaction_scope_id,
    compaction_source_ref,
    normalize_session_created_at,
    source_plan_digest,
    _checkpoint_from_receipt,
)
from agent.model_runtime.types import ModelUsage
from core.memory.markdown import CompactionMarkdownDraft
from session.memory_policy import excludes_memory
from session.store import (
    CompactionHead,
    CompactionPrepare,
    SessionCompaction,
    SessionStore,
)

if TYPE_CHECKING:
    from agent.core.runtime_support import SessionLike
    from core.memory.markdown import MarkdownMemoryMaintenance
    from session.manager import SessionManager


@dataclass(frozen=True)
class CompactionProjection:
    """Current session projection and the Store-owned ledger head."""

    segments: ContextPayloadSegments
    active: ActiveCompaction | None
    head: CompactionHead


class SessionCompactionPort(Protocol):
    """Narrow owner port consumed by DefaultReasoner."""

    async def projection(
        self,
        session: "SessionLike",
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection: ...

    async def recover_pending(self, session: "SessionLike") -> SessionCompaction | None: ...

    async def commit_checkpoint(
        self,
        session: "SessionLike",
        checkpoint: ContextCompaction,
        *,
        head: CompactionHead,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> SessionCompaction: ...


class SessionCompactionRuntime:
    """Own session projection plus Markdown/SQLite compaction commit saga."""

    def __init__(
        self,
        *,
        session_manager: "SessionManager",
        markdown: "MarkdownMemoryMaintenance",
    ) -> None:
        self._session_manager = session_manager
        self._store: SessionStore = session_manager.control_store
        self._markdown = markdown

    async def projection(
        self,
        session: "SessionLike",
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection:
        """Build the exact payload sections from ledger summary and canonical rows."""

        _ = await self.recover_pending(session)
        head = self._store.get_compaction_head(session.key)
        active_row = self._store.get_active_compaction(session.key)
        active: ActiveCompaction | None = None
        units: list[CommittedContextUnit]
        projected_prefix = list(prefix)
        if active_row is None:
            units = list(session.history_units(after_seq=-1))
        else:
            active = ActiveCompaction(
                generation=active_row.generation,
                summary=active_row.summary,
                source_from_seq=active_row.source_from_seq,
                consolidated_through_seq=active_row.consolidated_through_seq,
                source_message_ids=active_row.source_message_ids,
                retained_tail=active_row.retained_tail,
            )
            projected_prefix.extend(
                build_compaction_messages(
                    active.summary,
                    generation=active.generation,
                    source_ref=active_row.source_ref,
                )
            )
            tail_units = _retained_tail_units(active_row.retained_tail)
            if tail_units:
                tail_through_seq = max(
                    unit.consolidated_through_seq for unit in tail_units
                )
                appended_units = session.history_units(after_seq=tail_through_seq)
            else:
                appended_units = session.history_units(
                    after_seq=active.consolidated_through_seq
                )
            units = [*tail_units, *appended_units]
        return CompactionProjection(
            segments=ContextPayloadSegments(
                prefix=tuple(projected_prefix),
                committed_units=tuple(units),
                current_anchor=tuple(current_anchor),
                pending=tuple(pending),
            ),
            active=active,
            head=head,
        )

    async def recover_pending(self, session: "SessionLike") -> SessionCompaction | None:
        """Finish a receipt left between Markdown and ledger commits."""

        head = self._store.get_compaction_head(session.key)
        return await self._recover_pending(session, head)

    async def _recover_pending(
        self,
        session: "SessionLike",
        head: CompactionHead,
    ) -> SessionCompaction | None:
        source_ref = compaction_source_ref(
            compaction_scope_id(session.key, session.created_at),
            head.next_generation,
        )
        prepare = self._store.get_compaction_prepare(
            session.key,
            source_ref=source_ref,
        )
        receipt = self._markdown.read_compaction_receipt(source_ref)
        if receipt is None:
            if prepare is not None:
                # Receipt is the first cross-file effect; no receipt means the
                # prepare is still in the pre-effect window and may be released.
                self._store._clear_orphan_compaction_prepare(prepare)
            return None
        if prepare is None:
            raise RuntimeError("compaction receipt 存在但 durable prepare 缺失")
        if receipt.get("session_key") != session.key:
            raise ValueError("compaction receipt session_key 冲突")
        if receipt.get("parent_generation") != head.parent_generation or receipt.get(
            "next_generation"
        ) != head.next_generation:
            raise RuntimeError("compaction receipt 与当前 ledger head 冲突")
        raw_digest = receipt.get("selection_digest")
        raw_checkpoint = receipt.get("checkpoint")
        if receipt.get("version") != 2:
            raise ValueError("compaction receipt version 不支持安全恢复")
        if not isinstance(raw_digest, str) or not isinstance(raw_checkpoint, dict):
            raise ValueError("compaction receipt schema 无效")
        if receipt.get("session_created_at") != normalize_session_created_at(
            session.created_at
        ):
            raise RuntimeError("compaction receipt session incarnation 冲突")
        if receipt.get("digest") != _receipt_digest(receipt):
            raise ValueError("compaction receipt digest 校验失败")
        _, _, checkpoint = _checkpoint_from_receipt(
            receipt,
            source_ref=source_ref,
            selection_digest=raw_digest,
        )
        checkpoint, canonical_digest = self._canonicalize_checkpoint_source(
            session,
            checkpoint,
        )
        self._assert_prepare_matches_checkpoint(session, checkpoint, prepare)
        if receipt.get("source_plan_digest") != canonical_digest:
            raise RuntimeError("compaction receipt source plan 与当前 SessionDB 不一致")
        self._store.validate_compaction_provenance(
            session.key,
            source_message_ids=checkpoint.source_message_ids,
            retained_tail=checkpoint.retained_tail,
            source_from_seq=checkpoint.source_from_seq,
            consolidated_through_seq=checkpoint.consolidated_through_seq,
        )
        draft = _draft_from_receipt(receipt)
        if checkpoint.source_ref != draft.source_ref:
            raise ValueError("compaction receipt source_ref 冲突")
        # 1. Durable receipt 已证明这是 included path；恢复时 metadata 可能已变化，
        # 仍必须幂等重放原始 Markdown side effect。
        await self._markdown.commit_compaction_markdown(draft)
        after_markdown = self._store.get_compaction_head(session.key)
        if after_markdown != head:
            raise RuntimeError("compaction receipt recovery 时 ledger head 发生变化")
        # 2. Ledger insert、cursor advance 和 prepare clear 在同一 Store 事务内提交。
        row = self._persist_checkpoint(
            checkpoint,
            head=head,
            prepare=prepare,
        )
        session.last_consolidated = row.generation
        return row

    async def commit_checkpoint(
        self,
        session: "SessionLike",
        checkpoint: ContextCompaction,
        *,
        head: CompactionHead,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> SessionCompaction:
        """Commit receipt, Markdown effects, then the SQLite ledger row."""

        expected_source_ref = compaction_source_ref(
            compaction_scope_id(session.key, session.created_at),
            head.next_generation,
        )
        if checkpoint.source_ref != expected_source_ref:
            raise ValueError(
                "compaction checkpoint source_ref 与 session incarnation 不一致"
            )
        if excludes_memory(session.key, session.metadata):
            if session.key != head.session_key:
                raise ValueError("compaction session 与 ledger head 不一致")
            before_mutation_digest = self._source_mutation_digest(
                session.key,
                checkpoint,
            )
            checkpoint, _ = self._canonicalize_checkpoint_source(session, checkpoint)
            mutation_digest = self._source_mutation_digest(session.key, checkpoint)
            if mutation_digest != before_mutation_digest:
                raise RuntimeError("compaction source snapshot 在 canonicalize 期间发生变化")
            self._store.validate_compaction_provenance(
                session.key,
                source_message_ids=checkpoint.source_message_ids,
                retained_tail=checkpoint.retained_tail,
                source_from_seq=checkpoint.source_from_seq,
                consolidated_through_seq=checkpoint.consolidated_through_seq,
            )
            current = self._store.get_compaction_head(head.session_key)
            if current != head:
                raise RuntimeError("excluded compaction ledger head 在提交前已变化")
            row = self._persist_checkpoint(
                checkpoint,
                head=head,
                source_mutation_digest=mutation_digest,
            )
            session.last_consolidated = row.generation
            return row

        (
            markdown_draft,
            checkpoint,
            source_digest,
            mutation_digest,
        ) = await self._build_markdown_draft(
            session,
            checkpoint,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )
        if checkpoint.source_ref != markdown_draft.source_ref:
            raise ValueError("compaction checkpoint 与 Markdown source_ref 不一致")
        if session.key != head.session_key:
            raise ValueError("compaction session 与 ledger head 不一致")
        self._store.validate_compaction_provenance(
            session.key,
            source_message_ids=checkpoint.source_message_ids,
            retained_tail=checkpoint.retained_tail,
            source_from_seq=checkpoint.source_from_seq,
            consolidated_through_seq=checkpoint.consolidated_through_seq,
        )
        current = self._store.get_compaction_head(head.session_key)
        if current != head:
            raise RuntimeError("compaction ledger head 在 prepare 前已变化")
        before_prepare_mutation_digest = self._source_mutation_digest(
            session.key,
            checkpoint,
        )
        checkpoint, post_prepare_digest = self._canonicalize_checkpoint_source(
            session,
            checkpoint,
        )
        if post_prepare_digest != source_digest:
            raise RuntimeError("compaction source plan 在 Markdown prepare 后发生变化")
        post_prepare_mutation_digest = self._source_mutation_digest(
            session.key,
            checkpoint,
        )
        if (
            before_prepare_mutation_digest != mutation_digest
            or post_prepare_mutation_digest != mutation_digest
        ):
            raise RuntimeError("compaction source snapshot 在 Markdown prepare 后发生变化")
        prepare = self._store.prepare_compaction(
            session_key=session.key,
            session_created_at=self._session_created_at(session.key),
            generation=head.next_generation,
            parent_generation=head.parent_generation,
            source_ref=checkpoint.source_ref,
            source_from_seq=checkpoint.source_from_seq,
            consolidated_through_seq=checkpoint.consolidated_through_seq,
            source_message_ids=checkpoint.source_message_ids,
            retained_tail=checkpoint.retained_tail,
            source_mutation_digest=mutation_digest,
        )
        receipt = _receipt_payload(
            checkpoint,
            session_key=head.session_key,
            head=head,
            markdown_draft=markdown_draft,
            model_runtime_id=checkpoint.model_runtime_id,
            model=checkpoint.model,
            session_created_at=normalize_session_created_at(session.created_at),
        )
        existing = self._markdown.read_compaction_receipt(checkpoint.source_ref)
        if existing is None:
            self._markdown.write_compaction_receipt(checkpoint.source_ref, receipt)
        elif existing != receipt:
            raise ValueError("compaction receipt 内容冲突")
        before_commit_mutation_digest = self._source_mutation_digest(
            session.key,
            checkpoint,
        )
        _, pre_commit_digest = self._canonicalize_checkpoint_source(session, checkpoint)
        if pre_commit_digest != source_digest:
            raise RuntimeError("compaction source plan 在 Markdown commit 前发生变化")
        pre_commit_mutation_digest = self._source_mutation_digest(
            session.key,
            checkpoint,
        )
        if (
            before_commit_mutation_digest != mutation_digest
            or pre_commit_mutation_digest != mutation_digest
        ):
            raise RuntimeError("compaction source snapshot 在 Markdown commit 前发生变化")
        await self._markdown.commit_compaction_markdown(markdown_draft)
        after_markdown = self._store.get_compaction_head(head.session_key)
        if after_markdown != head:
            raise RuntimeError("Markdown 提交期间 compaction ledger head 发生变化")
        row = self._persist_checkpoint(
            checkpoint,
            head=head,
            prepare=prepare,
            source_mutation_digest=mutation_digest,
        )
        session.last_consolidated = row.generation
        return row

    async def _build_markdown_draft(
        self,
        session: "SessionLike",
        checkpoint: ContextCompaction,
        *,
        scope_channel: str,
        scope_chat_id: str,
    ) -> tuple[CompactionMarkdownDraft, ContextCompaction, str, str]:
        """Build Markdown input exclusively from Store-owned committed source rows."""

        if not checkpoint.selected_source_messages:
            raise ValueError("compaction Markdown source plan 不能为空")
        before_mutation_digest = self._source_mutation_digest(session.key, checkpoint)
        canonical_checkpoint, plan_digest = self._canonicalize_checkpoint_source(
            session,
            checkpoint,
        )
        mutation_digest = self._source_mutation_digest(
            session.key,
            canonical_checkpoint,
        )
        if mutation_digest != before_mutation_digest:
            raise RuntimeError("compaction source snapshot 在 canonicalize 期间发生变化")
        draft = await self._markdown.prepare_compaction_markdown(
            canonical_checkpoint.selected_source_messages,
            source_ref=checkpoint.source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )
        return draft, canonical_checkpoint, plan_digest, mutation_digest

    def _canonicalize_checkpoint_source(
        self,
        session: "SessionLike",
        checkpoint: ContextCompaction,
    ) -> tuple[ContextCompaction, str]:
        """Rebuild selected and retained plans from the latest SessionDB rows."""

        current = self._session_manager.get_existing(session.key)
        if current.created_at != session.created_at:
            raise RuntimeError("compaction source session incarnation changed")
        selected_requested = canonical_source_plan(checkpoint.selected_source_messages)
        retained_requested = canonical_source_plan(checkpoint.retained_tail)
        selected_ids = {str(item["id"]) for item in selected_requested}
        retained_ids = {str(item["id"]) for item in retained_requested}
        if selected_ids.intersection(retained_ids):
            raise RuntimeError("compaction selected 与 retained source 不得重叠")
        available: dict[
            tuple[str, int], list[tuple[int, int, dict[str, Any]]]
        ] = {}
        current_units = current.history_units(after_seq=-1)
        for unit_index, unit in enumerate(current_units):
            if len(unit.message_refs) != len(unit.messages):
                raise ValueError(
                    "compaction source history_refs 与 rendered messages 不一致"
                )
            for message_index, (message, (message_id, seq)) in enumerate(
                zip(unit.messages, unit.message_refs)
            ):
                canonical = canonical_source_plan(
                    [
                        {
                            "id": message_id,
                            "seq": seq,
                            "unit_ref": "canonical",
                            "message": message,
                        }
                    ]
                )[0]["message"]
                available.setdefault((message_id, seq), []).append(
                    (unit_index, message_index, canonical)
                )

        canonical_selected, selected_occurrences = _consume_checkpoint_source_plan(
            selected_requested,
            "selected",
            session.key,
            available,
        )
        canonical_retained, retained_occurrences = _consume_checkpoint_source_plan(
            retained_requested,
            "retained",
            session.key,
            available,
        )
        selected_unit_indices = _validate_checkpoint_source_units(
            selected_occurrences,
            current_units,
            "selected",
            require_generated_ref=True,
        )
        retained_unit_indices = _validate_checkpoint_source_units(
            retained_occurrences,
            current_units,
            "retained",
            require_generated_ref=True,
        )
        if set(selected_unit_indices).intersection(retained_unit_indices):
            raise RuntimeError("compaction selected 与 retained logical unit 不得重叠")
        if selected_unit_indices and retained_unit_indices:
            if max(selected_unit_indices) >= min(retained_unit_indices):
                raise RuntimeError("compaction selected 与 retained 必须按历史顺序")
        canonical_selected_ids = tuple(
            dict.fromkeys(str(item["id"]) for item in canonical_selected)
        )
        if canonical_selected_ids != tuple(checkpoint.source_message_ids):
            raise RuntimeError("compaction source plan 与 source_message_ids 不一致")
        digest = source_plan_digest(canonical_selected)
        return replace(
            checkpoint,
            selected_source_messages=tuple(canonical_selected),
            retained_tail=tuple(canonical_retained),
        ), digest

    def _assert_prepare_matches_checkpoint(
        self,
        session: "SessionLike",
        checkpoint: ContextCompaction,
        prepare: CompactionPrepare,
    ) -> None:
        """Reject a receipt whose durable fence does not own its checkpoint."""

        if (
            prepare.session_key != session.key
            or prepare.session_created_at != self._session_created_at(session.key)
            or prepare.generation != checkpoint.generation
            or prepare.parent_generation != checkpoint.parent_generation
            or prepare.source_ref != checkpoint.source_ref
            or prepare.source_from_seq != checkpoint.source_from_seq
            or prepare.consolidated_through_seq != checkpoint.consolidated_through_seq
            or prepare.source_message_ids != checkpoint.source_message_ids
            or prepare.retained_tail != checkpoint.retained_tail
        ):
            raise RuntimeError("compaction receipt 与 durable prepare identity 冲突")

    def _session_created_at(self, session_key: str) -> str:
        """Read the exact persisted incarnation string used by the prepare fence."""

        meta = self._store.get_session_meta(session_key)
        if meta is None:
            raise RuntimeError(f"compaction session 不存在: {session_key}")
        return str(meta["created_at"])

    def _source_mutation_digest(
        self,
        session_key: str,
        checkpoint: ContextCompaction,
    ) -> str:
        """Hash every canonical row covered by selected and retained source plans."""

        source_ids = tuple(
            dict.fromkeys(
                [
                    *checkpoint.source_message_ids,
                    *(str(item["id"]) for item in checkpoint.retained_tail),
                ]
            )
        )
        return self._store.source_mutation_digest(session_key, source_ids)

    def _persist_checkpoint(
        self,
        checkpoint: ContextCompaction,
        *,
        head: CompactionHead,
        prepare: CompactionPrepare | None = None,
        source_mutation_digest: str | None = None,
    ) -> SessionCompaction:
        canonical_plan = canonical_source_plan(checkpoint.selected_source_messages)
        plan_digest = source_plan_digest(canonical_plan)
        return self._store.persist_compaction(
            session_key=head.session_key,
            trigger=checkpoint.trigger,
            summary=checkpoint.summary,
            source_ref=checkpoint.source_ref,
            source_plan_digest=plan_digest,
            source_from_seq=checkpoint.source_from_seq,
            consolidated_through_seq=checkpoint.consolidated_through_seq,
            source_message_ids=checkpoint.source_message_ids,
            retained_tail=checkpoint.retained_tail,
            model_runtime_id=checkpoint.model_runtime_id,
            model=checkpoint.model,
            context_window=checkpoint.context_window,
            threshold_tokens=checkpoint.soft_limit_tokens,
            hard_input_tokens=checkpoint.hard_input_tokens,
            keep_recent_tokens=checkpoint.keep_recent_tokens,
            tokens_before=checkpoint.estimated_tokens_before,
            tokens_after=checkpoint.estimated_tokens_after,
            summary_usage=_usage_payload(checkpoint.summary_usage),
            parent_generation=head.parent_generation,
            generation=head.next_generation,
            summary_format_version=1,
            prepare=prepare,
            source_mutation_digest=source_mutation_digest,
        )


def _consume_checkpoint_source_plan(
    requested: tuple[dict[str, Any], ...],
    label: str,
    session_key: str,
    available: dict[tuple[str, int], list[tuple[int, int, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[tuple[int, int, str]]]:
    """Resolve one source plan against the latest rendered message occurrences."""

    canonical_plan: list[dict[str, Any]] = []
    occurrences: list[tuple[int, int, str]] = []
    for item in requested:
        key = (str(item["id"]), int(item["seq"]))
        candidates = available.get(key)
        if not candidates:
            raise ValueError(
                f"compaction {label} message 不存在或已被修改: "
                f"{session_key}:{item['id']}:{item['seq']}"
            )
        unit_index, message_index, canonical_message = candidates.pop(0)
        if item["message"] != canonical_message:
            raise RuntimeError(
                f"compaction {label} rendered message 已被修改: "
                f"{session_key}:{item['id']}:{item['seq']}"
            )
        occurrences.append((unit_index, message_index, str(item["unit_ref"])))
        canonical_plan.append(
            {
                "id": item["id"],
                "seq": item["seq"],
                "unit_ref": item["unit_ref"],
                "message": canonical_message,
            }
        )
    return canonical_plan, occurrences


def _validate_checkpoint_source_units(
    occurrences: list[tuple[int, int, str]],
    current_units: tuple[CommittedContextUnit, ...],
    label: str,
    *,
    require_generated_ref: bool,
) -> tuple[int, ...]:
    """Require complete logical units in chronological plan order."""

    grouped_units: dict[int, list[tuple[int, str]]] = {}
    seen_unit_indices: set[int] = set()
    previous_unit_index: int | None = None
    for unit_index, message_index, unit_ref in occurrences:
        if unit_index != previous_unit_index:
            if unit_index in seen_unit_indices:
                raise RuntimeError(f"compaction {label} logical units 不能交错")
            if previous_unit_index is not None and unit_index < previous_unit_index:
                raise RuntimeError(f"compaction {label} logical units 必须按历史顺序")
            seen_unit_indices.add(unit_index)
            previous_unit_index = unit_index
        grouped_units.setdefault(unit_index, []).append((message_index, unit_ref))
    for selection_index, (unit_index, unit_occurrences) in enumerate(
        grouped_units.items()
    ):
        unit = current_units[unit_index]
        expected_positions = list(range(len(unit.messages)))
        actual_positions = [message_index for message_index, _ in unit_occurrences]
        if actual_positions != expected_positions:
            raise RuntimeError(
                f"compaction {label} 必须覆盖完整 logical history unit"
            )
        if require_generated_ref:
            unit_refs = {unit_ref for _, unit_ref in unit_occurrences}
            expected_unit_ref = (
                f"{unit.source_from_seq}:{unit.consolidated_through_seq}:"
                f"{selection_index}"
            )
            if unit_refs != {expected_unit_ref}:
                raise RuntimeError(
                    f"compaction {label} unit_ref 与 logical unit 不一致"
                )
    return tuple(grouped_units)


def _retained_tail_units(
    retained_tail: tuple[dict[str, Any], ...],
) -> list[CommittedContextUnit]:
    if not retained_tail:
        return []
    grouped: dict[str, tuple[list[dict[str, Any]], list[tuple[str, int]]]] = {}
    for item in retained_tail:
        message = item.get("message")
        raw_id = item.get("id")
        raw_seq = item.get("seq")
        if not isinstance(message, dict) or not isinstance(raw_id, str) or not raw_id:
            raise ValueError("compaction retained_tail provenance 无效")
        if not isinstance(raw_seq, int) or isinstance(raw_seq, bool) or raw_seq < 0:
            raise ValueError("compaction retained_tail seq 无效")
        unit_ref = item.get("unit_ref")
        if not isinstance(unit_ref, str) or not unit_ref.strip():
            raise ValueError("compaction retained_tail 缺少 unit_ref")
        messages, refs = grouped.setdefault(unit_ref, ([], []))
        messages.append(dict(message))
        refs.append((raw_id, raw_seq))
    return [
        CommittedContextUnit(
            source_from_seq=min(seq for _, seq in refs),
            consolidated_through_seq=max(seq for _, seq in refs),
            source_message_ids=tuple(dict.fromkeys(ref[0] for ref in refs)),
            messages=tuple(messages),
            message_refs=tuple(refs),
        )
        for messages, refs in grouped.values()
    ]


def _usage_payload(usage: ModelUsage | None) -> dict[str, object]:
    if usage is None:
        return {}
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "request_count": usage.request_count,
        "covered_request_count": usage.covered_request_count,
        "coverage": usage.coverage.value,
    }


def _receipt_digest(receipt: dict[str, object]) -> str:
    identity = {
        "session_key": receipt.get("session_key"),
        "parent_generation": receipt.get("parent_generation"),
        "next_generation": receipt.get("next_generation"),
        "model_runtime_id": receipt.get("model_runtime_id"),
        "model": receipt.get("model"),
        "checkpoint": receipt.get("checkpoint"),
        "markdown_draft": receipt.get("markdown_draft"),
    }
    if receipt.get("version") == 2:
        identity["session_created_at"] = receipt.get("session_created_at")
        identity["source_plan_digest"] = receipt.get("source_plan_digest")
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _draft_from_receipt(receipt: dict[str, object]) -> CompactionMarkdownDraft:
    raw = receipt.get("markdown_draft")
    if not isinstance(raw, dict):
        raise ValueError("compaction receipt markdown_draft 无效")
    payloads = raw.get("history_entry_payloads", [])
    if not isinstance(payloads, list):
        raise ValueError("compaction receipt history_entry_payloads 无效")
    normalized: list[tuple[str, int]] = []
    for item in payloads:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("compaction receipt history entry 无效")
        summary, weight = item
        if not isinstance(summary, str) or not isinstance(weight, int):
            raise ValueError("compaction receipt history entry 类型无效")
        normalized.append((summary, weight))
    fields = ("source_ref", "pending_items", "conversation", "scope_channel", "scope_chat_id")
    values = {field: raw.get(field, "") for field in fields}
    if not all(isinstance(value, str) for value in values.values()):
        raise ValueError("compaction receipt markdown 字段类型无效")
    return CompactionMarkdownDraft(
        source_ref=str(values["source_ref"]),
        history_entry_payloads=tuple(normalized),
        pending_items=str(values["pending_items"]),
        conversation=str(values["conversation"]),
        scope_channel=str(values["scope_channel"]),
        scope_chat_id=str(values["scope_chat_id"]),
    )


def _receipt_payload(
    checkpoint: ContextCompaction,
    *,
    session_key: str,
    head: CompactionHead,
    markdown_draft: CompactionMarkdownDraft,
    model_runtime_id: str,
    model: str,
    session_created_at: str,
) -> dict[str, object]:
    draft = {
        "source_ref": markdown_draft.source_ref,
        "history_entry_payloads": [list(item) for item in markdown_draft.history_entry_payloads],
        "pending_items": markdown_draft.pending_items,
        "conversation": markdown_draft.conversation,
        "scope_channel": markdown_draft.scope_channel,
        "scope_chat_id": markdown_draft.scope_chat_id,
    }
    checkpoint_payload = checkpoint.to_payload()
    canonical_plan = canonical_source_plan(checkpoint.selected_source_messages)
    plan_digest = source_plan_digest(canonical_plan)
    identity = {
        "session_key": session_key,
        "session_created_at": normalize_session_created_at(session_created_at),
        "source_plan_digest": plan_digest,
        "parent_generation": head.parent_generation,
        "next_generation": head.next_generation,
        "model_runtime_id": model_runtime_id,
        "model": model,
        "checkpoint": checkpoint_payload,
        "markdown_draft": draft,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "version": 2,
        "source_ref": checkpoint.source_ref,
        "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "selection_digest": checkpoint.selection_digest,
        **identity,
    }
