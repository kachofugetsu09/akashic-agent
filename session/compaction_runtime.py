from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from agent.model_runtime.context_compaction import (
    ActiveCompaction,
    CommittedContextUnit,
    ContextCompaction,
    ContextPayloadSegments,
    build_compaction_messages,
    compaction_source_ref,
    _checkpoint_from_receipt,
)
from agent.model_runtime.types import ModelUsage
from core.memory.markdown import CompactionMarkdownDraft
from session.memory_policy import excludes_memory
from session.store import CompactionHead, SessionCompaction, SessionStore

if TYPE_CHECKING:
    from core.memory.markdown import MarkdownMemoryMaintenance
    from session.manager import Session
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
        session: "Session",
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection: ...

    async def recover_pending(self, session: "Session") -> SessionCompaction | None: ...

    async def commit_checkpoint(
        self,
        session: "Session",
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
        session: "Session",
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

    async def recover_pending(self, session: "Session") -> SessionCompaction | None:
        """Finish a receipt left between Markdown and ledger commits."""

        head = self._store.get_compaction_head(session.key)
        return await self._recover_pending(session, head)

    async def _recover_pending(
        self,
        session: "Session",
        head: CompactionHead,
    ) -> SessionCompaction | None:
        source_ref = compaction_source_ref(session.key, head.next_generation)
        receipt = self._markdown.read_compaction_receipt(source_ref)
        if receipt is None:
            return None
        if receipt.get("session_key") != session.key:
            raise ValueError("compaction receipt session_key 冲突")
        if receipt.get("parent_generation") != head.parent_generation or receipt.get(
            "next_generation"
        ) != head.next_generation:
            raise RuntimeError("compaction receipt 与当前 ledger head 冲突")
        raw_digest = receipt.get("selection_digest")
        raw_checkpoint = receipt.get("checkpoint")
        if receipt.get("version") != 1:
            raise ValueError("compaction receipt version 无效")
        if not isinstance(raw_digest, str) or not isinstance(raw_checkpoint, dict):
            raise ValueError("compaction receipt schema 无效")
        if receipt.get("digest") != _receipt_digest(receipt):
            raise ValueError("compaction receipt digest 校验失败")
        _, _, checkpoint = _checkpoint_from_receipt(
            receipt,
            source_ref=source_ref,
            selection_digest=raw_digest,
        )
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
        if not excludes_memory(session.key, session.metadata):
            # Markdown append uses its own source_ref index, so replay is idempotent.
            await self._markdown.commit_compaction_markdown(draft)
            after_markdown = self._store.get_compaction_head(session.key)
            if after_markdown != head:
                raise RuntimeError("compaction receipt recovery 时 ledger head 发生变化")
        else:
            # excluded session 仍推进自己的 compaction ledger，但不产生记忆副作用。
            if self._store.get_compaction_head(session.key) != head:
                raise RuntimeError("excluded compaction receipt recovery 时 ledger head 发生变化")
        row = self._persist_checkpoint(
            checkpoint,
            head=head,
        )
        session.last_consolidated = row.generation
        return row

    async def commit_checkpoint(
        self,
        session: "Session",
        checkpoint: ContextCompaction,
        *,
        head: CompactionHead,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> SessionCompaction:
        """Commit receipt, Markdown effects, then the SQLite ledger row."""

        if excludes_memory(session.key, session.metadata):
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
                raise RuntimeError("excluded compaction ledger head 在提交前已变化")
            row = self._persist_checkpoint(
                checkpoint,
                head=head,
            )
            session.last_consolidated = row.generation
            return row

        markdown_draft = await self._build_markdown_draft(
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
        receipt = _receipt_payload(
            checkpoint,
            session_key=head.session_key,
            head=head,
            markdown_draft=markdown_draft,
            model_runtime_id=checkpoint.model_runtime_id,
            model=checkpoint.model,
        )
        existing = self._markdown.read_compaction_receipt(checkpoint.source_ref)
        if existing is None:
            self._markdown.write_compaction_receipt(checkpoint.source_ref, receipt)
        elif existing != receipt:
            raise ValueError("compaction receipt 内容冲突")
        await self._markdown.commit_compaction_markdown(markdown_draft)
        after_markdown = self._store.get_compaction_head(head.session_key)
        if after_markdown != head:
            raise RuntimeError("Markdown 提交期间 compaction ledger head 发生变化")
        row = self._persist_checkpoint(
            checkpoint,
            head=head,
        )
        session.last_consolidated = row.generation
        return row

    async def _build_markdown_draft(
        self,
        session: "Session",
        checkpoint: ContextCompaction,
        *,
        scope_channel: str,
        scope_chat_id: str,
    ) -> CompactionMarkdownDraft:
        """Build Markdown input exclusively from Store-owned committed source rows."""

        if not checkpoint.selected_source_messages:
            raise ValueError("compaction Markdown source plan 不能为空")
        source_ids = set(checkpoint.source_message_ids)
        for item in checkpoint.selected_source_messages:
            message_id = item.get("id")
            raw_seq = item.get("seq")
            message = item.get("message")
            unit_ref = item.get("unit_ref")
            if (
                not isinstance(message_id, str)
                or message_id not in source_ids
                or not isinstance(raw_seq, int)
                or not isinstance(unit_ref, str)
                or not unit_ref.strip()
                or not isinstance(message, dict)
            ):
                raise ValueError("compaction Markdown source plan 无效")
            row = self._store.get_message(message_id)
            if (
                row is None
                or row.get("session_key") != session.key
                or row.get("seq") != raw_seq
            ):
                raise ValueError(
                    "compaction Markdown source message 不存在: "
                    f"{session.key}:{message_id}"
                )
        return await self._markdown.prepare_compaction_markdown(
            tuple(checkpoint.selected_source_messages),
            source_ref=checkpoint.source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )

    def _persist_checkpoint(
        self,
        checkpoint: ContextCompaction,
        *,
        head: CompactionHead,
    ) -> SessionCompaction:
        return self._store.persist_compaction(
            session_key=head.session_key,
            trigger=checkpoint.trigger,
            summary=checkpoint.summary,
            source_ref=checkpoint.source_ref,
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
        )


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
    identity = {
        "session_key": session_key,
        "parent_generation": head.parent_generation,
        "next_generation": head.next_generation,
        "model_runtime_id": model_runtime_id,
        "model": model,
        "checkpoint": checkpoint_payload,
        "markdown_draft": draft,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "version": 1,
        "source_ref": checkpoint.source_ref,
        "digest": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "selection_digest": checkpoint.selection_digest,
        **identity,
    }
