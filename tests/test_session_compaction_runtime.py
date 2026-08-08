from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextCompaction,
    ContextCompactor,
    ContextPayloadSegments,
    SUMMARY_HEADINGS,
    compaction_source_ref,
)
from core.memory.markdown import CompactionMarkdownDraft
from session.compaction_runtime import SessionCompactionRuntime, _receipt_payload
from session.manager import SessionManager
from session.store import CompactionHead


class _MarkdownReceiptProbe:
    def __init__(self) -> None:
        self.receipts: dict[str, dict[str, object]] = {}
        self.commit_count = 0
        self.fail_after_commit = False

    def read_compaction_receipt(self, source_ref: str):
        return self.receipts.get(source_ref)

    def write_compaction_receipt(self, source_ref: str, payload: dict[str, object]):
        self.receipts[source_ref] = dict(payload)
        return dict(payload)

    async def commit_compaction_markdown(self, draft: CompactionMarkdownDraft):
        self.commit_count += 1
        if self.fail_after_commit:
            raise RuntimeError("simulated crash after Markdown side effect")


class _CountingProvider:
    context_window = 1000
    runtime_id = "runtime"

    def __init__(self) -> None:
        self.calls = 0

    def estimate_context_tokens(self, messages, tools):
        return sum(int(message.get("tokens", 1)) for message in messages)

    def estimate_appended_message_tokens(self, messages):
        return sum(int(message.get("tokens", 1)) for message in messages)

    async def chat(self, **kwargs):
        self.calls += 1
        from agent.model_runtime.types import LLMResponse

        return LLMResponse(content="\n".join(SUMMARY_HEADINGS))


def _seed_receipt(tmp_path: Path) -> tuple[SessionManager, _MarkdownReceiptProbe, str]:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "persisted")
    manager.save(session)
    probe = _MarkdownReceiptProbe()
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(session.key, head.next_generation)
    checkpoint = ContextCompaction(
        summary="\n".join(SUMMARY_HEADINGS),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        trigger="soft_limit",
        context_window=100,
        soft_limit_tokens=74,
        hard_input_tokens=90,
        keep_recent_tokens=20,
        estimated_tokens_before=80,
        estimated_tokens_after=40,
        source_from_seq=0,
        consolidated_through_seq=0,
        source_message_ids=(str(session.messages[0]["id"]),),
        retained_tail=(),
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="selection",
    )
    draft = CompactionMarkdownDraft(source_ref=source_ref)
    probe.receipts[source_ref] = _receipt_payload(
        checkpoint,
        session_key=session.key,
        head=head,
        markdown_draft=draft,
        model_runtime_id="runtime",
        model="model",
    )
    return manager, probe, source_ref


def test_receipt_recovery_skips_provider_calls(tmp_path: Path) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path)
    runtime = SessionCompactionRuntime(session_manager=manager, markdown=markdown)  # type: ignore[arg-type]
    session = manager.get_existing("session")
    provider_calls = 0

    recovered = asyncio.run(runtime.recover_pending(session))

    assert recovered is not None
    assert provider_calls == 0
    assert session.last_consolidated == recovered.generation
    assert markdown.commit_count == 1


def test_receipt_recovery_after_markdown_is_idempotent_and_skips_provider(
    tmp_path: Path,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path)
    runtime = SessionCompactionRuntime(session_manager=manager, markdown=markdown)  # type: ignore[arg-type]
    session = manager.get_existing("session")
    provider_calls = 0

    markdown.fail_after_commit = True
    try:
        asyncio.run(runtime.recover_pending(session))
    except RuntimeError as exc:
        assert "simulated crash" in str(exc)
    else:
        raise AssertionError("expected simulated crash")
    # Restart after Markdown/event side effect but before ledger insert.
    assert markdown.receipts[source_ref]["source_ref"] == source_ref
    markdown.fail_after_commit = False
    original_persist = manager.control_store.persist_compaction
    persist_calls = 0

    def fail_once(*args, **kwargs):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 1:
            raise RuntimeError("simulated crash before SQLite commit")
        return original_persist(*args, **kwargs)

    manager.control_store.persist_compaction = fail_once  # type: ignore[method-assign]
    try:
        asyncio.run(runtime.recover_pending(session))
    except RuntimeError as exc:
        assert "SQLite" in str(exc)
    else:
        raise AssertionError("expected SQLite crash")
    manager.control_store.persist_compaction = original_persist  # type: ignore[method-assign]
    manager.invalidate(session.key)
    resumed = manager.get_existing(session.key)
    resumed_runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=markdown,  # type: ignore[arg-type]
    )
    recovered = asyncio.run(resumed_runtime.recover_pending(resumed))

    assert recovered is not None
    assert provider_calls == 0
    assert resumed.last_consolidated == recovered.generation
    assert markdown.commit_count == 3


def test_tampered_receipt_is_rejected_before_markdown_or_ledger(tmp_path: Path) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path)
    runtime = SessionCompactionRuntime(session_manager=manager, markdown=markdown)  # type: ignore[arg-type]
    checkpoint = markdown.receipts[source_ref]["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["summary"] = "tampered"
    session = manager.get_existing("session")

    with pytest.raises(ValueError, match="digest"):
        asyncio.run(runtime.recover_pending(session))

    assert manager.control_store.get_compaction_head("session").parent_generation == 0
    assert markdown.commit_count == 0


def test_deleted_receipt_source_is_rejected_without_ledger_write(tmp_path: Path) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path)
    runtime = SessionCompactionRuntime(session_manager=manager, markdown=markdown)  # type: ignore[arg-type]
    session = manager.get_existing("session")
    message_id = str(session.messages[0]["id"])
    # Keep the in-memory cache stale while the Store deletion is authoritative.
    assert manager.control_store.delete_messages_batch([message_id]) == 1

    with pytest.raises(ValueError, match="不存在"):
        asyncio.run(runtime.recover_pending(session))

    assert manager.control_store.get_compaction_head("session").parent_generation == 0
    assert markdown.commit_count == 0


def test_retained_tail_without_unit_ref_is_rejected(tmp_path: Path) -> None:
    manager, _, _ = _seed_receipt(tmp_path)
    head = manager.control_store.get_compaction_head("session")
    with pytest.raises(ValueError, match="unit_ref"):
        manager.control_store.persist_compaction(
            session_key="session",
            trigger="soft_limit",
            summary="\n".join(SUMMARY_HEADINGS),
            source_ref="bad-unit-ref",
            source_from_seq=0,
            consolidated_through_seq=0,
            source_message_ids=("session:0",),
            retained_tail=(
                {"id": "session:0", "seq": 0, "message": {"role": "user"}},
            ),
            model_runtime_id="runtime",
            model="model",
            context_window=100,
            threshold_tokens=74,
            hard_input_tokens=90,
            keep_recent_tokens=20,
            tokens_before=80,
            tokens_after=40,
            summary_usage={},
            parent_generation=head.parent_generation,
            generation=head.next_generation,
        )


def test_context_compactor_receipt_resume_does_not_call_summary_provider() -> None:
    provider = _CountingProvider()
    units = tuple(
        CommittedContextUnit(
            source_from_seq=index,
            consolidated_through_seq=index,
            source_message_ids=(f"m{index}",),
            messages=({"role": "user", "content": f"u{index}", "tokens": 400},),
            message_refs=((f"m{index}", index),),
        )
        for index in (0, 1)
    )
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=units,
        current_anchor=({"role": "user", "content": "query", "tokens": 1},),
    )
    first = ContextCompactor(
        provider=provider,
        model="model",
        scope_id="session",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
    )
    first_messages = segments.flatten()
    first_result = asyncio.run(
        first.prepare(first_messages, pending_start=3, tools=[], force=True)
    )
    assert first_result.checkpoint is not None
    assert provider.calls == 1
    head = CompactionHead(
        session_key="session",
        parent_generation=0,
        next_generation=1,
    )
    draft = CompactionMarkdownDraft(source_ref=first_result.checkpoint.source_ref)
    receipt = _receipt_payload(
        first_result.checkpoint,
        session_key="session",
        head=head,
        markdown_draft=draft,
        model_runtime_id="runtime",
        model="model",
    )
    second = ContextCompactor(
        provider=provider,
        model="model",
        scope_id="session",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
        receipt_loader=lambda source_ref: receipt,
    )
    second_messages = segments.flatten()
    resumed = asyncio.run(
        second.prepare(second_messages, pending_start=3, tools=[], force=True)
    )

    assert resumed.checkpoint is not None
    assert resumed.checkpoint.summary == first_result.checkpoint.summary
    assert provider.calls == 1
