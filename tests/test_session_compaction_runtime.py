from __future__ import annotations

import asyncio
from pathlib import Path

from agent.model_runtime.context_compaction import (
    ContextCompaction,
    SUMMARY_HEADINGS,
    compaction_source_ref,
)
from core.memory.markdown import CompactionMarkdownDraft
from session.compaction_runtime import SessionCompactionRuntime, _receipt_payload
from session.manager import SessionManager


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
