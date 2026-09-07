from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, cast

import pytest

from agent.control.context import running_turn_id
from agent.plugin_composition import (
    BoundModelDescriptor,
    ModelRequest,
)
from plugins.compaction.engine import (
    CommittedContextUnit,
    ContextCompaction,
    ContextCompactionError,
    ContextCompactor,
    ContextPayloadSegments,
    SUMMARY_HEADINGS,
    canonical_source_plan,
    compaction_scope_id,
    compaction_source_ref,
    normalize_session_created_at,
    source_plan_digest,
    _selection_digest,
)
from agent.plugin_composition import LLMResponse
from plugins.compaction.runtime import (
    SessionCompactionRuntime,
    _receipt_digest,
    _receipt_payload,
)
from agent.plugin_composition import SessionCompactionStorage


class _ReceiptAdapter:
    def __init__(self, markdown: object) -> None:
        self._markdown = markdown

    def read(self, source_ref: str) -> dict[str, object] | None:
        return self._markdown.read_compaction_receipt(source_ref)  # type: ignore[attr-defined,no-any-return]

    def write(
        self,
        source_ref: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return self._markdown.write_compaction_receipt(source_ref, payload)  # type: ignore[attr-defined,no-any-return]

    def list_all(self) -> tuple[dict[str, object], ...]:
        return tuple(  # type: ignore[attr-defined]
            dict(payload) for payload in self._markdown.receipts.values()
        )


def _runtime(
    manager: SessionManager,
    markdown: object,
    session_key: str = "session",
) -> SessionCompactionRuntime:
    if session_key == "session" and manager.control_store.get_session_meta(
        session_key
    ) is None:
        keys = tuple(manager._cache)
        if len(keys) != 1:
            raise RuntimeError("test runtime needs one explicit Session scope")
        session_key = keys[0]
    session = manager.get_existing(session_key)
    return SessionCompactionRuntime(
        storage=SessionCompactionStorage(manager).scope(
            session.issue_projection_grant(running_turn_id.get())
        ),
        receipts=_ReceiptAdapter(markdown),
    )
from session.manager import Session, SessionManager
from session.store import (
    CompactionHead,
    CompactionPrepare,
    SessionCompactionPrepareConflictError,
)
from tests.model_plugin_fakes import BoundChatModelFake

SessionManagerFactory = Callable[[Path], SessionManager]


@dataclass(frozen=True)
class CompactionMarkdownDraft:
    source_ref: str
    history_entry_payloads: tuple[tuple[str, int], ...] = ()
    pending_items: str = ""


@pytest.fixture(autouse=True)
def _active_projection_turn() -> Generator[None, None, None]:
    token = running_turn_id.set("test:compaction")
    try:
        yield
    finally:
        running_turn_id.reset(token)


@pytest.fixture
def session_manager_factory() -> Generator[SessionManagerFactory, None, None]:
    """Create and close every SessionManager owned by one test."""

    managers: list[SessionManager] = []

    def factory(path: Path) -> SessionManager:
        manager = SessionManager(path)
        managers.append(manager)
        return manager

    yield factory
    for manager in managers:
        manager.close()


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


class _MarkdownCompactionProbe(_MarkdownReceiptProbe):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_count = 0

    async def prepare_compaction_markdown(
        self,
        selected_source_messages,
        *,
        source_ref: str,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> CompactionMarkdownDraft:
        self.prepare_count += 1
        assert selected_source_messages
        return CompactionMarkdownDraft(source_ref=source_ref)


def _seed_two_unit_checkpoint(
    manager: SessionManager,
    session_key: str,
    *,
    suppress_post_commit: bool = False,
) -> tuple[Session, CompactionHead, ContextCompaction, str]:
    """Create one selected unit and one retained unit backed by canonical rows."""

    session = manager.get_or_create(session_key)
    effects = {"post_commit": "suppress"} if suppress_post_commit else None
    if effects is None:
        session.add_message("user", "old user", control_turn_id="turn-old")
        session.add_message("assistant", "old reply", control_turn_id="turn-old")
    else:
        session.add_message(
            "user", "old user", control_turn_id="turn-old", effects=effects
        )
        session.add_message(
            "assistant", "old reply", control_turn_id="turn-old", effects=effects
        )
    session.add_message("user", "tail user", control_turn_id="turn-tail")
    session.add_message("assistant", "tail reply", control_turn_id="turn-tail")
    manager.save(session)
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    selected_unit, retained_unit = session.history_units()
    selected = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": f"{selected_unit.source_from_seq}:"
            f"{selected_unit.consolidated_through_seq}:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(
            selected_unit.messages,
            selected_unit.message_refs,
        )
    )
    retained_tail = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": f"{retained_unit.source_from_seq}:"
            f"{retained_unit.consolidated_through_seq}:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(
            retained_unit.messages,
            retained_unit.message_refs,
        )
    )
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
        source_from_seq=selected_unit.source_from_seq,
        consolidated_through_seq=selected_unit.consolidated_through_seq,
        source_message_ids=selected_unit.source_message_ids,
        retained_tail=retained_tail,
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="source-fence",
        selected_source_messages=selected,
    )
    retained_id = str(retained_unit.message_refs[0][0])
    return session, head, checkpoint, retained_id


class _CountingProvider:
    context_window: int = 1000
    runtime_id: str = "runtime"

    def __init__(
        self, *, context_window: int = 1000, runtime_id: str = "runtime"
    ) -> None:
        self.context_window = context_window
        self.runtime_id = runtime_id
        self.calls = 0

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return sum(int(message.get("tokens", 1)) for message in messages)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return sum(int(message.get("tokens", 1)) for message in messages)

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="\n".join(SUMMARY_HEADINGS))

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return BoundChatModelFake(
            self,
            model=str(getattr(self, "model", "model")),
        ).descriptor

    @property
    def max_tool_schemas(self) -> int | None:
        return None

    async def complete(self, request: ModelRequest) -> LLMResponse:
        return await BoundChatModelFake(
            self,
            model=str(getattr(self, "model", "model")),
        ).complete(request)


def _checkpoint_source_mutation_digest(
    manager: SessionManager,
    session_key: str,
    checkpoint: ContextCompaction,
) -> str:
    source_ids = tuple(
        dict.fromkeys(
            [
                *checkpoint.source_message_ids,
                *(str(item["id"]) for item in checkpoint.retained_tail),
            ]
        )
    )
    return manager.control_store.source_mutation_digest(session_key, source_ids)


def _seed_receipt(
    tmp_path: Path,
    manager_factory: SessionManagerFactory,
    *,
    version: int = 4,
) -> tuple[SessionManager, _MarkdownReceiptProbe, str]:
    manager = manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "persisted")
    manager.save(session)
    probe = _MarkdownReceiptProbe()
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    unit = session.history_units()[0]
    selected_source_messages = tuple(
        {
            "id": message_id,
            "seq": seq,
            "unit_ref": "0:0:0",
            "message": dict(message),
        }
        for message, (message_id, seq) in zip(unit.messages, unit.message_refs)
    )
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
        source_from_seq=unit.source_from_seq,
        consolidated_through_seq=unit.consolidated_through_seq,
        source_message_ids=unit.source_message_ids,
        retained_tail=(),
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="selection",
        selected_source_messages=selected_source_messages,
    )
    manager.control_store.prepare_compaction(
        session_key=session.key,
        session_created_at=session.created_at.isoformat(),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        source_ref=source_ref,
        source_from_seq=checkpoint.source_from_seq,
        consolidated_through_seq=checkpoint.consolidated_through_seq,
        source_message_ids=checkpoint.source_message_ids,
        retained_tail=checkpoint.retained_tail,
    )
    receipt = _receipt_payload(
        checkpoint,
        session_key=session.key,
        head=head,
        model_runtime_id="runtime",
        model="model",
        session_created_at=normalize_session_created_at(session.created_at),
        source_mutation_digest=_checkpoint_source_mutation_digest(
            manager,
            session.key,
            checkpoint,
        ),
        scope_channel="",
        scope_chat_id="",
    )
    if version == 2:
        receipt["version"] = 2
        receipt["markdown_draft"] = {
            "source_ref": source_ref,
            "history_entry_payloads": [],
            "pending_items": "",
            "conversation": "",
            "scope_channel": "",
            "scope_chat_id": "",
        }
        receipt.pop("scope_channel")
        receipt.pop("scope_chat_id")
        receipt.pop("source_mutation_digest")
        receipt["digest"] = _receipt_digest(receipt)
    elif version == 3:
        receipt["version"] = 3
        receipt["digest"] = _receipt_digest(receipt)
    probe.receipts[source_ref] = receipt
    return manager, probe, source_ref


def test_compaction_scope_separates_session_incarnations_and_reloads_stably() -> None:
    provider = _CountingProvider()
    unit = CommittedContextUnit(
        source_from_seq=0,
        consolidated_through_seq=0,
        source_message_ids=("m0",),
        messages=({"role": "user", "content": "same"},),
        message_refs=(("m0", 0),),
    )
    first = compaction_scope_id("same-key", "2026-08-08T00:00:00+00:00")
    reloaded = compaction_scope_id("same-key", "2026-08-08T08:00:00+08:00")
    recreated = compaction_scope_id("same-key", "2026-08-09T00:00:00+00:00")
    assert first == reloaded
    assert first != recreated
    assert compaction_source_ref(first, 1) != compaction_source_ref(recreated, 1)
    digest_kwargs = {
        "provider": provider,
        "model": "model",
        "soft_limit_tokens": 74,
        "hard_input_tokens": 90,
        "keep_recent_tokens": 20,
    }
    assert _selection_digest(
        (unit,), (), scope_id=first, **digest_kwargs
    ) != _selection_digest((unit,), (), scope_id=recreated, **digest_kwargs)


def test_receipt_recovery_skips_provider_calls(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    runtime = _runtime(manager, markdown)
    session = manager.get_existing("session")
    provider_calls = 0

    recovered = asyncio.run(runtime.recover_pending(session))

    assert recovered is not None
    assert provider_calls == 0
    assert session.last_consolidated == recovered.generation
    assert markdown.commit_count == 0
    receipt = markdown.receipts[source_ref]
    raw_checkpoint = receipt["checkpoint"]
    assert isinstance(raw_checkpoint, dict)
    raw_plan = raw_checkpoint["selected_source_messages"]
    assert isinstance(raw_plan, list)
    persisted = manager.control_store.get_compaction("session", recovered.generation)
    assert persisted is not None
    assert persisted.source_plan_digest == source_plan_digest(
        canonical_source_plan(raw_plan)
    )
    assert (
        manager.control_store.get_compaction_prepare("session", source_ref=source_ref)
        is None
    )


def test_v2_receipt_recovery_defers_markdown_to_durable_fact_reader(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, _ = _seed_receipt(
        tmp_path,
        session_manager_factory,
        version=2,
    )
    runtime = _runtime(manager, markdown)

    recovered = asyncio.run(runtime.recover_pending(manager.get_existing("session")))

    assert recovered is not None
    assert markdown.commit_count == 0
    assert len(markdown.receipts) == 1


def test_v2_receipt_without_prepare_still_fails_loud(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(
        tmp_path,
        session_manager_factory,
        version=2,
    )
    prepare = manager.control_store.get_compaction_prepare(
        "session",
        source_ref=source_ref,
    )
    assert prepare is not None
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()
    runtime = _runtime(manager, markdown)

    with pytest.raises(RuntimeError, match="durable prepare 缺失"):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))


def _seed_orphan_prepare(
    tmp_path: Path,
    manager_factory: SessionManagerFactory,
) -> tuple[SessionManager, SessionCompactionRuntime, CompactionPrepare]:
    manager = manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "prepared")
    manager.save(session)
    head = manager.control_store.get_compaction_head(session.key)
    unit = session.history_units()[0]
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
    prepare = manager.control_store.prepare_compaction(
        session_key=session.key,
        session_created_at=session.created_at.isoformat(),
        generation=head.next_generation,
        parent_generation=head.parent_generation,
        source_ref=source_ref,
        source_from_seq=unit.source_from_seq,
        consolidated_through_seq=unit.consolidated_through_seq,
        source_message_ids=unit.source_message_ids,
        retained_tail=(),
    )
    runtime = _runtime(manager, _MarkdownReceiptProbe())
    return manager, runtime, prepare


def test_prepare_without_receipt_is_released_on_recovery(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, runtime, prepare = _seed_orphan_prepare(tmp_path, session_manager_factory)
    session = manager.get_existing("session")
    head = manager.control_store.get_compaction_head(session.key)

    assert asyncio.run(runtime.recover_pending(session)) is None
    assert (
        manager.control_store.get_compaction_prepare(
            session.key, source_ref=prepare.source_ref
        )
        is None
    )
    assert manager.control_store.get_compaction_head(session.key) == head


@pytest.mark.parametrize(
    ("column", "value", "match"),
    (
        ("source_message_ids_json", "[]", "source_message_ids"),
        (
            "retained_tail_json",
            '[{"id":"session:0","seq":true,"message":{},"unit_ref":"0:0:0"}]',
            "retained_tail",
        ),
        (
            "retained_tail_json",
            '[{"id":"session:0","seq":0,"message":{}}]',
            "retained_tail",
        ),
        ("session_created_at", "", "identity"),
        ("prepared_at", "", "prepared_at"),
    ),
)
def test_corrupt_prepare_without_receipt_fails_loud_and_keeps_row(
    tmp_path: Path,
    column: str,
    value: object,
    match: str,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, runtime, prepare = _seed_orphan_prepare(tmp_path, session_manager_factory)
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            f"UPDATE session_compaction_prepares SET {column} = ? "
            "WHERE session_key = ? AND generation = ?",
            (value, prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()

    with pytest.raises(ValueError, match=match):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))
    with manager.control_store._lock:
        raw = manager.control_store._conn.execute(
            "SELECT 1 FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        ).fetchone()
    assert raw is not None


def test_v3_receipt_without_prepare_is_audit_only(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    prepare = manager.control_store.get_compaction_prepare(
        "session", source_ref=source_ref
    )
    assert prepare is not None
    # Explicit SQL corruption simulates a receipt whose durable prepare vanished.
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()
    runtime = _runtime(manager, markdown)

    assert asyncio.run(runtime.recover_pending(manager.get_existing("session"))) is None
    assert markdown.commit_count == 0


def test_v3_receipt_without_prepare_still_validates_identity(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    prepare = manager.control_store.get_compaction_prepare(
        "session", source_ref=source_ref
    )
    assert prepare is not None
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )
        manager.control_store._conn.commit()
    receipt = markdown.receipts[source_ref]
    receipt["session_key"] = "other-session"
    receipt["digest"] = _receipt_digest(receipt)
    runtime = _runtime(manager, markdown)

    with pytest.raises(ValueError, match="session_key 冲突"):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))


def test_v3_recovery_rejects_raw_source_mutation(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    receipt = markdown.receipts[source_ref]
    checkpoint = cast(dict[str, Any], receipt["checkpoint"])
    source_ids = cast(list[str], checkpoint["source_message_ids"])
    with manager.control_store._lock:
        manager.control_store._conn.execute(
            "UPDATE messages SET ts = ts || '-tampered' WHERE id = ?",
            (source_ids[0],),
        )
        manager.control_store._conn.commit()
    runtime = _runtime(manager, markdown)

    with pytest.raises(RuntimeError, match="source snapshot"):
        asyncio.run(runtime.recover_pending(manager.get_existing("session")))
    assert (
        manager.control_store.get_compaction_prepare("session", source_ref=source_ref)
        is not None
    )


def test_suppressed_turn_commit_advances_ledger_without_markdown(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager = session_manager_factory(tmp_path)
    session = manager.get_or_create("session")
    session.add_message(
        "user",
        "excluded",
        effects={"post_commit": "suppress"},
    )
    manager.save(session)
    markdown = _MarkdownCompactionProbe()
    runtime = _runtime(manager, markdown)
    head = manager.control_store.get_compaction_head(session.key)
    unit = session.history_units()[0]
    message = unit.messages[0]
    message_id, message_seq = unit.message_refs[0]
    source_ref = compaction_source_ref(
        compaction_scope_id(session.key, session.created_at),
        head.next_generation,
    )
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
        source_from_seq=message_seq,
        consolidated_through_seq=message_seq,
        source_message_ids=(message_id,),
        retained_tail=(),
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="selection",
        selected_source_messages=(
            {
                "id": message_id,
                "seq": message_seq,
                "unit_ref": "0:0:0",
                "message": dict(message),
            },
        ),
    )

    row = asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))

    assert row.generation == 1
    expected_digest = source_plan_digest(
        canonical_source_plan(checkpoint.selected_source_messages)
    )
    assert row.source_plan_digest == expected_digest
    reloaded_row = manager.control_store.get_compaction(session.key, row.generation)
    assert reloaded_row is not None
    assert reloaded_row.source_plan_digest == expected_digest
    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    assert markdown.prepare_count == 0
    assert markdown.commit_count == 0
    assert (
        manager.control_store.get_compaction_prepare(session.key, source_ref=source_ref)
        is None
    )


def test_receipt_recovery_ignores_unrelated_session_metadata_change(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path, session_manager_factory)
    session = manager.get_existing("session")
    session.metadata["plugin_state"] = True
    manager.save(session)
    runtime = _runtime(manager, markdown)

    recovered = asyncio.run(runtime.recover_pending(session))

    assert recovered is not None
    assert recovered.generation == 1
    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    assert markdown.commit_count == 0
    persisted = manager.control_store.get_compaction(session.key, recovered.generation)
    assert persisted is not None
    assert persisted.source_plan_digest == str(
        markdown.receipts[recovered.source_ref]["source_plan_digest"]
    )


def test_v3_receipt_recovery_retries_ledger_without_markdown(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    runtime = _runtime(manager, markdown)
    session = manager.get_existing("session")
    provider_calls = 0

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
    assert (
        manager.control_store.get_compaction_prepare(session.key, source_ref=source_ref)
        is not None
    )
    manager.control_store.persist_compaction = original_persist  # type: ignore[method-assign]
    manager.invalidate(session.key)
    resumed = manager.get_existing(session.key)
    resumed_runtime = _runtime(manager, markdown)
    recovered = asyncio.run(resumed_runtime.recover_pending(resumed))

    assert recovered is not None
    assert provider_calls == 0
    assert resumed.last_consolidated == recovered.generation
    persisted = manager.control_store.get_compaction(
        resumed.key,
        recovered.generation,
    )
    assert persisted is not None
    assert persisted.source_plan_digest == str(
        markdown.receipts[source_ref]["source_plan_digest"]
    )
    assert markdown.commit_count == 0
    assert (
        manager.control_store.get_compaction_prepare(resumed.key, source_ref=source_ref)
        is None
    )


def test_tampered_receipt_is_rejected_before_markdown_or_ledger(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, source_ref = _seed_receipt(tmp_path, session_manager_factory)
    runtime = _runtime(manager, markdown)
    checkpoint = markdown.receipts[source_ref]["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["summary"] = "tampered"
    session = manager.get_existing("session")

    with pytest.raises(ValueError, match="digest"):
        asyncio.run(runtime.recover_pending(session))

    assert manager.control_store.get_compaction_head("session").parent_generation == 0
    assert markdown.commit_count == 0


def test_pending_prepare_rejects_source_deletion(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path, session_manager_factory)
    session = manager.get_existing("session")
    message_id = str(session.messages[0]["id"])
    with pytest.raises(
        SessionCompactionPrepareConflictError,
        match="pending compaction prepare",
    ):
        manager.control_store.delete_messages_batch([message_id])
    assert manager.control_store.get_compaction_head("session").parent_generation == 0
    assert markdown.commit_count == 0


def test_retained_tail_without_unit_ref_is_rejected(
    tmp_path: Path,
    session_manager_factory: SessionManagerFactory,
) -> None:
    manager, _, _ = _seed_receipt(tmp_path, session_manager_factory)
    head = manager.control_store.get_compaction_head("session")
    with pytest.raises(ValueError, match="unit_ref"):
        manager.control_store.persist_compaction(
            session_key="session",
            trigger="soft_limit",
            summary="\n".join(SUMMARY_HEADINGS),
            source_ref="bad-unit-ref",
            source_plan_digest="a" * 64,
            source_from_seq=0,
            consolidated_through_seq=0,
            source_message_ids=("session:0",),
            retained_tail=({"id": "session:0", "seq": 0, "message": {"role": "user"}},),
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
    receipt = _receipt_payload(
        first_result.checkpoint,
        session_key="session",
        head=head,
        model_runtime_id="runtime",
        model="model",
        session_created_at="2026-08-08T00:00:00+00:00",
        source_mutation_digest="0" * 64,
        scope_channel="",
        scope_chat_id="",
    )
    second = ContextCompactor(
        provider=provider,
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
