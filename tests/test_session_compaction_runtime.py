from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.core.passive_turn import DefaultReasoner
from agent.config_models import ContextCompactionConfig
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.looping.ports import LLMConfig
from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextCompaction,
    ContextCompactor,
    ContextPayloadSegments,
    SUMMARY_HEADINGS,
    compaction_source_ref,
)
from agent.model_runtime.types import LLMResponse
from agent.tools.registry import ToolRegistry
from core.memory.markdown import CompactionMarkdownDraft
from session.compaction_runtime import (
    CompactionProjection,
    SessionCompactionRuntime,
    _receipt_payload,
)
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
        return CompactionMarkdownDraft(
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
        )


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


class _GateProvider(_CountingProvider):
    context_window = 250
    runtime_id = "gate-runtime"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict[str, object]] = []

    def estimate_context_tokens(self, messages, tools):
        if len(messages) == 1 and str(messages[0].get("content", "")).startswith(
            "更新当前长任务"
        ):
            return 10
        total = 0
        for message in messages:
            role = message.get("role")
            if role == "system":
                total += 20
            elif role == "user":
                total += 80 if str(message.get("content", "")).startswith("old") else 10
            else:
                total += 10
        return total + len(tools)

    def estimate_appended_message_tokens(self, messages):
        return self.estimate_context_tokens(messages, [])

    async def chat(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        return LLMResponse(content="\n".join(SUMMARY_HEADINGS))


class _ScopedCompactionProvider(_GateProvider):
    context_window = 1_000

    def estimate_context_tokens(self, messages, tools):
        if any(
            "<session-context-compaction>" in str(message.get("content", ""))
            for message in messages
        ):
            return 5
        return 100

    def estimate_appended_message_tokens(self, messages):
        return 1

    async def chat(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        messages = kwargs.get("messages") or []
        content = "\n".join(str(message.get("content", "")) for message in messages)
        if "Closed history to consolidate" in content:
            marker = "A_SENTINEL" if "a-" in content else "B_SENTINEL"
            return LLMResponse(
                content="\n".join(SUMMARY_HEADINGS) + f"\n{marker}"
            )
        marker = "A_SENTINEL" if "a-" in content else "B_SENTINEL"
        return LLMResponse(content=f"reply-{marker}")


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


def test_excluded_session_commit_advances_ledger_without_markdown(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("session")
    session.metadata["skip_post_memory"] = True
    session.add_message("user", "excluded")
    manager.save(session)
    markdown = _MarkdownCompactionProbe()
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=markdown,  # type: ignore[arg-type]
    )
    head = manager.control_store.get_compaction_head(session.key)
    message = session.messages[0]
    message_id = str(message["id"])
    message_seq = int(message["seq"])
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
                "unit_ref": "session:unit:0",
                "message": {"role": "user", "content": "excluded"},
            },
        ),
    )

    row = asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))

    assert row.generation == 1
    assert manager.control_store.get_compaction_head(session.key).parent_generation == 1
    assert markdown.prepare_count == 0
    assert markdown.commit_count == 0


def test_excluded_receipt_recovery_advances_ledger_without_markdown(
    tmp_path: Path,
) -> None:
    manager, markdown, _ = _seed_receipt(tmp_path)
    session = manager.get_existing("session")
    session.metadata["skip_post_memory"] = True
    manager.save(session)
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=markdown,  # type: ignore[arg-type]
    )

    recovered = asyncio.run(runtime.recover_pending(session))

    assert recovered is not None
    assert recovered.generation == 1
    assert session.last_consolidated == 1
    assert markdown.commit_count == 0


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


def test_default_reasoner_gate_commits_real_runtime_before_provider_payload(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "old one")
    session.add_message("user", "old two")
    manager.save(session)
    markdown = _MarkdownCompactionProbe()
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=markdown,  # type: ignore[arg-type]
    )
    provider = _GateProvider()
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=provider, light_provider=provider),
        llm_config=LLMConfig(model="model", max_tokens=10),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        compaction_runtime=runtime,
    )
    projection = asyncio.run(
        runtime.projection(session, prefix=[], current_anchor=[], pending=[])
    )
    history = [
        message
        for unit in projection.segments.committed_units
        for message in unit.messages
    ]
    render_payload = [
        {"role": "system", "content": "root"},
        *history,
        {"role": "user", "content": "current request"},
    ]
    state = reasoner._build_compaction_state(
        session=session,
        projection=projection,
        initial_messages=render_payload,
        history_count=len(history),
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="chat",
    )
    state.compactor._keep_recent_tokens = 1
    call_result = asyncio.run(
        reasoner._call_provider(
            state,
            render_payload,
            tools=[],
            max_tokens=10,
        )
    )

    assert call_result.response.content == "\n".join(SUMMARY_HEADINGS)
    assert provider.calls == 2
    assert markdown.commit_count == 1
    assert session.last_consolidated == 1
    assert "current request" not in str(provider.requests[0]["messages"])
    assert provider.requests[1]["messages"] == render_payload
    assert any(
        message.get("role") == "system"
        and "<session-context-compaction>" in str(message.get("content"))
        for message in render_payload
    )


def test_projection_reload_does_not_duplicate_retained_tail_or_new_units(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "old user", control_turn_id="turn-old")
    session.add_message("assistant", "old reply", control_turn_id="turn-old")
    session.add_message("user", "tail user", control_turn_id="turn-tail")
    session.add_message("assistant", "tail reply", control_turn_id="turn-tail")
    manager.save(session)

    markdown = _MarkdownCompactionProbe()
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=markdown,  # type: ignore[arg-type]
    )
    head = manager.control_store.get_compaction_head(session.key)
    source_ref = compaction_source_ref(session.key, head.next_generation)
    selected = tuple(
        {
            "id": str(message["id"]),
            "seq": int(message["seq"]),
            "unit_ref": "turn-old",
            "message": dict(message),
        }
        for message in session.messages[:2]
    )
    retained_tail = tuple(
        {
            "id": str(message["id"]),
            "seq": int(message["seq"]),
            "unit_ref": "turn-tail",
            "message": dict(message),
        }
        for message in session.messages[2:]
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
        source_from_seq=0,
        consolidated_through_seq=1,
        source_message_ids=tuple(str(message["id"]) for message in session.messages[:2]),
        retained_tail=retained_tail,
        summary_usage=None,
        source_ref=source_ref,
        model_runtime_id="runtime",
        model="model",
        selection_digest="projection-reload",
        selected_source_messages=selected,
    )

    asyncio.run(runtime.commit_checkpoint(session, checkpoint, head=head))
    session.add_message("user", "new user", control_turn_id="turn-new")
    session.add_message("assistant", "new reply", control_turn_id="turn-new")
    manager.save(session)
    manager.invalidate(session.key)
    reloaded = manager.get_or_create(session.key)

    projection = asyncio.run(
        runtime.projection(reloaded, prefix=[], current_anchor=[], pending=[])
    )
    projected_ids = [
        message_id
        for unit in projection.segments.committed_units
        for message_id in unit.source_message_ids
    ]
    projected_content = [
        str(message.get("content"))
        for unit in projection.segments.committed_units
        for message in unit.messages
    ]

    assert projected_ids == [
        str(session.messages[2]["id"]),
        str(session.messages[3]["id"]),
        str(reloaded.messages[4]["id"]),
        str(reloaded.messages[5]["id"]),
    ]
    assert projected_content == [
        "tail user",
        "tail reply",
        "new user",
        "new reply",
    ]


def test_reasoner_builder_preserves_replay_tail_and_current_payload_order() -> None:
    provider = _GateProvider()
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=provider, light_provider=provider),
        llm_config=LLMConfig(model="m", max_tokens=100),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        compaction_runtime=object(),
    )
    committed = CommittedContextUnit(
        source_from_seq=1,
        consolidated_through_seq=1,
        source_message_ids=("canonical-1",),
        messages=({"role": "assistant", "content": "history"},),
        message_refs=(("canonical-1", 1),),
    )
    replay = [
        {"role": "user", "content": "U1"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1"}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        {"role": "assistant", "content": "[execution attempt interrupted]"},
    ]
    projection = CompactionProjection(
        segments=ContextPayloadSegments(
            prefix=({"role": "system", "content": "stable"},),
            committed_units=(committed,),
            current_anchor=(),
        ),
        active=None,
        head=CompactionHead(session_key="session", parent_generation=0, next_generation=1),
    )
    render_payload = [
        {"role": "system", "content": "root"},
        {"role": "system", "content": "stable"},
        *committed.messages,
        *replay,
        {"role": "user", "content": "<system-reminder data-system-context-frame=\"true\">memory</system-reminder>"},
        {"role": "user", "content": "U2"},
        {"role": "user", "content": "U3"},
    ]
    state = reasoner._build_compaction_state(
        session=SimpleNamespace(key="session"),
        projection=projection,
        initial_messages=render_payload,
        history_count=2,
        attempt_replay=replay,
        prior_tool_groups=1,
        channel="",
        chat_id="",
    )
    assert state.compactor._segments.flatten() == render_payload
    assert state.compactor._segments.current_anchor == ()
    assert state.compactor._segments.active_batches == (tuple(replay[:3]),)
    assert state.compactor._segments.pending == tuple(
        [replay[3], *render_payload[7:]]
    )
    assert state.compactor._current_query == (
        '{"logical_interaction_inputs":["U1","U2","U3"]}'
    )


def test_reasoner_compaction_state_is_call_local_per_session(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session_a = manager.get_or_create("session-a")
    session_b = manager.get_or_create("session-b")
    for session, prefix in ((session_a, "a"), (session_b, "b")):
        session.add_message("user", f"{prefix}-u1", control_turn_id=f"{prefix}-1")
        session.add_message(
            "assistant", f"{prefix}-a1", control_turn_id=f"{prefix}-1"
        )
        session.add_message("user", f"{prefix}-u2", control_turn_id=f"{prefix}-2")
        session.add_message(
            "assistant", f"{prefix}-a2", control_turn_id=f"{prefix}-2"
        )
        manager.save(session)
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=_MarkdownCompactionProbe(),  # type: ignore[arg-type]
    )
    provider = _GateProvider()
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=provider, light_provider=provider),
        llm_config=LLMConfig(model="model", max_tokens=10),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        compaction_runtime=runtime,
    )

    projections = [
        asyncio.run(runtime.projection(session, prefix=[], current_anchor=[], pending=[]))
        for session in (session_a, session_b)
    ]
    states = []
    for session, projection in zip((session_a, session_b), projections, strict=True):
        history = [
            message
            for unit in projection.segments.committed_units
            for message in unit.messages
        ]
        payload = [
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": f"{session.key}-current"},
        ]
        states.append(
            reasoner._build_compaction_state(
                session=session,
                projection=projection,
                initial_messages=payload,
                history_count=len(history),
                attempt_replay=[],
                prior_tool_groups=0,
                channel="test",
                chat_id=session.key,
            )
        )

    states[0].compactor.set_pending(
        [
            *states[0].compactor._segments.flatten(),
            {"role": "user", "content": "only-a"},
        ]
    )
    original_b_pending = states[1].compactor._segments.pending
    assert states[0].compactor._scope_id == "session-a"
    assert states[1].compactor._scope_id == "session-b"
    assert states[0].compactor._segments.pending[-1] == {
        "role": "user",
        "content": "only-a",
    }
    assert states[1].compactor._segments.pending == original_b_pending


def test_reasoner_binds_configured_main_fallback_with_distinct_provenance(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("session")
    session.add_message("user", "u", control_turn_id="turn-1")
    session.add_message("assistant", "a", control_turn_id="turn-1")
    session.add_message("user", "u2", control_turn_id="turn-2")
    session.add_message("assistant", "a2", control_turn_id="turn-2")
    manager.save(session)
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=_MarkdownCompactionProbe(),  # type: ignore[arg-type]
    )
    selected = _GateProvider()
    configured_main = _GateProvider()
    reasoner = DefaultReasoner(
        llm=LLMServices(
            provider=selected,
            light_provider=selected,
            fallback_provider=configured_main,
            fallback_model="main-model",
        ),
        llm_config=LLMConfig(model="agent-model", max_tokens=10),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        compaction_runtime=runtime,
    )
    projection = asyncio.run(
        runtime.projection(session, prefix=[], current_anchor=[], pending=[])
    )
    history = [
        message
        for unit in projection.segments.committed_units
        for message in unit.messages
    ]
    state = reasoner._build_compaction_state(
        session=session,
        projection=projection,
        initial_messages=[
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": "current"},
        ],
        history_count=len(history),
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="chat",
    )

    assert state.compactor._provider is selected
    assert state.compactor._fallback_provider is configured_main
    assert state.compactor._model == "agent-model"
    assert state.compactor._fallback_model == "main-model"


def test_two_session_compaction_commits_are_isolated_in_sqlite(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    sessions = []
    source_message_ids: dict[str, set[str]] = {}
    for key in ("session-a", "session-b"):
        session = manager.get_or_create(key)
        prefix = key[-1]
        session.add_message("user", f"{prefix}-u1", control_turn_id=f"{prefix}-1")
        session.add_message(
            "assistant", f"{prefix}-a1", control_turn_id=f"{prefix}-1"
        )
        session.add_message("user", f"{prefix}-u2", control_turn_id=f"{prefix}-2")
        session.add_message(
            "assistant", f"{prefix}-a2", control_turn_id=f"{prefix}-2"
        )
        manager.save(session)
        source_message_ids[key] = {
            str(message["id"])
            for message in session.messages
            if message.get("id")
        }
        sessions.append(session)
    markdown = _MarkdownCompactionProbe()
    runtime = SessionCompactionRuntime(
        session_manager=manager,
        markdown=markdown,  # type: ignore[arg-type]
    )
    provider = _ScopedCompactionProvider()
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=provider, light_provider=provider),
        llm_config=LLMConfig(model="model", max_tokens=10),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        compaction_runtime=runtime,
        context_compaction=ContextCompactionConfig(
            trigger_percent=0.01,
            keep_recent_tokens=1,
        ),
    )
    prepared = []
    for session in sessions:
        projection = asyncio.run(
            runtime.projection(session, prefix=[], current_anchor=[], pending=[])
        )
        history = [
            message
            for unit in projection.segments.committed_units
            for message in unit.messages
        ]
        payload = [
            {"role": "system", "content": "root"},
            *history,
            {"role": "user", "content": f"{session.key}-current"},
        ]
        state = reasoner._build_compaction_state(
            session=session,
            projection=projection,
            initial_messages=payload,
            history_count=len(history),
            attempt_replay=[],
            prior_tool_groups=0,
            channel="test",
            chat_id=session.key,
        )
        prepared.append(
            asyncio.run(
                reasoner._call_provider(
                    state,
                    payload,
                    tools=[],
                    max_tokens=10,
                )
            )
        )

    assert [item.prepared.checkpoint.generation for item in prepared if item.prepared and item.prepared.checkpoint] == [1, 1]
    active_a = manager._store.get_active_compaction("session-a")
    active_b = manager._store.get_active_compaction("session-b")
    assert active_a is not None and active_b is not None
    assert active_a.generation == active_b.generation == 1
    assert active_a.source_ref != active_b.source_ref
    assert set(active_a.source_message_ids) <= source_message_ids["session-a"]
    assert set(active_b.source_message_ids) <= source_message_ids["session-b"]
    assert set(active_a.source_message_ids).isdisjoint(source_message_ids["session-b"])
    assert set(active_b.source_message_ids).isdisjoint(source_message_ids["session-a"])
    assert manager.control_store.get_session_meta("session-a")["last_consolidated"] == 1
    assert manager.control_store.get_session_meta("session-b")["last_consolidated"] == 1
    assert "A_SENTINEL" in active_a.summary
    assert "B_SENTINEL" in active_b.summary
    assert "B_SENTINEL" not in str(prepared[0].prepared.checkpoint.summary)
    assert "A_SENTINEL" not in str(prepared[1].prepared.checkpoint.summary)
    assert markdown.commit_count == 2
