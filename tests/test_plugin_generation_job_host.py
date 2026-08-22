from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.runtime import ConversationRuntime
from agent.plugin_composition.background_jobs import (
    BackgroundJobBinding,
    BackgroundJobCatalog,
    BackgroundJobDefinition,
    BackgroundJobDescriptor,
    CoreEvent,
    CoreEventTrigger,
    IntervalTrigger,
    ProgrammaticTurnPreAdmissionError,
    ProgrammaticTurnUncertainError,
    RetryPolicy,
)
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.context import FiberHandle, HealthHandle
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation_job_host import (
    BackgroundJobActivityAdapter,
    DriftFinishedEvent,
)
from agent.plugins.job_outcome_ledger import (
    JobOutcomeIdentity,
    JobOutcomeLedger,
    JobOutcomePhase,
    JobOutcomeState,
    ProgrammaticTurnState,
)
from agent.plugins.proactive_documents import (
    PROACTIVE_CONTEXT,
    PROACTIVE_PENDING,
    DomainEffectReceipt,
    DomainEffectReceiptStore,
    ProactiveDocumentPair,
    ProactiveDocuments,
)
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from bus.events_lifecycle import DriftFinished
from bus.event_bus import EventBus
from bootstrap.tools import CoreRuntime
from session.store import SessionStore


class _Store:
    def __init__(
        self,
        snapshot: Any,
        candidate_plugin_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.snapshot = snapshot
        self.candidate_plugin_ids = candidate_plugin_ids
        self.leases = 0

    async def acquire(self, snapshot_id: str) -> RuntimeSnapshotLease:
        assert snapshot_id == self.snapshot.snapshot_id
        self.snapshot.lease_count += 1
        self.leases += 1
        return RuntimeSnapshotLease(
            cast(RuntimeSnapshotStore, self),
            cast(RuntimeSnapshot, self.snapshot),
            self.candidate_plugin_ids,
        )

    async def release_lease(self, snapshot: Any) -> None:
        snapshot.lease_count -= 1
        self.leases -= 1

    def fork_lease(self, source: RuntimeSnapshotLease) -> RuntimeSnapshotLease:
        assert source.active
        assert source.snapshot is self.snapshot
        self.snapshot.lease_count += 1
        self.leases += 1
        return RuntimeSnapshotLease(
            cast(RuntimeSnapshotStore, self),
            cast(RuntimeSnapshot, self.snapshot),
            source.validation_candidate_plugin_ids,
        )


@dataclass
class _Fiber:
    activation_token: object
    state: FiberState = FiberState.ACTIVE


@dataclass
class _Health:
    healthy: bool = True


def _module(
    handler: Any,
    *,
    name: str = "drift",
    domain_lookup: Any | None = None,
) -> ComposablePlugin:
    module = ModuleType(f"{name}_module")
    module.api_version = 3
    module.name = name
    module.version = "1.0.0"
    module.apply = _apply
    module.merge_pending = handler
    if domain_lookup is not None:
        module.lookup_emotion_effect = domain_lookup
    return ComposablePlugin.from_module(module)


async def _apply(ctx: Any, config: Any) -> None:
    return None


def _fixture(
    tmp_path,
    handler: Any,
    *,
    model_role: str | None = None,
    provider: object | None = None,
    debounce_seconds: int = 0,
    coalesce: bool = True,
    clock: Any | None = None,
    triggers: tuple[Any, ...] | None = None,
    ledger_path: Any | None = None,
    documents: bool = False,
    domain_lookup: Any | None = None,
    programmatic_turns: bool = False,
    conversation_runtime: object | None = None,
    programmatic_session_creator: Any | None = None,
    programmatic_session_reader: Any | None = None,
    conversation_runtime_binder: Any | None = None,
    validation_candidate_plugin_ids: frozenset[str] = frozenset(),
    retry_policy: RetryPolicy | None = None,
    plugin_id_override: str | None = None,
    job_name: str = "merge_pending",
):
    plugin_id = plugin_id_override or ("emotion" if documents else "drift")
    plugin = _module(handler, name=plugin_id, domain_lookup=domain_lookup)
    job_triggers = triggers or (CoreEventTrigger(CoreEvent.DRIFT_FINISHED),)
    definition = BackgroundJobDefinition(
        name=job_name,
        triggers=job_triggers,
        handler_export="merge_pending",
        model_role=model_role,
        debounce_seconds=debounce_seconds,
        coalesce=coalesce,
        documents_scope=(("emotion",) if documents else ()),
        domain_effect=("emotion.state" if documents else None),
        domain_effect_lookup_export=(
            "lookup_emotion_effect" if documents else None
        ),
        programmatic_turns=programmatic_turns,
        retry_policy=retry_policy or RetryPolicy(),
    )
    descriptor = BackgroundJobDescriptor(
        owner=plugin_id,
        name=definition.name,
        triggers=definition.triggers,
        debounce_seconds=definition.debounce_seconds,
        coalesce=definition.coalesce,
        handler_export=definition.handler_export,
        retry_policy=definition.retry_policy,
        documents_scope=definition.documents_scope,
        domain_effect=definition.domain_effect,
        domain_effect_lookup_export=definition.domain_effect_lookup_export,
        model_role=definition.model_role,
        programmatic_turns=definition.programmatic_turns,
    )
    fiber = _Fiber(object())
    binding = BackgroundJobBinding(
        generation_id="generation-1",
        plugin_id=plugin_id,
        name=definition.name,
        descriptor=descriptor,
        definition=definition,
        owner_fiber=cast(FiberHandle, fiber),
        activation_token=fiber.activation_token,
        required_health=cast(HealthHandle, _Health()),
    )
    catalog = BackgroundJobCatalog(
        {f"{plugin_id}:{job_name}": binding},
        root_instance_token=object(),
    )
    generation = SimpleNamespace(
        generation_id="generation-1",
        instance=plugin,
        source_revision="source-1",
    )
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-1",
        background_job_catalog=catalog,
        generations={plugin_id: generation},
        lease_count=0,
    )
    store = _Store(snapshot, validation_candidate_plugin_ids)
    snapshot.lease_count += 1
    store.leases += 1
    target_lease = RuntimeSnapshotLease(
        cast(RuntimeSnapshotStore, store),
        cast(RuntimeSnapshot, snapshot),
        validation_candidate_plugin_ids,
    )
    ledger = JobOutcomeLedger(ledger_path or tmp_path / "outcomes.sqlite")
    adapter = BackgroundJobActivityAdapter(
        EventBus(),
        cast(RuntimeSnapshotStore, store),
        model_provider=provider,
        ledger=ledger,
        clock=clock,
        workspace=(str(tmp_path / "workspace") if documents else None),
    )
    if conversation_runtime is not None:
        if conversation_runtime_binder is None:
            if programmatic_session_reader is None:
                programmatic_session_reader = getattr(
                    getattr(conversation_runtime, "_store", None),
                    "get_session_meta",
                    None,
                )
            adapter.bind_conversation_runtime(
                conversation_runtime,
                programmatic_session_creator=programmatic_session_creator,
                programmatic_session_reader=programmatic_session_reader,
            )
        else:
            conversation_runtime_binder(adapter, conversation_runtime)
    plan = adapter.prepare_components("tx-1", target_lease, catalog)
    return adapter, plan, target_lease, store, ledger


def _event(event_id: str) -> DriftFinished:
    return DriftFinished(
        event_id=event_id,
        session_key="session",
        skill_name="skill",
        status="completed",
        briefing="briefing",
        message_result="ok",
        timestamp=datetime.now(timezone.utc),
    )


def _event_payload(event: DriftFinished) -> dict[str, str]:
    return {
        "event_id": event.event_id,
        "session_key": event.session_key,
        "skill_name": event.skill_name,
        "status": event.status,
        "briefing": event.briefing,
        "message_result": event.message_result,
        "timestamp": event.timestamp.isoformat(),
    }


class _ProgrammaticSessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, object]] = {}

    def create_session(
        self,
        *,
        key: str,
        metadata: dict[str, object],
    ) -> None:
        if key in self.sessions:
            raise RuntimeError(f"duplicate session: {key}")
        self.sessions[key] = dict(metadata)

    def get_session_meta(self, key: str) -> dict[str, object] | None:
        metadata = self.sessions.get(key)
        if metadata is None:
            return None
        return {"key": key, "metadata": dict(metadata)}


class _ProgrammaticTurnHandle:
    def __init__(self, turn_id: str) -> None:
        self.id = turn_id
        self._terminal: asyncio.Future[object] = asyncio.get_running_loop().create_future()

    async def result(self) -> object:
        return await self._terminal

    def complete(self) -> None:
        if not self._terminal.done():
            self._terminal.set_result(object())


class _ProgrammaticConversationRuntime:
    def __init__(self) -> None:
        self._store = _ProgrammaticSessionStore()
        self.requests: list[tuple[Any, RuntimeSnapshotLease, _ProgrammaticTurnHandle]] = []

    def create_session(
        self,
        *,
        key: str,
        metadata: dict[str, object],
    ) -> None:
        self._store.create_session(key=key, metadata=metadata)

    async def start_turn(
        self,
        request: Any,
        *,
        runtime_snapshot_lease: RuntimeSnapshotLease,
        fresh_interaction: bool,
    ) -> _ProgrammaticTurnHandle:
        assert fresh_interaction is True
        handle = _ProgrammaticTurnHandle(f"turn:{len(self.requests) + 1}")
        self.requests.append((request, runtime_snapshot_lease, handle))
        return handle


def _domain_record(ctx: Any) -> dict[str, object]:
    return {
        "state": "committed",
        "invocation_id": ctx.invocation_id,
        "effect_id": ctx.effect_id,
        "idempotency_key": ctx.idempotency_key,
        "attempt": ctx.attempt,
        "result_digest": "emotion-domain-result",
    }


def _pending_identity(
    plan,
    *,
    invocation_id: str,
    event: DriftFinished | None = None,
    interval_bucket: str | None = None,
) -> JobOutcomeIdentity:
    return JobOutcomeIdentity(
        plugin_id="drift",
        job_name="merge_pending",
        invocation_id=invocation_id,
        event_id=None if event is None else event.event_id,
        interval_bucket=interval_bucket,
        snapshot_id=plan.snapshot_id,
        plugin_generation_id="generation-1",
        model_generation_id="execution-pending",
        artifact_identity=f"{plan.catalog_identity}:drift:merge_pending",
        source_revision="source-1",
        handler_export="merge_pending",
        lifecycle_revision="background-job-v3",
        api_revision="plugin-api-v3",
        event_payload=None if event is None else _event_payload(event),
    )


@pytest.mark.asyncio
async def test_prepare_is_pure_and_materialized_binding_is_closed(tmp_path) -> None:
    calls: list[object] = []

    async def handler(ctx) -> None:
        calls.append(ctx)

    adapter, plan, target_lease, store, ledger = _fixture(tmp_path, handler)
    assert adapter.handler_resolution_count == 0
    assert adapter.subscription_count == 0
    assert adapter.timer_count == 0

    runtime = await adapter.materialize_closed("tx-1", plan)
    assert adapter.handler_resolution_count == 1
    assert runtime.admission_open is False
    assert runtime.subscription_count == 1
    assert calls == []

    await adapter.enqueue_event(runtime, _event("event-1"))
    await asyncio.sleep(0)
    assert calls == []

    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("event-1"))
    await adapter.drain(runtime)
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="event-1",
    )
    assert len(calls) == 1
    assert outcome is not None
    assert outcome.state is JobOutcomeState.SUCCEEDED
    assert dict(outcome.event_payload or {}) == {
        "event_id": "event-1",
        "session_key": "session",
        "skill_name": "skill",
        "status": "completed",
        "briefing": "briefing",
        "message_result": "ok",
        "timestamp": calls[0].event.timestamp.isoformat(),
    }
    assert store.leases == 1

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()
    assert store.leases == 0
    assert adapter.subscription_count == 0
    assert adapter.timer_count == 0


@pytest.mark.asyncio
async def test_installed_emotion_domain_effect_is_in_core_allowlist(tmp_path) -> None:
    async def handler(ctx) -> None:
        return None

    adapter, plan, target_lease, store, _ = _fixture(
        tmp_path,
        handler,
        documents=True,
        domain_lookup=lambda ctx: None,
        plugin_id_override="emotion@github",
        job_name="merge_proactive_pending",
    )

    runtime = await adapter.materialize_closed("tx-1", plan)
    assert "emotion@github:merge_proactive_pending" in runtime.jobs
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()
    assert store.leases == 0


@pytest.mark.asyncio
async def test_emotion_documents_finish_after_in_process_post_effect_failure(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / PROACTIVE_CONTEXT).write_text("old context\n", encoding="utf-8")
    (workspace / PROACTIVE_PENDING).write_text("pending\n", encoding="utf-8")
    durable: dict[str, object] = {}

    def lookup(ctx):
        return durable.get(ctx.invocation_id)

    async def handler(ctx) -> None:
        assert ctx.documents is not None and ctx.domain_effects is not None
        expected, _ = ctx.documents.read_pair()
        intent = await ctx.documents.prepare_pair(
            expected,
            ProactiveDocumentPair(b"new context\n", b""),
        )

        async def transaction(effect_ctx) -> None:
            durable[effect_ctx.invocation_id] = _domain_record(effect_ctx)
            raise RuntimeError("callback failed after SQLite commit")

        _ = await ctx.domain_effects.run("emotion.state", transaction)
        assert intent.invocation_id
        raise RuntimeError("plugin failed after durable effect")

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        documents=True,
        domain_lookup=lookup,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("event-docs-in-process"))
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="emotion",
        job_name="merge_pending",
        event_id="event-docs-in-process",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.SUCCEEDED
    assert (workspace / PROACTIVE_CONTEXT).read_bytes() == b"new context\n"
    assert (workspace / PROACTIVE_PENDING).read_bytes() == b""
    await adapter.close_components("shutdown", runtime)
    await target_lease.release()
    assert store.leases == 0


@pytest.mark.asyncio
async def test_emotion_documents_restart_forwards_without_handler_replay(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / PROACTIVE_CONTEXT).write_text("old context\n", encoding="utf-8")
    (workspace / PROACTIVE_PENDING).write_text("pending\n", encoding="utf-8")
    durable: dict[str, object] = {}
    handler_calls = 0

    def lookup(ctx):
        return durable.get(ctx.invocation_id)

    async def handler(ctx) -> None:
        nonlocal handler_calls
        handler_calls += 1

    ledger_path = tmp_path / "outcomes.sqlite"
    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        documents=True,
        domain_lookup=lookup,
        ledger_path=ledger_path,
    )
    event = _event("event-docs-restart")
    identity = JobOutcomeIdentity(
        plugin_id="emotion",
        job_name="merge_pending",
        invocation_id="invocation-restart",
        event_id=event.event_id,
        snapshot_id=plan.snapshot_id,
        plugin_generation_id="generation-1",
        model_generation_id="execution-pending",
        artifact_identity=f"{plan.catalog_identity}:emotion:merge_pending",
        source_revision="source-1",
        handler_export="merge_pending",
        lifecycle_revision="background-job-v3",
        api_revision="plugin-api-v3",
        event_payload=_event_payload(event),
    )
    _ = ledger.admit(identity)
    running = ledger.transition(
        identity.invocation_id,
        JobOutcomeState.RUNNING,
        model_generation_id="provider",
    )
    documents = ProactiveDocuments(
        workspace,
        identity.invocation_id,
        idempotency_key=f"{running.semantic_job_id}:{running.trigger_identity}",
        effect_id="emotion.state",
    )
    expected, _ = documents.read_pair()
    _ = await documents.prepare_pair(
        expected,
        ProactiveDocumentPair(b"recovered context\n", b""),
    )
    effect_ctx = type(
        "EffectCtx",
        (),
        {
            "invocation_id": identity.invocation_id,
            "effect_id": "emotion.state",
            "idempotency_key": f"{running.semantic_job_id}:{running.trigger_identity}",
            "attempt": running.attempt,
        },
    )()
    durable[identity.invocation_id] = _domain_record(effect_ctx)

    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.drain(runtime)
    recovered = ledger.get(identity.invocation_id)
    assert recovered is not None and recovered.state is JobOutcomeState.SUCCEEDED
    assert handler_calls == 0
    assert (workspace / PROACTIVE_CONTEXT).read_bytes() == b"recovered context\n"
    assert (workspace / PROACTIVE_PENDING).read_bytes() == b""
    await adapter.close_components("shutdown", runtime)
    await target_lease.release()
    assert store.leases == 0


@pytest.mark.asyncio
async def test_emotion_documents_cancel_aborts_prepared_intent_before_terminal(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / PROACTIVE_CONTEXT).write_text("old context\n", encoding="utf-8")
    (workspace / PROACTIVE_PENDING).write_text("pending\n", encoding="utf-8")
    prepared = asyncio.Event()

    async def handler(ctx) -> None:
        assert ctx.documents is not None
        expected, _ = ctx.documents.read_pair()
        _ = await ctx.documents.prepare_pair(
            expected,
            ProactiveDocumentPair(b"cancelled context\n", b""),
        )
        prepared.set()
        await asyncio.Event().wait()

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        documents=True,
        domain_lookup=lambda _ctx: None,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("event-docs-cancel"))
    await prepared.wait()
    await adapter.cancel_running(runtime)

    outcome = ledger.find_by_event(
        plugin_id="emotion",
        job_name="merge_pending",
        event_id="event-docs-cancel",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.CANCELLED
    documents = ProactiveDocuments(
        workspace,
        outcome.invocation_id,
        idempotency_key=f"{outcome.semantic_job_id}:{outcome.trigger_identity}",
        effect_id="emotion.state",
    )
    assert documents.pending_intent_ids() == ()
    terminal = documents.load_terminal_receipt()
    assert terminal is not None and terminal.status.value == "aborted"
    assert (workspace / PROACTIVE_CONTEXT).read_bytes() == b"old context\n"
    assert (workspace / PROACTIVE_PENDING).read_bytes() == b"pending\n"
    await adapter.close_components("shutdown", runtime)
    await target_lease.release()
    assert store.leases == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_drift", [None, "domain", "document"])
async def test_emotion_documents_restart_closes_terminal_before_ledger_window(
    tmp_path,
    restart_drift: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / PROACTIVE_CONTEXT).write_text("old context\n", encoding="utf-8")
    (workspace / PROACTIVE_PENDING).write_text("pending\n", encoding="utf-8")
    ledger_path = tmp_path / "outcomes.sqlite"
    durable: dict[str, object] = {}

    async def handler(_ctx) -> None:
        raise AssertionError("terminal receipt recovery must not replay handler")

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        documents=True,
        domain_lookup=lambda _ctx: durable,
        ledger_path=ledger_path,
    )
    event = _event("event-docs-terminal-restart")
    identity = JobOutcomeIdentity(
        plugin_id="emotion",
        job_name="merge_pending",
        invocation_id="invocation-terminal-restart",
        event_id=event.event_id,
        snapshot_id=plan.snapshot_id,
        plugin_generation_id="generation-1",
        model_generation_id="execution-pending",
        artifact_identity=f"{plan.catalog_identity}:emotion:merge_pending",
        source_revision="source-1",
        handler_export="merge_pending",
        lifecycle_revision="background-job-v3",
        api_revision="plugin-api-v3",
        event_payload=_event_payload(event),
    )
    _ = ledger.admit(identity)
    running = ledger.transition(
        identity.invocation_id,
        JobOutcomeState.RUNNING,
        model_generation_id="provider",
    )
    pending = ledger.transition(
        identity.invocation_id,
        JobOutcomeState.RETRY_PENDING,
        phase=JobOutcomePhase.DOCUMENTS,
        error="process crashed before ledger terminal",
    )
    receipt_store = DomainEffectReceiptStore(tmp_path / "domain-effects.sqlite")
    receipt = receipt_store.record(
        DomainEffectReceipt(
            invocation_id=identity.invocation_id,
            effect_id="emotion.state",
            idempotency_key=f"{pending.semantic_job_id}:{pending.trigger_identity}",
            state="committed",
            result_digest="emotion-domain-result",
            attempt=running.attempt,
        )
    )
    durable.update(receipt.as_dict())
    documents = ProactiveDocuments(
        workspace,
        identity.invocation_id,
        idempotency_key=receipt.idempotency_key,
        effect_id=receipt.effect_id,
        receipt_lookup=receipt_store,
    )
    expected, _ = documents.read_pair()
    intent = await documents.prepare_pair(
        expected,
        ProactiveDocumentPair(b"terminal context\n", b""),
    )
    terminal = await documents.commit_after(intent, receipt)
    assert documents.pending_intent_ids() == ()

    if restart_drift == "domain":
        durable.clear()
    elif restart_drift == "document":
        (workspace / PROACTIVE_CONTEXT).write_text("third party\n", encoding="utf-8")

    runtime = await adapter.materialize_closed("tx-1", plan)
    if restart_drift is not None:
        with pytest.raises(RuntimeError):
            await adapter.open(runtime)
        retained = ledger.get(identity.invocation_id)
        assert retained is not None
        assert retained.state is JobOutcomeState.RETRY_PENDING
        assert retained.phase is JobOutcomePhase.DOCUMENTS
        await adapter.close_components("shutdown", runtime)
        await target_lease.release()
        assert store.leases == 0
        return

    await adapter.open(runtime)
    await adapter.drain(runtime)
    recovered = ledger.get(identity.invocation_id)
    assert recovered is not None and recovered.state is JobOutcomeState.SUCCEEDED
    assert recovered.terminal_result_digest == terminal.document_digest
    assert (workspace / PROACTIVE_CONTEXT).read_bytes() == b"terminal context\n"
    assert (workspace / PROACTIVE_PENDING).read_bytes() == b""
    await adapter.close_components("shutdown", runtime)
    await target_lease.release()
    assert store.leases == 0


@pytest.mark.asyncio
async def test_missing_job_catalog_materializes_closed_noop_without_worker(tmp_path) -> None:
    snapshot = SimpleNamespace(
        snapshot_id="empty-jobs",
        background_job_catalog=None,
        generations={},
        lease_count=1,
    )
    store = _Store(snapshot)
    target_lease = RuntimeSnapshotLease(
        cast(RuntimeSnapshotStore, store), cast(RuntimeSnapshot, snapshot)
    )
    adapter = BackgroundJobActivityAdapter(
        EventBus(),
        cast(RuntimeSnapshotStore, store),
        ledger=JobOutcomeLedger(tmp_path / "outcomes.sqlite"),
    )

    plan = adapter.prepare_components(
        "tx-empty",
        target_lease,
        SimpleNamespace(background_jobs=None),
    )
    binding = await adapter.materialize_closed("tx-empty", plan)

    assert binding.jobs == {}
    assert binding.worker_task is None
    assert binding.subscription_count == 0
    assert binding.timer_count == 0
    adapter.finalize_components("tx-empty", binding)
    assert binding.admission_open

    await adapter.close_components("shutdown", binding)


@pytest.mark.asyncio
async def test_exact_handler_and_event_id_dedupe_use_first_binding(tmp_path) -> None:
    calls: list[str] = []

    async def first(ctx) -> None:
        calls.append("first")

    async def second(ctx) -> None:
        calls.append("second")

    adapter, plan, target_lease, _store, ledger = _fixture(tmp_path, first)
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    event = _event("same-event")
    await adapter.enqueue_event(runtime, event)
    await adapter.enqueue_event(runtime, event)
    await adapter.drain(runtime)
    assert calls == ["first"]
    assert len(ledger.list_all()) == 1

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_debounce_uses_core_clock_without_changing_event_dedupe(tmp_path) -> None:
    now = datetime(2026, 8, 17, 4, 0, tzinfo=timezone.utc)
    calls: list[str] = []

    async def handler(ctx) -> None:
        calls.append(ctx.event.event_id)

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        debounce_seconds=60,
        clock=lambda: now,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("debounce-1"))
    await adapter.drain(runtime)
    await adapter.enqueue_event(runtime, _event("debounce-2"))
    await adapter.drain(runtime)
    assert calls == ["debounce-1"]
    assert len(ledger.list_all()) == 2
    assert all(record.state is not JobOutcomeState.RUNNING for record in ledger.list_all())

    now = now.replace(minute=1, second=1)
    await adapter.enqueue_event(runtime, _event("debounce-3"))
    await adapter.drain(runtime)
    assert calls == ["debounce-1", "debounce-3"]
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_llm_lease_is_invocation_scoped_and_invalid_after_handler(tmp_path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs: object) -> object:
            self.calls += 1
            return SimpleNamespace(content="model-answer", usage=None)

    provider = Provider()
    saved: list[object] = []

    async def handler(ctx) -> None:
        saved.append(ctx.llm)
        assert await ctx.llm.generate_text(prompt="hello") == "model-answer"

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        provider=provider,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("llm-event"))
    await adapter.drain(runtime)
    assert provider.calls == 1
    with pytest.raises(RuntimeError, match="已失效"):
        await saved[0].generate_text(prompt="after terminal")
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="llm-event",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.SUCCEEDED
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_port_is_only_exposed_to_declared_job_and_releases_child_lease(
    tmp_path,
) -> None:
    conversation = _ProgrammaticConversationRuntime()
    captured: list[Any] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        captured.append(ctx.turns)
        session_id = await ctx.turns.create_session(metadata={"label": "watch"})
        receipt = await ctx.turns.submit(session_id, "inspect this event")
        assert receipt.session_id == session_id
        assert receipt.turn_id == "turn:1"

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("programmatic-event"))
    await adapter.drain(runtime)

    assert len(conversation.requests) == 1
    request, child_lease, handle = conversation.requests[0]
    assert request.thread_id.startswith("programmatic:")
    assert request.metadata["plugin_id"] == "drift"
    assert request.metadata["job_name"] == "merge_pending"
    assert request.metadata["generation_id"] == "generation-1"
    assert request.metadata["snapshot_id"] == "snapshot-1"
    assert request.metadata["event_id"] == "programmatic-event"
    assert request.metadata["skip_post_memory"] is True
    assert conversation._store.sessions[request.thread_id]["label"] == "watch"
    assert store.leases == 2

    handle.complete()
    for _ in range(100):
        if store.leases == 1:
            break
        await asyncio.sleep(0)
    assert store.leases == 1
    with pytest.raises(RuntimeError, match="已结算"):
        await captured[0].create_session(metadata={})
    assert ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="programmatic-event",
    ).state is JobOutcomeState.SUCCEEDED
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()
    assert not child_lease.active


@pytest.mark.asyncio
async def test_programmatic_turn_uses_bootstrap_bound_real_conversation_runtime(
    tmp_path,
) -> None:
    session_store = SessionStore(tmp_path / "real-sessions.db")

    async def execute(request) -> str:
        return f"reply:{request.input}"

    conversation = ConversationRuntime(session_store, execute)
    receipts: list[Any] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={"source": "real"})
        receipts.append(await ctx.turns.submit(session_id, "real runtime"))

    def bind_from_core(host: Any, owner: object) -> None:
        core = cast(Any, object.__new__(CoreRuntime))
        core.background_job_host = host
        core.session_manager = SimpleNamespace(control_store=session_store)
        core.bind_conversation_runtime(owner)

    adapter, plan, target_lease, snapshot_store, _ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        conversation_runtime_binder=bind_from_core,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    try:
        await adapter.enqueue_event(runtime, _event("real-programmatic"))
        await adapter.drain(runtime)
        assert len(receipts) == 1
        receipt = receipts[0]
        result = await conversation.wait_result(receipt.session_id, receipt.turn_id)
        assert result.final_response == "reply:real runtime"
        metadata = session_store.get_session_meta(receipt.session_id)
        assert metadata is not None
        assert metadata["metadata"]["programmatic"] is True
        assert metadata["metadata"]["plugin_id"] == "drift"
        record = session_store.read_turn(receipt.turn_id)
        assert record is not None
        assert record.thread_id == receipt.session_id
        for _ in range(100):
            if snapshot_store.leases == 1:
                break
            await asyncio.sleep(0)
        assert snapshot_store.leases == 1
    finally:
        await adapter.close_components("tx-1", runtime)
        await target_lease.release()
        await conversation.shutdown()
        session_store.close()


@pytest.mark.asyncio
async def test_programmatic_turn_port_rejects_reserved_metadata_and_foreign_session(
    tmp_path,
) -> None:
    conversation = _ProgrammaticConversationRuntime()
    conversation.create_session(
        key="programmatic:foreign",
        metadata={
            "programmatic": True,
            "plugin_id": "other-plugin",
            "job_name": "merge_pending",
        },
    )
    saved: list[Any] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        saved.append(ctx.turns)
        with pytest.raises(ValueError, match="不能覆盖 Core 字段"):
            await ctx.turns.create_session(metadata={"plugin_id": "forged"})
        session_id = await ctx.turns.create_session(metadata={})
        with pytest.raises(
            ProgrammaticTurnPreAdmissionError,
            match="provenance 不匹配",
        ) as caught:
            await ctx.turns.submit("programmatic:foreign", "no")
        assert caught.value.reason == "session_provenance_mismatch"
        handle = conversation.requests
        assert session_id.startswith("programmatic:")
        assert handle == []

    adapter, plan, target_lease, store, _ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("programmatic-validation"))
    await adapter.drain(runtime)
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_reuses_durable_session_across_invocations(
    tmp_path,
) -> None:
    conversation = _ProgrammaticConversationRuntime()
    session_ids: list[str] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        if not session_ids:
            session_ids.append(await ctx.turns.create_session(metadata={"source": "watch"}))
            return
        receipt = await ctx.turns.submit(session_ids[0], "second poll")
        assert receipt.turn_id == "turn:1"

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("create-programmatic-session"))
    await adapter.drain(runtime)
    await adapter.enqueue_event(runtime, _event("reuse-programmatic-session"))
    await adapter.drain(runtime)

    assert len(conversation.requests) == 1
    request, _lease, handle = conversation.requests[0]
    assert request.thread_id == session_ids[0]
    assert request.metadata["event_id"] == "reuse-programmatic-session"
    assert request.metadata["invocation_id"] != conversation._store.sessions[
        session_ids[0]
    ]["invocation_id"]
    assert request.metadata["source"] == "watch"
    reused = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="reuse-programmatic-session",
    )
    assert reused is not None
    assert reused.programmatic_turn_state is ProgrammaticTurnState.ADMITTED
    handle.complete()
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_caught_uncertain_turn_receipt_still_fails_invocation_for_manual_reconcile(
    tmp_path,
    monkeypatch,
) -> None:
    conversation = _ProgrammaticConversationRuntime()
    caught: list[ProgrammaticTurnUncertainError] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={})
        try:
            await ctx.turns.submit(session_id, "uncertain receipt")
        except ProgrammaticTurnUncertainError as error:
            caught.append(error)

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
        retry_policy=RetryPolicy(max_attempts=2),
    )

    real_commit = ledger.commit_programmatic_turn

    def fail_commit(invocation_id: str, turn_id: str):
        _ = real_commit(invocation_id, turn_id)
        raise OSError("receipt database unavailable")

    monkeypatch.setattr(ledger, "commit_programmatic_turn", fail_commit)
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("uncertain-receipt"))
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="uncertain-receipt",
    )
    assert len(caught) == 1
    assert len(conversation.requests) == 1
    assert outcome is not None
    assert outcome.state is JobOutcomeState.FAILED
    assert outcome.attempt == 1
    assert outcome.programmatic_turn_state is ProgrammaticTurnState.ADMITTED
    assert outcome.programmatic_turn_id == "turn:1"
    assert outcome.error is not None and "manual reconcile" in outcome.error
    conversation.requests[0][2].complete()
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_post_persist_turn_start_failure_is_manual_and_not_retried(
    tmp_path,
    monkeypatch,
) -> None:
    session_store = SessionStore(tmp_path / "post-persist-sessions.db")

    async def execute(_request) -> str:
        return "unused"

    conversation = ConversationRuntime(session_store, execute)
    caught: list[ProgrammaticTurnUncertainError] = []
    session_ids: list[str] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={})
        session_ids.append(session_id)
        try:
            await ctx.turns.submit(session_id, "post-persist failure")
        except ProgrammaticTurnUncertainError as error:
            caught.append(error)

    def bind_from_core(host: Any, owner: object) -> None:
        core = cast(Any, object.__new__(CoreRuntime))
        core.background_job_host = host
        core.session_manager = SimpleNamespace(control_store=session_store)
        core.bind_conversation_runtime(owner)

    adapter, plan, target_lease, snapshot_store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        conversation_runtime_binder=bind_from_core,
        retry_policy=RetryPolicy(max_attempts=2),
    )

    def fail_publish_user(_thread_id: str, _turn_id: str, _item: object) -> None:
        raise RuntimeError("publish failed after durable admission")

    monkeypatch.setattr(conversation, "_publish_user_item", fail_publish_user)
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("post-persist-turn"))
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="post-persist-turn",
    )
    assert len(caught) == 1
    assert len(session_ids) == 1
    assert outcome is not None
    assert outcome.state is JobOutcomeState.FAILED
    assert outcome.attempt == 1
    assert outcome.programmatic_turn_state is ProgrammaticTurnState.SUBMITTING
    assert outcome.error is not None and "manual reconcile" in outcome.error
    turns = session_store.list_turns(session_ids[0])
    assert len(turns) == 1 and turns[0].status.value == "failed"
    assert snapshot_store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()
    await conversation.shutdown()
    session_store.close()


@pytest.mark.asyncio
async def test_pre_admission_reset_failure_releases_child_lease(
    tmp_path,
    monkeypatch,
) -> None:
    class RejectingConversation(_ProgrammaticConversationRuntime):
        async def start_turn(
            self, request, *, runtime_snapshot_lease, fresh_interaction
        ):
            assert fresh_interaction is True
            raise RuntimeError("rejected before Turn persistence")

    conversation = RejectingConversation()
    caught: list[ProgrammaticTurnUncertainError] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={})
        try:
            await ctx.turns.submit(session_id, "pre-admission")
        except ProgrammaticTurnUncertainError as error:
            caught.append(error)

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )

    def fail_reset(_invocation_id: str):
        raise OSError("ledger reset failed")

    monkeypatch.setattr(ledger, "reset_programmatic_turn", fail_reset)
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("reset-failure"))
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="reset-failure",
    )
    assert len(caught) == 1
    assert store.leases == 1
    assert outcome is not None and outcome.state is JobOutcomeState.FAILED
    assert outcome.programmatic_turn_state is ProgrammaticTurnState.SUBMITTING
    assert outcome.error is not None and "manual reconcile" in outcome.error
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_port_is_none_for_ordinary_job(tmp_path) -> None:
    captured: list[Any] = []

    async def handler(ctx) -> None:
        captured.append(ctx.turns)

    adapter, plan, target_lease, store, _ledger = _fixture(tmp_path, handler)
    conversation = _ProgrammaticConversationRuntime()
    adapter.bind_conversation_runtime(conversation)
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("ordinary-job"))
    await adapter.drain(runtime)
    assert captured == [None]
    assert conversation.requests == []
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_port_is_not_exposed_anywhere_in_candidate_snapshot(
    tmp_path,
) -> None:
    captured: list[Any] = []

    async def handler(ctx) -> None:
        captured.append(ctx.turns)

    adapter, plan, target_lease, store, _ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        validation_candidate_plugin_ids=frozenset({"other-plugin"}),
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("candidate-job"))
    await adapter.drain(runtime)
    assert captured == [None]
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_admission_forbids_handler_retry(tmp_path) -> None:
    conversation = _ProgrammaticConversationRuntime()

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={})
        _ = await ctx.turns.submit(session_id, "admit once")
        raise RuntimeError("failure after admission")

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("admitted-no-retry"))
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="admitted-no-retry",
    )
    assert len(conversation.requests) == 1
    assert outcome is not None
    assert outcome.state is JobOutcomeState.FAILED
    assert outcome.attempt == 1
    assert outcome.programmatic_turn_id == "turn:1"
    assert outcome.error is not None and "manual reconcile" in outcome.error
    conversation.requests[0][2].complete()
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_repeated_cancel_finishes_admission_and_receipt(
    tmp_path,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingConversation(_ProgrammaticConversationRuntime):
        async def start_turn(
            self, request, *, runtime_snapshot_lease, fresh_interaction
        ):
            assert fresh_interaction is True
            started.set()
            await release.wait()
            return await super().start_turn(
                request,
                runtime_snapshot_lease=runtime_snapshot_lease,
                fresh_interaction=fresh_interaction,
            )

    conversation = BlockingConversation()
    submit_tasks: list[asyncio.Task[Any]] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={})
        task = asyncio.create_task(ctx.turns.submit(session_id, "cancel twice"))
        submit_tasks.append(task)
        await task

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("cancelled-admission"))
    await started.wait()
    submit_tasks[0].cancel()
    await asyncio.sleep(0)
    submit_tasks[0].cancel()
    release.set()
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="cancelled-admission",
    )
    assert len(conversation.requests) == 1
    assert outcome is not None
    assert outcome.state is JobOutcomeState.FAILED
    assert outcome.programmatic_turn_id == "turn:1"
    assert store.leases == 2
    conversation.requests[0][2].complete()
    for _ in range(100):
        if store.leases == 1:
            break
        await asyncio.sleep(0)
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_cancelled_pre_admission_failure_does_not_retry_handler(tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class RejectingConversation(_ProgrammaticConversationRuntime):
        calls = 0

        async def start_turn(
            self, request, *, runtime_snapshot_lease, fresh_interaction
        ):
            assert fresh_interaction is True
            self.calls += 1
            started.set()
            await release.wait()
            raise RuntimeError("rejected before Turn persistence")

    conversation = RejectingConversation()
    submit_tasks: list[asyncio.Task[Any]] = []
    handler_calls = 0

    async def handler(ctx) -> None:
        nonlocal handler_calls
        handler_calls += 1
        assert ctx.turns is not None
        session_id = await ctx.turns.create_session(metadata={})
        task = asyncio.create_task(ctx.turns.submit(session_id, "cancel before admission"))
        submit_tasks.append(task)
        await task

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("cancelled-pre-admission"))
    await started.wait()
    submit_tasks[0].cancel()
    release.set()
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="cancelled-pre-admission",
    )
    assert handler_calls == 1
    assert conversation.calls == 1
    assert outcome is not None and outcome.state is JobOutcomeState.CANCELLED
    assert outcome.attempt == 1
    assert outcome.programmatic_turn_state is None
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_programmatic_turn_owner_is_required_before_formal_materialization(
    tmp_path,
) -> None:
    async def handler(_ctx) -> None:
        return None

    with pytest.raises(RuntimeError, match="ConversationRuntime"):
        _fixture(
            tmp_path,
            handler,
            programmatic_turns=True,
        )


@pytest.mark.asyncio
async def test_programmatic_turn_port_failure_closes_without_lease_residue(tmp_path) -> None:
    conversation = _ProgrammaticConversationRuntime()
    captured: list[Any] = []

    async def handler(ctx) -> None:
        assert ctx.turns is not None
        captured.append(ctx.turns)
        await ctx.turns.create_session(metadata={})
        raise RuntimeError("programmatic handler failed")

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_event(runtime, _event("programmatic-failure"))
    await adapter.drain(runtime)
    assert store.leases == 1
    with pytest.raises(RuntimeError, match="已结算"):
        await captured[0].create_session(metadata={})
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="programmatic-failure",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.FAILED
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_cancel_running_releases_snapshot_lease_and_marks_cancelled(tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Provider:
        async def chat(self, **kwargs: object) -> object:
            started.set()
            await release.wait()
            return SimpleNamespace(content="never", usage=None)

    async def handler(ctx) -> None:
        await ctx.llm.generate_text(prompt="blocked")

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        provider=Provider(),
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("cancel-event"))
    for _ in range(100):
        if runtime.running:
            break
        await asyncio.sleep(0)
    await started.wait()
    assert store.leases == 2
    await adapter.cancel_running(runtime)
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="cancel-event",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.CANCELLED
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_completed_child_failure_prevents_handler_success(tmp_path) -> None:
    async def fail_child() -> None:
        raise RuntimeError("child failed")

    async def handler(ctx) -> None:
        ctx.spawn_child(fail_child(), name="failing-child")
        await asyncio.sleep(0)

    adapter, plan, target_lease, _store, ledger = _fixture(tmp_path, handler)
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("child-failure"))
    await adapter.drain(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="child-failure",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.FAILED
    assert outcome.error == "RuntimeError: child failed"
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_cancel_intent_wins_when_handler_swallows_cancelled_error(tmp_path) -> None:
    started = asyncio.Event()

    async def handler(ctx) -> str | None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "swallowed"

    adapter, plan, target_lease, store, ledger = _fixture(tmp_path, handler)
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("swallowed-cancel"))
    await started.wait()

    await adapter.cancel_running(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="swallowed-cancel",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.CANCELLED
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_close_components_drains_queued_request_after_running_cancel(tmp_path) -> None:
    started = asyncio.Event()

    class Provider:
        async def chat(self, **kwargs: object) -> object:
            started.set()
            await asyncio.Event().wait()
            return SimpleNamespace(content="never", usage=None)

    async def handler(ctx) -> None:
        await ctx.llm.generate_text(prompt="blocked")

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        provider=Provider(),
        coalesce=False,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("running-close"))
    await started.wait()
    await adapter.enqueue_event(runtime, _event("queued-close"))
    assert len(runtime.queued) == 1

    await asyncio.wait_for(adapter.close_components("tx-1", runtime), timeout=1)

    assert runtime.closed is True
    assert runtime.worker_task is None
    assert ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="running-close",
    ).state is JobOutcomeState.CANCELLED
    assert ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="queued-close",
    ).state is JobOutcomeState.CANCELLED
    assert store.leases == 1
    await target_lease.release()


@pytest.mark.asyncio
async def test_queued_request_selects_model_generation_at_execution_start(tmp_path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    model_ids: list[str] = []

    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.registry = SimpleNamespace(
                current=SimpleNamespace(generation_id="model-1")
            )

        async def chat(self, **kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                started.set()
                await release.wait()
            return SimpleNamespace(content="ok", usage=None)

    provider = Provider()

    async def handler(ctx) -> None:
        model_ids.append(ctx.model_generation_id)
        await ctx.llm.generate_text(prompt="model")

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        provider=provider,
        coalesce=False,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.enqueue_event(runtime, _event("model-running"))
    await started.wait()
    await adapter.enqueue_event(runtime, _event("model-queued"))
    provider.registry.current.generation_id = "model-2"
    release.set()
    await adapter.drain(runtime)

    assert model_ids == ["model-1", "model-2"]
    queued_outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="model-queued",
    )
    assert queued_outcome is not None
    assert queued_outcome.state is JobOutcomeState.SUCCEEDED
    assert queued_outcome.model_generation_id == "model-2"
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_event_accepted_before_pause_is_admitted_on_old_binding(tmp_path) -> None:
    calls: list[str] = []

    async def handler(ctx) -> None:
        calls.append(ctx.event.event_id)

    adapter, plan, target_lease, store, ledger = _fixture(tmp_path, handler)
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)

    source_lease = await store.acquire("snapshot-1")
    token = bind_runtime_snapshot(source_lease)
    try:
        await adapter.pause(runtime)
        adapter._on_event(runtime, _event("accepted-before-pause"))
    finally:
        reset_runtime_snapshot(token)
        await source_lease.release()

    await adapter.stop_components("tx-1", runtime)

    assert calls == ["accepted-before-pause"]
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        event_id="accepted-before-pause",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.SUCCEEDED
    await adapter.close_components("shutdown", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_requeues_exact_queued_event_once_with_original_invocation(tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    path = tmp_path / "restart-event.sqlite"

    async def handler(ctx) -> None:
        calls.append((ctx.event.event_id, ctx.reason))

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    event = _event("restart-event")
    first_ledger.admit(
        _pending_identity(first_plan, invocation_id="restart-invocation", event=event)
    )
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.drain(runtime)
    await adapter.enqueue_event(runtime, event)
    await adapter.drain(runtime)

    outcome = ledger.get("restart-invocation")
    assert calls == [("restart-event", "event")]
    assert outcome is not None
    assert outcome.state is JobOutcomeState.SUCCEEDED
    assert outcome.invocation_id == "restart-invocation"
    assert len(ledger.list_all()) == 1

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_requeues_exact_queued_interval_once(tmp_path) -> None:
    calls: list[tuple[object, str]] = []
    path = tmp_path / "restart-interval.sqlite"
    bucket = "2026-08-17T03:00:00+00:00"

    async def handler(ctx) -> None:
        calls.append((ctx.event, ctx.reason))

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        triggers=(IntervalTrigger(60),),
        ledger_path=path,
    )
    first_ledger.admit(
        _pending_identity(
            first_plan,
            invocation_id="restart-interval-invocation",
            interval_bucket=bucket,
        )
    )
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        triggers=(IntervalTrigger(60),),
        ledger_path=path,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    for _ in range(100):
        outcome = ledger.get("restart-interval-invocation")
        if outcome is not None and outcome.state is JobOutcomeState.SUCCEEDED:
            break
        await asyncio.sleep(0)

    outcome = ledger.get("restart-interval-invocation")
    assert calls == [(None, "interval")]
    assert outcome is not None
    assert outcome.state is JobOutcomeState.SUCCEEDED
    assert outcome.interval_bucket == bucket

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_process_restart_marks_submitting_programmatic_turn_for_reconcile(
    tmp_path,
) -> None:
    calls: list[object] = []
    path = tmp_path / "restart-programmatic.sqlite"
    conversation = _ProgrammaticConversationRuntime()

    async def handler(ctx) -> None:
        calls.append(ctx)

    _first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    event = _event("restart-programmatic")
    invocation_id = "restart-programmatic-invocation"
    _ = first_ledger.admit(
        _pending_identity(first_plan, invocation_id=invocation_id, event=event)
    )
    _ = first_ledger.transition(
        invocation_id,
        JobOutcomeState.RUNNING,
        model_generation_id="model-restart",
    )
    _ = first_ledger.begin_programmatic_turn(invocation_id)
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)

    recovered = ledger.require(invocation_id)
    assert calls == []
    assert recovered.state is JobOutcomeState.FAILED
    assert recovered.programmatic_turn_state is ProgrammaticTurnState.SUBMITTING
    assert recovered.error is not None and "manual reconcile" in recovered.error
    assert any("manual reconcile" in report for report in adapter.recovery_reports)
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_retains_running_retry_and_documents_without_handler_replay(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "restart-pending.sqlite"

    async def handler(ctx) -> None:
        calls.append(ctx.event.event_id)

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    for suffix, phase in (
        ("running", None),
        ("provider-retry", JobOutcomePhase.PROVIDER),
        ("documents-retry", JobOutcomePhase.DOCUMENTS),
    ):
        event = _event(f"restart-{suffix}")
        invocation_id = f"restart-{suffix}-invocation"
        first_ledger.admit(
            _pending_identity(first_plan, invocation_id=invocation_id, event=event)
        )
        first_ledger.transition(
            invocation_id,
            JobOutcomeState.RUNNING,
            model_generation_id="model-restart",
        )
        if phase is not None:
            first_ledger.transition(
                invocation_id,
                JobOutcomeState.RETRY_PENDING,
                phase=phase,
                error=f"{suffix} interrupted",
            )
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.drain(runtime)

    assert calls == []
    assert len(adapter.recovery_reports) == 3
    assert any("running/handler" in report for report in adapter.recovery_reports)
    assert any("retry_pending/provider" in report for report in adapter.recovery_reports)
    assert any("documents phase retained" in report for report in adapter.recovery_reports)
    assert all(
        record.state in {JobOutcomeState.RUNNING, JobOutcomeState.RETRY_PENDING}
        for record in ledger.list_pending()
    )

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_rejects_each_changed_binding_identity_without_current_fallback(
    tmp_path,
) -> None:
    calls: list[str] = []
    path = tmp_path / "restart-identity.sqlite"
    fields = (
        "snapshot_id",
        "plugin_generation_id",
        "artifact_identity",
        "source_revision",
        "handler_export",
        "lifecycle_revision",
        "api_revision",
    )

    async def handler(ctx) -> None:
        calls.append(ctx.event.event_id)

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    for index, field in enumerate(fields):
        event = _event(f"identity-mismatch-{field}")
        identity = _pending_identity(
            first_plan,
            invocation_id=f"identity-mismatch-{index}",
            event=event,
        )
        identity = replace(identity, **{field: f"wrong-{field}"})
        first_ledger.admit(identity)
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.drain(runtime)

    assert calls == []
    assert len(adapter.recovery_reports) == len(fields)
    for field in fields:
        assert any(f"{field} expected=" in report for report in adapter.recovery_reports)
    assert all(record.state is JobOutcomeState.QUEUED for record in ledger.list_pending())

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_reports_pending_outcome_without_exact_current_job_binding(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "restart-missing-job.sqlite"

    async def handler(ctx) -> None:
        calls.append(ctx.event.event_id)

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    event = _event("missing-job-event")
    missing_job_identity = replace(
        _pending_identity(
            first_plan,
            invocation_id="missing-job-invocation",
            event=event,
        ),
        job_name="retired_job",
        semantic_job_id=None,
        handler_export="retired_handler",
    )
    first_ledger.admit(missing_job_identity)
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.drain(runtime)

    assert calls == []
    assert any(
        "exact job binding unavailable" in report
        for report in adapter.recovery_reports
    )
    outcome = ledger.get("missing-job-invocation")
    assert outcome is not None and outcome.state is JobOutcomeState.QUEUED

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_terminalizes_programmatic_state_before_missing_job_lookup(
    tmp_path,
) -> None:
    path = tmp_path / "restart-missing-programmatic-job.sqlite"

    async def handler(_ctx) -> None:
        raise AssertionError("retired programmatic job must not replay")

    _first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
        programmatic_turns=True,
        conversation_runtime=_ProgrammaticConversationRuntime(),
        programmatic_session_creator=lambda **_kwargs: None,
    )
    event = _event("missing-programmatic-job")
    identity = replace(
        _pending_identity(
            first_plan,
            invocation_id="missing-programmatic-invocation",
            event=event,
        ),
        job_name="retired_job",
        semantic_job_id=None,
        handler_export="retired_handler",
    )
    _ = first_ledger.admit(identity)
    _ = first_ledger.transition(
        identity.invocation_id,
        JobOutcomeState.RUNNING,
        model_generation_id="model-restart",
    )
    _ = first_ledger.begin_programmatic_turn(identity.invocation_id)
    await first_lease.release()

    conversation = _ProgrammaticConversationRuntime()
    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
        programmatic_turns=True,
        conversation_runtime=conversation,
        programmatic_session_creator=conversation.create_session,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)

    recovered = ledger.require(identity.invocation_id)
    assert recovered.state is JobOutcomeState.FAILED
    assert recovered.programmatic_turn_state is ProgrammaticTurnState.SUBMITTING
    assert recovered.error is not None and "manual reconcile" in recovered.error
    assert not ledger.list_pending()
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_rejects_malformed_event_payload_without_running_handler(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "restart-payload.sqlite"

    async def handler(ctx) -> None:
        calls.append(ctx.event.event_id)

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    event = _event("malformed-restart-event")
    identity = replace(
        _pending_identity(
            first_plan,
            invocation_id="malformed-restart-invocation",
            event=event,
        ),
        event_payload={"event_id": event.event_id},
    )
    first_ledger.admit(identity)
    await first_lease.release()

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open(runtime)
    await adapter.drain(runtime)

    assert calls == []
    assert any("payload rejected" in report for report in adapter.recovery_reports)
    outcome = ledger.get("malformed-restart-invocation")
    assert outcome is not None and outcome.state is JobOutcomeState.QUEUED

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_materialize_rejects_non_async_or_wrong_handler_signature(tmp_path) -> None:
    def sync_handler(ctx) -> None:
        return None

    adapter, plan, target_lease, _store, _ledger = _fixture(tmp_path, sync_handler)
    with pytest.raises(TypeError, match="必须是 async"):
        await adapter.materialize_closed("tx-1", plan)
    assert adapter.subscription_count == 0
    assert adapter.timer_count == 0
    await target_lease.release()


def test_drift_event_requires_non_empty_event_id() -> None:
    with pytest.raises(ValueError, match="event_id"):
        _event(" ")
