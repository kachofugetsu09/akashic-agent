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
    IntervalTrigger,
    ProgrammaticTurnPreAdmissionError,
    ProgrammaticTurnUncertainError,
    RetryPolicy,
)
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.context import FiberHandle, HealthHandle
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation_job_host import BackgroundJobActivityAdapter
from agent.plugins.job_outcome_ledger import (
    JobOutcomeIdentity,
    JobOutcomeLedger,
    JobOutcomePhase,
    JobOutcomeState,
    ProgrammaticTurnState,
)
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
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


def _module(handler: Any, *, name: str = "drift") -> ComposablePlugin:
    module = ModuleType(f"{name}_module")
    module.api_version = 3
    module.name = name
    module.version = "1.0.0"
    module.apply = _apply
    module.merge_pending = handler
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
    plugin_id = plugin_id_override or "drift"
    plugin = _module(handler, name=plugin_id)
    job_triggers = triggers or (IntervalTrigger(60),)
    definition = BackgroundJobDefinition(
        name=job_name,
        triggers=job_triggers,
        handler_export="merge_pending",
        model_role=model_role,
        debounce_seconds=debounce_seconds,
        coalesce=coalesce,
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
        cast(RuntimeSnapshotStore, store),
        model_provider=provider,
        ledger=ledger,
        clock=clock,
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
        self._terminal: asyncio.Future[object] = (
            asyncio.get_running_loop().create_future()
        )

    async def result(self) -> object:
        return await self._terminal

    def complete(self) -> None:
        if not self._terminal.done():
            self._terminal.set_result(object())


class _ProgrammaticConversationRuntime:
    def __init__(self) -> None:
        self._store = _ProgrammaticSessionStore()
        self.requests: list[
            tuple[Any, RuntimeSnapshotLease, _ProgrammaticTurnHandle]
        ] = []

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


async def _wait_for_accepted_work(runtime: Any) -> None:
    """Wait for accepted requests without stopping the interval producer."""

    for _ in range(1000):
        if (
            not runtime.pending_admission
            and not runtime.queued
            and not runtime.running
            and runtime.queue.empty()
        ):
            return
        await asyncio.sleep(0)
    raise AssertionError("background job did not settle")


def _pending_identity(
    plan,
    *,
    invocation_id: str,
    interval_bucket: str | None = None,
) -> JobOutcomeIdentity:
    return JobOutcomeIdentity(
        plugin_id="drift",
        job_name="merge_pending",
        invocation_id=invocation_id,
        event_id=None,
        interval_bucket=interval_bucket,
        snapshot_id=plan.snapshot_id,
        plugin_generation_id="generation-1",
        model_generation_id="execution-pending",
        artifact_identity=f"{plan.catalog_identity}:drift:merge_pending",
        source_revision="source-1",
        handler_export="merge_pending",
        lifecycle_revision="background-job-v3",
        api_revision="plugin-api-v3",
        event_payload=None,
    )


@pytest.mark.asyncio
async def test_prepare_is_pure_and_materialized_binding_is_closed(tmp_path) -> None:
    calls: list[object] = []

    async def handler(ctx) -> None:
        calls.append(ctx)

    adapter, plan, target_lease, store, ledger = _fixture(tmp_path, handler)
    assert adapter.handler_resolution_count == 0
    assert adapter.timer_count == 0

    runtime = await adapter.materialize_closed("tx-1", plan)
    assert adapter.handler_resolution_count == 1
    assert runtime.admission_open is False
    assert calls == []

    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="event-1"
    )
    await asyncio.sleep(0)
    assert calls == []

    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="event-1"
    )
    await _wait_for_accepted_work(runtime)
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="event-1",
    )
    assert len(calls) == 1
    assert outcome is not None
    assert outcome.state is JobOutcomeState.SUCCEEDED
    assert outcome.event_payload is None
    assert store.leases == 1

    await adapter.close_components("tx-1", runtime)
    await target_lease.release()
    assert store.leases == 0
    assert adapter.timer_count == 0


@pytest.mark.asyncio
async def test_missing_job_catalog_materializes_closed_noop_without_worker(
    tmp_path,
) -> None:
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
    assert binding.timer_count == 0
    adapter.finalize_components("tx-empty", binding)
    assert binding.admission_open

    await adapter.close_components("shutdown", binding)


@pytest.mark.asyncio
async def test_llm_lease_is_invocation_scoped_and_invalid_after_handler(
    tmp_path,
) -> None:
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="llm-event"
    )
    await _wait_for_accepted_work(runtime)
    assert provider.calls == 1
    with pytest.raises(RuntimeError, match="已失效"):
        await saved[0].generate_text(prompt="after terminal")
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="llm-event",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="programmatic-event"
    )
    await _wait_for_accepted_work(runtime)

    assert len(conversation.requests) == 1
    request, child_lease, handle = conversation.requests[0]
    assert request.thread_id.startswith("programmatic:")
    assert request.metadata["plugin_id"] == "drift"
    assert request.metadata["job_name"] == "merge_pending"
    assert request.metadata["generation_id"] == "generation-1"
    assert request.metadata["snapshot_id"] == "snapshot-1"
    assert request.metadata["event_id"] is None
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
    assert (
        ledger.find_by_event(
            plugin_id="drift",
            job_name="merge_pending",
            interval_bucket="programmatic-event",
        ).state
        is JobOutcomeState.SUCCEEDED
    )
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
        await adapter.enqueue_interval(
            runtime, "drift:merge_pending", interval_bucket="real-programmatic"
        )
        await _wait_for_accepted_work(runtime)
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="programmatic-validation"
    )
    await _wait_for_accepted_work(runtime)
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
            session_ids.append(
                await ctx.turns.create_session(metadata={"source": "watch"})
            )
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="create-programmatic-session"
    )
    await _wait_for_accepted_work(runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="reuse-programmatic-session"
    )
    await _wait_for_accepted_work(runtime)

    assert len(conversation.requests) == 1
    request, _lease, handle = conversation.requests[0]
    assert request.thread_id == session_ids[0]
    assert request.metadata["event_id"] is None
    assert (
        request.metadata["invocation_id"]
        != conversation._store.sessions[session_ids[0]]["invocation_id"]
    )
    assert request.metadata["source"] == "watch"
    reused = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="reuse-programmatic-session",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="uncertain-receipt"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="uncertain-receipt",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="post-persist-turn"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="post-persist-turn",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="reset-failure"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="reset-failure",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="ordinary-job"
    )
    await _wait_for_accepted_work(runtime)
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="candidate-job"
    )
    await _wait_for_accepted_work(runtime)
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="admitted-no-retry"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="admitted-no-retry",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="cancelled-admission"
    )
    await started.wait()
    submit_tasks[0].cancel()
    await asyncio.sleep(0)
    submit_tasks[0].cancel()
    release.set()
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="cancelled-admission",
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
        task = asyncio.create_task(
            ctx.turns.submit(session_id, "cancel before admission")
        )
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="cancelled-pre-admission"
    )
    await started.wait()
    submit_tasks[0].cancel()
    release.set()
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="cancelled-pre-admission",
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
async def test_programmatic_turn_port_failure_closes_without_lease_residue(
    tmp_path,
) -> None:
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="programmatic-failure"
    )
    await _wait_for_accepted_work(runtime)
    assert store.leases == 1
    with pytest.raises(RuntimeError, match="已结算"):
        await captured[0].create_session(metadata={})
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="programmatic-failure",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.FAILED
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_cancel_running_releases_snapshot_lease_and_marks_cancelled(
    tmp_path,
) -> None:
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="cancel-event"
    )
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
        interval_bucket="cancel-event",
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="child-failure"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="child-failure",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.FAILED
    assert outcome.error == "RuntimeError: child failed"
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_cancel_intent_wins_when_handler_swallows_cancelled_error(
    tmp_path,
) -> None:
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="swallowed-cancel"
    )
    await started.wait()

    await adapter.cancel_running(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="swallowed-cancel",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.CANCELLED
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_close_components_drains_queued_request_after_running_cancel(
    tmp_path,
) -> None:
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="running-close"
    )
    await started.wait()
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="queued-close"
    )
    assert len(runtime.queued) == 1

    await asyncio.wait_for(adapter.close_components("tx-1", runtime), timeout=1)

    assert runtime.closed is True
    assert runtime.worker_task is None
    assert (
        ledger.find_by_event(
            plugin_id="drift",
            job_name="merge_pending",
            interval_bucket="running-close",
        ).state
        is JobOutcomeState.CANCELLED
    )
    assert (
        ledger.find_by_event(
            plugin_id="drift",
            job_name="merge_pending",
            interval_bucket="queued-close",
        ).state
        is JobOutcomeState.CANCELLED
    )
    assert store.leases == 1
    await target_lease.release()


@pytest.mark.asyncio
async def test_queued_request_selects_model_generation_at_execution_start(
    tmp_path,
) -> None:
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
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="model-running"
    )
    await started.wait()
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="model-queued"
    )
    provider.registry.current.generation_id = "model-2"
    release.set()
    await _wait_for_accepted_work(runtime)

    assert model_ids == ["model-1", "model-2"]
    queued_outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="model-queued",
    )
    assert queued_outcome is not None
    assert queued_outcome.state is JobOutcomeState.SUCCEEDED
    assert queued_outcome.model_generation_id == "model-2"
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_restart_requeues_exact_queued_interval_once(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "restart-interval.sqlite"
    bucket = "2026-08-17T03:00:00+00:00"

    async def handler(ctx) -> None:
        calls.append(ctx.reason)

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
    assert calls == ["interval"]
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
    invocation_id = "restart-programmatic-invocation"
    _ = first_ledger.admit(
        _pending_identity(
            first_plan,
            invocation_id=invocation_id,
            interval_bucket="restart-programmatic",
        )
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
async def test_materialize_rejects_non_async_or_wrong_handler_signature(
    tmp_path,
) -> None:
    def sync_handler(ctx) -> None:
        return None

    adapter, plan, target_lease, _store, _ledger = _fixture(tmp_path, sync_handler)
    with pytest.raises(TypeError, match="必须是 async"):
        await adapter.materialize_closed("tx-1", plan)
    assert adapter.timer_count == 0
    await target_lease.release()
