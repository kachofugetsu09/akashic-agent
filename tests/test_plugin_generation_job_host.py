from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.ports import ControlExecutionResult
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
from agent.plugin_composition import (
    CHAT_MODELS,
    BoundModelDescriptor,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelRequest,
    ModelRole,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation_activity_host import ActivityHost
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
    get_current_runtime_snapshot,
    lease_current_runtime_snapshot,
    reset_runtime_snapshot,
)
from bootstrap.tools import CoreRuntime
from agent.turn_effects import PostCommitEffect
from session.store import SessionStore


class _NoopModel:
    runtime_id = "job-model"

    async def complete(self, _request: ModelRequest) -> LLMResponse:
        raise AssertionError("job unexpectedly called the model")


class _RecordingBoundModel:
    def __init__(self, responder: object, role: ModelRole) -> None:
        runtime_id = str(getattr(responder, "runtime_id", "job-model"))
        self.responder = responder
        self.requests: list[ModelRequest] = []
        self._descriptor = BoundModelDescriptor(
            binding_id=f"job-binding:{runtime_id}:{role.value}",
            plugin_snapshot_id="snapshot-1",
            model_revision=1,
            model_id=runtime_id,
            connection_id="job-connection",
            driver_id="job-driver",
            driver_contract_version="1",
            auth_identity="job-test",
            model=runtime_id,
            role=role,
            reasoning_effort=None,
            capabilities=ModelCapabilities(),
            capability_sources=CapabilitySources(),
            capability_digest="job-capabilities",
        )

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return self._descriptor

    async def complete(self, request: ModelRequest) -> LLMResponse:
        self.requests.append(request)
        return await self.responder.complete(request)  # type: ignore[attr-defined]

    def estimate_context_tokens(self, messages, tools=()) -> int:
        return len(messages) + len(tools)

    def estimate_appended_message_tokens(self, messages) -> int:
        return len(messages)

    @property
    def max_tool_schemas(self) -> int | None:
        return None


class _StrictChatModels:
    """Model facade that proves exact snapshot binding and lease cleanup."""

    def __init__(self, responder: object | None = None) -> None:
        self.responder = responder or _NoopModel()
        self.execution_enters = 0
        self.execution_exits = 0
        self.roles: list[ModelRole] = []
        self.models: list[_RecordingBoundModel] = []

    @asynccontextmanager
    async def execution(self, **_selection: object):
        snapshot = get_current_runtime_snapshot()
        assert snapshot is not None and snapshot.snapshot_id == "snapshot-1"
        fork = lease_current_runtime_snapshot()
        assert fork is not None and fork.snapshot is snapshot
        self.execution_enters += 1
        facade = self

        class _Execution:
            def chat(self, role: ModelRole) -> _RecordingBoundModel:
                facade.roles.append(role)
                model = _RecordingBoundModel(facade.responder, role)
                facade.models.append(model)
                return model

        try:
            yield _Execution()
        finally:
            self.execution_exits += 1
            await fork.release()


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
    model_responder: object | None = None,
    chat_models: _StrictChatModels | None = None,
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
    models = chat_models or _StrictChatModels(model_responder)
    required_services: list[object] = []

    def require_service(key: object) -> object:
        required_services.append(key)
        assert key is CHAT_MODELS
        return models

    snapshot = SimpleNamespace(
        snapshot_id="snapshot-1",
        background_job_catalog=catalog,
        generations={plugin_id: generation},
        lease_count=0,
        composition_root=SimpleNamespace(
            context=SimpleNamespace(
                require=require_service,
            )
        ),
        required_services=required_services,
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
        self.execution_scopes: list[Any] = []

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
        execution_scope: Any = None,
    ) -> _ProgrammaticTurnHandle:
        assert fresh_interaction is True
        assert execution_scope is not None
        self.execution_scopes.append(execution_scope)
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
    class ModelResponder:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> LLMResponse:
            assert request.messages[-1]["content"] == "hello"
            self.calls += 1
            return LLMResponse(content="model-answer")

    responder = ModelResponder()
    saved: list[object] = []

    async def handler(ctx) -> None:
        saved.append(ctx.llm)
        response = await ctx.llm.complete(
            ModelRequest(messages=[{"role": "user", "content": "hello"}])
        )
        assert response.content == "model-answer"

    chat_models = _StrictChatModels(responder)
    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        chat_models=chat_models,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="llm-event"
    )
    await _wait_for_accepted_work(runtime)
    assert responder.calls == 1
    assert store.snapshot.required_services == [CHAT_MODELS]
    assert chat_models.execution_enters == 1
    assert chat_models.execution_exits == 1
    assert chat_models.roles == [ModelRole.AGENT]
    with pytest.raises(RuntimeError, match="已失效"):
        await saved[0].complete(
            ModelRequest(messages=[{"role": "user", "content": "after terminal"}])
        )
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="llm-event",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.SUCCEEDED
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_model_retry_reuses_one_exact_binding_and_execution(tmp_path) -> None:
    class ModelResponder:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, _request: ModelRequest) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("retry me")
            return LLMResponse(content="done")

    responder = ModelResponder()
    chat_models = _StrictChatModels(responder)
    bindings: list[str] = []

    async def handler(ctx) -> None:
        bindings.append(ctx.llm.descriptor.binding_id)
        await ctx.llm.complete(
            ModelRequest(messages=[{"role": "user", "content": "retry"}])
        )

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        chat_models=chat_models,
        retry_policy=RetryPolicy(max_attempts=2),
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="model-retry"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="model-retry",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.SUCCEEDED
    assert bindings == [bindings[0], bindings[0]]
    assert outcome.model_generation_id == bindings[0]
    assert responder.calls == 2
    assert chat_models.execution_enters == 1
    assert chat_models.execution_exits == 1
    assert len(chat_models.models) == 1
    assert len(chat_models.models[0].requests) == 2
    assert store.leases == 1
    await adapter.close_components("tx-1", runtime)
    await target_lease.release()


@pytest.mark.asyncio
async def test_model_view_rejects_inherited_child_task(tmp_path) -> None:
    async def child(model) -> None:
        await model.complete(
            ModelRequest(messages=[{"role": "user", "content": "child"}])
        )

    async def handler(ctx) -> None:
        ctx.spawn_child(child(ctx.llm), name="model-child")

    chat_models = _StrictChatModels()
    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        chat_models=chat_models,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="model-child"
    )
    await _wait_for_accepted_work(runtime)

    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="model-child",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.FAILED
    assert "不能由子 task 继承" in str(outcome.error)
    assert chat_models.execution_enters == 1
    assert chat_models.execution_exits == 1
    assert store.leases == 1
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
    assert (
        conversation.execution_scopes[0].post_commit_effect is PostCommitEffect.SUPPRESS
    )
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

    async def execute(request) -> ControlExecutionResult:
        return ControlExecutionResult(response=f"reply:{request.input}")

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
        assert record.metadata["inboundMetadata"] == {
            "disabled_prompt_sections": ["memory"],
            "effects": {"post_commit": "suppress"},
        }
        assert record.items[0].data["metadata"] == record.metadata["inboundMetadata"]
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

    async def execute(_request) -> ControlExecutionResult:
        return ControlExecutionResult(response="unused")

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
            self,
            request,
            *,
            runtime_snapshot_lease,
            fresh_interaction,
            execution_scope=None,
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
            self,
            request,
            *,
            runtime_snapshot_lease,
            fresh_interaction,
            execution_scope=None,
        ):
            assert fresh_interaction is True
            started.set()
            await release.wait()
            return await super().start_turn(
                request,
                runtime_snapshot_lease=runtime_snapshot_lease,
                fresh_interaction=fresh_interaction,
                execution_scope=execution_scope,
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
            self,
            request,
            *,
            runtime_snapshot_lease,
            fresh_interaction,
            execution_scope=None,
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

    class ModelResponder:
        async def complete(self, _request: ModelRequest) -> LLMResponse:
            started.set()
            await release.wait()
            return LLMResponse(content="never")

    async def handler(ctx) -> None:
        await ctx.llm.complete(
            ModelRequest(messages=[{"role": "user", "content": "blocked"}])
        )

    chat_models = _StrictChatModels(ModelResponder())
    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        chat_models=chat_models,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="cancel-event"
    )
    for _ in range(100):
        if runtime.running:
            break
        await asyncio.sleep(0)
    await started.wait()
    assert store.leases == 3
    await adapter.cancel_running(runtime)
    outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="cancel-event",
    )
    assert outcome is not None and outcome.state is JobOutcomeState.CANCELLED
    assert chat_models.execution_enters == 1
    assert chat_models.execution_exits == 1
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
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
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
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
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

    class ModelResponder:
        async def complete(self, _request: ModelRequest) -> LLMResponse:
            started.set()
            await asyncio.Event().wait()
            return LLMResponse(content="never")

    async def handler(ctx) -> None:
        await ctx.llm.complete(
            ModelRequest(messages=[{"role": "user", "content": "blocked"}])
        )

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        model_responder=ModelResponder(),
        coalesce=False,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
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

    class ModelResponder:
        def __init__(self) -> None:
            self.calls = 0
            self.runtime_id = "model-1"

        async def complete(self, _request: ModelRequest) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                started.set()
                await release.wait()
            return LLMResponse(content="ok")

    responder = ModelResponder()

    async def handler(ctx) -> None:
        model_ids.append(ctx.model_generation_id)
        await ctx.llm.complete(
            ModelRequest(messages=[{"role": "user", "content": "model"}])
        )

    adapter, plan, target_lease, _store, ledger = _fixture(
        tmp_path,
        handler,
        model_responder=responder,
        coalesce=False,
    )
    runtime = await adapter.materialize_closed("tx-1", plan)
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="model-running"
    )
    await started.wait()
    await adapter.enqueue_interval(
        runtime, "drift:merge_pending", interval_bucket="model-queued"
    )
    responder.runtime_id = "model-2"
    release.set()
    await _wait_for_accepted_work(runtime)

    assert len(model_ids) == 2
    assert ":model-1:" in model_ids[0]
    assert ":model-2:" in model_ids[1]
    queued_outcome = ledger.find_by_event(
        plugin_id="drift",
        job_name="merge_pending",
        interval_bucket="model-queued",
    )
    assert queued_outcome is not None
    assert queued_outcome.state is JobOutcomeState.SUCCEEDED
    assert ":model-2:" in queued_outcome.model_generation_id
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
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)
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
async def test_restart_closes_orphaned_running_outcome_without_replay(tmp_path) -> None:
    calls: list[str] = []
    path = tmp_path / "restart-running.sqlite"

    async def handler(ctx) -> None:
        calls.append(ctx.reason)

    first, first_plan, first_lease, _store, first_ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    invocation_id = "restart-running-invocation"
    _ = first_ledger.admit(
        _pending_identity(
            first_plan,
            invocation_id=invocation_id,
            interval_bucket="restart-running",
        )
    )
    _ = first_ledger.transition(
        invocation_id,
        JobOutcomeState.RUNNING,
        model_generation_id="model-restart",
    )
    await first_lease.release()

    adapter, plan, target_lease, store, ledger = _fixture(
        tmp_path,
        handler,
        ledger_path=path,
    )
    adapter.discard_plan("tx-1", plan)

    class _BlockingRecoveryAdapter(BackgroundJobActivityAdapter):
        def __init__(self) -> None:
            super().__init__(cast(RuntimeSnapshotStore, store), ledger=ledger)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def _recover_pending(self, runtime):
            self.entered.set()
            await self.release.wait()
            return await super()._recover_pending(runtime)

    recovering = _BlockingRecoveryAdapter()
    host = ActivityHost((recovering,))
    transaction = await host.prepare_transaction(target_lease)
    await host.pause_and_drain(transaction)
    staged = await host.materialize_closed(transaction)
    host.finalize(transaction)
    opening = asyncio.create_task(host.open(transaction))
    await recovering.entered.wait()

    runtime = staged.child_bindings[recovering.name]
    assert not staged.admission_open
    assert not runtime.admission_open
    recovering.release.set()
    await opening

    recovered = ledger.require(invocation_id)
    assert calls == []
    assert recovered.state is JobOutcomeState.FAILED
    assert (
        recovered.error is not None
        and "external effects are unknown" in recovered.error
    )
    assert any(
        "automatic replay is disabled" in report
        for report in recovering.recovery_reports
    )
    await host.close()


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
    await adapter.open_components("tx-1", runtime)
    adapter.finalize_components("tx-1", runtime)

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
