"""Generation-scoped execution host for the v3 background-job catalog.

This module is intentionally independent from ``PluginJobRuntime``.  A job is
bound to one committed snapshot and one exact ComposablePlugin module; all
execution state (leases, queue entries, child tasks and ledger transitions) is
owned by that binding.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import secrets
from contextvars import ContextVar, Token
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, TYPE_CHECKING, cast

from agent.control.models import TurnRequest
from agent.control.scoped_turn import ScopedTurnHandle, ScopedTurnPort
from agent.control.turn_scope import TurnExecutionScope
from agent.turn_effects import PostCommitEffect
from agent.control.errors import TurnAdmissionUncertainError
from agent.plugin_composition import (
    CHAT_MODELS,
    BoundModelDescriptor,
    LLMResponse,
    ModelRequest,
    ModelRole,
)
from agent.plugin_composition.models import BoundChatModel, ChatModels
from agent.plugin_composition.background_jobs import (
    BackgroundJobBinding,
    BackgroundJobCatalog,
    IntervalTrigger,
    ProgrammaticTurnPort,
    ProgrammaticTurnPreAdmissionError,
    ProgrammaticTurnReceipt,
    ProgrammaticTurnUncertainError,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.job_outcome_ledger import (
    JobOutcomeIdentity,
    JobOutcomeLedger,
    JobOutcomePhase,
    JobOutcomeRecord,
    JobOutcomeState,
    ProgrammaticTurnState,
)
from agent.plugins.snapshot import (
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)

if TYPE_CHECKING:
    from agent.plugins.generation_activity_host import ActivityCatalog


_JOB_LIFECYCLE_REVISION = "background-job-v3"
_JOB_API_REVISION = "plugin-api-v3"
_PROGRAMMATIC_SESSION_RESERVED_FIELDS = frozenset(
    {
        "event_id",
        "generation_id",
        "invocation_id",
        "job_name",
        "plugin_id",
        "programmatic",
        "runtime",
        "snapshot_id",
    }
)


_CURRENT_INVOCATION_TOKEN: ContextVar[object | None] = ContextVar(
    "background_job_invocation_token",
    default=None,
)


class _JobBoundChatModel:
    """Fence one public BoundChatModel to an exact job invocation."""

    __slots__ = (
        "_model",
        "_snapshot_lease",
        "_snapshot_id",
        "_plugin_generation_id",
        "_invocation_token",
        "_owner_task",
        "_invalidated",
        "_provider_called",
    )

    def __init__(
        self,
        model: BoundChatModel,
        *,
        snapshot_lease: RuntimeSnapshotLease,
        snapshot_id: str,
        plugin_generation_id: str,
        invocation_token: object,
    ) -> None:
        self._model = model
        self._snapshot_lease = snapshot_lease
        self._snapshot_id = snapshot_id
        self._plugin_generation_id = plugin_generation_id
        self._invocation_token = invocation_token
        self._owner_task = asyncio.current_task()
        self._invalidated = False
        self._provider_called = False

    @property
    def provider_called(self) -> bool:
        return self._provider_called

    @property
    def invocation_token(self) -> object:
        return self._invocation_token

    def invalidate(self) -> None:
        self._invalidated = True

    @property
    def descriptor(self) -> BoundModelDescriptor:
        self._require_live()
        return self._model.descriptor

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """Complete one request while the invocation and snapshot remain live."""

        self._require_live()
        self._provider_called = True
        response = await self._model.complete(request)
        self._require_live()
        return response

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        self._require_live()
        return self._model.estimate_context_tokens(messages, tools)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        self._require_live()
        return self._model.estimate_appended_message_tokens(messages)

    @property
    def max_tool_schemas(self) -> int | None:
        self._require_live()
        return self._model.max_tool_schemas

    def _require_live(self) -> None:
        if self._invalidated:
            raise RuntimeError("BackgroundJob BoundChatModel 已失效")
        if asyncio.current_task() is not self._owner_task:
            raise RuntimeError("BackgroundJob BoundChatModel 不能由子 task 继承")
        if not self._snapshot_lease.active:
            raise RuntimeError("BackgroundJob BoundChatModel 的 snapshot lease 已释放")
        if self._snapshot_lease.snapshot.snapshot_id != self._snapshot_id:
            raise RuntimeError("BackgroundJob BoundChatModel 的 snapshot identity 不匹配")
        generations = getattr(self._snapshot_lease.snapshot, "generations", {})
        if not any(
            str(getattr(generation, "generation_id", "")) == self._plugin_generation_id
            for generation in generations.values()
        ):
            raise RuntimeError("BackgroundJob BoundChatModel 的 plugin generation 不匹配")
        if _CURRENT_INVOCATION_TOKEN.get() is not self._invocation_token:
            raise RuntimeError("BackgroundJob BoundChatModel 的 invocation token 不匹配")


@dataclass(slots=True)
class BackgroundJobContext:
    """Invocation-scoped context exposed to one exact async plugin handler."""

    plugin_id: str
    reason: str
    triggered_at: datetime
    snapshot_id: str
    generation_id: str
    plugin_generation_id: str
    model_generation_id: str
    llm: BoundChatModel
    activation_token: object
    turns: ProgrammaticTurnPort | None = None
    _children: set[asyncio.Future[None]] = field(
        default_factory=set, init=False, repr=False
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def spawn_child(self, awaitable: Awaitable[None], *, name: str) -> None:
        """Start a Core-owned child task that is drained before handler terminal state."""

        if self._closed:
            raise RuntimeError("BackgroundJobContext 已结算，不能 spawn child")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("child name 必须是非空字符串")
        if inspect.iscoroutine(awaitable):
            task: asyncio.Future[None] = asyncio.create_task(
                cast(Coroutine[Any, Any, None], awaitable),
                name=f"background_job:{name}",
            )
        else:
            task = asyncio.ensure_future(awaitable)
        self._children.add(task)

    async def drain_children(self) -> None:
        """Wait for all children and preserve their first failure."""

        self._closed = True
        while self._children:
            tasks = tuple(self._children)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._children.difference_update(tasks)
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    async def cancel_children(self) -> None:
        """Cancel every child after handler failure and wait for child cleanup."""

        self._closed = True
        tasks = tuple(self._children)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(frozen=True, slots=True)
class BackgroundJobPlan:
    """Pure materialization plan fixed to one exact snapshot and catalog."""

    transaction_id: str
    snapshot_id: str
    catalog_identity: str
    target_lease: RuntimeSnapshotLease
    bindings: tuple[BackgroundJobBinding, ...]
    snapshot_store: RuntimeSnapshotStore


@dataclass(slots=True)
class _MaterializedJob:
    key: str
    binding: BackgroundJobBinding
    handler: Callable[[BackgroundJobContext], Awaitable[object]]
    source_revision: str
    artifact_identity: str
    snapshot_id: str

    @property
    def admission_key(self) -> tuple[str, str, object, str]:
        """Return the non-persistent binding identity used by queue admission."""

        return (
            self.snapshot_id,
            self.binding.generation_id,
            self.binding.activation_token,
            self.key,
        )


@dataclass(slots=True)
class _JobRequest:
    binding: "BackgroundJobRuntimeBinding"
    job: _MaterializedJob
    reason: str
    interval_bucket: str | None
    invocation_id: str
    snapshot_lease: RuntimeSnapshotLease
    outcome: JobOutcomeRecord
    cancelled: bool = False
    lease_released: bool = False
    programmatic_turn_uncertain: bool = False


class _ProgrammaticTurnPort:
    """Admit programmatic Turns through one exact job invocation lease."""

    def __init__(
        self,
        runtime: object,
        request: _JobRequest,
        session_creator: Callable[..., object],
        session_reader: Callable[[str], object],
        ledger: JobOutcomeLedger,
    ) -> None:
        self._runtime = runtime
        self._request = request
        self._turn_port = ScopedTurnPort(
            runtime,
            request.snapshot_lease,
            execution_scope=TurnExecutionScope(
                disabled_prompt_sections=frozenset({"memory"}),
                post_commit_effect=PostCommitEffect.SUPPRESS,
                tool_source="background_job",
            ),
        )
        self._session_creator = session_creator
        self._session_reader = session_reader
        self._ledger = ledger
        self._invocation_token: object | None = None
        self._sessions: dict[str, dict[str, object]] = {}
        self._turn_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def _bind_invocation_token(self, token: object) -> None:
        """Bind the non-forgeable Context token before exposing the port."""

        if self._invocation_token is not None:
            raise RuntimeError("programmatic Turn invocation token 已绑定")
        self._invocation_token = token

    async def create_session(self, *, metadata: Mapping[str, object]) -> str:
        """Persist one Core-named programmatic session with immutable provenance."""

        self._require_live()
        payload = _programmatic_session_metadata(self._request, metadata)
        key = "programmatic:" + secrets.token_hex(16)
        try:
            result = self._session_creator(key=key, metadata=payload)
            if inspect.isawaitable(result):
                await result
        except BaseException as error:
            raise ProgrammaticTurnPreAdmissionError(
                "programmatic session 未完成持久化"
            ) from error
        self._sessions[key] = payload
        return key

    async def submit(
        self,
        session_id: str,
        content: str,
    ) -> ProgrammaticTurnReceipt:
        """Admit one Turn on a port-created session and release its lease at terminal."""

        self._require_live()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("programmatic session_id 必须是非空字符串")
        metadata = await self._require_owned_session(session_id)
        if not isinstance(content, str):
            raise TypeError("programmatic Turn content 必须是字符串")
        try:
            self._ledger.begin_programmatic_turn(self._request.invocation_id)
        except BaseException as error:
            raise ProgrammaticTurnPreAdmissionError(
                "programmatic Turn 无法建立 durable admission boundary"
            ) from error
        try:
            task = asyncio.create_task(
                self._admit(session_id, content, metadata),
                name=f"programmatic_turn_admission:{session_id}",
            )
        except BaseException as error:
            try:
                self._ledger.reset_programmatic_turn(self._request.invocation_id)
            except BaseException as reset_error:
                self._request.programmatic_turn_uncertain = True
                raise ProgrammaticTurnUncertainError(
                    "programmatic Turn admission task 失败且 receipt 无法重置"
                ) from reset_error
            raise ProgrammaticTurnPreAdmissionError(
                "programmatic Turn admission task 未创建"
            ) from error
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        try:
            handle = task.result()
        except BaseException as error:
            if isinstance(error, TurnAdmissionUncertainError):
                self._request.programmatic_turn_uncertain = True
                raise ProgrammaticTurnUncertainError(
                    "programmatic Turn 已持久化，但未取得 handle"
                ) from error
            try:
                self._ledger.reset_programmatic_turn(self._request.invocation_id)
            except BaseException as reset_error:
                self._request.programmatic_turn_uncertain = True
                raise ProgrammaticTurnUncertainError(
                    "programmatic Turn pre-admission receipt 无法重置"
                ) from reset_error
            if cancelled:
                raise asyncio.CancelledError
            raise ProgrammaticTurnPreAdmissionError(
                "programmatic Turn 在取得 receipt 前失败"
            ) from error
        self._retain_turn_cleanup(handle, session_id)
        try:
            turn_id = _turn_handle_id(handle)
            self._ledger.commit_programmatic_turn(
                self._request.invocation_id,
                turn_id,
            )
        except BaseException as error:
            self._request.programmatic_turn_uncertain = True
            raise ProgrammaticTurnUncertainError(
                "programmatic Turn 已取得 handle，但 durable receipt 未确认"
            ) from error
        if cancelled:
            raise asyncio.CancelledError
        return ProgrammaticTurnReceipt(session_id=session_id, turn_id=turn_id)

    async def _require_owned_session(self, session_id: str) -> dict[str, object]:
        """Verify Session ownership and rebuild provenance for the current Turn."""

        if session_id in self._sessions:
            metadata: Mapping[str, object] = self._sessions[session_id]
        else:
            try:
                result = self._session_reader(session_id)
                if inspect.isawaitable(result):
                    result = await result
            except BaseException as error:
                raise ProgrammaticTurnPreAdmissionError(
                    "programmatic Turn session provenance 无法读取"
                ) from error
            if not isinstance(result, Mapping):
                raise ProgrammaticTurnPreAdmissionError(
                    "programmatic Turn session 不存在或不属于当前插件"
                )
            stored_metadata = result.get("metadata")
            if not isinstance(stored_metadata, Mapping):
                raise ProgrammaticTurnPreAdmissionError(
                    "programmatic Turn session 缺少 Core provenance"
                )
            if any(not isinstance(key, str) for key in stored_metadata):
                raise ProgrammaticTurnPreAdmissionError(
                    "programmatic Turn session provenance key 无效"
                )
            metadata = cast(Mapping[str, object], stored_metadata)
        if (
            metadata.get("programmatic") is not True
            or metadata.get("plugin_id") != self._request.job.binding.plugin_id
            or metadata.get("job_name") != self._request.job.binding.name
        ):
            raise ProgrammaticTurnPreAdmissionError(
                "programmatic Turn session provenance 不匹配",
                reason="session_provenance_mismatch",
            )
        plugin_metadata = {
            str(key): value
            for key, value in metadata.items()
            if isinstance(key, str) and key not in _PROGRAMMATIC_SESSION_RESERVED_FIELDS
        }
        return _programmatic_session_metadata(self._request, plugin_metadata)

    async def _admit(
        self,
        session_id: str,
        content: str,
        metadata: Mapping[str, object],
    ) -> ScopedTurnHandle:
        request = TurnRequest(session_id, content, dict(metadata))
        return await self._turn_port.start(request)

    def _close(self) -> None:
        """Close this invocation port without cancelling already admitted Turns."""

        self._closed = True

    def _retain_turn_cleanup(
        self,
        handle: ScopedTurnHandle,
        session_id: str,
    ) -> None:
        task = asyncio.create_task(
            handle.cleanup(),
            name=f"programmatic_turn_cleanup:{session_id}",
        )
        self._turn_tasks.add(task)

        def finish(completed: asyncio.Task[None]) -> None:
            self._turn_tasks.discard(completed)
            # Retrieve the exception so a failed child Turn does not become an orphan task.
            _ = completed.exception() if not completed.cancelled() else None

        task.add_done_callback(finish)

    def _require_live(self) -> None:
        if self._closed:
            raise RuntimeError("ProgrammaticTurnPort 已结算")
        if self._invocation_token is None:
            raise RuntimeError("ProgrammaticTurnPort invocation token 未绑定")
        if _CURRENT_INVOCATION_TOKEN.get() is not self._invocation_token:
            raise RuntimeError("ProgrammaticTurnPort invocation token 不匹配")
        lease = self._request.snapshot_lease
        if not lease.active:
            raise RuntimeError("ProgrammaticTurnPort 的 RuntimeSnapshot lease 已释放")
        if lease.snapshot.snapshot_id != self._request.binding.snapshot_id:
            raise RuntimeError("ProgrammaticTurnPort snapshot identity 不匹配")


@dataclass(slots=True)
class _InvocationResources:
    turns: _ProgrammaticTurnPort | None

    async def finalize(self) -> None:
        if self.turns is not None:
            self.turns._close()


@dataclass(slots=True)
class BackgroundJobRuntimeBinding:
    """Closed or open resources materialized from one BackgroundJobPlan."""

    snapshot_id: str
    catalog_identity: str
    jobs: Mapping[str, _MaterializedJob]
    snapshot_store: RuntimeSnapshotStore
    interval_task: asyncio.Task[None] | None = None
    worker_task: asyncio.Task[None] | None = None
    queue: asyncio.Queue[_JobRequest | None] = field(default_factory=asyncio.Queue)
    pending_admission: set[asyncio.Task[None]] = field(default_factory=set)
    admission_errors: list[BaseException] = field(default_factory=list, repr=False)
    queued: dict[str, _JobRequest] = field(default_factory=dict)
    running: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    running_requests: dict[str, _JobRequest] = field(default_factory=dict)
    running_job_keys: dict[str, tuple[str, str, object, str]] = field(
        default_factory=dict
    )
    recovery_scanned: bool = False
    recovery_reports: list[str] = field(default_factory=list)
    admission_open: bool = False
    closed: bool = False

    @property
    def timer_count(self) -> int:
        return int(self.interval_task is not None and not self.interval_task.done())


class BackgroundJobActivityAdapter:
    """Materialize, admit, execute and drain generation-bound background jobs."""

    name = "background_jobs"

    def __init__(
        self,
        snapshot_store: RuntimeSnapshotStore | None = None,
        *,
        ledger: JobOutcomeLedger | None = None,
        ledger_path: str | None = None,
        workspace: str | None = None,
        clock: Callable[[], datetime] | None = None,
        invocation_id_factory: Callable[[], str] | None = None,
        interval_poll_seconds: float = 0.05,
        conversation_runtime: object | None = None,
        programmatic_session_creator: Callable[..., object] | None = None,
        programmatic_session_reader: Callable[[str], object] | None = None,
    ) -> None:
        if ledger is not None and ledger_path is not None:
            raise TypeError("ledger 不能与 ledger_path 同时提供")
        if interval_poll_seconds <= 0:
            raise ValueError("interval_poll_seconds 必须为正数")
        self._snapshot_store = snapshot_store
        if ledger is not None:
            self._ledger = ledger
        elif ledger_path is not None:
            self._ledger = JobOutcomeLedger(ledger_path)
        elif workspace is not None:
            self._ledger = JobOutcomeLedger.for_workspace(workspace)
        else:
            self._ledger = None
        self._conversation_runtime: object | None = None
        self._programmatic_session_creator: Callable[..., object] | None = None
        self._programmatic_session_reader: Callable[[str], object] | None = None
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._invocation_id_factory = invocation_id_factory or (
            lambda: "invocation-" + secrets.token_hex(16)
        )
        self._interval_poll_seconds = interval_poll_seconds
        self._bindings: dict[str, BackgroundJobRuntimeBinding] = {}
        self._plans: dict[str, BackgroundJobPlan] = {}
        self._handler_resolution_count = 0
        self._active: BackgroundJobRuntimeBinding | None = None

        if conversation_runtime is not None:
            self.bind_conversation_runtime(
                conversation_runtime,
                programmatic_session_creator=programmatic_session_creator,
                programmatic_session_reader=programmatic_session_reader,
            )

    @property
    def ledger(self) -> JobOutcomeLedger | None:
        return self._ledger

    @property
    def active_binding(self) -> BackgroundJobRuntimeBinding | None:
        return self._active

    @property
    def conversation_runtime(self) -> object | None:
        return self._conversation_runtime

    def bind_conversation_runtime(
        self,
        runtime: object,
        *,
        programmatic_session_creator: Callable[..., object] | None = None,
        programmatic_session_reader: Callable[[str], object] | None = None,
    ) -> None:
        """Bind the unique Core ConversationRuntime before job admission opens."""

        if self._conversation_runtime is not None:
            if (
                self._conversation_runtime is runtime
                and (
                    programmatic_session_creator is None
                    or programmatic_session_creator
                    is self._programmatic_session_creator
                )
                and (
                    programmatic_session_reader is None
                    or programmatic_session_reader is self._programmatic_session_reader
                )
            ):
                if (
                    self._programmatic_session_creator is None
                    and programmatic_session_creator is not None
                ):
                    self._programmatic_session_creator = programmatic_session_creator
                if (
                    self._programmatic_session_reader is None
                    and programmatic_session_reader is not None
                ):
                    self._programmatic_session_reader = programmatic_session_reader
                return
            raise RuntimeError("BackgroundJob ConversationRuntime owner 已绑定")
        if not callable(getattr(runtime, "start_turn", None)):
            raise TypeError("BackgroundJob ConversationRuntime 缺少 start_turn")
        if programmatic_session_creator is not None and not callable(
            programmatic_session_creator
        ):
            raise TypeError("BackgroundJob programmatic session creator 必须可调用")
        if programmatic_session_reader is not None and not callable(
            programmatic_session_reader
        ):
            raise TypeError("BackgroundJob programmatic session reader 必须可调用")
        self._conversation_runtime = runtime
        self._programmatic_session_creator = programmatic_session_creator
        self._programmatic_session_reader = programmatic_session_reader

    @property
    def handler_resolution_count(self) -> int:
        return self._handler_resolution_count

    @property
    def timer_count(self) -> int:
        return sum(binding.timer_count for binding in self._bindings.values())

    @property
    def recovery_reports(self) -> tuple[str, ...]:
        """Return restart recovery findings retained by every materialized binding."""

        return tuple(
            report
            for binding in self._bindings.values()
            for report in binding.recovery_reports
        )

    def prepare_components(
        self,
        transaction_id: str,
        target_lease: RuntimeSnapshotLease,
        target_catalog: object,
    ) -> BackgroundJobPlan:
        """Validate immutable target identity without starting handlers or resources."""

        # 1. Validate only frozen snapshot/catalog identity; no handler or provider lookup is allowed here.
        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError("transaction_id 必须是非空字符串")
        if not target_lease.active:
            raise RuntimeError("BackgroundJob target snapshot lease 已失效")
        catalog = _background_catalog(target_catalog)
        snapshot = target_lease.snapshot
        snapshot_catalog = snapshot.background_job_catalog
        if snapshot_catalog is not catalog:
            if catalog is None or snapshot_catalog is None:
                raise RuntimeError("BackgroundJob target catalog 与 snapshot 不匹配")
            if snapshot_catalog.identity != catalog.identity:
                raise RuntimeError("BackgroundJob target catalog 与 snapshot 不匹配")
        store = self._snapshot_store or _lease_store(target_lease)
        if store is None:
            raise RuntimeError("BackgroundJob 需要 RuntimeSnapshotStore")
        bindings = () if catalog is None else tuple(catalog.values())
        candidate_snapshot = bool(target_lease.validation_candidate_plugin_ids)
        if (
            any(binding.definition.programmatic_turns for binding in bindings)
            and not candidate_snapshot
            and (
                self._conversation_runtime is None
                or self._programmatic_session_creator is None
                or self._programmatic_session_reader is None
            )
        ):
            raise RuntimeError(
                "BackgroundJob programmatic_turns 需要在 PluginManager.load_all 前绑定 ConversationRuntime 与 SessionStore creator"
            )
        for binding in bindings:
            generation = snapshot.generations.get(binding.plugin_id)
            if generation is None:
                raise RuntimeError(
                    f"BackgroundJob owner 不属于 target snapshot: {binding.plugin_id}"
                )
            if generation.generation_id != binding.generation_id:
                raise RuntimeError(
                    f"BackgroundJob generation identity 不匹配: {binding.plugin_id}:{binding.name}"
                )
        plan = BackgroundJobPlan(
            transaction_id=transaction_id,
            snapshot_id=snapshot.snapshot_id,
            catalog_identity="" if catalog is None else catalog.identity,
            target_lease=target_lease,
            bindings=bindings,
            snapshot_store=store,
        )
        self._plans[transaction_id] = plan
        return plan

    async def materialize_closed(
        self,
        transaction_id: str,
        plan: BackgroundJobPlan,
    ) -> BackgroundJobRuntimeBinding:
        """Resolve exact async exports and create closed resources without admission."""

        # 1. Resolve exports only from the target snapshot's exact ComposablePlugin modules.
        expected = self._plans.get(transaction_id)
        if expected is not plan or plan.transaction_id != transaction_id:
            raise RuntimeError("BackgroundJob materialization plan 已失效")
        if not plan.target_lease.active:
            raise RuntimeError("BackgroundJob target snapshot lease 已释放")
        jobs: dict[str, _MaterializedJob] = {}
        snapshot = plan.target_lease.snapshot
        try:
            if (
                any(binding.definition.programmatic_turns for binding in plan.bindings)
                and not plan.target_lease.validation_candidate_plugin_ids
                and (
                    self._conversation_runtime is None
                    or self._programmatic_session_creator is None
                    or self._programmatic_session_reader is None
                )
            ):
                raise RuntimeError(
                    "BackgroundJob programmatic_turns 缺少 ConversationRuntime owner 或 SessionStore creator"
                )
            for binding in plan.bindings:
                generation = snapshot.generations[binding.plugin_id]
                handler = self._resolve_handler(
                    generation.instance, binding.handler_export
                )
                source_revision = str(generation.source_revision)
                key = f"{binding.plugin_id}:{binding.name}"
                jobs[key] = _MaterializedJob(
                    key=key,
                    binding=binding,
                    handler=handler,
                    source_revision=source_revision,
                    artifact_identity=f"{plan.catalog_identity}:{key}",
                    snapshot_id=plan.snapshot_id,
                )
            runtime = BackgroundJobRuntimeBinding(
                snapshot_id=plan.snapshot_id,
                catalog_identity=plan.catalog_identity,
                jobs=MappingProxyType(jobs),
                snapshot_store=plan.snapshot_store,
            )
            # 2. Build producer resources while the binding remains closed.
            if not jobs:
                self._bindings[plan.snapshot_id] = runtime
                return runtime
            if any(
                isinstance(trigger, IntervalTrigger)
                for job in jobs.values()
                for trigger in job.binding.definition.triggers
            ):
                runtime.interval_task = asyncio.create_task(
                    self._interval_loop(runtime),
                    name=f"background_job_intervals:{plan.snapshot_id}",
                )
            runtime.worker_task = asyncio.create_task(
                self._worker_loop(runtime),
                name=f"background_job_worker:{plan.snapshot_id}",
            )
            self._bindings[plan.snapshot_id] = runtime
            self._handler_resolution_count += len(jobs)
            return runtime
        except BaseException:
            if "runtime" in locals():
                await self.close_components(transaction_id, runtime)
            raise

    async def stop_components(
        self,
        transaction_id: str,
        old_binding: BackgroundJobRuntimeBinding,
    ) -> None:
        """Stop new admissions and drain accepted old requests before publication swap."""

        self._require_binding(transaction_id, old_binding)
        old_binding.admission_open = False
        self._close_producers(old_binding)
        try:
            await self._drain_binding(old_binding, cancel_running=False)
        finally:
            await self._stop_worker(old_binding)

    async def restore_components(
        self,
        transaction_id: str,
        old_binding: BackgroundJobRuntimeBinding,
    ) -> None:
        """Reopen a previously stopped binding during publication rollback."""

        self._require_binding(transaction_id, old_binding)
        if old_binding.closed:
            raise RuntimeError("BackgroundJob old binding 已关闭，不能 restore")
        self._ensure_producers(old_binding)
        self._ensure_worker(old_binding)
        old_binding.admission_open = True
        self._active = old_binding

    async def close_components(
        self,
        transaction_id: str,
        binding: BackgroundJobRuntimeBinding,
    ) -> None:
        """Cancel and fully clean one closed or discarded binding."""

        self._require_binding(transaction_id, binding, allow_missing_plan=True)
        if binding.closed:
            return
        binding.admission_open = False
        self._close_producers(binding)
        admission_errors = list(await self._wait_pending_admissions(binding))
        await self.cancel_queued(binding)
        await self.cancel_running(binding)
        try:
            await self._drain_binding(binding, cancel_running=False)
        finally:
            binding.closed = True
            await self._stop_worker(binding)
            self._bindings.pop(binding.snapshot_id, None)
            if self._active is binding:
                self._active = None
        if admission_errors:
            raise admission_errors[0]

    async def open_components(
        self,
        transaction_id: str,
        binding: BackgroundJobRuntimeBinding,
    ) -> None:
        """Recover durable work after the shared publication pointer commits."""

        self._require_binding(transaction_id, binding, allow_missing_plan=True)
        if binding.closed:
            raise RuntimeError("BackgroundJob binding 已关闭")
        if binding.snapshot_id not in self._bindings:
            raise RuntimeError("BackgroundJob binding 不属于当前 adapter")
        self._ensure_producers(binding)
        self._ensure_worker(binding)
        if not binding.recovery_scanned:
            recovered = await self._recover_pending(binding)
            try:
                for request in recovered:
                    if (
                        request.invocation_id in binding.queued
                        or request.invocation_id in binding.running
                    ):
                        await self._release_request_lease(request)
                        continue
                    binding.queued[request.invocation_id] = request
                    await binding.queue.put(request)
            except BaseException:
                for request in recovered:
                    await self._release_request_lease(request)
                raise
            binding.recovery_scanned = True
    def finalize_components(
        self,
        transaction_id: str,
        binding: BackgroundJobRuntimeBinding,
    ) -> None:
        """Synchronously open a finalized child at the shared pointer commit boundary."""

        self._require_binding(transaction_id, binding)
        if binding.closed:
            raise RuntimeError("BackgroundJob binding 已关闭")
        if binding.snapshot_id not in self._bindings:
            raise RuntimeError("BackgroundJob binding 不属于当前 adapter")
        if binding.jobs and (binding.worker_task is None or binding.worker_task.done()):
            raise RuntimeError("BackgroundJob worker 尚未 materialize")
        binding.admission_open = True
        self._active = binding
        self._plans.pop(transaction_id, None)

    def discard_plan(self, transaction_id: str, plan: BackgroundJobPlan) -> None:
        """Discard a plan when a later Activity child rejects the transaction."""

        if self._plans.get(transaction_id) is plan:
            self._plans.pop(transaction_id, None)

    def pause_components(self, binding: BackgroundJobRuntimeBinding) -> None:
        """Synchronously reject new producer admissions after committed cleanup failed."""

        if self._bindings.get(binding.snapshot_id) is not binding:
            raise RuntimeError("BackgroundJob binding 不属于当前 adapter")
        binding.admission_open = False

    async def pause(self, binding: BackgroundJobRuntimeBinding) -> None:
        """Close only new admission while retaining exact accepted requests."""

        if binding.closed:
            return
        binding.admission_open = False
        self._close_producers(binding)

    async def drain(self, binding: BackgroundJobRuntimeBinding) -> None:
        """Wait until queued and running accepted requests release all leases."""

        await self._drain_binding(binding, cancel_running=False)

    async def cancel_queued(self, binding: BackgroundJobRuntimeBinding) -> None:
        """Durably cancel queued requests and release their exact snapshot leases."""

        requests = tuple(binding.queued.values())
        for request in requests:
            if request.cancelled:
                continue
            request.cancelled = True
            record = self._require_ledger().get(request.invocation_id)
            if record is not None and record.state is JobOutcomeState.QUEUED:
                self._transition_outcome(
                    request.invocation_id,
                    JobOutcomeState.CANCELLED,
                    error=None,
                )
            await self._release_request_lease(request)

    async def cancel_running(self, binding: BackgroundJobRuntimeBinding) -> None:
        """Cancel handlers/children and wait for their provider and lease cleanup."""

        for request in binding.running_requests.values():
            request.cancelled = True
        tasks = tuple(binding.running.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def enqueue(
        self,
        binding: BackgroundJobRuntimeBinding,
        job_key: str,
        *,
        reason: str,
        interval_bucket: str | None = None,
        accepted_snapshot_lease: RuntimeSnapshotLease | None = None,
    ) -> asyncio.Task[None]:
        """Schedule one exact binding admission without consulting a mutable current catalog."""

        if binding.closed:
            raise RuntimeError("BackgroundJob binding 已关闭")
        task = asyncio.create_task(
            self._admit(
                binding,
                job_key,
                reason=reason,
                interval_bucket=interval_bucket,
                accepted_snapshot_lease=accepted_snapshot_lease,
            ),
            name=f"background_job_admission:{job_key}",
        )
        binding.pending_admission.add(task)
        task.add_done_callback(
            lambda completed: self._on_admission_done(binding, completed)
        )
        return task

    async def enqueue_interval(
        self,
        binding: BackgroundJobRuntimeBinding,
        job_key: str,
        *,
        interval_bucket: str | None = None,
    ) -> None:
        """Admit one deterministic interval bucket for tests and Core timer ownership."""

        job = _job_for(binding, job_key)
        interval = next(
            trigger.seconds
            for trigger in job.binding.definition.triggers
            if isinstance(trigger, IntervalTrigger)
        )
        bucket = interval_bucket or _interval_bucket(self._clock(), interval)
        await self.enqueue(
            binding,
            job_key,
            reason="interval",
            interval_bucket=bucket,
        )

    async def aclose(self) -> None:
        """Stop and clean every binding owned by this adapter."""

        for binding in tuple(self._bindings.values()):
            await self.close_components("shutdown", binding)

    async def _admit(
        self,
        runtime: BackgroundJobRuntimeBinding,
        job_key: str,
        *,
        reason: str,
        interval_bucket: str | None,
        accepted_snapshot_lease: RuntimeSnapshotLease | None = None,
    ) -> None:
        # 1. Validate trigger identity before acquiring durable execution resources.
        job = _job_for(runtime, job_key)
        if interval_bucket is not None and not _has_interval_trigger(job.binding):
            raise RuntimeError(f"BackgroundJob 不接受 interval trigger: {job_key}")
        if interval_bucket is None:
            raise ValueError("BackgroundJob admission 必须有 interval_bucket")
        accepted = accepted_snapshot_lease is not None
        if not runtime.admission_open and not accepted:
            return
        store = runtime.snapshot_store
        snapshot_lease = accepted_snapshot_lease
        if snapshot_lease is None:
            snapshot_lease = await self._acquire_exact_lease(store, runtime.snapshot_id)
        request: _JobRequest | None = None
        try:
            if runtime.closed:
                return
            if snapshot_lease.snapshot.snapshot_id != runtime.snapshot_id:
                raise RuntimeError("BackgroundJob admission snapshot identity 不匹配")
            if not accepted and not runtime.admission_open:
                return
            if not job.binding.is_live():
                return
            ledger = self._require_ledger()
            invocation_id = self._invocation_id_factory()
            identity = JobOutcomeIdentity(
                plugin_id=job.binding.plugin_id,
                job_name=job.binding.name,
                invocation_id=invocation_id,
                event_id=None,
                interval_bucket=interval_bucket,
                snapshot_id=runtime.snapshot_id,
                plugin_generation_id=job.binding.generation_id,
                # The model lease is deliberately selected at execution start.
                # The ledger schema is immutable, so this marker is not used as
                # an execution fence; _execute_request records the actual lease
                # in BackgroundJobContext and holds it through all retries.
                model_generation_id="execution-pending",
                artifact_identity=job.artifact_identity,
                source_revision=job.source_revision,
                handler_export=job.binding.handler_export,
                lifecycle_revision=_JOB_LIFECYCLE_REVISION,
                api_revision=_JOB_API_REVISION,
                event_payload=None,
            )
            outcome = ledger.admit(identity, now=self._clock())
            if outcome.invocation_id != invocation_id:
                # Dedupe retains the first generation's exact handler and lease owner.
                return
            if outcome.state is not JobOutcomeState.QUEUED:
                return
            if job.binding.definition.coalesce and (
                any(
                    queued.job.admission_key == job.admission_key
                    for queued in runtime.queued.values()
                )
                or any(
                    running_key == job.admission_key
                    for running_key in runtime.running_job_keys.values()
                )
            ):
                self._transition_outcome(
                    invocation_id,
                    JobOutcomeState.CANCELLED,
                    error=None,
                )
                return
            if self._debounced(job):
                self._transition_outcome(
                    invocation_id,
                    JobOutcomeState.CANCELLED,
                    error=None,
                )
                return
            request = _JobRequest(
                binding=runtime,
                job=job,
                reason=reason,
                interval_bucket=interval_bucket,
                invocation_id=invocation_id,
                snapshot_lease=snapshot_lease,
                outcome=outcome,
            )
            runtime.queued[invocation_id] = request
            await runtime.queue.put(request)
            snapshot_lease = None  # type: ignore[assignment]
        finally:
            if snapshot_lease is not None and snapshot_lease.active:
                await snapshot_lease.release()

    async def _recover_pending(
        self,
        runtime: BackgroundJobRuntimeBinding,
    ) -> tuple[_JobRequest, ...]:
        """Rebuild only exact queued outcomes and report every unsafe pending state."""

        ledger = self._require_ledger()
        jobs = {
            f"{job.binding.plugin_id}:{job.binding.name}": job
            for job in runtime.jobs.values()
        }
        requests: list[_JobRequest] = []
        try:
            # 1. Scan durable pending outcomes without consulting the current catalog.
            for record in ledger.list_pending():
                if record.programmatic_turn_state is not None:
                    detail = _programmatic_turn_reconcile_error(record)
                    _ = self._transition_outcome(
                        record.invocation_id,
                        JobOutcomeState.FAILED,
                        error=detail,
                    )
                    _report_recovery(runtime, record, detail)
                    continue
                if record.state is JobOutcomeState.RUNNING:
                    detail = (
                        "runtime restarted before the handler wrote a terminal outcome; "
                        "external effects are unknown and automatic replay is disabled"
                    )
                    _ = self._transition_outcome(
                        record.invocation_id,
                        JobOutcomeState.FAILED,
                        error=detail,
                    )
                    _report_recovery(runtime, record, detail)
                    continue
                job = jobs.get(record.semantic_job_id)
                if job is None:
                    _report_recovery(runtime, record, "exact job binding unavailable")
                    continue
                mismatches = _recovery_identity_mismatches(runtime, job, record)
                if mismatches:
                    _report_recovery(
                        runtime,
                        record,
                        "exact identity mismatch: " + ", ".join(mismatches),
                    )
                    continue
                if record.state is not JobOutcomeState.QUEUED:
                    detail = (
                        f"{record.state.value}/{record.phase.value} retained; "
                        "automatic handler replay disabled"
                    )
                    _report_recovery(runtime, record, detail)
                    continue
                if record.model_generation_id != "execution-pending":
                    _report_recovery(
                        runtime,
                        record,
                        "queued outcome has a bound model generation",
                    )
                    continue
                try:
                    reason = _recovery_trigger(record, job)
                except (TypeError, ValueError, RuntimeError) as error:
                    _report_recovery(runtime, record, f"payload rejected: {error}")
                    continue
                snapshot_lease = await self._acquire_exact_lease(
                    runtime.snapshot_store,
                    runtime.snapshot_id,
                )
                requests.append(
                    _JobRequest(
                        binding=runtime,
                        job=job,
                        reason=reason,
                        interval_bucket=record.interval_bucket,
                        invocation_id=record.invocation_id,
                        snapshot_lease=snapshot_lease,
                        outcome=record,
                    )
                )
        except BaseException:
            for request in requests:
                await self._release_request_lease(request)
            raise
        return tuple(requests)

    async def _worker_loop(self, runtime: BackgroundJobRuntimeBinding) -> None:
        while True:
            request = await runtime.queue.get()
            try:
                if request is None:
                    return
                runtime.queued.pop(request.invocation_id, None)
                if request.cancelled:
                    await self._release_request_lease(request)
                    continue
                task = asyncio.create_task(
                    self._execute_request(request),
                    name=f"background_job_run:{request.job.key}:{request.invocation_id}",
                )
                runtime.running[request.invocation_id] = task
                runtime.running_requests[request.invocation_id] = request
                runtime.running_job_keys[request.invocation_id] = (
                    request.job.admission_key
                )
                try:
                    result = (await asyncio.gather(task, return_exceptions=True))[0]
                    if isinstance(result, asyncio.CancelledError):
                        if not request.cancelled:
                            raise result
                    elif isinstance(result, BaseException):
                        raise result
                finally:
                    runtime.running.pop(request.invocation_id, None)
                    runtime.running_requests.pop(request.invocation_id, None)
                    runtime.running_job_keys.pop(request.invocation_id, None)
            finally:
                runtime.queue.task_done()

    def _invocation_resources(
        self,
        request: _JobRequest,
    ) -> _InvocationResources:
        """Bind the optional programmatic Turn port to one invocation."""

        job = request.job
        turns = None
        is_candidate = bool(request.snapshot_lease.validation_candidate_plugin_ids)
        if job.binding.definition.programmatic_turns and not is_candidate:
            if (
                self._conversation_runtime is None
                or self._programmatic_session_creator is None
                or self._programmatic_session_reader is None
            ):
                raise RuntimeError(
                    "BackgroundJob programmatic_turns 缺少 ConversationRuntime owner 或 SessionStore creator"
                )
            turns = _ProgrammaticTurnPort(
                self._conversation_runtime,
                request,
                self._programmatic_session_creator,
                self._programmatic_session_reader,
                self._require_ledger(),
            )
        return _InvocationResources(turns)

    async def _finish_invocation_resources(
        self,
        resources: _InvocationResources,
    ) -> None:
        """Close invocation-scoped ports after handler cleanup."""

        await resources.finalize()

    async def _execute_request(self, request: _JobRequest) -> None:
        ledger = self._require_ledger()
        binding = request.binding
        runtime_token = bind_runtime_snapshot(request.snapshot_lease)
        try:
            chat_models = self._chat_models_for(request)
            role = ModelRole(request.job.binding.definition.model_role or "agent")
            # 1. Bind one exact model execution at actual job execution start.
            async with chat_models.execution() as model_execution:
                model = model_execution.chat(role)
                model_generation_id = model.descriptor.binding_id
                while True:
                    self._transition_outcome(
                        request.invocation_id,
                        JobOutcomeState.RUNNING,
                        model_generation_id=model_generation_id,
                    )
                    resources = self._invocation_resources(request)
                    llm = _JobBoundChatModel(
                        model,
                        snapshot_lease=request.snapshot_lease,
                        snapshot_id=binding.snapshot_id,
                        plugin_generation_id=request.job.binding.generation_id,
                        invocation_token=object(),
                    )
                    if resources.turns is not None:
                        resources.turns._bind_invocation_token(llm.invocation_token)
                    ctx = BackgroundJobContext(
                        plugin_id=request.job.binding.plugin_id,
                        reason=request.reason,
                        triggered_at=self._clock(),
                        snapshot_id=binding.snapshot_id,
                        generation_id=request.job.binding.generation_id,
                        plugin_generation_id=request.job.binding.generation_id,
                        model_generation_id=model_generation_id,
                        llm=llm,
                        activation_token=request.job.binding.activation_token,
                        turns=resources.turns,
                    )
                    invocation_token = _CURRENT_INVOCATION_TOKEN.set(
                        llm.invocation_token
                    )
                    try:
                        try:
                            from agent.plugin_composition.diagnostics import (
                                plugin_entrypoint,
                            )

                            with plugin_entrypoint(
                                plugin_id=request.job.binding.plugin_id,
                                generation_id=request.job.binding.generation_id,
                                fiber=request.job.binding.plugin_id,
                                operation="background_job.call",
                                entrypoint=request.job.binding.name,
                            ):
                                result = await request.job.handler(ctx)
                            await ctx.drain_children()
                            await self._finish_invocation_resources(resources)
                            if request.cancelled:
                                self._transition_outcome(
                                    request.invocation_id,
                                    JobOutcomeState.CANCELLED,
                                    error=None,
                                )
                                return
                        except asyncio.CancelledError as cancelled_error:
                            llm.invalidate()
                            await ctx.cancel_children()
                            try:
                                await self._finish_invocation_resources(resources)
                            except BaseException:
                                raise cancelled_error
                            current = ledger.get(request.invocation_id)
                            if (
                                current is not None
                                and not current.terminal
                                and current.programmatic_turn_state is not None
                            ):
                                self._transition_outcome(
                                    request.invocation_id,
                                    JobOutcomeState.FAILED,
                                    error=_programmatic_turn_reconcile_error(current),
                                )
                                raise
                            if current is not None and not current.terminal:
                                self._transition_outcome(
                                    request.invocation_id,
                                    JobOutcomeState.CANCELLED,
                                    error=None,
                                )
                            raise
                        except BaseException as error:
                            llm.invalidate()
                            await ctx.cancel_children()
                            await self._finish_invocation_resources(resources)
                            current = ledger.get(request.invocation_id)
                            if current is None or current.terminal:
                                raise
                            if current.programmatic_turn_state is not None:
                                self._transition_outcome(
                                    request.invocation_id,
                                    JobOutcomeState.FAILED,
                                    error=_programmatic_turn_reconcile_error(current),
                                )
                                return
                            retry_policy = request.job.binding.definition.retry_policy
                            if current.attempt < retry_policy.max_attempts:
                                phase = (
                                    JobOutcomePhase.PROVIDER
                                    if llm.provider_called
                                    else JobOutcomePhase.HANDLER
                                )
                                self._transition_outcome(
                                    request.invocation_id,
                                    JobOutcomeState.RETRY_PENDING,
                                    phase=phase,
                                    error=_error_text(error),
                                )
                                delay = min(
                                    retry_policy.max_delay_seconds,
                                    retry_policy.base_delay_seconds
                                    * (2 ** max(0, current.attempt - 1)),
                                )
                                if delay:
                                    await asyncio.sleep(delay)
                                continue
                            phase = (
                                JobOutcomePhase.PROVIDER
                                if llm.provider_called
                                else JobOutcomePhase.HANDLER
                            )
                            self._transition_outcome(
                                request.invocation_id,
                                JobOutcomeState.FAILED,
                                phase=phase,
                                error=_error_text(error),
                            )
                            return
                    finally:
                        llm.invalidate()
                        _CURRENT_INVOCATION_TOKEN.reset(invocation_token)
                    current = ledger.get(request.invocation_id)
                    if request.programmatic_turn_uncertain or (
                        current is not None
                        and current.programmatic_turn_state
                        is ProgrammaticTurnState.SUBMITTING
                    ):
                        if current is None:
                            raise RuntimeError("programmatic Turn outcome 丢失")
                        self._transition_outcome(
                            request.invocation_id,
                            JobOutcomeState.FAILED,
                            error=_programmatic_turn_reconcile_error(current),
                        )
                        return
                    digest = hashlib.sha256(repr(result).encode("utf-8")).hexdigest()
                    self._transition_outcome(
                        request.invocation_id,
                        JobOutcomeState.SUCCEEDED,
                        terminal_result_digest=digest,
                    )
                    return
        except asyncio.CancelledError:
            current = ledger.get(request.invocation_id)
            if current is not None and not current.terminal:
                self._transition_outcome(
                    request.invocation_id,
                    JobOutcomeState.CANCELLED,
                    error=None,
                )
            raise
        except BaseException as error:
            current = ledger.get(request.invocation_id)
            if current is not None and not current.terminal:
                self._transition_outcome(
                    request.invocation_id,
                    JobOutcomeState.FAILED,
                    phase=JobOutcomePhase.PROVIDER,
                    error=_error_text(error),
                )
        finally:
            reset_runtime_snapshot(runtime_token)
            await self._release_request_lease(request)

    async def _interval_loop(self, runtime: BackgroundJobRuntimeBinding) -> None:
        seen: set[tuple[str, str]] = set()
        due: dict[tuple[str, int], float] = {}
        while not runtime.closed:
            await asyncio.sleep(self._interval_poll_seconds)
            if not runtime.admission_open:
                continue
            now = self._clock()
            monotonic_now = asyncio.get_running_loop().time()
            for key, job in runtime.jobs.items():
                for index, trigger in enumerate(job.binding.definition.triggers):
                    if not isinstance(trigger, IntervalTrigger):
                        continue
                    schedule_key = (key, index)
                    deadline = due.setdefault(
                        schedule_key,
                        monotonic_now + trigger.seconds,
                    )
                    if monotonic_now < deadline:
                        continue
                    bucket = _interval_bucket(now, trigger.seconds)
                    marker = (key, bucket)
                    if marker not in seen:
                        seen.add(marker)
                        self.enqueue(
                            runtime,
                            key,
                            reason="interval",
                            interval_bucket=bucket,
                        )
                    due[schedule_key] = monotonic_now + trigger.seconds

    def _resolve_handler(
        self,
        instance: object,
        handler_export: str,
    ) -> Callable[[BackgroundJobContext], Awaitable[object]]:
        if not isinstance(instance, ComposablePlugin):
            raise RuntimeError("BackgroundJob owner 不是 ComposablePlugin")
        value: object = instance.module
        for segment in handler_export.replace(":", ".").split("."):
            if not segment:
                raise RuntimeError(f"handler_export 无效: {handler_export}")
            try:
                value = getattr(value, segment)
            except AttributeError as error:
                raise RuntimeError(
                    f"BackgroundJob handler_export 不存在: {instance.name}:{handler_export}"
                ) from error
        if not inspect.iscoroutinefunction(value):
            raise TypeError(f"BackgroundJob handler 必须是 async: {handler_export}")
        signature = inspect.signature(value)
        try:
            signature.bind(cast(object, None))
        except TypeError as error:
            raise TypeError(
                f"BackgroundJob handler 必须精确接受一个 ctx: {handler_export}"
            ) from error
        if len(signature.parameters) != 1:
            raise TypeError(
                f"BackgroundJob handler 必须精确接受一个 ctx: {handler_export}"
            )
        return cast(Callable[[BackgroundJobContext], Awaitable[object]], value)

    def _chat_models_for(self, request: _JobRequest) -> ChatModels:
        """Resolve CHAT_MODELS from the request's already-frozen composition Root."""

        root = request.snapshot_lease.snapshot.composition_root
        if root is None:
            raise RuntimeError("BackgroundJob snapshot 缺少 composition Root")
        return cast(ChatModels, root.context.require(CHAT_MODELS))

    def _require_ledger(self) -> JobOutcomeLedger:
        if self._ledger is None:
            raise RuntimeError("BackgroundJob 需要 Core-owned JobOutcomeLedger")
        return self._ledger

    def _transition_outcome(
        self,
        invocation_id: str,
        state: JobOutcomeState,
        *,
        phase: JobOutcomePhase | None = None,
        model_generation_id: str | None = None,
        error: str | None = None,
        terminal_result_digest: str | None = None,
    ) -> JobOutcomeRecord:
        return self._require_ledger().transition(
            invocation_id,
            state,
            phase=phase,
            model_generation_id=model_generation_id,
            error=error,
            terminal_result_digest=terminal_result_digest,
            now=self._clock(),
        )

    def _require_binding(
        self,
        transaction_id: str,
        binding: BackgroundJobRuntimeBinding,
        *,
        allow_missing_plan: bool = False,
    ) -> None:
        if binding.closed:
            raise RuntimeError("BackgroundJob binding 已关闭")
        plan = self._plans.get(transaction_id)
        if plan is None and not allow_missing_plan and transaction_id != "shutdown":
            raise RuntimeError("BackgroundJob transaction 不存在")
        if self._bindings.get(binding.snapshot_id) is not binding:
            raise RuntimeError("BackgroundJob binding 不属于当前 adapter")

    def _close_producers(self, binding: BackgroundJobRuntimeBinding) -> None:
        task = binding.interval_task
        if task is not None and not task.done():
            task.cancel()

    def _ensure_producers(self, binding: BackgroundJobRuntimeBinding) -> None:
        if binding.interval_task is None or binding.interval_task.done():
            if any(
                isinstance(trigger, IntervalTrigger)
                for job in binding.jobs.values()
                for trigger in job.binding.definition.triggers
            ):
                binding.interval_task = asyncio.create_task(
                    self._interval_loop(binding),
                    name=f"background_job_intervals:{binding.snapshot_id}",
                )

    async def _drain_binding(
        self,
        binding: BackgroundJobRuntimeBinding,
        *,
        cancel_running: bool,
    ) -> None:
        interval_task = binding.interval_task
        if interval_task is not None:
            await asyncio.gather(interval_task, return_exceptions=True)
            binding.interval_task = None
        admission_errors = list(await self._wait_pending_admissions(binding))
        if cancel_running:
            await self.cancel_running(binding)
        await binding.queue.join()
        running = tuple(binding.running.values())
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        if admission_errors:
            raise admission_errors[0]

    def _on_admission_done(
        self,
        binding: BackgroundJobRuntimeBinding,
        task: asyncio.Task[None],
    ) -> None:
        binding.pending_admission.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and not any(
            existing is error for existing in binding.admission_errors
        ):
            binding.admission_errors.append(error)

    async def _wait_pending_admissions(
        self,
        binding: BackgroundJobRuntimeBinding,
    ) -> tuple[BaseException, ...]:
        """Await every producer admission and return errors for the owner to raise."""

        # 1. Wait for already-created admissions; closing producers prevents new ones.
        pending = tuple(binding.pending_admission)
        results: tuple[object, ...] = ()
        if pending:
            results = tuple(await asyncio.gather(*pending, return_exceptions=True))
            binding.pending_admission.difference_update(pending)
            await asyncio.sleep(0)

        # 2. Reconcile callback-captured errors so observer failures are never silent.
        errors = list(binding.admission_errors)
        binding.admission_errors.clear()
        for result in results:
            if not isinstance(result, BaseException) or isinstance(
                result, asyncio.CancelledError
            ):
                continue
            if not any(existing is result for existing in errors):
                errors.append(result)
        return tuple(errors)

    async def _acquire_exact_lease(
        self,
        store: object,
        snapshot_id: str,
    ) -> RuntimeSnapshotLease:
        lease = getattr(store, "lease", None)
        if callable(lease):
            return cast(RuntimeSnapshotLease, lease(snapshot_id))
        acquire = getattr(store, "acquire", None)
        if not callable(acquire):
            raise RuntimeError("RuntimeSnapshotStore 缺少 exact lease acquisition")
        pending = cast(Awaitable[RuntimeSnapshotLease], acquire(snapshot_id))
        return await pending

    def _debounced(self, job: _MaterializedJob) -> bool:
        seconds = job.binding.definition.debounce_seconds
        if seconds <= 0:
            return False
        ledger = self._require_ledger()
        records = tuple(
            record
            for record in ledger.list_all()
            if record.semantic_job_id == f"{job.binding.plugin_id}:{job.binding.name}"
            and record.state is JobOutcomeState.SUCCEEDED
        )
        if not records:
            return False
        latest = max(records, key=lambda record: record.updated_at)
        try:
            last = datetime.fromisoformat(latest.updated_at)
        except ValueError as error:
            raise RuntimeError("JobOutcomeLedger updated_at 无效") from error
        if last.tzinfo is None:
            raise RuntimeError("JobOutcomeLedger updated_at 缺少时区")
        now = self._clock().astimezone(timezone.utc)
        return (now - last.astimezone(timezone.utc)).total_seconds() < seconds

    async def _stop_worker(self, binding: BackgroundJobRuntimeBinding) -> None:
        task = binding.worker_task
        if task is None or task.done():
            binding.worker_task = None
            return
        await binding.queue.put(None)
        await asyncio.gather(task, return_exceptions=True)
        binding.worker_task = None

    def _ensure_worker(self, binding: BackgroundJobRuntimeBinding) -> None:
        if binding.closed:
            raise RuntimeError("BackgroundJob binding 已关闭")
        if binding.worker_task is None or binding.worker_task.done():
            binding.worker_task = asyncio.create_task(
                self._worker_loop(binding),
                name=f"background_job_worker:{binding.snapshot_id}",
            )

    async def _release_request_lease(self, request: _JobRequest) -> None:
        if request.lease_released:
            return
        request.lease_released = True
        if request.snapshot_lease.active:
            await request.snapshot_lease.release()


GenerationJobHost = BackgroundJobActivityAdapter


def _background_catalog(value: object) -> BackgroundJobCatalog | None:
    catalog = getattr(value, "background_jobs", value)
    if catalog is None:
        return None
    if not isinstance(catalog, BackgroundJobCatalog):
        raise TypeError("target catalog 不是 BackgroundJobCatalog")
    return catalog


def _lease_store(lease: RuntimeSnapshotLease) -> RuntimeSnapshotStore | None:
    store = getattr(lease, "_store", None)
    return store if isinstance(store, RuntimeSnapshotStore) else None


def _job_for(binding: BackgroundJobRuntimeBinding, key: str) -> _MaterializedJob:
    job = binding.jobs.get(key)
    if job is None:
        raise KeyError(f"BackgroundJob 不存在: {key}")
    return job


def _has_interval_trigger(binding: BackgroundJobBinding) -> bool:
    return any(
        isinstance(trigger, IntervalTrigger) for trigger in binding.definition.triggers
    )


def _programmatic_session_metadata(
    request: _JobRequest,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate plugin metadata and append immutable Core provenance."""

    if not isinstance(metadata, Mapping):
        raise TypeError("programmatic session metadata 必须是 JSON object")
    payload = dict(metadata)
    if any(not isinstance(key, str) for key in payload):
        raise TypeError("programmatic session metadata key 必须是字符串")
    reserved = sorted(_PROGRAMMATIC_SESSION_RESERVED_FIELDS.intersection(payload))
    if reserved:
        raise ValueError(
            "programmatic session metadata 不能覆盖 Core 字段: " + ", ".join(reserved)
        )
    _validate_json_value(payload, "metadata")
    payload.update(
        {
            "event_id": None,
            "generation_id": request.job.binding.generation_id,
            "invocation_id": request.invocation_id,
            "job_name": request.job.binding.name,
            "plugin_id": request.job.binding.plugin_id,
            "programmatic": True,
            "snapshot_id": request.binding.snapshot_id,
        }
    )
    return payload


def _programmatic_turn_reconcile_error(record: JobOutcomeRecord) -> str:
    """Describe a durable Turn boundary that forbids automatic handler replay."""

    state = record.programmatic_turn_state
    if state is None:
        raise RuntimeError("programmatic Turn reconcile 缺少 admission state")
    if state is ProgrammaticTurnState.ADMITTED:
        return (
            "programmatic Turn 已提交，禁止自动重跑；manual reconcile turn_id="
            f"{record.programmatic_turn_id}"
        )
    return "programmatic Turn 提交结果不确定，禁止自动重跑；manual reconcile"


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} 必须是有限 JSON number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} 的 object key 必须是字符串")
            _validate_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} 包含不可 JSON 序列化的值")


def _turn_handle_id(handle: object) -> str:
    turn_id = getattr(handle, "id", None)
    if not isinstance(turn_id, str) or not turn_id:
        raise RuntimeError("ConversationRuntime TurnHandle 缺少已持久化 turn id")
    return turn_id


def _recovery_identity_mismatches(
    runtime: BackgroundJobRuntimeBinding,
    job: _MaterializedJob,
    record: JobOutcomeRecord,
) -> tuple[str, ...]:
    expected = {
        "semantic_job_id": f"{job.binding.plugin_id}:{job.binding.name}",
        "snapshot_id": runtime.snapshot_id,
        "plugin_generation_id": job.binding.generation_id,
        "artifact_identity": job.artifact_identity,
        "source_revision": job.source_revision,
        "handler_export": job.binding.handler_export,
        "lifecycle_revision": _JOB_LIFECYCLE_REVISION,
        "api_revision": _JOB_API_REVISION,
    }
    return tuple(
        f"{field} expected={value!r} actual={getattr(record, field)!r}"
        for field, value in expected.items()
        if getattr(record, field) != value
    )


def _report_recovery(
    runtime: BackgroundJobRuntimeBinding,
    record: JobOutcomeRecord,
    detail: str,
) -> None:
    runtime.recovery_reports.append(
        "background job restart recovery degraded "
        f"invocation={record.invocation_id} semantic_job_id={record.semantic_job_id}: "
        f"{detail}"
    )


def _recovery_trigger(
    record: JobOutcomeRecord,
    job: _MaterializedJob,
) -> str:
    if record.event_id is not None or record.event_payload is not None:
        raise ValueError("interval job outcome 不得包含 event identity")
    if record.interval_bucket is not None:
        if not _has_interval_trigger(job.binding):
            raise RuntimeError("durable interval 不属于当前 job trigger")
        return "interval"
    raise ValueError("durable outcome 缺少 interval identity")


def _interval_bucket(value: datetime, seconds: int) -> str:
    timestamp = value.astimezone(timezone.utc).timestamp()
    bucket = int(timestamp // seconds) * seconds
    return datetime.fromtimestamp(bucket, timezone.utc).isoformat()


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


__all__ = [
    "BackgroundJobActivityAdapter",
    "BackgroundJobContext",
    "BackgroundJobPlan",
    "BackgroundJobRuntimeBinding",
    "GenerationJobHost",
    "ProgrammaticTurnPort",
    "ProgrammaticTurnReceipt",
]
