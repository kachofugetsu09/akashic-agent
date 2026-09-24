from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import secrets
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from time import monotonic_ns
from types import MappingProxyType
from uuid import uuid4
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    AsyncIterator,
    Mapping,
    Protocol,
    Sequence,
    cast,
)

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.tasks import register_task_bound_context

from agent.plugin_composition import (
    BoundChatModel,
    BoundEmbeddingModel,
    BoundModelDescriptor,
    CHAT_MODELS,
    ChatModelSelection,
    ConnectionDescriptor,
    Context,
    DriverConnection,
    DriverConnectionDescriptor,
    DriverChatModel,
    DriverEmbeddingModel,
    DriverUnavailableError,
    DiscoveredModel,
    Effect,
    EMBEDDINGS,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    LLMResponse,
    MODEL_CATALOG,
    MODEL_DRIVERS,
    ModelAvailability,
    ModelCatalogSnapshot,
    ModelDescriptor,
    ModelDriverDefinition,
    ModelExecution,
    ModelKind,
    ModelRequest,
    ModelUnavailableError,
    FiberState,
    SavedEmbedding,
    ServiceKey,
)

from .settings import (
    AddConnection,
    AddModel,
    CancelConnectionAuth,
    CreateConnectionWithModel,
    DisableConnection,
    FinishConnectionAuth,
    MODEL_SETTINGS,
    ModelChange,
    SetDefaultModel,
    SettingsReceipt,
    StartConnectionAuth,
    SyncModels,
    UpdateConnection,
)
from agent.plugin_composition.models import ModelContinuation, ModelUsage, ToolCall
from .store import (
    MODEL_ROLES,
    ModelsStore,
    StoredConnection,
    StoredModel,
    StoredSnapshot,
    _request_digest,
)

logger = logging.getLogger(__name__)
_PROCESS_INSTANCE = secrets.token_hex(8)
_LOCAL_ROOT = secrets.token_hex(8)
# 本进程存活的 attempt 登记先于 started 记录返回，Task 退出时注销；
# 同 key 的 started 记录只有不属于任何活 attempt 才算孤儿。
_LIVE_CALLS: set[str] = set()
_RUN_ADMISSION = threading.Lock()
_AUTH_ATTEMPT_TTL_SECONDS = 15 * 60
_DEFAULT_ROLE = "default"
_AGENT_ROLE = "agent"
_VISION_ROLE = "vision"


class _CapabilityCatalog(Protocol):
    async def enrich(
        self,
        discovered: tuple[DiscoveredModel, ...],
        *,
        provider_id: str,
    ) -> tuple[DiscoveredModel, ...]: ...


def _decode_response(payload: object) -> LLMResponse:
    """从调用账重建可重放响应；损坏的持久正文在边界明确失败。"""
    if not isinstance(payload, Mapping):
        raise ValueError("Model 响应记录损坏")
    data = cast(Mapping[str, Any], payload)
    continuation = data.get("continuation")
    if continuation is not None and not isinstance(continuation, Mapping):
        raise ValueError("Model continuation 记录损坏")
    calls = data.get("tool_calls") or ()
    if not isinstance(calls, Sequence) or isinstance(calls, str):
        raise ValueError("Model tool_calls 记录损坏")
    return LLMResponse(
        cast(str | None, data.get("content")),
        tool_calls=[
            ToolCall(
                cast(str, item["id"]), cast(str, item["name"]),
                cast(Mapping[str, Any], item["arguments"]),
            )
            for item in cast(Sequence[Mapping[str, Any]], calls)
        ],
        thinking=cast(str | None, data.get("thinking")),
        finish_reason=cast(str | None, data.get("finish_reason")),
        continuation=(
            None
            if continuation is None
            else ModelContinuation(
                cast(str, continuation["binding_id"]),
                cast(Mapping[str, Any], continuation["payload"]),
            )
        ),
    )


class _BoundChat:
    def __init__(
        self,
        descriptor: BoundModelDescriptor,
        driver: DriverChatModel,
        store: ModelsStore,
        *,
        root_instance: str = _LOCAL_ROOT,
        max_attempts: int = 1,
    ) -> None:
        self._descriptor = descriptor
        self._driver = driver
        self._store = store
        self._root_instance = root_instance
        self._max_attempts = max(1, max_attempts)

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return self._descriptor

    async def complete(self, request: ModelRequest) -> LLMResponse:
        """先登记真实调用，再计时并按实际响应结算。"""
        continuation = request.continuation
        if (
            continuation is not None
            and continuation.binding_id != self._descriptor.binding_id
        ):
            raise ModelUnavailableError("continuation 不属于当前 model binding")
        digest = _request_digest(request)
        if request.request_key is None:
            # 无 key 调用是独立效果身份：单次尝试记账，不共享回执也不占用重试预算。
            return await self._attempts(
                request, f"anonymous:{secrets.token_hex(8)}", digest, budget=1
            )
        request_key = request.request_key
        # 活 run 合并只发生在同一权威账本内；不同 store 的同 key 是独立调用。
        live_runs = cast(
            "dict[tuple[str, str], tuple[asyncio.Future[LLMResponse], str]]",
            self._store.live_runs,
        )
        run_key = (request_key, self._descriptor.binding_id)
        owner = False
        shared: asyncio.Future[LLMResponse]
        with _RUN_ADMISSION:
            entry = live_runs.get(run_key)
            if entry is not None and not entry[0].done():
                if entry[1] != digest:
                    raise ValueError("同一模型请求 key 的请求内容不一致")
                shared = entry[0]
            else:
                replayed = self._scan(request_key, digest)
                if replayed is not None:
                    return replayed
                # attempt owner 内联执行 provider 调用，取消如实送达当前 await；
                # 并发同 key 等待者只分享同一个记账结果，不杀死真实 attempt。
                shared = asyncio.get_running_loop().create_future()
                shared.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception()
                )
                live_runs[run_key] = (shared, digest)
                owner = True
        if not owner:
            return await asyncio.shield(shared)
        try:
            result = await self._attempts(request, request_key, digest)
        except BaseException as error:
            if not shared.done():
                shared.set_exception(error)
            raise
        else:
            if not shared.done():
                shared.set_result(result)
            return result
        finally:
            live_runs.pop(run_key, None)

    def key_recovery(self, request_key: str) -> str:
        """同 key 最近耐久记录的恢复裁决，Models 独占分类、调用方只消费决定：

        - "open"：无终结结算（无记录、成功、在途或仍有耐久退避额度）；
        - "rejected"：provider 明确容量拒绝（ContextLengthError），可证明
          该请求未被处理——本代可续跑有界缩减，真实 resume 也可开新准备；
        - "answered"：其他可证明失败——provider 明确应答或请求可证明
          未发出，真实 resume 后允许新准备如实付费；
        - "uncertain"：取消、孤儿、传输/超时与一切未知名目——远端效果
          不可证，resume 不得据此重付，只有新 Input 作为真正新工作可运行。

        终结（rejected/answered/uncertain）的 key 不因重启/重调获得新预算。
        send_evidence 是发送边界的正面证据：provider 明确 HTTP 拒绝应答
        记 "rejected"，连接未建立/发送前校验失败记 "unsent"；缺失（含
        旧记录）一律按远端效果不确定处理——异常名与"未见 delta"都不能
        充当未处理证明。"""
        records = self._store.calls_for_key(request_key)
        if not records:
            return "open"
        last = records[-1]
        if last["state"] != "error":
            return "open"
        if last.get("next_attempt_at") is not None and len(records) < self._max_attempts:
            return "open"
        evidence = last.get("send_evidence")
        failure = last.get("failure")
        if evidence == "rejected":
            # provider 明确拒绝应答是可证明失败；容量拒绝额外保留有界缩减。
            return "rejected" if failure == "ContextLengthError" else "answered"
        if evidence == "unsent":
            # 连接未建立/发送前校验失败：可证明请求从未到达 provider。
            return "answered"
        return "uncertain"

    def _scan(self, request_key: str, digest: str) -> LLMResponse | None:
        """同 key 账目核对：成功重放；孤儿结算；存活或身份不明的 attempt 阻断。"""
        records = self._store.calls_for_key(request_key)
        orphan_found = False
        for record in records:
            if record["request_digest"] != digest:
                raise ValueError("同一模型请求 key 的请求内容不一致")
            binding = record["binding"]
            if not isinstance(binding, Mapping) or (
                binding.get("binding_id") != self._descriptor.binding_id
            ):
                raise ValueError("同一模型请求 key 的 binding 不一致")
            if record["state"] == "success" and record.get("response") is not None:
                replayed = _decode_response(record["response"])
                replayed.call_record_id = cast(str, record["id"])
                replayed.usage = (
                    None if record.get("usage") is None else ModelUsage(**record["usage"])
                )
                return replayed
            if record["state"] != "started":
                continue
            call_id = cast(str, record["id"])
            if call_id in _LIVE_CALLS:
                raise ModelUnavailableError("同一请求的活 attempt 正在结算，请稍后显式重试")
            if not self._owner_dead(record):
                raise ModelUnavailableError("无法确认先前调用的执行 owner 已死亡，结果不确定")
            try:
                self._store.finish_call(
                    call_id, usage=None,
                    failure="orphaned: 原执行 owner 已退出，真实结果不确定",
                )
            except Exception:
                logger.warning("孤儿 Model 调用结算失败 call_id=%s", call_id, exc_info=True)
            orphan_found = True
        if orphan_found:
            raise ModelUnavailableError(
                "同一请求的先前调用结果不确定；孤儿记录已结算，请显式重试"
            )
        return None

    def _owner_dead(self, record: Mapping[str, Any]) -> bool:
        """owner 身份为 epoch:进程:Root:attempt；只凭真实死亡证据结算。

        更早 host_epoch 的 owner 在本 store 持有独占宿主锁时可证明死亡；
        同纪元内只有本进程签发且 attempt 无活登记者仍属不明，异构或
        其他进程 token 一律不当作死亡证据。
        """
        owner = record.get("owner_id")
        if not isinstance(owner, str):
            return False
        parts = owner.split(":")
        if len(parts) != 4:
            return False
        try:
            record_epoch = int(parts[0])
        except ValueError:
            return False
        host_epoch = self._store.host_epoch
        if host_epoch is not None and record_epoch < host_epoch:
            # 独占宿主锁成立时，旧纪元的写方已经退出，started 永不再结算。
            return self._store.holds_host_lock
        return False

    async def _attempts(
        self, request: ModelRequest, request_key: str, digest: str, *,
        budget: int | None = None,
    ) -> LLMResponse:
        """Models 独占重试预算：一次 complete 内有界自动重试；每个真实 attempt
        先记账再结算，失败写耐久 next_attempt_at，重试前重新核对准入与孤儿。"""
        budget = self._max_attempts if budget is None else max(1, budget)
        while True:
            replayed = self._scan(request_key, digest)
            if replayed is not None:
                return replayed
            records = self._store.calls_for_key(request_key)
            # 预算是耐久事实：连续 complete、关闭重开、进程重启都不刷新；
            # 显式恢复只能以新准备身份（新 key）进入，同 key 重入不重新付费。
            if len(records) >= budget:
                raise ModelUnavailableError("模型调用重试预算耗尽")
            last = records[-1] if records else None
            if (
                last is not None
                and last["state"] == "error"
                and last.get("next_attempt_at") is None
            ):
                # 不可重试/取消的失败是终结裁决：取消只证明本地等待被取消，
                # 不能证明 provider 未接收或未计费；同 key 重调不得再发送，
                # 恢复只能由调用方以新请求身份（新业务边界）显式进入。
                raise ModelUnavailableError(
                    "该请求 key 的最近调用已终结失败，同 key 不得重新付费"
                )
            next_at = None if last is None else last.get("next_attempt_at")
            if isinstance(next_at, (int, float)) and not isinstance(next_at, bool):
                delay = float(next_at) - time.time()
                if delay > 0:
                    # 退避可取消；取消后 attempt 记录保持 started，结果不确定。
                    await asyncio.sleep(delay)
            call_id = self._store.resume_call(
                self._descriptor, request,
                request_key=request_key,
                owner_id=(
                    f"{self._store.host_epoch or 0}:{_PROCESS_INSTANCE}"
                    f":{self._root_instance}:{secrets.token_hex(8)}"
                ),
            )
            _LIVE_CALLS.add(call_id)
            started: int | None = None
            first_token = False

            async def delta(value: dict[str, str]) -> None:
                nonlocal first_token
                assert started is not None
                if not first_token and (
                    value.get("content_delta") or value.get("thinking_delta")
                ):
                    self._store.record_first_token(
                        call_id, (monotonic_ns() - started) / 1_000_000
                    )
                    first_token = True
                if request.on_delta is not None:
                    await request.on_delta(value)

            try:
                try:
                    # driver 恒单次尝试：accounted 调用统一置 key，重试预算只由 Models 持有。
                    driver_request = replace(
                        request, on_delta=None if request.on_delta is None else delta,
                        request_key=request_key,
                    )
                    if request.on_delta is not None:
                        await request.on_delta({"call_record_id": call_id})
                    started = monotonic_ns()
                    response = await self._driver.complete(driver_request)
                except BaseException as failure:
                    # 网络请求可能已经到达 provider；本地异常不证明没有计费。
                    # 重发只允许建立在发送边界的正面证据上：driver 明确置位
                    # send_evidence="rejected"（provider HTTP 拒绝应答）或
                    # "unsent"（连接未建立/发送前校验失败）才可进入自动重试；
                    # HTTP 200 流内失败、读/写错误、超时、取消一律无证据，
                    # 无论是否观察到 delta 都不得重发同一请求。
                    partial = bool(getattr(failure, "response_delta_seen", False))
                    evidence = getattr(failure, "send_evidence", None)
                    retryable = evidence in ("rejected", "unsent") and bool(
                        getattr(failure, "retry_safe", False)
                        or getattr(failure, "retryable", False)
                    )
                    retry_at = None
                    if retryable and len(records) + 1 < budget:
                        # Retry-After 优先于本地退避，且随失败记录耐久保存。
                        hint = getattr(failure, "retry_after", None)
                        retry_at = time.time() + (
                            float(hint)
                            if isinstance(hint, (int, float)) and not isinstance(hint, bool)
                            else min(8.0, 0.5 * (2 ** (len(records) + 1)))
                        )
                    try:
                        self._store.finish_call(
                            call_id, usage=None, failure=type(failure).__name__,
                            duration_ms=None if started is None else (monotonic_ns() - started) / 1_000_000,
                            next_attempt_at=retry_at,
                            partial_response=partial,
                            send_evidence=evidence,
                        )
                    except Exception as record_failure:
                        raise failure from record_failure
                    if retry_at is None:
                        raise
                    continue
                self._store.finish_call(
                    call_id, usage=response.usage, failure=None,
                    duration_ms=(monotonic_ns() - started) / 1_000_000,
                    response=response,
                )
                response.call_record_id = call_id
                return response
            finally:
                _LIVE_CALLS.discard(call_id)

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return self._driver.estimate_context_tokens(messages, tools)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return self._driver.estimate_appended_message_tokens(messages)

    @property
    def max_tool_schemas(self) -> int | None:
        return self._driver.max_tool_schemas


class _BoundEmbedding:
    def __init__(
        self,
        descriptor: EmbeddingSpaceDescriptor,
        driver: DriverEmbeddingModel,
    ) -> None:
        self._descriptor = descriptor
        self._driver = driver

    @property
    def descriptor(self) -> EmbeddingSpaceDescriptor:
        return self._descriptor

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        result = await self._driver.embed(texts)
        if any(len(vector) != self._descriptor.dimensions for vector in result.vectors):
            raise ModelUnavailableError("embedding 返回维度与绑定空间不一致")
        return result


@dataclass
class _DriverRegistration:
    """Keep one provider Context, definition, and lifecycle identity together."""

    context: Context
    definition: ModelDriverDefinition
    registration_id: str


@dataclass(frozen=True, slots=True)
class _SelectedDriver:
    connection_id: str
    registration: _DriverRegistration


@dataclass
class _AuthAttempt:
    connection_id: str
    registration: _DriverRegistration
    state: Mapping[str, Any] | None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancelled: bool = False
    expiry_task: asyncio.Task[None] | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _SelectedChat:
    role: str
    model_id: str
    effort: str | None


@dataclass(slots=True)
class _ConnectionHolder:
    driver: DriverConnection | None = None


async def _close_connection(holder: _ConnectionHolder) -> None:
    """Close one driver connection inside its owning Fiber Effect."""

    driver = holder.driver
    if driver is not None and driver.close is not None:
        await driver.close()


class _DriverScope:
    """Hold selected driver scopes and pre-registered connection Effects."""

    def __init__(
        self,
        selected: tuple[_SelectedDriver, ...],
    ) -> None:
        self.selected = {
            item.connection_id: item for item in selected
        }
        self.opened: dict[str, DriverConnection] = {}
        self._holders = {
            connection_id: _ConnectionHolder()
            for connection_id in self.selected
        }
        self._effects: list[Effect] = []
        self._runtime_scopes: list[AsyncContextManager[None]] = []

    async def enter(self) -> None:
        """Acquire every selected owner before any driver I/O can suspend."""

        try:
            contexts: list[Context] = []
            for item in self.selected.values():
                context = item.registration.context
                if not any(existing is context for existing in contexts):
                    contexts.append(context)
            for context in contexts:
                scope = context.runtime_scope()
                await scope.__aenter__()
                self._runtime_scopes.append(scope)
            for connection_id, item in self.selected.items():
                holder = self._holders[connection_id]
                effect = await item.registration.context.effect(
                    lambda holder=holder: lambda: _close_connection(holder),
                    label=f"model-connection:{connection_id}",
                )
                self._effects.append(effect)
        except BaseException as error:
            try:
                await self.close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "model driver scope 准备和清理均失败",
                    [error, cleanup_error],
                ) from None
            raise

    async def close(self) -> None:
        """Close every registered connection and then release driver scopes."""

        errors: list[BaseException] = []
        for effect in reversed(self._effects):
            try:
                await effect.aclose()
            except BaseException as error:
                errors.append(error)
        self._effects.clear()
        for scope in reversed(self._runtime_scopes):
            try:
                await scope.__aexit__(None, None, None)
            except BaseException as error:
                errors.append(error)
        self._runtime_scopes.clear()
        if not errors:
            return
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup("model driver scope 清理失败", errors)

    def holder(self, connection_id: str) -> _ConnectionHolder:
        """Return the pre-registered holder for one selected connection."""

        return self._holders[connection_id]


@asynccontextmanager
async def _driver_scope(
    state: ModelsState,
    connections: Sequence[StoredConnection],
    *,
    selected: tuple[_SelectedDriver, ...] | None = None,
) -> AsyncGenerator[_DriverScope]:
    """Protect selected driver owners and register their connection cleanup first."""

    fixed = (
        state._select_driver_records(connections)
        if selected is None
        else selected
    )
    scope = _DriverScope(fixed)
    await scope.enter()
    try:
        yield scope
    finally:
        await scope.close()


class _Execution:
    def __init__(
        self,
        state: ModelsState,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        model_id: str | None,
        reasoning_effort: str | None,
        chat: Mapping[str, BoundChatModel],
    ) -> None:
        self.owner_task = asyncio.current_task()
        self.state = state
        self.plugin_snapshot_id = plugin_snapshot_id
        self.snapshot = snapshot
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self._chat = MappingProxyType(dict(chat))

    def chat(self, role: str) -> BoundChatModel:
        try:
            return self._chat[role]
        except KeyError as exc:
            raise ModelUnavailableError(f"模型角色不可用: {role}") from exc


_CURRENT_EXECUTION: ContextVar[_Execution | None] = ContextVar(
    "models_current_execution",
    default=None,
)
# 独立 Task 不得继承父任务已绑定的 execution；由 Task 创建点统一清空。
register_task_bound_context(_CURRENT_EXECUTION)


def _check_vision_binding(snapshot: StoredSnapshot) -> None:
    """Reject a corrupt historical vision binding before any driver opens."""

    model_id = snapshot.role_bindings.get(_VISION_ROLE)
    if model_id is None:
        return
    model = snapshot.models.get(model_id)
    if model is None or model.kind is not ModelKind.CHAT or not model.enabled:
        raise ModelUnavailableError(f"vision model 不可用: {model_id}")
    connection = snapshot.connections.get(model.connection_id)
    if connection is None or not connection.enabled:
        raise ModelUnavailableError(
            f"vision model connection 不可用: {model.connection_id}"
        )
    if "image" not in model.capabilities.input_modalities:
        raise ModelUnavailableError("vision role requires an image-capable model")


class _DriversView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    async def register(
        self,
        ctx: Context,
        definition: ModelDriverDefinition,
    ) -> Effect:
        return await self._state.register_driver(ctx, definition)


class _ChatModelsView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    def execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ):
        return self._state.execution(model_id, reasoning_effort)

    def independent_execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ):
        return self._state.independent_execution(model_id, reasoning_effort)


class _EmbeddingsView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    def save_binding(self, bindings: Bindings, *, model_id: str | None = None) -> str:
        return self._state.save_embedding_binding(bindings, model_id)

    def describe(self, *, model_id: str | None = None) -> EmbeddingSpaceDescriptor:
        return self._state.describe_embedding(model_id)

    def bind(self, *, model_id: str | None = None):
        return self._state.embedding_scope(model_id)


class _CatalogView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    def snapshot(self) -> ModelCatalogSnapshot:
        return self._state.catalog_snapshot()

    def validate_chat_selection(
        self,
        selection: ChatModelSelection,
    ) -> ChatModelSelection:
        return self._state.validate_chat_selection(selection)


class _SettingsView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    async def discover(self, connection: AddConnection) -> tuple[DiscoveredModel, ...]:
        return await self._state.discover_models(connection)

    async def apply(self, command: ModelChange) -> SettingsReceipt:
        return await self._state.apply_change(command)


class _MemoryCredential:
    def __init__(
        self,
        connection_id: str,
        auth_identity: str,
        payload: Mapping[str, str],
    ) -> None:
        self.connection_id = connection_id
        self.auth_identity = auth_identity
        self._payload = dict(payload)

    async def read(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._payload))

    async def refresh(self, payload: Mapping[str, str]) -> None:
        self._payload = dict(payload)

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        yield


class ModelsState:
    """Own one model store and the current driver registrations for this Root."""

    def __init__(
        self,
        store: ModelsStore,
        *,
        context: Context,
        capability_catalog: _CapabilityCatalog | None = None,
    ) -> None:
        self.store = store
        self._instance_id = uuid4().hex
        self.context = context
        self.capability_catalog = capability_catalog
        self._registrations: dict[str, _DriverRegistration] = {}
        self.drivers = _DriversView(self)
        self.chat_models = _ChatModelsView(self)
        self.embeddings = _EmbeddingsView(self)
        self.catalog = _CatalogView(self)
        self.settings = _SettingsView(self)
        self._auth_attempts: dict[str, _AuthAttempt] = {}
        # Root 实例身份进入每次真实 attempt 的 owner_id，进程重启后不再继承。
        self._root_instance = secrets.token_hex(8)

    async def register_driver(
        self,
        ctx: Context,
        definition: ModelDriverDefinition,
    ) -> Effect:
        """Register one driver as an Effect of its provider Fiber."""

        if (ctx.root_instance_token is not self.context.root_instance_token
                or ctx.require(MODEL_DRIVERS) is not self.drivers):
            raise RuntimeError("model driver 与 MODEL_DRIVERS 不属于同一个 Root")
        if not definition.driver_id.strip() or not definition.contract_version.strip():
            raise ValueError("model driver identity 不能为空")

        def setup():
            if definition.driver_id in self._registrations:
                raise ValueError(f"model driver 重复注册: {definition.driver_id}")
            record = _DriverRegistration(ctx, definition, uuid4().hex)
            self._registrations[definition.driver_id] = record

            async def cleanup() -> None:
                await self._cleanup_registration(record)

            return cleanup

        return await ctx.effect(setup, label=f"model-driver:{definition.driver_id}")

    def _registration_required(self, driver_id: str) -> _DriverRegistration:
        record = self._registrations.get(driver_id)
        if record is None or record.context.fiber.state is not FiberState.ACTIVE:
            raise DriverUnavailableError(f"model driver 不可用: {driver_id}")
        return record

    def _select_driver_records(
        self,
        connections: Sequence[StoredConnection],
    ) -> tuple[_SelectedDriver, ...]:
        """Freeze current registrations before any selected driver can await."""

        selected: list[_SelectedDriver] = []
        seen: set[str] = set()
        for connection in connections:
            if connection.connection_id in seen:
                continue
            seen.add(connection.connection_id)
            selected.append(
                _SelectedDriver(
                    connection_id=connection.connection_id,
                    registration=self._registration_required(connection.driver_id),
                )
            )
        return tuple(selected)

    def _runtime_namespace(self, selected: Sequence[_SelectedDriver]) -> str:
        """Derive a stable identity from this Models instance and selected registrations."""

        identities = sorted(
            {
                (item.registration.definition.driver_id, item.registration.registration_id)
                for item in selected
            }
        )
        payload = "\0".join(
            (self._instance_id, *(f"{driver}\0{registration}" for driver, registration in identities))
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def _cleanup_registration(self, record: _DriverRegistration) -> None:
        """Retire one registration and retry its auth cleanup under its Effect binder."""

        if self._registrations.get(record.definition.driver_id) is record:
            del self._registrations[record.definition.driver_id]
        attempts = [
            (attempt_id, attempt)
            for attempt_id, attempt in tuple(self._auth_attempts.items())
            if attempt.registration is record
        ]
        for _attempt_id, attempt in attempts:
            attempt.cancelled = True
        current = asyncio.current_task()
        for _attempt_id, attempt in attempts:
            task = attempt.expiry_task
            attempt.expiry_task = None
            if task is not None and task is not current:
                task.cancel()
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    pass
        failures: list[BaseException] = []
        for attempt_id, attempt in attempts:
            async with attempt.lock:
                if self._auth_attempts.get(attempt_id) is not attempt:
                    continue
                if attempt.state is None:
                    self._auth_attempts.pop(attempt_id, None)
                    continue
                cancel = record.definition.cancel_auth
                if cancel is None:
                    self._auth_attempts.pop(attempt_id, None)
                    continue
                try:
                    await cancel(attempt.state)
                except BaseException as error:
                    failures.append(error)
                    continue
                self._auth_attempts.pop(attempt_id, None)
        if failures:
            raise BaseExceptionGroup("model auth attempt 清理失败", failures)

    def catalog_snapshot(self) -> ModelCatalogSnapshot:
        self._check_snapshot_service(MODEL_CATALOG, self.catalog)
        snapshot = self._snapshot_or_empty()
        connections = tuple(
            ConnectionDescriptor(
                connection_id=item.connection_id,
                name=item.name,
                driver_id=item.driver_id,
                auth_identity=item.auth_identity,
                availability=self._availability(item),
            )
            for item in snapshot.connections.values()
        )
        models = tuple(
            self._model_descriptor(snapshot, item) for item in snapshot.models.values()
        )
        return ModelCatalogSnapshot(
            revision=snapshot.revision,
            connections=connections,
            models=models,
            role_bindings=dict(snapshot.role_bindings),
            default_embedding_model_id=snapshot.default_embedding_model_id,
        )

    def validate_chat_selection(
        self,
        selection: ChatModelSelection,
    ) -> ChatModelSelection:
        self._check_snapshot_service(CHAT_MODELS, self.chat_models)
        if selection.reasoning_effort and selection.model_id is None:
            raise ValueError("推理强度必须绑定显式模型")
        if selection.model_id is None:
            return ChatModelSelection()
        snapshot = self._snapshot_required()
        model = snapshot.models.get(selection.model_id)
        if model is None or model.kind is not ModelKind.CHAT or not model.enabled:
            raise ModelUnavailableError(f"聊天模型不可用: {selection.model_id}")
        connection = snapshot.connections[model.connection_id]
        if self._availability(connection) is not ModelAvailability.AVAILABLE:
            raise ModelUnavailableError(f"聊天模型连接不可用: {selection.model_id}")
        efforts = model.capabilities.supported_reasoning_efforts
        if (
            selection.reasoning_effort
            and efforts
            and selection.reasoning_effort not in efforts
        ):
            raise ValueError(f"模型不支持推理强度: {selection.reasoning_effort}")
        return selection

    def _select_chat_models(
        self,
        snapshot: StoredSnapshot,
        explicit_model_id: str | None,
        reasoning_effort: str | None,
    ) -> tuple[_SelectedChat, ...]:
        """Freeze role-to-model choices before any driver connection can await."""

        _check_vision_binding(snapshot)
        selected: list[_SelectedChat] = []
        for role in MODEL_ROLES:
            model_id = snapshot.role_bindings.get(role)
            binding_role = role
            if explicit_model_id is not None and role == _AGENT_ROLE:
                model_id = explicit_model_id
            if model_id is None:
                if role == _DEFAULT_ROLE:
                    raise ModelUnavailableError("尚未配置 default 聊天模型")
                default_id = snapshot.role_bindings.get(_DEFAULT_ROLE)
                if role == _VISION_ROLE:
                    if (
                        default_id is None
                        or "image"
                        not in snapshot.models[default_id].capabilities.input_modalities
                    ):
                        continue
                model_id = default_id
                binding_role = _DEFAULT_ROLE
            if model_id is None:
                continue
            effort = (
                reasoning_effort
                if explicit_model_id and role == _AGENT_ROLE
                else snapshot.role_reasoning_efforts.get(binding_role)
                or snapshot.models[model_id].default_reasoning_effort
            )
            selected.append(_SelectedChat(role, model_id, effort))
        return tuple(selected)

    @asynccontextmanager
    async def execution(
        self,
        model_id: str | None,
        reasoning_effort: str | None,
    ) -> AsyncIterator[ModelExecution]:
        inherited = _CURRENT_EXECUTION.get()
        if inherited is not None and inherited.owner_task is not asyncio.current_task():
            raise RuntimeError("model execution 不能由子 task 继承")
        async with self.context.runtime_scope():
            self._check_snapshot_service(CHAT_MODELS, self.chat_models)
            existing = inherited
            if existing is not None:
                if existing.state is not self:
                    raise RuntimeError("同一执行不能绑定两个 models Service")
                if model_id is None and reasoning_effort is None:
                    yield existing
                    return
                if (
                    existing.model_id != model_id
                    or existing.reasoning_effort != reasoning_effort
                ):
                    raise RuntimeError("嵌套 model execution 选择冲突")
                yield existing
                return
            selection = self.validate_chat_selection(
                ChatModelSelection(model_id, reasoning_effort)
            )
            snapshot = self._snapshot_required()
            selected_models = self._select_chat_models(
                snapshot,
                selection.model_id,
                selection.reasoning_effort,
            )
            connections = tuple(
                snapshot.connections[snapshot.models[item.model_id].connection_id]
                for item in selected_models
            )
            selected_drivers = self._select_driver_records(connections)
            namespace = self._runtime_namespace(selected_drivers)
            async with _driver_scope(
                self, connections, selected=selected_drivers
            ) as opened:
                execution = await self._build_execution(
                    namespace,
                    snapshot,
                    selection.model_id,
                    selection.reasoning_effort,
                    selected_models,
                    opened,
                )
                token = _CURRENT_EXECUTION.set(execution)
                try:
                    yield execution
                finally:
                    _CURRENT_EXECUTION.reset(token)

    @asynccontextmanager
    async def independent_execution(
        self,
        model_id: str | None,
        reasoning_effort: str | None,
    ) -> AsyncIterator[ModelExecution]:
        """Open an execution without a parent task's inherited binding."""

        inherited = _CURRENT_EXECUTION.get()
        if inherited is not None and inherited.owner_task is asyncio.current_task():
            raise RuntimeError("当前 task 已绑定 model execution")
        token = _CURRENT_EXECUTION.set(None)
        try:
            async with self.execution(model_id, reasoning_effort) as execution:
                yield execution
        finally:
            _CURRENT_EXECUTION.reset(token)

    @asynccontextmanager
    async def embedding_scope(
        self,
        model_id: str | None,
    ) -> AsyncIterator[BoundEmbeddingModel]:
        inherited = _CURRENT_EXECUTION.get()
        if inherited is not None and inherited.owner_task is not asyncio.current_task():
            raise RuntimeError("model execution 不能由子 task 继承")
        async with self.context.runtime_scope():
            self._check_snapshot_service(EMBEDDINGS, self.embeddings)
            existing = inherited
            if existing is not None:
                if existing.state is not self:
                    raise RuntimeError("同一执行不能绑定两个 models Service")
                selected = model_id or existing.snapshot.default_embedding_model_id
                if selected is None:
                    raise ModelUnavailableError("尚未配置默认 embedding 模型")
                snapshot = existing.snapshot
            else:
                snapshot = self._snapshot_required()
                selected = model_id or snapshot.default_embedding_model_id
                if selected is None:
                    raise ModelUnavailableError("尚未配置默认 embedding 模型")
            connection = self._embedding_connection(snapshot, selected)
            selected_drivers = self._select_driver_records((connection,))
            plugin_snapshot_id = self._runtime_namespace(selected_drivers)
            async with _driver_scope(
                self, (connection,), selected=selected_drivers
            ) as opened:
                bound = await self._bind_embedding(
                    plugin_snapshot_id,
                    snapshot,
                    selected,
                    opened,
                )
                yield bound

    def chat_contributors(self) -> tuple[Context, ...]:
        """只归档可选聊天模型所需的实际 driver，不夹带独立 embedding 或未配置的 driver。"""
        self._check_snapshot_service(CHAT_MODELS, self.chat_models)
        snapshot = self._snapshot_or_empty()
        drivers = {
            snapshot.connections[model.connection_id].driver_id
            for model in snapshot.models.values()
            if model.kind == ModelKind.CHAT
            and model.enabled
            and snapshot.connections[model.connection_id].enabled
        }
        return tuple(
            record.context
            for driver_id, record in self._registrations.items()
            if driver_id in drivers
            and record.context.fiber.state is FiberState.ACTIVE
        )

    def save_embedding_binding(self, bindings: Bindings, model_id: str | None) -> str:
        """由实际注册表选择 driver owner，调用者不能自己拼归档闭包。"""
        descriptor = self.describe_embedding(model_id)
        saved = SavedEmbedding(model_id=descriptor.model_id, space_identity=descriptor.identity,
                               dimensions=descriptor.dimensions)
        registration = self._registration_required(descriptor.driver_id)
        return bindings.bind(EMBEDDINGS, saved.model_dump(),
            contributors=(registration.context,))

    def describe_embedding(self, model_id: str | None) -> EmbeddingSpaceDescriptor:
        """描述已配置空间，不读取凭据或执行外部 I/O。"""

        self._check_snapshot_service(EMBEDDINGS, self.embeddings)
        snapshot = self._snapshot_required()
        selected = model_id or snapshot.default_embedding_model_id
        if selected is None:
            raise ModelUnavailableError("尚未配置默认 embedding 模型")
        model = snapshot.models.get(selected)
        if model is None or model.kind is not ModelKind.EMBEDDING or not model.enabled:
            raise ModelUnavailableError(f"embedding 模型不可用: {selected}")
        connection = snapshot.connections[model.connection_id]
        if not connection.enabled:
            raise ModelUnavailableError(f"模型连接已禁用: {connection.connection_id}")
        registration = self._registration_required(connection.driver_id)
        namespace = self._runtime_namespace(
            (_SelectedDriver(connection.connection_id, registration),)
        )
        return _embedding_descriptor(
            namespace,
            snapshot,
            connection,
            model,
            registration.definition,
        )

    def _embedding_connection(
        self,
        snapshot: StoredSnapshot,
        model_id: str,
    ) -> StoredConnection:
        """Resolve the one embedding connection before entering its owner scope."""

        model = snapshot.models.get(model_id)
        if model is None or model.kind is not ModelKind.EMBEDDING or not model.enabled:
            raise ModelUnavailableError(f"embedding 模型不可用: {model_id}")
        dimensions = model.capabilities.embedding_dimensions
        if dimensions is None or dimensions <= 0:
            raise ModelUnavailableError(f"embedding 模型缺少 dimensions: {model_id}")
        connection = snapshot.connections[model.connection_id]
        if not connection.enabled:
            raise ModelUnavailableError(f"模型连接已禁用: {connection.connection_id}")
        return connection

    async def _build_execution(
        self,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        explicit_model_id: str | None,
        reasoning_effort: str | None,
        selected: tuple[_SelectedChat, ...],
        opened: _DriverScope,
    ) -> _Execution:
        chat: dict[str, BoundChatModel] = {}
        for item in selected:
            chat[item.role] = await self._bind_chat(
                plugin_snapshot_id,
                snapshot,
                item.model_id,
                item.role,
                item.effort,
                opened,
            )
        return _Execution(
            self,
            plugin_snapshot_id,
            snapshot,
            explicit_model_id,
            reasoning_effort,
            chat,
        )

    async def _bind_chat(
        self,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        model_id: str,
        role: str,
        effort: str | None,
        opened: _DriverScope,
    ) -> BoundChatModel:
        model = snapshot.models.get(model_id)
        if model is None or model.kind is not ModelKind.CHAT or not model.enabled:
            raise ModelUnavailableError(f"聊天模型不可用: {model_id}")
        connection = snapshot.connections[model.connection_id]
        definition, driver = await self._open_driver(connection, opened)
        capability_digest = _capability_digest(model)
        descriptor = BoundModelDescriptor(
            binding_id=_binding_id(
                plugin_snapshot_id,
                snapshot.revision,
                definition.contract_version,
                connection,
                model,
                effort,
                capability_digest,
            ),
            plugin_snapshot_id=plugin_snapshot_id,
            model_revision=snapshot.revision,
            model_id=model.model_id,
            connection_id=connection.connection_id,
            driver_id=connection.driver_id,
            driver_contract_version=definition.contract_version,
            auth_identity=connection.auth_identity,
            model=model.model,
            role=role,
            reasoning_effort=effort,
            capabilities=model.capabilities,
            capability_sources=model.capability_sources,
            capability_digest=capability_digest,
        )
        # Models 重试预算集中在连接配置边界解析：显式 max_attempts 优先；
        # 旧 max_retries 迁移为 N+1 次 attempt；非法值 fail-loud，不静默回 1。
        max_attempts = _retry_budget(connection.driver_config)
        return _BoundChat(
            descriptor,
            driver.bind_chat(descriptor, model.driver_config),
            self.store,
            root_instance=self._root_instance,
            max_attempts=max_attempts,
        )

    async def _bind_embedding(
        self,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        model_id: str,
        opened: _DriverScope,
    ) -> BoundEmbeddingModel:
        model = snapshot.models.get(model_id)
        if model is None or model.kind is not ModelKind.EMBEDDING or not model.enabled:
            raise ModelUnavailableError(f"embedding 模型不可用: {model_id}")
        dimensions = model.capabilities.embedding_dimensions
        if dimensions is None or dimensions <= 0:
            raise ModelUnavailableError(f"embedding 模型缺少 dimensions: {model_id}")
        connection = snapshot.connections[model.connection_id]
        definition, driver = await self._open_driver(connection, opened)
        descriptor = _embedding_descriptor(
            plugin_snapshot_id,
            snapshot,
            connection,
            model,
            definition,
        )
        return _BoundEmbedding(
            descriptor,
            driver.bind_embedding(descriptor, model.driver_config),
        )

    async def _open_driver(
        self,
        connection: StoredConnection,
        scope: _DriverScope,
        *,
        credential: Any | None = None,
    ) -> tuple[ModelDriverDefinition, DriverConnection]:
        if not connection.enabled:
            raise ModelUnavailableError(f"模型连接已禁用: {connection.connection_id}")
        selected = scope.selected.get(connection.connection_id)
        if selected is None:
            raise DriverUnavailableError(f"model driver 不可用: {connection.driver_id}")
        if selected.registration.definition.driver_id != connection.driver_id:
            raise DriverUnavailableError(f"model driver 不可用: {connection.driver_id}")
        definition = selected.registration.definition
        driver = scope.opened.get(connection.connection_id)
        if driver is None:
            driver = await definition.open(
                _driver_connection_descriptor(connection),
                credential
                if credential is not None
                else self.store.credential_handle(
                    connection.connection_id, connection.auth_identity
                ),
            )
            scope.holder(connection.connection_id).driver = driver
            scope.opened[connection.connection_id] = driver
        return definition, driver

    async def apply_change(self, command: ModelChange) -> SettingsReceipt:
        """Validate driver I/O under Models ownership before one store CAS."""

        async with self.context.runtime_scope():
            self._check_snapshot_service(MODEL_SETTINGS, self.settings)
            return await self._apply_change(command)

    async def discover_models(
        self,
        connection: AddConnection,
    ) -> tuple[DiscoveredModel, ...]:
        """Discover one unsaved connection without publishing durable state."""

        async with self.context.runtime_scope():
            self._check_snapshot_service(MODEL_SETTINGS, self.settings)
            return await self._discover_new_connection(connection)

    def _check_snapshot_service(
        self,
        key: ServiceKey[object],
        expected: object,
    ) -> None:
        """Reject a service object that is not provided by this Models Context."""

        if self.context.require(key) is not expected:
            raise RuntimeError("models Service 不属于当前 Context")

    async def _apply_change(self, command: ModelChange) -> SettingsReceipt:
        if isinstance(command, AddConnection):
            await self._probe_new_connection(command)
            revision = self.store.add_connection(command)
        elif isinstance(command, CreateConnectionWithModel):
            self._check_initial_model_identity(command)
            await self._check_new_connection_model(command)
            revision = self.store.create_connection_with_model(command)
        elif isinstance(command, UpdateConnection):
            await self._probe_updated_connection(command)
            revision = self.store.update_connection(command)
        elif isinstance(command, DisableConnection):
            revision = self.store.disable_connection(command)
        elif isinstance(command, AddModel):
            await self._check_model(command)
            revision = self.store.add_model(command)
        elif isinstance(command, SetDefaultModel):
            revision = self.store.set_default(command)
        elif isinstance(command, SyncModels):
            revision = await self._sync_models(command)
        elif isinstance(command, StartConnectionAuth):
            return await self._start_auth(command)
        elif isinstance(command, FinishConnectionAuth):
            return await self._finish_auth(command)
        elif isinstance(command, CancelConnectionAuth):
            return await self._cancel_auth(command)
        else:
            raise TypeError(f"不支持的 ModelChange: {type(command).__name__}")
        return SettingsReceipt(revision=revision, status="committed")

    async def _sync_models(self, command: SyncModels) -> int:
        """Discover outside SQLite, then publish one catalog revision with CAS."""

        snapshot = self._snapshot_required()
        connection = snapshot.connections.get(command.connection_id)
        if connection is None or not connection.enabled:
            raise ModelUnavailableError(f"模型连接不可用: {command.connection_id}")
        registration = self._registration_required(connection.driver_id)
        definition = registration.definition
        if definition.discover is None:
            raise ValueError(f"driver 不支持模型发现: {connection.driver_id}")
        async with registration.context.runtime_scope():
            discovered = await definition.discover(
                _driver_connection_descriptor(connection),
                self.store.credential_handle(
                    connection.connection_id,
                    connection.auth_identity,
                ),
            )
        if self.capability_catalog is not None:
            discovered = await self.capability_catalog.enrich(
                discovered,
                provider_id=_capability_provider_id(
                    connection.driver_config,
                    connection.driver_id,
                ),
            )
        return self.store.sync_models(
            command.expected_revision,
            connection.connection_id,
            discovered,
        )

    async def _discover_new_connection(
        self,
        connection: AddConnection,
    ) -> tuple[DiscoveredModel, ...]:
        """Read and enrich a provider catalog using an in-memory credential."""

        registration = self._registration_required(connection.driver_id)
        definition = registration.definition
        if definition.discover is None:
            raise ValueError(f"driver 不支持模型发现: {connection.driver_id}")
        descriptor = DriverConnectionDescriptor(
            connection_id=connection.connection_id,
            name=connection.name,
            driver_id=connection.driver_id,
            endpoint=connection.endpoint,
            auth_identity=connection.auth_identity,
            config=connection.driver_config,
        )
        credential = _MemoryCredential(
            connection.connection_id,
            connection.auth_identity,
            connection.credential,
        )
        async with registration.context.runtime_scope():
            discovered = await definition.discover(descriptor, credential)
        if self.capability_catalog is not None:
            discovered = await self.capability_catalog.enrich(
                discovered,
                provider_id=_capability_provider_id(
                    connection.driver_config,
                    connection.driver_id,
                ),
            )
        return discovered

    async def _probe_new_connection(self, command: AddConnection) -> None:
        connection = StoredConnection(
            connection_id=command.connection_id,
            name=command.name,
            driver_id=command.driver_id,
            endpoint=command.endpoint,
            auth_identity=command.auth_identity,
            driver_config=command.driver_config,
            enabled=True,
        )
        credential = _MemoryCredential(
            command.connection_id, command.auth_identity, command.credential
        )
        selected = self._select_driver_records((connection,))
        async with _driver_scope(self, (connection,), selected=selected) as opened:
            await self._probe_connection(connection, credential, opened)

    async def _probe_updated_connection(self, command: UpdateConnection) -> None:
        snapshot = self._snapshot_required()
        existing = snapshot.connections.get(command.connection_id)
        if existing is None:
            raise ModelUnavailableError(f"模型连接不存在: {command.connection_id}")
        connection = replace(
            existing,
            name=command.name,
            endpoint=command.endpoint or existing.endpoint,
            auth_identity=command.auth_identity,
            driver_config=(
                command.driver_config
                if command.driver_config is not None
                else existing.driver_config
            ),
        )
        credential = (
            _MemoryCredential(
                connection.connection_id, command.auth_identity, command.credential
            )
            if command.credential is not None
            else self.store.credential_handle(
                connection.connection_id, command.auth_identity
            )
        )
        selected = self._select_driver_records((connection,))
        async with _driver_scope(self, (connection,), selected=selected) as opened:
            await self._probe_connection(connection, credential, opened)

    async def _probe_connection(
        self,
        connection: StoredConnection,
        credential: Any,
        scope: _DriverScope,
    ) -> None:
        """Probe or open one connection using the already selected registration."""

        selected = scope.selected.get(connection.connection_id)
        if selected is None:
            raise DriverUnavailableError(f"model driver 不可用: {connection.driver_id}")
        definition = selected.registration.definition
        descriptor = _driver_connection_descriptor(connection)
        if definition.probe is not None:
            await definition.probe(descriptor, credential)
        else:
            await self._open_driver(connection, scope, credential=credential)

    async def _check_model(self, command: AddModel) -> None:
        snapshot = self._snapshot_required()
        connection = snapshot.connections.get(command.connection_id)
        if connection is None:
            raise ModelUnavailableError(f"模型连接不存在: {command.connection_id}")
        selected = self._select_driver_records((connection,))
        async with _driver_scope(self, (connection,), selected=selected) as opened:
            definition, driver = await self._open_driver(connection, opened)
            await self._check_bound_model(
                snapshot,
                connection,
                StoredModel.from_command(command),
                definition,
                driver,
            )

    async def _check_new_connection_model(
        self,
        command: CreateConnectionWithModel,
    ) -> None:
        """Validate the first model against the uncommitted connection draft."""

        connection_change = command.connection
        connection = StoredConnection(
            connection_id=connection_change.connection_id,
            name=connection_change.name,
            driver_id=connection_change.driver_id,
            endpoint=connection_change.endpoint,
            auth_identity=connection_change.auth_identity,
            driver_config=connection_change.driver_config,
            enabled=True,
        )
        credential = _MemoryCredential(
            connection.connection_id,
            connection.auth_identity,
            connection_change.credential,
        )
        selected = self._select_driver_records((connection,))
        async with _driver_scope(self, (connection,), selected=selected) as opened:
            definition = selected[0].registration.definition
            if definition.probe is not None:
                await definition.probe(
                    _driver_connection_descriptor(connection), credential
                )
            definition, driver = await self._open_driver(
                connection, opened, credential=credential
            )
            await self._check_bound_model(
                self._snapshot_or_empty(),
                connection,
                StoredModel.from_command(command.model),
                definition,
                driver,
            )

    async def _check_bound_model(
        self,
        snapshot: StoredSnapshot,
        connection: StoredConnection,
        model: StoredModel,
        definition: ModelDriverDefinition,
        driver: DriverConnection,
    ) -> None:
        """Bind one model and probe embedding output before any durable write."""

        if model.kind is ModelKind.CHAT:
            descriptor = self._temporary_chat_descriptor(
                snapshot, connection, model, definition
            )
            _ = driver.bind_chat(descriptor, model.driver_config)
        else:
            descriptor = self._temporary_embedding_descriptor(
                snapshot, connection, model, definition
            )
            bound = _BoundEmbedding(
                descriptor,
                driver.bind_embedding(descriptor, model.driver_config),
            )
            _ = await bound.embed(("Akashic embedding setup check",))

    @staticmethod
    def _check_initial_model_identity(command: CreateConnectionWithModel) -> None:
        if command.connection.expected_revision != command.model.expected_revision:
            raise ValueError("connection and initial model revisions differ")
        if command.connection.connection_id != command.model.connection_id:
            raise ValueError("initial model belongs to a different connection")

    async def _start_auth(self, command: StartConnectionAuth) -> SettingsReceipt:
        registration = self._registration_required(command.driver_id)
        definition = registration.definition
        if definition.start_auth is None:
            raise ValueError(f"driver 不支持登录: {command.driver_id}")
        attempt_id = secrets.token_urlsafe(18)
        attempt = _AuthAttempt(
            connection_id=command.connection_id,
            registration=registration,
            state=None,
        )
        self._auth_attempts[attempt_id] = attempt
        try:
            async with attempt.lock:
                async with registration.context.runtime_scope():
                    result = await definition.start_auth(dict(command.input))
                    state, challenge = _auth_state_and_challenge(result)
                    attempt.state = state
                    self._require_live_attempt(attempt_id, attempt)
                    attempt.expiry_task = asyncio.create_task(
                        self._expire_auth_attempt(attempt_id, attempt),
                        name=f"model-auth-expiry:{attempt_id}",
                    )
            return SettingsReceipt(
                revision=self._snapshot_or_empty().revision,
                status="pending",
                attempt_id=attempt_id,
                challenge=cast(Mapping[str, Any] | None, challenge),
            )
        except BaseException:
            if (
                self._auth_attempts.get(attempt_id) is attempt
                and attempt.state is None
            ):
                self._auth_attempts.pop(attempt_id, None)
            raise

    async def _finish_auth(self, command: FinishConnectionAuth) -> SettingsReceipt:
        attempt = self._auth_attempts.get(command.attempt_id)
        if attempt is None:
            raise ValueError(f"auth attempt 不存在: {command.attempt_id}")
        async with attempt.lock:
            self._require_live_attempt(command.attempt_id, attempt)
            registration = attempt.registration
            definition = registration.definition
            if definition.finish_auth is None:
                raise ValueError(f"driver 不支持完成登录: {definition.driver_id}")
            if attempt.state is None:
                raise RuntimeError("driver auth 尚未返回可完成的 state")
            selected = (_SelectedDriver(attempt.connection_id, registration),)
            async with _driver_scope(self, (), selected=selected) as opened:
                result = await definition.finish_auth(attempt.state)
                if str(result.get("status") or "") != "complete":
                    next_state, challenge = _auth_state_and_challenge(result)
                    attempt.state = next_state
                    self._require_live_attempt(command.attempt_id, attempt)
                    return SettingsReceipt(
                        revision=self._snapshot_or_empty().revision,
                        status="pending",
                        attempt_id=command.attempt_id,
                        challenge=cast(Mapping[str, Any] | None, challenge),
                    )
                self._require_live_attempt(command.attempt_id, attempt)
                fields = _auth_connection_fields(result)
                current = self.store.read_snapshot()
                existing = (
                    None
                    if current is None
                    else current.connections.get(attempt.connection_id)
                )
                if existing is not None and existing.driver_id != definition.driver_id:
                    raise ValueError("auth driver 与已有 connection 不一致")
                connection = StoredConnection(
                    connection_id=attempt.connection_id,
                    name=fields["name"],
                    driver_id=definition.driver_id,
                    endpoint=fields["endpoint"],
                    auth_identity=fields["auth_identity"],
                    driver_config=fields["driver_config"],
                    enabled=True,
                )
                credential: Any = _MemoryCredential(
                    connection.connection_id,
                    connection.auth_identity,
                    fields["credential"],
                )
                await self._probe_connection(connection, credential, opened)
                self._require_live_attempt(command.attempt_id, attempt)
            self._require_live_attempt(command.attempt_id, attempt)
            if existing is None:
                change: AddConnection | UpdateConnection = AddConnection(
                    expected_revision=command.expected_revision,
                    connection_id=connection.connection_id,
                    name=connection.name,
                    driver_id=connection.driver_id,
                    endpoint=connection.endpoint,
                    auth_identity=connection.auth_identity,
                    credential=fields["credential"],
                    driver_config=connection.driver_config,
                )
                self._require_live_attempt(command.attempt_id, attempt)
                revision = self.store.add_connection(change)
            else:
                change = UpdateConnection(
                    expected_revision=command.expected_revision,
                    connection_id=connection.connection_id,
                    name=connection.name,
                    endpoint=connection.endpoint,
                    auth_identity=connection.auth_identity,
                    credential=fields["credential"],
                    driver_config=connection.driver_config,
                )
                self._require_live_attempt(command.attempt_id, attempt)
                revision = self.store.update_connection(change)
            self._stop_auth_expiry(attempt)
            self._auth_attempts.pop(command.attempt_id, None)
            return SettingsReceipt(revision=revision, status="committed")

    async def _cancel_auth(self, command: CancelConnectionAuth) -> SettingsReceipt:
        attempt = self._auth_attempts.get(command.attempt_id)
        if attempt is None:
            raise ValueError(f"auth attempt 不存在: {command.attempt_id}")
        attempt.cancelled = True
        async with attempt.lock:
            if self._auth_attempts.get(command.attempt_id) is not attempt:
                return SettingsReceipt(
                    revision=self._snapshot_or_empty().revision,
                    status="cancelled",
                    attempt_id=command.attempt_id,
                )
            if attempt.state is None:
                self._stop_auth_expiry(attempt)
                self._auth_attempts.pop(command.attempt_id, None)
            else:
                registration = attempt.registration
                if self._registrations.get(registration.definition.driver_id) is not registration:
                    raise DriverUnavailableError(
                        f"model driver 不可用: {registration.definition.driver_id}"
                    )
                if registration.context.fiber.state is not FiberState.ACTIVE:
                    raise DriverUnavailableError(
                        f"model driver 不可用: {registration.definition.driver_id}"
                    )
                async with registration.context.runtime_scope():
                    cancel = registration.definition.cancel_auth
                    if cancel is not None:
                        await cancel(attempt.state)
                self._stop_auth_expiry(attempt)
                self._auth_attempts.pop(command.attempt_id, None)
        return SettingsReceipt(
            revision=self._snapshot_or_empty().revision,
            status="cancelled",
            attempt_id=command.attempt_id,
        )

    async def _expire_auth_attempt(
        self,
        attempt_id: str,
        attempt: _AuthAttempt,
    ) -> None:
        """Cancel one abandoned provider login after its bounded lifetime."""

        try:
            await asyncio.sleep(_AUTH_ATTEMPT_TTL_SECONDS)
            if self._auth_attempts.get(attempt_id) is not attempt:
                return
            await self._cancel_auth(CancelConnectionAuth(attempt_id))
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("expired model auth attempt cleanup failed: %s", attempt_id)

    @staticmethod
    def _stop_auth_expiry(attempt: _AuthAttempt) -> None:
        task = attempt.expiry_task
        attempt.expiry_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _require_live_attempt(
        self,
        attempt_id: str,
        attempt: _AuthAttempt,
    ) -> None:
        if (
            attempt.cancelled
            or self._auth_attempts.get(attempt_id) is not attempt
            or self._registrations.get(attempt.registration.definition.driver_id)
            is not attempt.registration
            or attempt.registration.context.fiber.state is not FiberState.ACTIVE
        ):
            raise ValueError(f"auth attempt 已取消: {attempt_id}")

    def _snapshot_required(self) -> StoredSnapshot:
        snapshot = self.store.read_snapshot()
        if snapshot is None:
            raise ModelUnavailableError("尚未配置任何模型")
        return snapshot

    def _snapshot_or_empty(self) -> StoredSnapshot:
        return self.store.read_snapshot() or StoredSnapshot.empty()

    def _availability(self, connection: StoredConnection) -> ModelAvailability:
        if not connection.enabled:
            return ModelAvailability.DISABLED
        record = self._registrations.get(connection.driver_id)
        if record is None or record.context.fiber.state is not FiberState.ACTIVE:
            return ModelAvailability.DRIVER_UNAVAILABLE
        return ModelAvailability.AVAILABLE

    def _model_descriptor(
        self, snapshot: StoredSnapshot, model: StoredModel
    ) -> ModelDescriptor:
        connection = snapshot.connections[model.connection_id]
        availability = self._availability(connection)
        if not model.enabled:
            availability = ModelAvailability.DISABLED
        return ModelDescriptor(
            model_id=model.model_id,
            connection_id=model.connection_id,
            kind=model.kind,
            model=model.model,
            default_reasoning_effort=model.default_reasoning_effort,
            capabilities=model.capabilities,
            capability_sources=model.capability_sources,
            availability=availability,
        )

    def _temporary_chat_descriptor(
        self,
        snapshot: StoredSnapshot,
        connection: StoredConnection,
        model: StoredModel,
        definition: ModelDriverDefinition,
    ) -> BoundModelDescriptor:
        digest = _capability_digest(model)
        return BoundModelDescriptor(
            binding_id="settings-probe",
            plugin_snapshot_id="settings-probe",
            model_revision=snapshot.revision,
            model_id=model.model_id,
            connection_id=connection.connection_id,
            driver_id=connection.driver_id,
            driver_contract_version=definition.contract_version,
            auth_identity=connection.auth_identity,
            model=model.model,
            role=_DEFAULT_ROLE,
            reasoning_effort=None,
            capabilities=model.capabilities,
            capability_sources=model.capability_sources,
            capability_digest=digest,
        )

    def _temporary_embedding_descriptor(
        self,
        snapshot: StoredSnapshot,
        connection: StoredConnection,
        model: StoredModel,
        definition: ModelDriverDefinition,
    ) -> EmbeddingSpaceDescriptor:
        return _embedding_descriptor(
            "settings-probe",
            snapshot,
            connection,
            model,
            definition,
        )


def _driver_connection_descriptor(
    connection: StoredConnection,
) -> DriverConnectionDescriptor:
    return DriverConnectionDescriptor(
        connection_id=connection.connection_id,
        name=connection.name,
        driver_id=connection.driver_id,
        endpoint=connection.endpoint,
        auth_identity=connection.auth_identity,
        config=connection.driver_config,
    )


def _retry_budget(config: Mapping[str, Any]) -> int:
    """连接配置中的 Models 重试预算：max_attempts 显式优先，旧 max_retries
    迁移为 N+1 次 attempt（N 次重试 = 首次 + N 次重试）；非法值直接报错。"""
    configured = config.get("max_attempts")
    if configured is not None:
        if not isinstance(configured, int) or isinstance(configured, bool) or configured < 1:
            raise ValueError("max_attempts must be a positive integer")
        return configured
    legacy = config.get("max_retries")
    if legacy is None:
        return 1
    if not isinstance(legacy, int) or isinstance(legacy, bool) or legacy < 0:
        raise ValueError("max_retries must be a non-negative integer")
    return legacy + 1


def _capability_provider_id(
    config: Mapping[str, Any],
    driver_id: str,
) -> str:
    """Use the optional catalog identity without accepting malformed config."""

    value = config.get("catalog_provider_id")
    if value is None or value == "":
        return driver_id
    if not isinstance(value, str):
        raise ValueError("driver_config.catalog_provider_id 必须是字符串")
    if value != value.strip():
        raise ValueError("driver_config.catalog_provider_id 不能包含首尾空白")
    return value


def _embedding_descriptor(
    plugin_snapshot_id: str,
    snapshot: StoredSnapshot,
    connection: StoredConnection,
    model: StoredModel,
    definition: ModelDriverDefinition,
) -> EmbeddingSpaceDescriptor:
    """为已配置向量空间生成唯一公开身份。"""

    dimensions = model.capabilities.embedding_dimensions
    if dimensions is None or dimensions <= 0:
        raise ModelUnavailableError(f"embedding 模型缺少 dimensions: {model.model_id}")
    return EmbeddingSpaceDescriptor(
        plugin_snapshot_id=plugin_snapshot_id,
        model_revision=snapshot.revision,
        model_id=model.model_id,
        connection_id=connection.connection_id,
        driver_id=connection.driver_id,
        driver_contract_version=definition.contract_version,
        auth_identity=connection.auth_identity,
        connection_fingerprint=_connection_fingerprint(connection),
        model=model.model,
        dimensions=dimensions,
        normalization=model.capabilities.embedding_normalization or "none",
        capability_digest=_capability_digest(model),
    )


def _capability_digest(model: StoredModel) -> str:
    payload = {
        "capabilities": asdict(model.capabilities),
        "sources": asdict(model.capability_sources),
        "driver_config": _plain_json(model.driver_config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _connection_fingerprint(connection: StoredConnection) -> str:
    payload = {
        "endpoint": connection.endpoint,
        "config": _plain_json(connection.driver_config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _binding_id(
    plugin_snapshot_id: str,
    revision: int,
    contract_version: str,
    connection: StoredConnection,
    model: StoredModel,
    effort: str | None,
    capability_digest: str,
) -> str:
    value = "\0".join(
        (
            plugin_snapshot_id,
            str(revision),
            connection.driver_id,
            contract_version,
            connection.connection_id,
            connection.auth_identity,
            model.model_id,
            effort or "",
            capability_digest,
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _auth_connection_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the provider-neutral fields returned by a completed login."""

    fields: dict[str, Any] = {}
    for name in ("name", "endpoint", "auth_identity"):
        value = result.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"driver auth 缺少 {name}")
        fields[name] = value.strip()
    credential = result.get("credential")
    if not isinstance(credential, Mapping) or not credential:
        raise RuntimeError("driver auth 缺少 credential")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in credential.items()
    ):
        raise RuntimeError("driver auth credential 必须只含 string")
    driver_config = result.get("driver_config", {})
    if not isinstance(driver_config, Mapping):
        raise RuntimeError("driver auth driver_config 必须是 object")
    fields["credential"] = cast(Mapping[str, str], credential)
    fields["driver_config"] = cast(Mapping[str, Any], driver_config)
    return fields


def _auth_state_and_challenge(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    state = result.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("driver auth 必须返回 private state object")
    challenge = result.get("challenge")
    if challenge is not None and not isinstance(challenge, Mapping):
        raise RuntimeError("driver auth challenge 必须是 object")
    return (
        cast(Mapping[str, Any], _freeze_json(state)),
        (
            None
            if challenge is None
            else cast(Mapping[str, Any], _freeze_json(challenge))
        ),
    )


def _freeze_json(value: Any) -> Any:
    active: set[int] = set()

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError("auth state 不允许循环引用")
            active.add(identity)
            try:
                frozen: dict[str, Any] = {}
                for key, nested in item.items():
                    if not isinstance(key, str):
                        raise TypeError("auth state key 必须是 string")
                    frozen[key] = freeze(nested)
                return MappingProxyType(frozen)
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("auth state 不允许循环引用")
            active.add(identity)
            try:
                return tuple(freeze(nested) for nested in item)
            finally:
                active.remove(identity)
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("auth state number 必须是有限值")
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"auth state 不是 JSON value: {type(item).__name__}")

    return freeze(value)
