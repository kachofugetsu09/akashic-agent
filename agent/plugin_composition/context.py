from __future__ import annotations

# pyright: reportPrivateUsage=false
import asyncio
import contextvars
import hashlib
import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncGenerator, TypeVar, cast

from agent.plugin_composition.access import CompositionAudit
from agent.plugin_composition.diagnostics import (
    CorePluginDiagnostics,
    PluginDiagnostics,
    plugin_entrypoint,
)
from agent.plugin_composition.effect import (
    Effect,
    EffectSetup,
    _join_cleanup as _await_critical,
)
from agent.plugin_composition.events import (
    Bail,
    EmitEventKey,
    EventKey,
    EventListener,
    EventRegistry,
    ObserveEventKey,
    ParallelEventKey,
    SerialEventKey,
    TransformEventKey,
)
from agent.plugin_composition.executor import reject_executor_context_access
from agent.plugin_composition.model import (
    CompositionError,
    CompositionReceipt,
    FiberState,
    FiberView,
    HealthView,
    IncidentView,
    PluginRuntime,
    ServiceKey,
    TopologyFiberView,
    TopologyView,
)
from agent.plugin_composition.runtime_lifecycle import (
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    RUNTIME_STOPPING,
    RuntimeStarted,
    RuntimeStarting,
    RuntimeStopping,
)

T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., object])
PluginApply = Callable[["Context"], object]


_lifecycle_binding: contextvars.ContextVar[
    "tuple[Context, asyncio.Task[object]] | None"
] = contextvars.ContextVar("plugin_lifecycle_binding", default=None)


@contextmanager
def _lifecycle_bound(context: Context) -> Iterator[None]:
    """Core-only：把 (Context, 实际执行 Task) 生命周期借用显式绑到当前 Task。

    ContextVar 值会随 create_task 继承，但借用要求 binding 里的 Task
    正是 current_task——原生后台 Task 继承该值也不获权。由实际执行
    回调的 Task 建立并 finally reset，不做隐式授权传播。
    """

    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("生命周期借用需要实际 Task")
    token = _lifecycle_binding.set((context, task))
    try:
        yield
    finally:
        _lifecycle_binding.reset(token)


@dataclass(slots=True)
class _Provider:
    key: ServiceKey[Any]
    value: object
    owner: Fiber
    revision: int
    binding_contributors: Callable[[], tuple[Context, ...]] | None = None
    revoking: bool = False


@dataclass(slots=True)
class _HealthEntry:
    owner: Fiber
    name: str
    required: bool
    reason: str | None = None
    active: bool = True


class Context:
    """Expose composition operations bound to one owning Fiber."""

    def __init__(self, root: CompositionRoot, fiber: Fiber) -> None:
        self._root = root
        self._fiber = fiber
        self._fiber_handle = FiberHandle(fiber)

    def _require_current(self) -> None:
        """对象身份拒绝被替换 activation 的旧 Context；不用 token 比较，
        避免误拒排空期仍需读自己 dependency_store 的已保护工作。"""

        if self._fiber.context is not self:
            raise CompositionError(
                "STALE_ACTIVATION",
                f"{self._fiber.name} 的 Context 属于已被替换的 activation",
            )

    @property
    def fiber(self) -> FiberHandle:
        reject_executor_context_access()
        self._require_current()
        return self._fiber_handle

    @property
    def generation_id(self) -> str:
        reject_executor_context_access()
        return self._root.generation_id

    @property
    def root_instance_token(self) -> object:
        """Return the opaque identity of this exact composition Root."""

        reject_executor_context_access()
        return self._root.instance_token

    def _root_instance_token(self) -> object:
        """Return the Core-only identity of this Context's Root."""

        reject_executor_context_access()
        return self._root.instance_token

    def _declared_dependencies(self) -> tuple[ServiceKey[Any], ...]:
        """供 Core 请求边界冻结声明 Fiber 的能力集合。"""
        reject_executor_context_access()
        return self._fiber.dependencies

    def _plugin_module(self) -> ModuleType | None:
        """Return the exact module mounted on this Fiber when one exists."""

        reject_executor_context_access()
        return self._fiber.plugin_module

    def _reserve_scope(self) -> RuntimeScope | None:
        """在派发时保留准确 owner；生命周期与 Root 由其外层寿命保护。"""
        reject_executor_context_access()
        self._require_current()
        binding = _lifecycle_binding.get()
        if self._fiber._is_root or binding == (self, asyncio.current_task()):
            return None
        owned = self._fiber._call_owned_by_current_task()
        call = (owned._retain() if owned is not None
                else self._fiber._begin_call(self._fiber._activation_token))
        return RuntimeScope(call)

    @contextmanager
    def _call_scope(self) -> Iterator[None]:
        """统一保护同步和异步入口。"""
        scope = self._reserve_scope()
        with scope if scope is not None else nullcontext():
            yield

    @asynccontextmanager
    async def runtime_scope(self) -> AsyncGenerator[None]:
        """为需要显式延长资源寿命的业务边界保留同一次 activation。"""
        with self._call_scope():
            yield

    def entrypoint(self, operation: F) -> F:
        """把同步或异步 callable 绑定到本 provider 的执行入口。"""
        self._require_current()
        if not callable(operation) or inspect.isgeneratorfunction(operation) or inspect.isasyncgenfunction(operation):
            raise TypeError("执行入口必须是同步函数或 async 函数，资源生成器需显式管理寿命")
        if inspect.iscoroutinefunction(operation):
            @wraps(operation)
            async def run_async(*args: Any, **kwargs: Any) -> object:
                with self._call_scope():
                    return await operation(*args, **kwargs)
            return cast(F, run_async)

        @wraps(operation)
        def run_sync(*args: Any, **kwargs: Any) -> object:
            with self._call_scope():
                result = operation(*args, **kwargs)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    elif isinstance(result, asyncio.Future):
                        result.cancel()
                    raise TypeError("同步入口不能返回 awaitable；请用 async 函数声明入口")
                return result
        return cast(F, run_sync)

    @contextmanager
    def borrow(self, key: ServiceKey[T]) -> Iterator[T | None]:
        """临时查询可选服务，借用期间保护实际 provider；缺席返回 None。"""
        reject_executor_context_access()
        self._require_current()
        provider = self._root._active_provider(cast(ServiceKey[Any], key))
        if provider is None:
            yield None
            return
        with provider.owner.context._call_scope():
            yield cast(T, provider.value)

    def require_runtime_owner(self, key: ServiceKey[Any], service: object) -> str:
        """验证当前 scope 的实际服务与 Context，返回 Core 分配的插件 owner。"""
        reject_executor_context_access()
        self._require_current()
        task = asyncio.current_task()
        binding = _lifecycle_binding.get()
        if not (
            self._fiber._call_owned_by_current_task() is not None
            or (
                binding is not None
                and binding[0] is self
                and binding[1] is task
            )
        ):
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                "授权需要当前 Context 的 OwnerCall 或生命周期借用",
            )
        if self.require(key) is not service:
            raise CompositionError(
                "SERVICE_SCOPE_MISMATCH",
                "授权服务不属于当前 Context 的 dependency store",
            )
        return self.runtime.plugin_id

    def require_declared_runtime_owner(self, key: ServiceKey[Any], service: object) -> str:
        """先核对 Context 有效性和声明，再核对实际调用许可。"""
        reject_executor_context_access()
        self._require_current()
        if key not in self._fiber.dependencies:
            raise CompositionError("UNDECLARED_SERVICE", f"当前 Fiber 未声明依赖: {key.name}")
        return self.require_runtime_owner(key, service)

    def capture_runtime_scope(self) -> RuntimeScope:
        """把当前 Task 已接纳的许可延长成一份可移交子 Task 的 scope。"""

        reject_executor_context_access()
        self._require_current()
        owned = self._fiber._call_owned_by_current_task()
        if owned is None:
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                "当前 task 未持有本 Fiber 的调用许可，不能 capture scope",
            )
        return RuntimeScope(owned._retain())

    @property
    def config(self) -> Mapping[str, object]:
        """当前插件的固定配置输入，字段含义由插件解释。"""

        return self.runtime.config

    @property
    def runtime(self) -> PluginRuntime:
        """Return the Core-owned runtime identity for this plugin tree."""

        reject_executor_context_access()
        runtime = self._fiber.runtime
        if runtime is None:
            raise CompositionError(
                "PLUGIN_RUNTIME_UNAVAILABLE",
                f"{self._fiber.name} 没有绑定插件运行环境",
            )
        return runtime

    @property
    def diagnostics(self) -> PluginDiagnostics:
        """Return diagnostics bound to this exact plugin Fiber and generation."""

        reject_executor_context_access()
        runtime = self.runtime
        return CorePluginDiagnostics(
            plugin_id=runtime.plugin_id,
            generation_id=runtime.generation_id,
            fiber=self._fiber.path,
        )

    @property
    def data_root(self) -> Path:
        """返回 Core 为当前插件 generation 分配的数据根。"""

        return self.runtime.data_dir

    def workspace_root(self, name: str) -> Path:
        """返回 Core 为当前 generation 投影的声明式 workspace root。"""

        reject_executor_context_access()
        return self.runtime.workspace_root(name)

    def workspace_file(self, name: str) -> Path:
        """Return one Core-projected product file declared by the plugin."""

        reject_executor_context_access()
        return self.runtime.workspace_file(name)

    async def mount(
        self,
        plugin: PluginApply,
        *,
        name: str | None = None,
        inject: Iterable[ServiceKey[Any]] | None = None,
        required_for_readiness: bool = True,
    ) -> FiberHandle:
        reject_executor_context_access()
        self._require_current()
        if not callable(plugin) or hasattr(plugin, "apply"):
            raise TypeError("Context.mount 只接受 child callable")
        fiber = await self._root._mount(
            parent=self._fiber,
            plugin=plugin,
            name=name,
            inject=inject,
            required_for_readiness=required_for_readiness,
            runtime=self._fiber.runtime,
            plugin_module=self._fiber.plugin_module,
        )
        return FiberHandle(fiber)

    async def inject(
        self,
        dependencies: Iterable[ServiceKey[Any]],
        apply: PluginApply,
        *,
        name: str | None = None,
    ) -> FiberHandle:
        """挂载独立随依赖激活的子 Fiber，不阻塞父 Fiber 就绪。"""

        reject_executor_context_access()
        return await self.mount(
            apply,
            name=name or getattr(apply, "__name__", "inject"),
            inject=dependencies,
            required_for_readiness=False,
        )

    async def provide(self, key: ServiceKey[T], value: T, *,
                      binding_contributors: Callable[[], tuple[Context, ...]] | None = None) -> Effect:
        """服务 owner 可声明归档时实际需要的动态注册 Context，生命周期随同一 Effect。"""
        reject_executor_context_access()
        self._require_current()
        typed_key = cast(ServiceKey[Any], key)
        if self._fiber.state == FiberState.ACTIVE:
            # ACTIVE late provide must reject synchronously before ownership changes.
            self._root._check_provider_registration(typed_key)
            self._root._guard_service_notify(typed_key, self._fiber)

        registration: _Provider | None = None

        def close_guard() -> None:
            if registration is None:
                raise RuntimeError("Service registration 尚未建立 close guard")
            self._root._guard_service_close(registration)

        def setup() -> Callable[[], Awaitable[None]]:
            nonlocal registration
            registration = self._root._register_provider(
                typed_key,
                value,
                self._fiber,
                binding_contributors=binding_contributors,
            )

            async def cleanup() -> None:
                if registration is None:
                    raise RuntimeError("Service registration cleanup 缺少 registration")
                await self._root._remove_provider(
                    registration,
                )

            return cleanup

        effect = await self._fiber.add_effect(
            setup,
            label=f"service:{key.name}",
            close_guard=close_guard,
        )
        if self._fiber.state == FiberState.ACTIVE:
            if registration is None:
                raise RuntimeError("ACTIVE Service registration 缺少 registration")
            notification = self._root._notify_provider_registered(registration)
            try:
                try:
                    notification_task = asyncio.create_task(
                        notification,
                        name=f"plugin-service-notify:{key.name}",
                    )
                except BaseException:
                    notification.close()
                    raise
                await _await_critical(notification_task)
            except BaseException as error:
                try:
                    await effect.aclose()
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        f"Service 发布与撤销失败: {key.name}", [error, cleanup_error],
                    ) from None
                raise
        return effect

    def get(self, key: ServiceKey[T]) -> T | None:
        reject_executor_context_access()
        self._require_current()
        provider = self._fiber.dependency_store.get(cast(ServiceKey[Any], key))
        if provider is None:
            # LOADING 中 owner 只读自己已登记的 provide；跨 owner 仍要求
            # provider owner 已 ACTIVE。
            provider = self._root._providers.get(cast(ServiceKey[Any], key))
            if provider is not None and provider.owner is self._fiber:
                if (
                    provider.revoking
                    and not self._root._provider_owner_accessible(provider, self)
                ):
                    provider = None
            elif self._fiber._is_root:
                provider = self._root._active_provider(cast(ServiceKey[Any], key))
            else:
                raise CompositionError(
                    "UNDECLARED_SERVICE",
                    f"{self._fiber.path} 未声明 Service {key.name}；可选调用请使用 borrow",
                )
        return cast(T | None, None if provider is None else provider.value)

    def require(self, key: ServiceKey[T]) -> T:
        reject_executor_context_access()
        value = self.get(key)
        if value is None:
            raise CompositionError(
                "INACTIVE_SERVICE",
                f"当前 Fiber 无法取得 Service: {key.name}",
            )
        return value

    async def effect(self, setup: EffectSetup, *, label: str = "effect") -> Effect:
        reject_executor_context_access()
        self._require_current()
        return await self._fiber.add_effect(setup, label=label)

    async def health(
        self,
        name: str,
        *,
        required: bool = True,
    ) -> HealthHandle:
        """注册一个由当前 Fiber Effect 持有的健康项。"""

        reject_executor_context_access()
        self._require_current()
        entry = self._root._new_health_entry(
            self._fiber,
            name=name,
            required=required,
        )
        handle = HealthHandle(self._root, entry)

        def setup() -> Callable[[], None]:
            self._root._register_health(entry)
            return lambda: self._root._remove_health(entry)

        _ = await self.effect(setup, label=f"health:{name}")
        return handle

    def report_incident(self, kind: str, message: str) -> IncidentView:
        """记录一条结构化 Incident，但不隐式改变当前 Health。"""

        reject_executor_context_access()
        self._require_current()
        if not kind or kind.strip() != kind:
            raise ValueError("Incident kind 必须是非空且无首尾空白的字符串")
        if not message or message.strip() != message:
            raise ValueError("Incident message 必须是非空且无首尾空白的字符串")
        return self._root._report_incident(
            self._fiber,
            kind=kind,
            message=message,
        )

    async def on(
        self,
        key: (
            EmitEventKey[T]
            | SerialEventKey[T, R]
            | ParallelEventKey[T]
            | TransformEventKey[T]
            | ObserveEventKey[T]
        ),
        listener: Callable[[T], object],
    ) -> Effect:
        """Register one typed listener as an Effect of the current Fiber."""

        reject_executor_context_access()
        raw_key = cast(EventKey, key)
        raw_listener = cast(EventListener, listener)
        return await self.effect(
            lambda: self._root._events.register(
                self._fiber,
                raw_key,
                raw_listener,
            ),
            label=f"event:{type(key).__name__}:{key.name}",
        )

    def emit(self, key: EmitEventKey[T], payload: T) -> None:
        reject_executor_context_access()
        self._require_current()
        self._root._events.emit(key, payload)

    async def serial(
        self,
        key: SerialEventKey[T, R],
        payload: T,
    ) -> Bail[R] | None:
        reject_executor_context_access()
        self._require_current()
        return await self._root._events.serial(key, payload)

    async def parallel(self, key: ParallelEventKey[T], payload: T) -> None:
        reject_executor_context_access()
        self._require_current()
        await self._root._events.parallel(key, payload)

    async def transform(self, key: TransformEventKey[T], payload: T) -> T:
        reject_executor_context_access()
        self._require_current()
        return await self._root._events.transform(key, payload)

    async def observe(self, key: ObserveEventKey[T], payload: T) -> None:
        reject_executor_context_access()
        self._require_current()
        await self._root._events.observe(key, payload)

    async def spawn(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str,
    ) -> asyncio.Task[T]:
        """Start one Fiber-owned task and expose failures to Core readiness."""

        reject_executor_context_access()
        try:
            self._require_current()
        except BaseException:
            # A stale Context rejects the operation before creating an owned
            # Task; close the caller's coroutine so the rejected request has
            # no unobserved CORO_CREATED resource left behind.
            coroutine.close()
            raise
        if not name or name.strip() != name:
            coroutine.close()
            raise ValueError("任务名称必须是非空且无首尾空白的字符串")
        task: asyncio.Task[T] | None = None
        # 就绪闸捕获当次 activation：换代后旧 coroutine 不会等到新
        # activation 的 ready。user_state 是 wrapper 与 cleanup 共享的唯一
        # 结算事实：not_started 由 cleanup 关闭，awaiting 已交给 Task 执行，
        # closed 已由用户 coroutine 的 finally 完成，避免双关/漏关。
        ready = self._fiber._activation_ready
        user_state = "not_started"

        def setup() -> Callable[[], Awaitable[None]]:
            nonlocal task
            runtime = self._fiber.runtime

            async def run_user() -> T:
                nonlocal user_state
                # 这里没有可被其他 Task 插入的 await；一旦进入该函数，
                # 下一条 await 就会把用户 coroutine 交给同一个 Task。
                user_state = "awaiting"
                try:
                    return await coroutine
                finally:
                    user_state = "closed"

            async def run_owned_task() -> T:
                await ready.wait()
                with self._call_scope():
                    if runtime is None:
                        return await run_user()
                    with plugin_entrypoint(
                        plugin_id=runtime.plugin_id,
                        generation_id=runtime.generation_id,
                        fiber=self._fiber.path,
                        operation="task.run",
                        entrypoint=name,
                    ):
                        return await run_user()

            owned_coroutine = run_owned_task()
            try:
                task = asyncio.create_task(
                    owned_coroutine,
                    name=f"plugin-task:{name}",
                )
            except BaseException:
                owned_coroutine.close()
                coroutine.close()
                raise
            self._fiber._tasks.add(task)
            task.add_done_callback(self._fiber._tasks.discard)
            task.add_done_callback(
                lambda completed: self._root._record_task_result(
                    self._fiber,
                    name,
                    cast(asyncio.Task[object], completed),
                )
            )

            async def cleanup() -> None:
                nonlocal user_state
                assert task is not None
                if not task.done():
                    _ = task.cancel()
                _ = await asyncio.gather(task, return_exceptions=True)
                # 原生 Task 首指令前取消不执行 wrapper finally；未开始的
                # 用户 coroutine 由 cleanup 侧关闭。
                if user_state == "not_started":
                    coroutine.close()
                    user_state = "closed"

            return cleanup

        try:
            _ = await self.effect(setup, label=f"task:{name}")
        except BaseException:
            if task is None:
                coroutine.close()
            raise
        assert task is not None
        return task


class FiberHandle:
    """暴露生命周期控制，但不暴露 Core-owned 可变集合。"""

    __slots__ = ("_fiber",)

    def __init__(self, fiber: Fiber) -> None:
        self._fiber = fiber

    @property
    def name(self) -> str:
        reject_executor_context_access()
        return self._fiber.name

    @property
    def state(self) -> FiberState:
        reject_executor_context_access()
        return self._fiber.state

    @property
    def activation_token(self) -> object | None:
        """返回当前 Fiber activation 的不透明身份令牌。"""

        reject_executor_context_access()
        return self._fiber._activation_token

    def acquire_call(self, expected_activation: object) -> OwnerCall:
        """Atomically admit one call bound to the caller's expected activation.

        旧绑定在 owner 换代后必须显式失败，不默认替调用方改绑最新实例。
        """

        reject_executor_context_access()
        return self._fiber._begin_call(expected_activation)

    async def dispose(self) -> None:
        reject_executor_context_access()
        await self._fiber.dispose()


class OwnerCall:
    """One admitted, not-yet-released local call permit.

    只能由 FiberHandle.acquire_call 或 Core-only 的 `_retain` 返回；
    排空期间许可仍保护其 owner 的资源直到 release。它不代表新调用
    资格，也不报告 owner 当前状态；不提供自行构造或跨 Task 转移的
    保障。
    """

    __slots__ = ("_fiber", "_activation", "_admission_closed", "_released")

    def __init__(
        self,
        fiber: Fiber,
        activation: object,
        admission_closed: asyncio.Event,
    ) -> None:
        self._fiber = fiber
        self._activation = activation
        self._admission_closed = admission_closed
        self._released = False

    @property
    def activation(self) -> object:
        """返回接纳本次调用的 activation 身份，不跟随后续换代。"""

        return self._activation

    async def __aenter__(self) -> OwnerCall:
        if self._released:
            raise CompositionError(
                "OWNER_CALL_RELEASED", "调用句柄不能重复进入"
            )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.release()

    def _retain(self) -> OwnerCall:
        """Core-only：从本许可派生同 activation 的独立许可。

        延长已被接纳调用的保护，不是新调用接纳：不判断 ACTIVE，不比
        较 Fiber 当前 token——UNLOADING 期间源许可仍受保护，故仍可
        保留。仅持有源许可的 Task 可调用；源/派生许可各自独立
        release，最后一份释放才允许资源清理。当前不承诺跨 Task
        转移。
        """

        if self._released:
            raise CompositionError(
                "OWNER_CALL_RELEASED", "已释放的调用句柄不能再保留"
            )
        return self._fiber._retain_call(self)

    def release(self) -> None:
        """结束本次调用并释放 owner 的排空等待。"""

        if self._released:
            raise CompositionError(
                "OWNER_CALL_RELEASED", "调用句柄不能重复释放"
            )
        self._released = True
        self._fiber._end_call(self)


_runtime_scope_binding: contextvars.ContextVar["RuntimeScope | None"] = (
    contextvars.ContextVar("plugin_runtime_scope_binding", default=None)
)


def _current_runtime_scope() -> "RuntimeScope | None":
    """Return the scope actually entered by the current task, if any.

    ContextVar 隐式继承到子 Task 不算授权：只有 entered_task 即当前
    Task 且未关闭的绑定才算数。
    """

    scope = _runtime_scope_binding.get()
    if (
        scope is None
        or scope._closed
        or scope._entered_task is not asyncio.current_task()
    ):
        return None
    return scope


class RuntimeScope:
    """一份独占 OwnerCall 的绑定/交接/结算 scope。

    构造消费一份当前 Task 独占、未释放的 call——调用方随后不得再
    释放或再次移交这同一份 call；这是 Core 调用代码遵守的合同，
    不为违约加反向指针/票据/所有权表。资源计数唯一由
    Fiber._in_flight_calls 持有；本类只保存许可、ContextVar reset
    token、进入 Task 与关闭事实，不持 lease/版本选择器/注册表。
    ContextVar 绑定不授权生命周期借用、跨 owner 借用或后台许可。
    """

    __slots__ = ("_call", "_entered_task", "_binding_token", "_closed")

    def __init__(self, call: OwnerCall) -> None:
        # 调用方把这份许可的结算责任移交本 scope；构造时要求当前 Task
        # 仍持有它（可来自 acquire_call 或 _retain）。
        current = asyncio.current_task()
        if call._released:
            raise CompositionError(
                "OWNER_CALL_RELEASED", "已释放的调用句柄不能绑定 scope"
            )
        if current is None or (
            call._fiber._in_flight_calls.get(call) is not current
        ):
            raise CompositionError(
                "OWNER_CALL_CONTEXT", "call scope 需要持有该许可的 Task"
            )
        self._call = call
        self._entered_task: asyncio.Task[object] | None = None
        self._binding_token: contextvars.Token[RuntimeScope | None] | None = None
        self._closed = False

    def capture(self) -> "RuntimeScope":
        """同步 retain 当前 scope 的许可，返回独立未 enter 的 scope。

        保护在返回前即存在，没有 capture→enter 空窗；源 scope 与
        captured scope 各自独立结算，不互相代替。
        """

        if (
            self._closed
            or self._entered_task is None
            or self._entered_task is not asyncio.current_task()
        ):
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                "只有已 enter 且未关闭的 scope 能由进入它的 Task capture",
            )
        return RuntimeScope(self._call._retain())

    async def wait_admission_closed(self) -> None:
        """等待本 activation 停止新接纳；不代表已接纳调用已结束。"""

        await self._call._admission_closed.wait()

    def __enter__(self) -> "RuntimeScope":
        if self._closed or self._entered_task is not None:
            raise RuntimeError("call scope 只能进入一次")
        task = asyncio.current_task()
        if task is None:
            raise CompositionError(
                "OWNER_CALL_CONTEXT", "call scope 需要实际 Task 上下文"
            )
        # 同步完成：许可仍在途且未释放 → Fiber 记账改归属当前 Task →
        # 绑定 ContextVar。不判断 ACTIVE、不比较 Fiber 最新 token——
        # 排空期间已持许可仍有效。
        self._call._fiber._adopt_call(self._call)
        self._binding_token = _runtime_scope_binding.set(self)
        self._entered_task = task
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._close()

    async def __aenter__(self) -> "RuntimeScope":
        return self.__enter__()

    async def __aexit__(self, *exc_info: object) -> None:
        self._close()

    def _close(self) -> None:
        """Synchronously settle this scope without duplicating release logic.

        错误 Task 在任何 closed 标记/释放/reset 之前被拒绝，合法
        Task 之后仍可完成清理。未 enter 的 scope 允许持有对象者关闭，
        只释放其独立许可，无 ContextVar 要 reset。
        """

        if self._closed:
            return
        entered = self._entered_task
        if entered is not None and asyncio.current_task() is not entered:
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                "已 enter 的 call scope 只能由进入它的 Task 关闭",
            )
        self._closed = True
        if entered is not None:
            token = self._binding_token
            if token is not None:
                _runtime_scope_binding.reset(token)
                self._binding_token = None
        self._call.release()

    async def close(self) -> None:
        """幂等；已 enter 的 scope 只能由 entered_task 关闭。"""

        self._close()


class HealthHandle:
    """允许插件显式降级或恢复一个 Effect-owned 健康项。"""

    __slots__ = ("_root", "_entry")

    def __init__(self, root: CompositionRoot, entry: _HealthEntry) -> None:
        self._root = root
        self._entry = entry

    @property
    def healthy(self) -> bool:
        reject_executor_context_access()
        return self._entry.active and self._entry.reason is None

    @property
    def reason(self) -> str | None:
        reject_executor_context_access()
        return self._entry.reason

    def degrade(self, reason: str) -> None:
        reject_executor_context_access()
        self._root._degrade_health(self._entry, reason)

    def recover(self) -> None:
        reject_executor_context_access()
        self._root._recover_health(self._entry)


class Fiber:
    """Activate one plugin against a stable dependency epoch."""

    def __init__(
        self,
        *,
        root: CompositionRoot,
        fiber_id: int,
        name: str,
        apply: PluginApply,
        dependencies: tuple[ServiceKey[Any], ...],
        parent: Fiber | None,
        required_for_readiness: bool,
        runtime: PluginRuntime | None,
        plugin_module: ModuleType | None,
        is_root: bool = False,
    ) -> None:
        self.root = root
        self.fiber_id = fiber_id
        self.name = name
        self.apply = apply
        self.dependencies = dependencies
        self.parent = parent
        self.required_for_readiness = required_for_readiness
        self.runtime = runtime
        self.plugin_module = plugin_module
        self.state = FiberState.ACTIVE if is_root else FiberState.PENDING
        self.context = Context(root, self)
        self.dependency_store: dict[ServiceKey[Any], _Provider] = {}
        self.effects: list[Effect] = []
        self.children: list[Fiber] = []
        self.error: BaseException | None = None
        self._task_failures: dict[str, str] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._epoch: tuple[tuple[str, int], ...] | None = () if is_root else None
        self._activation_token: object | None = object() if is_root else None
        self._admission_closed = asyncio.Event()
        self._transition = asyncio.Lock()
        self._transition_owner: asyncio.Task[object] | None = None
        self._dispose_requested = False
        self._dispose_task: asyncio.Task[None] | None = None
        self._is_root = is_root
        # 本次 activation 已接纳的在途调用；只对受影响 owner 等待。
        # 接纳条件即 state==ACTIVE 且 activation token 匹配，不另设独立事实。
        # 许可对象本身即键——嵌套 scope 可据此取回当前 Task 的真实许可。
        self._in_flight_calls: dict[OwnerCall, asyncio.Task[object]] = {}
        self._calls_idle = asyncio.Event()
        self._calls_idle.set()
        # activation-local 事实：就绪闸与 STOPPING 完成标志随每次 _load 重建。
        self._activation_ready = asyncio.Event()
        if is_root:
            self._activation_ready.set()
        self._stopping_completed = False
        self._lifecycle_started = False

    @property
    def path(self) -> str:
        """诊断与拓扑使用父路径；局部名字只由当前父 Fiber 管理。"""
        segment = self.name.replace("~", "~0").replace("/", "~1")
        if self.parent is None or self.parent._is_root:
            return segment
        return f"{self.parent.path}/{segment}"

    @property
    def missing_services(self) -> tuple[str, ...]:
        return tuple(
            key.name
            for key in self.dependencies
            if self.root._active_provider(key) is None
        )

    async def add_effect(
        self,
        setup: EffectSetup,
        *,
        label: str,
        close_guard: Callable[[], object] | None = None,
    ) -> Effect:
        """Register ownership before setup and expose only live Fiber states."""

        if self.state in {FiberState.UNLOADING, FiberState.DISPOSED}:
            raise CompositionError(
                "INACTIVE_EFFECT",
                f"{self.name} 在 {self.state.value} 状态不能注册 Effect",
            )
        runtime = self.runtime
        context = self.context
        effect = Effect(
            label=label,
            remove_from_owner=self._remove_effect,
            plugin_id="" if runtime is None else runtime.plugin_id,
            generation_id="" if runtime is None else runtime.generation_id,
            fiber=self.name,
            lifecycle_binder=lambda: _lifecycle_bound(context),
            close_guard=close_guard,
        )
        self.effects.append(effect)
        return await effect.start(setup)

    def _begin_call(self, expected_activation: object) -> OwnerCall:
        """Atomically admit one call bound to the expected activation."""

        if self._is_root:
            raise CompositionError(
                "ROOT_CALL_ADMISSION",
                "Root Fiber 不提供调用接纳",
            )
        current = asyncio.current_task()
        if current is None:
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                f"{self.name} 的调用接纳需要实际 Task 上下文",
            )
        if self.state != FiberState.ACTIVE or self._activation_token is None:
            raise CompositionError(
                "OWNER_UNAVAILABLE",
                f"{self.name} 当前 activation 不接纳新调用",
            )
        if expected_activation is not self._activation_token:
            raise CompositionError(
                "STALE_ACTIVATION",
                f"{self.name} 的调用仍绑定旧 activation，不得静默重绑",
            )
        call = OwnerCall(self, self._activation_token, self._admission_closed)
        self._in_flight_calls[call] = current
        self._calls_idle.clear()
        return call

    def _call_owned_by_current_task(self) -> OwnerCall | None:
        """返回当前 Task 实际持有的本 Fiber 在途许可，无则 None。"""

        current = asyncio.current_task()
        if current is None:
            return None
        for call, task in self._in_flight_calls.items():
            if task is current:
                return call
        return None

    def _retain_call(self, source: OwnerCall) -> OwnerCall:
        """Register one extra permit derived from a still-held call.

        只信任调用方仍持有源许可：源许可的在途 entry 必须归属当前
        Task。检查先于任何新增，不做跨 Task 转交或死锁探测。
        """

        current = asyncio.current_task()
        if current is None or (
            self._in_flight_calls.get(source) is not current
        ):
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                f"{self.name} 的调用保留需要持有源许可的 Task 上下文",
            )
        call = OwnerCall(
            self,
            source._activation,
            source._admission_closed,
        )
        self._in_flight_calls[call] = current
        self._calls_idle.clear()
        return call

    def _adopt_call(self, call: OwnerCall) -> None:
        """Core-only：把一份在途许可的执行归属转交当前 Task。

        仅供 RuntimeScope.__aenter__ 对自有许可使用；不是通用任意
        Task 转交 API。许可已释放或已不在在途记账时 fail-loud。
        """

        current = asyncio.current_task()
        if current is None:
            raise CompositionError(
                "OWNER_CALL_CONTEXT", f"{self.name} 的许可接管需要实际 Task 上下文"
            )
        if call._released or self._in_flight_calls.get(call) is None:
            raise CompositionError(
                "OWNER_CALL_RELEASED", "许可已释放，不能接管"
            )
        self._in_flight_calls[call] = current

    def _end_call(self, call: OwnerCall) -> None:
        if self._in_flight_calls.pop(call, None) is None:
            raise CompositionError(
                "OWNER_CALL_RELEASED", f"{self.name} 的调用句柄不能重复释放"
            )
        if not self._in_flight_calls:
            self._calls_idle.set()

    def _reject_self_call_wait(self) -> None:
        """持有本 activation 在途调用的任务不能等待它自己的卸载。"""

        current = asyncio.current_task()
        if current is not None and any(
            task is current for task in self._in_flight_calls.values()
        ):
            raise CompositionError(
                "REENTRANT_CALL_WAIT",
                f"{self.name} 的卸载不能等待该任务自己仍持有的在途调用",
            )

    async def reconcile(self) -> None:
        """Move to the state implied by the newest dependency epoch."""

        self._reject_direct_reentrant_wait("reconcile")
        # 仅当 transition 已被另一操作持有、本任务又持有本 owner 的
        # 在途调用时，等待会构成同 owner 自等待；无锁/无在途的
        # reconcile（包括同 epoch no-op）不受影响。
        if self._transition.locked() and self._in_flight_calls:
            self._reject_self_call_wait()
        async with self._locked_transition():
            await self._reconcile()

    async def _reconcile(self) -> None:
        """在调用方持有转换锁时完成依赖装配。"""
        if self._dispose_requested or self._is_root:
            return
        providers = self.root._dependency_snapshot(self.dependencies)
        target_epoch = self.root._provider_epoch(providers)
        if providers is None:
            if self.state in {FiberState.ACTIVE, FiberState.FAILED, FiberState.UNLOADING}:
                await self._unload(next_state=FiberState.PENDING)
            return
        assert target_epoch is not None
        if self.state == FiberState.ACTIVE and self._epoch == target_epoch:
            return
        if self.state in {FiberState.ACTIVE, FiberState.FAILED, FiberState.UNLOADING}:
            await self._unload(next_state=FiberState.PENDING)
        await self._load(providers, target_epoch)

    async def dispose(self) -> None:
        """Permanently unload this Fiber and join all child/effect cleanup."""

        self._reject_direct_reentrant_wait("dispose")
        if self._in_flight_calls:
            self._reject_self_call_wait()
        if self._dispose_task is None or self._dispose_task.done():
            self._dispose_task = asyncio.create_task(
                self._dispose(),
                name=f"plugin-fiber-dispose:{self.name}",
            )
        await _await_critical(self._dispose_task)

    async def _dispose(self) -> None:
        async with self._locked_transition():
            if self.state == FiberState.DISPOSED:
                return
            self._dispose_requested = True
            await self._unload(next_state=FiberState.DISPOSED)
            self.root._remove_fiber(self)
            if self.parent is not None and self in self.parent.children:
                self.parent.children.remove(self)

    async def _load(
        self,
        providers: dict[ServiceKey[Any], _Provider],
        epoch: tuple[tuple[str, int], ...],
    ) -> None:
        # 1. Freeze the dependency values for this activation.
        self.state = FiberState.LOADING
        self._activation_token = object()
        self._admission_closed = asyncio.Event()
        # 就绪闸与停止事实属本次 activation；旧 coroutine 不会等到
        # 下一次 activation 的 ready。
        self._activation_ready = asyncio.Event()
        self._stopping_completed = False
        self._lifecycle_started = False
        # 每次 activation 换 Context：旧 Context/旧 bound method 不随
        # Fiber 重载获得新能力；Root Context 保持稳定身份。
        if not self._is_root:
            self.context = Context(self.root, self)
        self.dependency_store = providers
        self.error = None
        await asyncio.sleep(0)
        if (
            self._dispose_requested
            or self.root._provider_epoch_if_active(self.dependencies)
            != epoch
        ):
            await self._unload(next_state=FiberState.PENDING)
            return

        # 2. Apply + 本 owner 生命周期 STARTING/STARTED + required health；
        #    任一失败都先清理已获资源（清理成功 FAILED，失败保留句柄）。
        try:
            runtime = self.runtime
            boundary = (
                nullcontext()
                if runtime is None
                else plugin_entrypoint(
                    plugin_id=runtime.plugin_id,
                    generation_id=runtime.generation_id,
                    fiber=self.name,
                    operation="lifecycle.apply",
                )
            )
            with _lifecycle_bound(self.context):
                with boundary:
                    result = self.apply(self.context)
                    if inspect.isawaitable(result):
                        await result
                # apply 已成功；从第一个 STARTING listener 开始，activation
                # 已取得需要由 STOPPING/Effect 释放的资源责任。
                self._lifecycle_started = True
                # 生命周期事件按准确 owner 派发（LOADING 中不经普通过滤）；
                # 不接受 Bail，同一 activation 只发一次。
                starting = await self.root._events.serial_for_owner(
                    self, RUNTIME_STARTING, RuntimeStarting(),
                )
                if starting is not None:
                    raise CompositionError(
                        "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                        f"{RUNTIME_STARTING.name} 不接受 Bail",
                    )
                started = await self.root._events.serial_for_owner(
                    self, RUNTIME_STARTED, RuntimeStarted(),
                )
                if started is not None:
                    raise CompositionError(
                        "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                        f"{RUNTIME_STARTED.name} 不接受 Bail",
                    )
                self.root._check_required_health(self)
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise asyncio.CancelledError
        except asyncio.CancelledError as error:
            cleanup_task = asyncio.create_task(
                self._unload(next_state=FiberState.PENDING),
                name=f"plugin-fiber-cancel-cleanup:{self.name}",
            )
            try:
                await _await_critical(cleanup_task)
            except asyncio.CancelledError:
                raise
            except BaseException as cleanup_error:
                failure = BaseExceptionGroup(
                    f"插件初始化取消且清理失败: {self.name}", [error, cleanup_error]
                )
                self.error = failure
                self.root._record_error(self, cleanup_error)
                raise failure from None
            raise
        except Exception as error:
            self.error = error
            self.root._record_error(self, error)
            try:
                await self._unload(next_state=FiberState.FAILED)
            except BaseException as cleanup_error:
                failure = BaseExceptionGroup(
                    f"插件初始化和清理均失败: {self.name}", [error, cleanup_error]
                )
                self.error = failure
                self.root._record_error(self, cleanup_error)
                raise failure from None
            return
        if (
            self._dispose_requested
            or self.root._provider_epoch_if_active(self.dependencies)
            != epoch
        ):
            await self._unload(next_state=FiberState.PENDING)
            return
        self._epoch = epoch
        self.state = FiberState.ACTIVE
        self._activation_ready.set()
        await self.root._owner_became_active(self)

    async def _unload(self, *, next_state: FiberState) -> None:
        # 1. 持本 activation 在途调用的任务不得驱动本次卸载；在撤销
        #    任何状态之前拒绝，避免失败后残留半卸载状态。
        self._reject_self_call_wait()
        # 2. 撤销本 activation 身份使新调用接纳关闭，再让消费者观察到不可用。
        self._activation_token = None
        self.state = FiberState.UNLOADING
        self._admission_closed.set()
        await self.root._owner_became_inactive(self)
        # 后台循环由 Fiber 停止；先取消并等待，避免排空等待它自己长期持有的许可。
        tasks = tuple(self._tasks)
        for task in tasks:
            _ = task.cancel()
        if tasks:
            async def join_tasks() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)
            await _await_critical(asyncio.create_task(join_tasks()))

        # 3. 依赖方先退出；本 owner 已接纳的实际调用结束后再释放资源。
        #    取消等待不等于资源已退出；超时与拒绝策略由调用方负责。
        if self._in_flight_calls:
            await _await_critical(asyncio.ensure_future(self._calls_idle.wait()))

        # 4. 子作用域失败时保留父资源；无关子分支仍尝试关闭。
        errors: list[BaseException] = []
        for child in reversed(tuple(self.children)):
            try:
                await child.dispose()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup(f"Fiber 子作用域关闭失败: {self.name}", errors)
        # 5. 本 owner 生命周期停止只在实际进入 STARTING 的 activation 上派发；
        #    STOPPING 成功且非 Bail 才记完成——失败在 Effect 释放前传播，
        #    保留 owner/资源/固定依赖，显式 retry 只重试失败的阶段。
        if self._lifecycle_started and not self._stopping_completed:
            with _lifecycle_bound(self.context):
                result = await self.root._events.serial_for_owner(
                    self, RUNTIME_STOPPING, RuntimeStopping()
                )
            if result is not None:
                raise CompositionError(
                    "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED",
                    f"{RUNTIME_STOPPING.name} 不接受 Bail",
                )
            self._stopping_completed = True
        # 6. 后取得的资源仍未关闭时，不提前释放它可能依赖的旧资源。
        for effect in reversed(tuple(self.effects)):
            await effect.aclose()
        self.dependency_store = {}
        self._task_failures.clear()
        self._epoch = None
        self.state = next_state

    @asynccontextmanager
    async def _locked_transition(self) -> AsyncGenerator[None]:
        """Own one lifecycle transition and expose direct self-waits."""

        async with self._transition:
            owner = asyncio.current_task()
            self._transition_owner = cast(asyncio.Task[object] | None, owner)
            try:
                yield
            finally:
                self._transition_owner = None

    def _reject_direct_reentrant_wait(self, operation: str) -> None:
        current = asyncio.current_task()
        if current is not None and current is self._transition_owner:
            raise CompositionError(
                "REENTRANT_LIFECYCLE_WAIT",
                f"{self.name} 不能在自身生命周期过渡中直接等待 {operation}；"
                "请用 asyncio.create_task 调度",
            )

    def _remove_effect(self, effect: Effect) -> None:
        if effect in self.effects:
            self.effects.remove(effect)


class CompositionRoot:
    """拥有依赖图、局部激活与排空，以及整棵资源树的退出。"""

    RECENT_INCIDENT_LIMIT = 128

    def __init__(
        self,
        generation_id: str,
        *,
        audit: CompositionAudit | None = None,
    ) -> None:
        if not generation_id:
            raise ValueError("generation_id 不能为空")
        self.generation_id = generation_id
        self._instance_token = object()
        self._next_fiber_id = 1
        self._next_provider_revision = 1
        self._composition_revision = 0
        self._fibers: dict[int, Fiber] = {}
        self._providers: dict[ServiceKey[Any], _Provider] = {}
        self._health_entries: dict[tuple[int, str], _HealthEntry] = {}
        self._incident_sequence = 0
        self._incident_counts: dict[tuple[int, str], int] = {}
        self._recent_incidents: deque[IncidentView] = deque(maxlen=self.RECENT_INCIDENT_LIMIT)
        self._audit = audit or CompositionAudit()
        self._events = EventRegistry(
            self._bump_composition_revision,
            self._record_listener_failure,
        )
        self._internal_cleanups: list[tuple[str, Callable[[], object]]] = []
        self._runtime_scope_acquirer: Callable[[], Awaitable[Any]] | None = None
        self._dispose_task: asyncio.Task[None] | None = None
        self.root_fiber = Fiber(
            root=self,
            fiber_id=0,
            name="root",
            apply=lambda _: None,
            dependencies=(),
            parent=None,
            required_for_readiness=True,
            runtime=None,
            plugin_module=None,
            is_root=True,
        )
        self.context = self.root_fiber.context

    @property
    def instance_token(self) -> object:
        """标识单个 Root 实例，不参与可持久化拓扑身份。"""

        return self._instance_token

    def _bind_runtime_scope_acquirer(
        self,
        acquire: Callable[[], Awaitable[Any]],
    ) -> None:
        """Bind the Core-owned exact-Root lease source before mounting plugins."""

        if self._runtime_scope_acquirer is not None:
            raise RuntimeError("composition Root runtime scope 已绑定")
        self._runtime_scope_acquirer = acquire

    async def _acquire_runtime_scope(self):
        acquire = self._runtime_scope_acquirer
        if acquire is None:
            raise RuntimeError("composition Root runtime scope 不可用")
        return await acquire()

    async def mount(
        self,
        plugin: PluginApply,
        *,
        name: str | None = None,
        inject: Iterable[ServiceKey[Any]] | None = None,
        runtime: PluginRuntime | None = None,
    ) -> Fiber:
        return await self._mount(
            parent=self.root_fiber,
            plugin=plugin,
            name=name,
            inject=inject,
            required_for_readiness=True,
            runtime=runtime,
            plugin_module=None,
        )

    async def _mount_module(
        self,
        plugin: PluginApply,
        *,
        name: str,
        inject: Iterable[ServiceKey[Any]],
        runtime: PluginRuntime,
        plugin_module: ModuleType,
    ) -> Fiber:
        """Mount one Manager-validated V3 module adapter."""

        return await self._mount(
            parent=self.root_fiber,
            plugin=plugin,
            name=name,
            inject=inject,
            required_for_readiness=True,
            runtime=runtime,
            plugin_module=plugin_module,
        )

    async def dispose(self) -> None:
        if self._dispose_task is None or self._dispose_task.done():
            self._dispose_task = asyncio.create_task(
                self._dispose(),
                name=f"plugin-composition-dispose:{self.generation_id}",
            )
        await _await_critical(self._dispose_task)

    def _defer_internal_cleanup(
        self,
        resource: str,
        cleanup: Callable[[], object],
    ) -> None:
        """登记不进入拓扑身份的 Core-owned cleanup。"""

        self._internal_cleanups.append((resource, cleanup))

    async def _dispose(self) -> None:
        self.root_fiber.state = FiberState.UNLOADING
        errors: list[BaseException] = []
        for child in reversed(tuple(self.root_fiber.children)):
            try:
                await child.dispose()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("Root 子作用域关闭失败", errors)
        for effect in reversed(tuple(self.root_fiber.effects)):
            await effect.aclose()
        while self._internal_cleanups:
            _, cleanup = self._internal_cleanups[-1]
            result = cleanup()
            if inspect.isawaitable(result):
                await result
            self._internal_cleanups.pop()
        self.root_fiber.state = FiberState.DISPOSED

    def receipt(self) -> CompositionReceipt:
        fibers = tuple(self._fiber_view(fiber) for fiber in self._fibers.values())
        incident_counts: dict[str, int] = {}
        for (_fiber_id, owner), count in self._incident_counts.items():
            incident_counts[owner] = incident_counts.get(owner, 0) + count
        external_effects = self._audit.external_effects
        required_pending = tuple(
            view.name
            for view in fibers
            if view.required_for_readiness and view.state != FiberState.ACTIVE
        )
        optional_pending = tuple(
            view.name
            for view in fibers
            if not view.required_for_readiness and view.state != FiberState.ACTIVE
        )
        effects = tuple(
            f"{fiber.path}:{effect.label}"
            for fiber in (self.root_fiber, *self._fibers.values())
            for effect in fiber.effects
        )
        health = self._health_view()
        required_degraded = tuple(
            f"{item.owner}:{item.name}"
            for item in health
            if item.required and not item.healthy
        )
        return CompositionReceipt(
            generation_id=self.generation_id,
            ready=(
                self.root_fiber.state == FiberState.ACTIVE
                and not required_pending
                and not required_degraded
                and not external_effects
            ),
            fibers=fibers,
            services=tuple(sorted(key.name for key in self._providers)),
            effects=effects,
            required_pending=required_pending,
            optional_pending=optional_pending,
            health=health,
            required_degraded=required_degraded,
            incidents=self.recent_incidents(),
            incident_sequence=self._incident_sequence,
            incident_counts=tuple(sorted(incident_counts.items())),
            incident_overflowed=False,
            writes=self._audit.writes,
            external_effects=external_effects,
        )

    def topology_view(
        self,
        *,
        plugin_ids: frozenset[str] | None = None,
    ) -> TopologyView:
        """Freeze the current logical topology as a content-addressed value."""

        # 1. 结构身份排除 Fiber 状态、错误和普通 Effect。
        selected = tuple(
            fiber
            for fiber in self._fibers.values()
            if plugin_ids is None
            or (fiber.runtime is not None and fiber.runtime.plugin_id in plugin_ids)
        )
        fibers = tuple(
            sorted(
                (
                    TopologyFiberView(
                        name=fiber.path,
                        parent=(
                            None
                            if fiber.parent is self.root_fiber
                            else cast(Fiber, fiber.parent).path
                        ),
                        required_for_readiness=fiber.required_for_readiness,
                        dependencies=tuple(
                            sorted(key.name for key in fiber.dependencies)
                        ),
                    )
                    for fiber in selected
                ),
                key=lambda item: item.name,
            )
        )
        effects = tuple(
            sorted(
                f"{fiber.path}:{effect.label}"
                for fiber in (
                    (self.root_fiber, *selected) if plugin_ids is None else selected
                )
                for effect in fiber.effects
            )
        )
        listeners = self._events.registrations(plugin_ids=plugin_ids)
        services = tuple(
            sorted(
                key.name
                for key, provider in self._providers.items()
                if plugin_ids is None
                or provider.owner.runtime is None
                or provider.owner.runtime.plugin_id in plugin_ids
            )
        )

        # 2. 内容 hash 与单调 revision 分别回答“是什么”和“是否变过”。
        identity_payload: dict[str, object] = {
            "fibers": [
                {
                    "name": fiber.name,
                    "parent": fiber.parent,
                    "required": fiber.required_for_readiness,
                    "dependencies": fiber.dependencies,
                }
                for fiber in fibers
            ],
            "services": services,
            "listeners": listeners,
        }
        encoded = json.dumps(
            identity_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        identity = hashlib.sha256(encoded).hexdigest()
        return TopologyView(
            generation_id=self.generation_id,
            identity=identity,
            composition_revision=self._composition_revision,
            fibers=fibers,
            services=services,
            effects=effects,
            listeners=listeners,
        )

    def topology_identity(self) -> str:
        return self.topology_view().identity

    def active_plugin_ids(self) -> frozenset[str]:
        """返回此 Root 实际挂载的顶层插件身份，不解释业务启用状态。"""

        return frozenset(
            runtime.plugin_id
            for fiber in self.root_fiber.children
            if (runtime := fiber.runtime) is not None
        )

    def fibers(self, *, plugin_id: str | None = None, generation_id: str | None = None) -> tuple[Fiber, ...]:
        """查询实际登记的 Fiber，可限定准确插件和 generation。"""
        return tuple(
            fiber for fiber in self._fibers.values()
            if (plugin_id is None or fiber.runtime is not None and fiber.runtime.plugin_id == plugin_id)
            and (generation_id is None or fiber.runtime is not None and fiber.runtime.generation_id == generation_id)
        )

    def consumers(self, owners: Iterable[Fiber], *, declared: bool = False) -> tuple[Fiber, ...]:
        """计算子作用域与依赖闭包；换代前读固定边，换代后读当前声明。"""
        initial = set(owners)
        affected = set(initial)
        changed = True
        while changed:
            changed = False
            for candidate in self._fibers.values():
                if candidate in affected or candidate.state == FiberState.DISPOSED:
                    continue
                providers = (
                    (self._providers.get(key) for key in candidate.dependencies)
                    if declared else candidate.dependency_store.values()
                )
                if candidate.parent in affected or any(
                    provider is not None and provider.owner in affected for provider in providers
                ):
                    affected.add(candidate)
                    changed = True
        return tuple(fiber for fiber in self._fibers.values() if fiber in affected - initial)

    def require_ready(self, fibers: Iterable[Fiber]) -> None:
        """核对实际登记的必需 Fiber 与其 health；可选分支不扩大就绪边界。"""
        for candidate in set(fibers):
            if self._fibers.get(candidate.fiber_id) is not candidate or not candidate.required_for_readiness:
                continue
            if candidate.state != FiberState.ACTIVE:
                raise RuntimeError(f"目标依赖未 ACTIVE: {candidate.path} state={candidate.state}")
            if candidate.error is not None:
                raise RuntimeError(f"目标依赖启动失败: {candidate.path}") from candidate.error
            degraded = tuple(entry.name for entry in self._health_entries.values()
                             if entry.owner is candidate and entry.required and entry.reason is not None)
            if degraded:
                raise RuntimeError(f"目标依赖 required health 失败: {candidate.path}:{','.join(degraded)}")

    def plugin_runtime(self, plugin_id: str) -> PluginRuntime:
        """返回顶层插件 Fiber 使用的 Core-owned runtime。"""

        matches = tuple(
            runtime
            for fiber in self.root_fiber.children
            if (runtime := fiber.runtime) is not None and runtime.plugin_id == plugin_id
        )
        if len(matches) != 1:
            raise CompositionError(
                "PLUGIN_RUNTIME_UNAVAILABLE",
                f"Root 中没有唯一插件 runtime: {plugin_id}",
            )
        return matches[0]

    def service_value(
        self,
        key: ServiceKey[T],
        *,
        plugin_ids: frozenset[str] | None = None,
    ) -> T | None:
        """Read one active provider selected by top-level plugin owner."""

        provider = self._active_provider(cast(ServiceKey[Any], key))
        if provider is None:
            return None
        runtime = provider.owner.runtime
        if (
            plugin_ids is not None
            and runtime is not None
            and runtime.plugin_id not in plugin_ids
        ):
            return None
        return cast(T, provider.value)

    def _service_provider(self, key: ServiceKey[T]) -> tuple[Context, T]:
        """Return the still-registered provider Context and exact value."""

        provider = self._providers.get(cast(ServiceKey[Any], key))
        if provider is None:
            raise RuntimeError(f"当前 runtime scope 不提供服务: {key.name}")
        if provider.revoking and not self._provider_owner_accessible(
            provider,
            provider.owner.context,
        ):
            raise RuntimeError(f"当前 runtime scope 不提供服务: {key.name}")
        return provider.owner.context, cast(T, provider.value)

    @staticmethod
    def _provider_owner_accessible(
        provider: _Provider,
        context: Context,
    ) -> bool:
        """Allow only an exact owner permit or lifecycle borrowing task."""

        current = asyncio.current_task()
        if current is None:
            return False
        if provider.owner._call_owned_by_current_task() is not None:
            return True
        binding = _lifecycle_binding.get()
        return (
            binding is not None
            and binding[0] is context
            and binding[1] is current
        )

    def provided_services(
        self,
        *,
        plugin_ids: frozenset[str] | None = None,
    ) -> Mapping[ServiceKey[Any], object]:
        """Freeze active services selected by top-level plugin owner."""

        return {
            key: provider.value
            for key, provider in self._providers.items()
            if provider.owner.state == FiberState.ACTIVE
            if not provider.revoking
            if plugin_ids is None
            or provider.owner.runtime is None
            or provider.owner.runtime.plugin_id in plugin_ids
        }

    def plugin_service_owners(self) -> Mapping[ServiceKey[Any], str]:
        """Freeze active plugin-owned provider identities for graph checks."""

        return {
            key: runtime.plugin_id
            for key, provider in self._providers.items()
            if provider.owner.state == FiberState.ACTIVE
            if not provider.revoking
            if (runtime := provider.owner.runtime) is not None
        }

    def binding_contributors(self, key: ServiceKey[Any]) -> tuple[Context, ...]:
        """归档依赖来自当前服务 provider，不另存动态注册状态。"""
        provider = self._active_provider(key)
        if provider is None:
            raise RuntimeError(f"归档服务已失效: {key.name}")
        return () if provider.binding_contributors is None else provider.binding_contributors()

    def context_owner(self, context: Context) -> str | None:
        """只识别本 Root 实际存活的插件 Context，不接受重建的身份字段。"""
        for fiber in self._fibers.values():
            if fiber.context is context:
                if fiber.state != FiberState.ACTIVE or fiber.runtime is None:
                    raise RuntimeError("注册 Context 已失效或没有插件 owner")
                return fiber.runtime.plugin_id
        return None

    def plugin_dependencies(self) -> Mapping[str, frozenset[ServiceKey[Any]]]:
        """收集各插件及子 Fiber 声明的服务依赖。"""
        dependencies: dict[str, set[ServiceKey[Any]]] = {}
        for fiber in self._fibers.values():
            if fiber.runtime is not None:
                dependencies.setdefault(fiber.runtime.plugin_id, set()).update(fiber.dependencies)
        return {owner: frozenset(keys) for owner, keys in dependencies.items()}

    async def _mount(
        self,
        *,
        parent: Fiber,
        plugin: PluginApply,
        name: str | None,
        inject: Iterable[ServiceKey[Any]] | None,
        required_for_readiness: bool,
        runtime: PluginRuntime | None,
        plugin_module: ModuleType | None,
    ) -> Fiber:
        """Publish only after parent ownership exists, then reconcile."""

        # 1. Resolve the narrow apply(ctx) contract.
        apply, resolved_name, dependencies = self._resolve_plugin(
            plugin,
            name=name,
            inject=inject,
        )
        if parent.state in {FiberState.UNLOADING, FiberState.DISPOSED}:
            raise CompositionError(
                "INACTIVE_PLUGIN_OWNER",
                f"{parent.name} 不能挂载子插件",
            )
        if any(fiber.name == resolved_name for fiber in parent.children):
            raise CompositionError(
                "DUPLICATE_PLUGIN",
                f"同一父 Fiber 下不能重复挂载插件: {parent.name}/{resolved_name}",
            )

        # 2. Parent ownership is visible before publication observers run.
        fiber = Fiber(
            root=self,
            fiber_id=self._next_fiber_id,
            name=resolved_name,
            apply=apply,
            dependencies=dependencies,
            parent=parent,
            required_for_readiness=required_for_readiness,
            runtime=runtime,
            plugin_module=plugin_module,
        )
        self._next_fiber_id += 1
        parent.children.append(fiber)
        self._fibers[fiber.fiber_id] = fiber
        self._bump_composition_revision()
        try:
            async with fiber._locked_transition():
                if parent.state not in {FiberState.UNLOADING, FiberState.DISPOSED}:
                    await fiber._reconcile()
        except BaseException as error:
            await self._rollback_mount(fiber, error)
            raise
        if parent.state in {FiberState.UNLOADING, FiberState.DISPOSED}:
            try:
                await fiber.dispose()
            except BaseException as error:
                fiber.error = error
                self._record_error(fiber, error)
                raise
            return fiber
        return fiber

    async def _rollback_mount(self, fiber: Fiber, error: BaseException) -> None:
        """保留挂载错误；清理失败的 Fiber 继续拥有资源和名称。"""

        fiber.error = error
        self._record_error(fiber, error)
        if fiber.state == FiberState.UNLOADING:
            # 初始化已经尝试清理；不在异常回退中隐式重试失败资源。
            return
        try:
            await fiber.dispose()
        except BaseException as cleanup_error:
            failure = BaseExceptionGroup(
                f"插件挂载和清理均失败: {fiber.name}", [error, cleanup_error]
            )
            fiber.error = failure
            self._record_error(fiber, cleanup_error)
            raise failure from None

    def _resolve_plugin(
        self,
        plugin: PluginApply,
        *,
        name: str | None,
        inject: Iterable[ServiceKey[Any]] | None,
    ) -> tuple[PluginApply, str, tuple[ServiceKey[Any], ...]]:
        if not callable(plugin) or hasattr(plugin, "apply"):
            raise TypeError("插件必须是 callable")
        apply = plugin
        resolved_name = name or str(getattr(plugin, "name", "")).strip()
        resolved_name = resolved_name or getattr(apply, "__name__", "plugin")
        raw_dependencies = inject
        if raw_dependencies is None:
            raw_dependencies = ()
        dependencies = tuple(cast(Iterable[ServiceKey[Any]], raw_dependencies))
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"插件依赖重复: {resolved_name}")
        return apply, resolved_name, dependencies

    def _register_provider(
        self,
        key: ServiceKey[Any],
        value: object,
        owner: Fiber,
        *, binding_contributors: Callable[[], tuple[Context, ...]] | None = None,
    ) -> _Provider:
        self._check_provider_registration(key)
        provider = _Provider(
            key=key,
            value=value,
            owner=owner,
            revision=self._next_provider_revision,
            binding_contributors=binding_contributors,
        )
        self._providers[key] = provider
        self._next_provider_revision += 1
        self._bump_composition_revision()
        return provider

    def _check_provider_registration(self, key: ServiceKey[Any]) -> None:
        """Check frozen and duplicate errors before creating registration state."""

        existing = self._providers.get(key)
        if existing is not None:
            raise CompositionError(
                "DUPLICATE_SERVICE",
                f"Service {key.name} 已由 {existing.owner.name} 提供",
            )

    async def _remove_provider(
        self,
        registration: _Provider,
    ) -> None:
        current = self._providers.get(registration.key)
        if current is not registration:
            return
        if not registration.revoking:
            registration.revoking = True
            self._bump_composition_revision()

        # Keep the exact record in the table until every consumer has released it.
        await self._reconcile_dependents(
            (registration.key,),
            exclude=registration.owner,
        )
        pending = tuple(
            fiber.name
            for fiber in self._fibers.values()
            if fiber is not registration.owner
            and fiber.state != FiberState.DISPOSED
            and any(
                provider is registration
                for provider in fiber.dependency_store.values()
            )
        )
        if pending:
            raise CompositionError(
                "DEPENDENT_CLEANUP_PENDING",
                f"{registration.owner.name} 仍被未关闭的消费者使用: {', '.join(pending)}",
            )
        if self._providers.get(registration.key) is not registration:
            return
        del self._providers[registration.key]
        self._bump_composition_revision()

    def _active_provider(self, key: ServiceKey[Any]) -> _Provider | None:
        provider = self._providers.get(key)
        if (
            provider is None
            or provider.revoking
            or provider.owner.state != FiberState.ACTIVE
        ):
            return None
        return provider

    def _dependency_snapshot(
        self,
        dependencies: tuple[ServiceKey[Any], ...],
    ) -> dict[ServiceKey[Any], _Provider] | None:
        providers: dict[ServiceKey[Any], _Provider] = {}
        for key in dependencies:
            provider = self._active_provider(key)
            if provider is None:
                return None
            providers[key] = provider
        return providers

    @staticmethod
    def _provider_epoch(
        providers: Mapping[ServiceKey[Any], _Provider] | None,
    ) -> tuple[tuple[str, int], ...] | None:
        if providers is None:
            return None
        return tuple(
            sorted((key.name, provider.revision) for key, provider in providers.items())
        )

    def _provider_epoch_if_active(
        self,
        dependencies: tuple[ServiceKey[Any], ...],
    ) -> tuple[tuple[str, int], ...] | None:
        return self._provider_epoch(self._dependency_snapshot(dependencies))

    async def _owner_became_active(self, owner: Fiber) -> None:
        keys = tuple(
            key for key, provider in self._providers.items() if provider.owner is owner
        )
        await self._reconcile_dependents(keys, exclude=owner)

    async def _notify_provider_registered(self, registration: _Provider) -> None:
        """Reconcile only the exact live registration after ACTIVE publication."""

        if (
            self._providers.get(registration.key) is not registration
            or registration.revoking
            or registration.owner.state != FiberState.ACTIVE
        ):
            return
        await self._reconcile_dependents(
            (registration.key,),
            exclude=registration.owner,
        )

    async def _owner_became_inactive(self, owner: Fiber) -> None:
        keys = tuple(
            key for key, provider in self._providers.items() if provider.owner is owner
        )
        await self._reconcile_dependents(keys, exclude=owner)
        # 已请求 dispose 的消费者不会再 reconcile，但它仍可能欠着资源关闭。
        pending = [
            fiber.name
            for fiber in self._fibers.values()
            if fiber is not owner
            and any(provider.owner is owner for provider in fiber.dependency_store.values())
        ]
        if pending:
            raise CompositionError(
                "DEPENDENT_CLEANUP_PENDING",
                f"{owner.name} 仍被未关闭的消费者使用: {', '.join(pending)}",
            )

    def _dependent_fibers(
        self,
        keys: tuple[ServiceKey[Any], ...],
        *,
        exclude: Fiber,
    ) -> list[Fiber]:
        """Scan the same direct wait set used by provider reconciliation."""

        return [
            fiber
            for fiber in tuple(self._fibers.values())
            if fiber is not exclude
            and fiber.state != FiberState.DISPOSED
            and any(key in fiber.dependencies for key in keys)
        ]

    def _service_wait_set(self, registration: _Provider) -> tuple[Fiber, ...]:
        """Expand one registration to children and downstream hard consumers."""

        return self._service_wait_set_for(
            registration.key,
            registration.owner,
            registration=registration,
        )

    def _service_wait_set_for(
        self,
        key: ServiceKey[Any],
        owner: Fiber,
        *,
        registration: _Provider | None = None,
    ) -> tuple[Fiber, ...]:
        """Build the shared wait set without inventing a provider record."""

        queue = deque(
            self._dependent_fibers(
                (key,),
                exclude=owner,
            )
        )
        if registration is not None:
            for fiber in tuple(self._fibers.values()):
                if (
                    fiber is not owner
                    and fiber.state != FiberState.DISPOSED
                    and any(
                        provider is registration
                        for provider in fiber.dependency_store.values()
                    )
                ):
                    queue.append(fiber)

        affected: set[Fiber] = set()
        while queue:
            fiber = queue.popleft()
            if fiber in affected or fiber.state == FiberState.DISPOSED:
                continue
            affected.add(fiber)
            for child in tuple(fiber.children):
                if child.state != FiberState.DISPOSED:
                    queue.append(child)
            owned_keys = tuple(
                key
                for key, provider in self._providers.items()
                if provider.owner is fiber
            )
            for consumer in self._dependent_fibers(
                owned_keys,
                exclude=fiber,
            ):
                queue.append(consumer)
            for consumer in tuple(self._fibers.values()):
                if (
                    consumer is not fiber
                    and consumer.state != FiberState.DISPOSED
                    and any(
                        provider.owner is fiber
                        for provider in consumer.dependency_store.values()
                    )
                ):
                    queue.append(consumer)
        return tuple(affected)

    def _guard_service_close(self, registration: _Provider) -> None:
        """Reject the original caller before service revocation can self-wait."""

        if self._providers.get(registration.key) is not registration:
            return
        self._guard_service_wait(
            registration.key,
            registration.owner,
            registration=registration,
            operation="撤销",
        )

    def _guard_service_notify(
        self,
        key: ServiceKey[Any],
        owner: Fiber,
    ) -> None:
        """Reject an ACTIVE notification that would wait on its own lifecycle."""

        self._guard_service_wait(key, owner, operation="通知")

    def _guard_service_wait(
        self,
        key: ServiceKey[Any],
        owner: Fiber,
        *,
        registration: _Provider | None = None,
        operation: str,
    ) -> None:
        """Check the shared exact owner/call/lifecycle wait set."""

        current = asyncio.current_task()
        if current is None:
            raise CompositionError(
                "OWNER_CALL_CONTEXT",
                "Service registration close 需要实际 Task 上下文",
            )
        for fiber in self._service_wait_set_for(
            key,
            owner,
            registration=registration,
        ):
            if fiber._call_owned_by_current_task() is not None:
                raise CompositionError(
                    "REENTRANT_CALL_WAIT",
                    f"{fiber.name} 的 Service {operation}不能等待该任务仍持有的在途调用",
                )
            if fiber._transition_owner is current:
                raise CompositionError(
                    "REENTRANT_LIFECYCLE_WAIT",
                    f"{fiber.name} 的 Service {operation}不能等待该任务持有的生命周期过渡",
                )
            binding = _lifecycle_binding.get()
            if (
                binding is not None
                and binding[0] is fiber.context
                and binding[1] is current
            ):
                raise CompositionError(
                    "REENTRANT_LIFECYCLE_WAIT",
                    f"{fiber.name} 的 Service {operation}不能等待该任务借用的生命周期",
                )

    async def _reconcile_dependents(
        self,
        keys: tuple[ServiceKey[Any], ...],
        *,
        exclude: Fiber,
    ) -> None:
        """服务变化只重新协调实际消费者。"""

        if not keys:
            return
        affected = self._dependent_fibers(keys, exclude=exclude)
        if affected:
            results = await asyncio.gather(
                *(fiber.reconcile() for fiber in affected),
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise BaseExceptionGroup("依赖 Fiber 协调失败", errors)

    def _new_health_entry(
        self,
        owner: Fiber,
        *,
        name: str,
        required: bool,
    ) -> _HealthEntry:
        if not name or name.strip() != name:
            raise ValueError("健康项名称必须是非空且无首尾空白的字符串")
        if (owner.fiber_id, name) in self._health_entries:
            raise CompositionError(
                "DUPLICATE_HEALTH",
                f"Fiber {owner.name} 已注册健康项: {name}",
            )
        return _HealthEntry(owner=owner, name=name, required=required)

    def _check_required_health(self, owner: Fiber) -> None:
        """ACTIVE 前检查本 owner 已登记的 required 健康项均可用。"""

        degraded = [
            entry.name
            for entry in self._health_entries.values()
            if entry.owner is owner and entry.required and entry.reason is not None
        ]
        if degraded:
            raise CompositionError(
                "UNHEALTHY_OWNER",
                f"{owner.name} 的必需健康项未就绪: {', '.join(degraded)}",
            )

    def _register_health(self, entry: _HealthEntry) -> None:
        key = (entry.owner.fiber_id, entry.name)
        if key in self._health_entries:
            raise CompositionError(
                "DUPLICATE_HEALTH",
                f"Fiber {entry.owner.name} 已注册健康项: {entry.name}",
            )
        self._health_entries[key] = entry

    def _remove_health(self, entry: _HealthEntry) -> None:
        key = (entry.owner.fiber_id, entry.name)
        if self._health_entries.get(key) is not entry:
            return
        del self._health_entries[key]
        entry.active = False
        entry.reason = None

    @staticmethod
    def _require_active_health(entry: _HealthEntry) -> None:
        if not entry.active:
            raise CompositionError(
                "INACTIVE_HEALTH",
                f"健康项已经注销: {entry.owner.name}:{entry.name}",
            )

    def _degrade_health(self, entry: _HealthEntry, reason: str) -> None:
        self._require_active_health(entry)
        if not reason or reason.strip() != reason:
            raise ValueError("健康降级原因必须是非空且无首尾空白的字符串")
        entry.reason = reason

    def _recover_health(self, entry: _HealthEntry) -> None:
        self._require_active_health(entry)
        entry.reason = None

    def _health_view(self) -> tuple[HealthView, ...]:
        entries = [
            HealthView(
                owner=entry.owner.path,
                name=entry.name,
                required=entry.required,
                healthy=entry.reason is None,
                reason=entry.reason,
            )
            for entry in self._health_entries.values()
        ]
        for fiber in (self.root_fiber, *self._fibers.values()):
            entries.extend(
                HealthView(
                    owner=fiber.path,
                    name=f"task:{name}",
                    required=fiber.required_for_readiness,
                    healthy=False,
                    reason=reason,
                )
                for name, reason in fiber._task_failures.items()
            )
        return tuple(sorted(entries, key=lambda item: (item.owner, item.name)))

    def _report_incident(
        self,
        fiber: Fiber,
        *,
        kind: str,
        message: str,
        error_type: str | None = None,
    ) -> IncidentView:
        self._incident_sequence += 1
        count_key = (fiber.fiber_id, fiber.path)
        self._incident_counts[count_key] = self._incident_counts.get(count_key, 0) + 1
        incident = IncidentView(
            fiber_id=fiber.fiber_id,
            sequence=self._incident_sequence,
            owner=fiber.path,
            kind=kind,
            message=message,
            error_type=error_type,
        )
        self._recent_incidents.append(incident)
        return incident

    def recent_incidents(self) -> tuple[IncidentView, ...]:
        return tuple(self._recent_incidents)

    @property
    def incident_sequence(self) -> int:
        return self._incident_sequence

    def _record_error(self, fiber: Fiber, error: BaseException) -> None:
        if isinstance(error, CompositionError):
            _ = self._report_incident(
                fiber,
                kind="composition_error",
                message=f"{error.code}: {error}",
                error_type=type(error).__name__,
            )
            return
        _ = self._report_incident(
            fiber,
            kind="runtime_error",
            message=_error_message(error),
            error_type=type(error).__name__,
        )

    def _record_listener_failure(
        self,
        fiber: Fiber,
        kind: str,
        error: BaseException,
    ) -> None:
        _ = self._report_incident(
            fiber,
            kind=kind,
            message=_error_message(error),
            error_type=type(error).__name__,
        )

    def _record_task_result(
        self,
        fiber: Fiber,
        name: str,
        task: asyncio.Task[object],
    ) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            reason = _error_message(error)
            fiber._task_failures[name] = reason
            _ = self._report_incident(
                fiber,
                kind="task_failure",
                message=reason,
                error_type=type(error).__name__,
            )

    def _remove_fiber(self, fiber: Fiber) -> None:
        if self._fibers.pop(fiber.fiber_id, None) is not None:
            self._bump_composition_revision()

    def _bump_composition_revision(self) -> None:
        self._composition_revision += 1

    @property
    def composition_revision(self) -> int:
        return self._composition_revision

    def _fiber_view(self, fiber: Fiber) -> FiberView:
        return FiberView(
            fiber_id=fiber.fiber_id,
            name=fiber.path,
            state=fiber.state,
            required_for_readiness=fiber.required_for_readiness,
            missing_services=fiber.missing_services,
            error=(
                None
                if fiber.error is None
                else f"{type(fiber.error).__name__}: {fiber.error}"
            ),
        )


def _error_message(error: BaseException) -> str:
    """把任意插件异常转换成不会再次失败的 Incident 文本。"""

    try:
        message = str(error)
    except BaseException:
        return f"<unprintable {type(error).__name__}>"
    return message or type(error).__name__
