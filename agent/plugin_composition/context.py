from __future__ import annotations

# pyright: reportPrivateUsage=false

import asyncio
import hashlib
import inspect
import json
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, AsyncGenerator, Literal, Protocol, TypeVar, cast

from agent.plugin_composition.effect import Effect, EffectSetup
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
from agent.plugin_composition.access import CompositionAudit
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

T = TypeVar("T")
R = TypeVar("R")
PluginApply = Callable[["Context"], object]
FiberObserver = Callable[["Fiber"], object]


class Plugin(Protocol):
    def apply(self, ctx: Context) -> object: ...


@dataclass(slots=True)
class _Provider:
    key: ServiceKey[object]
    value: object
    owner: Fiber
    revision: int


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

    @property
    def fiber(self) -> FiberHandle:
        reject_executor_context_access()
        return self._fiber_handle

    @property
    def generation_id(self) -> str:
        reject_executor_context_access()
        return self._root.generation_id

    def _root_instance_token(self) -> object:
        """Return the Core-only identity of this Context's Root."""

        reject_executor_context_access()
        return self._root.instance_token

    def _plugin_module(self) -> ModuleType | None:
        """Return the exact module mounted on this Fiber when one exists."""

        reject_executor_context_access()
        return self._fiber.plugin_module

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
    def data_root(self) -> Path:
        """返回 Core 为当前插件 generation 分配的数据根。"""

        return self.runtime.data_dir

    @property
    def data_access(self) -> Literal["read_write", "read_only"]:
        """Return the access mode assigned to this exact plugin Root."""

        return self.runtime.data_access

    def workspace_root(self, name: str) -> Path:
        """返回 Core 为当前 generation 投影的声明式 workspace root。"""

        reject_executor_context_access()
        return self.runtime.workspace_root(name)

    def workspace_file(self, name: str) -> Path:
        """Return one Core-projected top-level product file declared by the plugin."""

        reject_executor_context_access()
        return self.runtime.workspace_file(name)

    def _set_static_active(self, active: bool) -> None:
        """把 adapter 决定的静态贡献状态冻结在当前 Fiber。"""

        reject_executor_context_access()
        self._root._set_static_active(self._fiber, active)

    async def mount(
        self,
        plugin: Plugin | PluginApply,
        *,
        name: str | None = None,
        inject: Iterable[ServiceKey[object]] | None = None,
        required_for_readiness: bool = True,
    ) -> FiberHandle:
        reject_executor_context_access()
        fiber = await self._root._mount(
            parent=self._fiber,
            plugin=plugin,
            name=name,
            inject=inject,
            required_for_readiness=required_for_readiness,
            runtime=self._fiber.runtime,
        )
        return FiberHandle(fiber)

    async def inject(
        self,
        dependencies: Iterable[ServiceKey[object]],
        apply: PluginApply,
        *,
        name: str | None = None,
    ) -> FiberHandle:
        """Mount an optional child that activates only while deps exist."""

        reject_executor_context_access()
        return await self.mount(
            apply,
            name=name or getattr(apply, "__name__", "inject"),
            inject=dependencies,
            required_for_readiness=False,
        )

    async def provide(self, key: ServiceKey[T], value: T) -> Effect:
        reject_executor_context_access()

        async def setup() -> Callable[[], Awaitable[None]]:
            self._root._register_provider(
                cast(ServiceKey[object], key),
                value,
                self._fiber,
            )

            async def cleanup() -> None:
                await self._root._remove_provider(
                    cast(ServiceKey[object], key),
                    self._fiber,
                )

            return cleanup

        return await self.effect(setup, label=f"service:{key.name}")

    def get(self, key: ServiceKey[T]) -> T | None:
        reject_executor_context_access()
        provider = self._fiber.dependency_store.get(cast(ServiceKey[object], key))
        if provider is None:
            provider = self._root._active_provider(cast(ServiceKey[object], key))
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
        return await self._fiber.add_effect(setup, label=label)

    async def health(
        self,
        name: str,
        *,
        required: bool = True,
    ) -> HealthHandle:
        """注册一个由当前 Fiber Effect 持有的健康项。"""

        reject_executor_context_access()
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
        self._root._events.emit(key, payload)

    async def serial(
        self,
        key: SerialEventKey[T, R],
        payload: T,
    ) -> Bail[R] | None:
        reject_executor_context_access()
        return await self._root._events.serial(key, payload)

    async def parallel(self, key: ParallelEventKey[T], payload: T) -> None:
        reject_executor_context_access()
        await self._root._events.parallel(key, payload)

    async def transform(self, key: TransformEventKey[T], payload: T) -> T:
        reject_executor_context_access()
        return await self._root._events.transform(key, payload)

    async def observe(self, key: ObserveEventKey[T], payload: T) -> None:
        reject_executor_context_access()
        await self._root._events.observe(key, payload)

    async def spawn(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str,
    ) -> asyncio.Task[T]:
        """Start one Fiber-owned task and expose failures to Core readiness."""

        reject_executor_context_access()
        if not name or name.strip() != name:
            coroutine.close()
            raise ValueError("任务名称必须是非空且无首尾空白的字符串")
        task: asyncio.Task[T] | None = None

        def setup() -> Callable[[], Awaitable[None]]:
            nonlocal task
            task = asyncio.create_task(coroutine, name=f"plugin-task:{name}")
            task.add_done_callback(
                lambda completed: self._root._record_task_result(
                    self._fiber,
                    name,
                    cast(asyncio.Task[object], completed),
                )
            )

            async def cleanup() -> None:
                assert task is not None
                if not task.done():
                    _ = task.cancel()
                _ = await asyncio.gather(task, return_exceptions=True)

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
    def fiber_id(self) -> int:
        reject_executor_context_access()
        return self._fiber.fiber_id

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

    async def restart(self) -> None:
        reject_executor_context_access()
        await self._fiber.restart()

    async def dispose(self) -> None:
        reject_executor_context_access()
        await self._fiber.dispose()


class HealthHandle:
    """允许插件显式降级或恢复一个 Effect-owned 健康项。"""

    __slots__ = ("_root", "_entry")

    def __init__(self, root: CompositionRoot, entry: _HealthEntry) -> None:
        self._root = root
        self._entry = entry

    @property
    def name(self) -> str:
        reject_executor_context_access()
        return self._entry.name

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
        dependencies: tuple[ServiceKey[object], ...],
        parent: Fiber | None,
        required_for_readiness: bool,
        runtime: PluginRuntime | None,
        plugin_module: ModuleType | None,
        static_active: bool = True,
        is_root: bool = False,
    ) -> None:
        self.root = root
        self.fiber_id = fiber_id
        self.name = name
        self.apply = apply
        self.dependencies = dependencies
        self._activation_dependencies = dependencies if static_active else ()
        self.parent = parent
        self.required_for_readiness = required_for_readiness
        self.runtime = runtime
        self.plugin_module = plugin_module
        self.state = FiberState.ACTIVE if is_root else FiberState.PENDING
        self.context = Context(root, self)
        self.dependency_store: dict[ServiceKey[object], _Provider] = {}
        self.effects: list[Effect] = []
        self.children: list[Fiber] = []
        self.error: BaseException | None = None
        self._task_failures: dict[str, str] = {}
        self.static_active = static_active
        self._epoch: tuple[tuple[str, int], ...] | None = () if is_root else None
        self._activation_token: object | None = object() if is_root else None
        self._transition = asyncio.Lock()
        self._transition_owner: asyncio.Task[object] | None = None
        self._dispose_requested = False
        self._dispose_task: asyncio.Task[None] | None = None
        self._restart_task: asyncio.Task[None] | None = None
        self._is_root = is_root

    @property
    def missing_services(self) -> tuple[str, ...]:
        return tuple(
            key.name
            for key in self._activation_dependencies
            if self.root._active_provider(key) is None
        )

    async def add_effect(self, setup: EffectSetup, *, label: str) -> Effect:
        """Register ownership before setup and expose only live Fiber states."""

        if self.state in {FiberState.UNLOADING, FiberState.DISPOSED}:
            raise CompositionError(
                "INACTIVE_EFFECT",
                f"{self.name} 在 {self.state.value} 状态不能注册 Effect",
            )
        effect = Effect(label=label, remove_from_owner=self._remove_effect)
        self.effects.append(effect)
        return await effect.start(setup)

    async def reconcile(self) -> None:
        """Move to the state implied by the newest dependency epoch."""

        async with self._locked_transition():
            if self._dispose_requested or self._is_root:
                return
            providers = self.root._dependency_snapshot(self._activation_dependencies)
            target_epoch = self.root._provider_epoch(providers)
            if providers is None:
                if self.state in {FiberState.ACTIVE, FiberState.FAILED}:
                    await self._unload(next_state=FiberState.PENDING)
                return
            assert target_epoch is not None
            if self.state == FiberState.ACTIVE and self._epoch == target_epoch:
                return
            if self.state in {FiberState.ACTIVE, FiberState.FAILED}:
                await self._unload(next_state=FiberState.PENDING)
            await self._load(providers, target_epoch)

    async def restart(self) -> None:
        self._reject_direct_reentrant_wait("restart")
        if self._restart_task is None or self._restart_task.done():
            self._restart_task = asyncio.create_task(
                self._restart(),
                name=f"plugin-fiber-restart:{self.name}",
            )
        await _await_critical(self._restart_task)

    async def _restart(self) -> None:
        async with self._locked_transition():
            if self._dispose_requested or self._is_root:
                return
            if self.state in {FiberState.ACTIVE, FiberState.FAILED}:
                await self._unload(next_state=FiberState.PENDING)
        await self.reconcile()

    async def dispose(self) -> None:
        """Permanently unload this Fiber and join all child/effect cleanup."""

        self._reject_direct_reentrant_wait("dispose")
        if self._dispose_task is None:
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
            unload_error: BaseException | None = None
            try:
                if self.state != FiberState.UNLOADING:
                    await self._unload(next_state=FiberState.DISPOSED)
            except BaseException as error:
                unload_error = error
            finally:
                self.state = FiberState.DISPOSED
                self.root._remove_fiber(self)
                if self.parent is not None and self in self.parent.children:
                    self.parent.children.remove(self)
        await self.root._notify_disposed(self)
        if unload_error is not None:
            raise unload_error

    async def _load(
        self,
        providers: dict[ServiceKey[object], _Provider],
        epoch: tuple[tuple[str, int], ...],
    ) -> None:
        # 1. Freeze the dependency values for this activation.
        self.state = FiberState.LOADING
        self._activation_token = object()
        self.dependency_store = providers
        self.error = None
        await self.root._notify_status(self)
        await asyncio.sleep(0)
        if (
            self._dispose_requested
            or self.root._provider_epoch_if_active(self._activation_dependencies)
            != epoch
        ):
            await self._unload(next_state=FiberState.PENDING)
            return

        # 2. Apply the plugin and publish its services only after success.
        try:
            result = self.apply(self.context)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                self._unload(next_state=FiberState.PENDING),
                name=f"plugin-fiber-cancel-cleanup:{self.name}",
            )
            await _await_critical(cleanup_task)
            raise
        except Exception as error:
            self.error = error
            self.root._record_error(self, error)
            await self._unload(next_state=FiberState.FAILED)
            return
        if (
            self._dispose_requested
            or self.root._provider_epoch_if_active(self._activation_dependencies)
            != epoch
        ):
            await self._unload(next_state=FiberState.PENDING)
            return
        self._epoch = epoch
        self.state = FiberState.ACTIVE
        await self.root._notify_status(self)
        await self.root._owner_became_active(self)

    async def _unload(self, *, next_state: FiberState) -> None:
        # 1. Make owned services unavailable before dependents clean up.
        self._activation_token = None
        self.state = FiberState.UNLOADING
        await self.root._notify_status(self)
        errors: list[BaseException] = []
        try:
            await self.root._owner_became_inactive(self)
        except BaseException as error:
            errors.append(error)

        # 2. Children and effects are fully drained in reverse ownership order.
        for child in reversed(tuple(self.children)):
            try:
                await child.dispose()
            except BaseException as error:
                errors.append(error)
        for effect in reversed(tuple(self.effects)):
            try:
                await effect.aclose()
            except BaseException as error:
                errors.append(error)
        self.dependency_store = {}
        self._task_failures.clear()
        self._epoch = None
        self.state = next_state
        await self.root._notify_status(self)
        if errors:
            raise BaseExceptionGroup(f"Fiber 卸载失败: {self.name}", errors)

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
    """Own one generation topology and derive its validation receipt."""

    RECENT_INCIDENT_LIMIT = 128

    def __init__(
        self,
        generation_id: str,
        *,
        audit: CompositionAudit | None = None,
        candidate_incident_limit: int | None = None,
    ) -> None:
        if not generation_id:
            raise ValueError("generation_id 不能为空")
        if candidate_incident_limit is not None and candidate_incident_limit <= 0:
            raise ValueError("candidate_incident_limit 必须大于零")
        self.generation_id = generation_id
        self._instance_token = object()
        self._next_fiber_id = 1
        self._next_provider_revision = 1
        self._composition_revision = 0
        self._fibers: dict[int, Fiber] = {}
        self._providers: dict[ServiceKey[object], _Provider] = {}
        self._mount_observers: list[FiberObserver] = []
        self._status_observers: list[FiberObserver] = []
        self._dispose_observers: list[FiberObserver] = []
        self._health_entries: dict[tuple[int, str], _HealthEntry] = {}
        self._incident_sequence = 0
        self._incident_counts: dict[str, int] = {}
        self._candidate_incident_limit = candidate_incident_limit
        self._incident_overflowed = False
        self._recent_incidents: deque[IncidentView] = deque(
            maxlen=(
                candidate_incident_limit
                if candidate_incident_limit is not None
                else self.RECENT_INCIDENT_LIMIT
            )
        )
        self._audit = audit or CompositionAudit()
        self._events = EventRegistry(
            self._bump_composition_revision,
            self._record_listener_failure,
        )
        self._internal_cleanups: list[tuple[str, Callable[[], object]]] = []
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
            static_active=True,
            is_root=True,
        )
        self.context = self.root_fiber.context

    @property
    def instance_token(self) -> object:
        """标识单个 Root 实例，不参与可持久化拓扑身份。"""

        return self._instance_token

    def on_mount(self, observer: FiberObserver) -> Callable[[], None]:
        return self._add_observer(self._mount_observers, observer)

    def on_status(self, observer: FiberObserver) -> Callable[[], None]:
        return self._add_observer(self._status_observers, observer)

    def on_dispose(self, observer: FiberObserver) -> Callable[[], None]:
        return self._add_observer(self._dispose_observers, observer)

    async def mount(
        self,
        plugin: Plugin | PluginApply,
        *,
        name: str | None = None,
        inject: Iterable[ServiceKey[object]] | None = None,
        runtime: PluginRuntime | None = None,
    ) -> Fiber:
        return await self._mount(
            parent=self.root_fiber,
            plugin=plugin,
            name=name,
            inject=inject,
            required_for_readiness=True,
            runtime=runtime,
        )

    async def dispose(self) -> None:
        if self._dispose_task is None:
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
        for effect in reversed(tuple(self.root_fiber.effects)):
            try:
                await effect.aclose()
            except BaseException as error:
                errors.append(error)
        for resource, cleanup in reversed(self._internal_cleanups):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except BaseException as error:
                errors.append(
                    BaseExceptionGroup(
                        f"Core cleanup 失败: {resource}",
                        [error],
                    )
                )
        self._internal_cleanups.clear()
        self.root_fiber.state = FiberState.DISPOSED
        if errors:
            raise BaseExceptionGroup("Root Context 清理失败", errors)

    def receipt(self) -> CompositionReceipt:
        fibers = tuple(self._fiber_view(fiber) for fiber in self._fibers.values())
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
            f"{fiber.name}:{effect.label}"
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
                and not self._incident_overflowed
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
            incident_counts=tuple(sorted(self._incident_counts.items())),
            incident_overflowed=self._incident_overflowed,
            writes=self._audit.writes,
            external_effects=external_effects,
        )

    def topology_view(self) -> TopologyView:
        """Freeze the current logical topology as a content-addressed value."""

        # 1. 结构身份排除 Fiber 状态、错误和普通 Effect。
        receipt = self.receipt()
        fibers = tuple(
            sorted(
                (
                    TopologyFiberView(
                        name=fiber.name,
                        parent=(
                            None
                            if fiber.parent is self.root_fiber
                            else cast(Fiber, fiber.parent).name
                        ),
                        required_for_readiness=fiber.required_for_readiness,
                        dependencies=tuple(
                            sorted(key.name for key in fiber.dependencies)
                        ),
                        static_active=fiber.static_active,
                    )
                    for fiber in self._fibers.values()
                ),
                key=lambda item: item.name,
            )
        )
        effects = tuple(sorted(receipt.effects))
        listeners = self._events.registrations()

        # 2. 内容 hash 与单调 revision 分别回答“是什么”和“是否变过”。
        identity_payload: dict[str, object] = {
            "fibers": [
                {
                    "name": fiber.name,
                    "parent": fiber.parent,
                    "required": fiber.required_for_readiness,
                    "dependencies": fiber.dependencies,
                    "static_active": fiber.static_active,
                }
                for fiber in fibers
            ],
            "services": receipt.services,
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
            services=receipt.services,
            effects=effects,
            listeners=listeners,
        )

    def topology_identity(self) -> str:
        return self.topology_view().identity

    def active_plugin_ids(self) -> frozenset[str]:
        """返回当前 Root 中 active 的顶层插件身份。"""

        return frozenset(
            runtime.plugin_id
            for fiber in self.root_fiber.children
            if fiber.static_active
            if (runtime := fiber.runtime) is not None
        )

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

    def validation_identity(self) -> str:
        """Bind the Core-observed topology and audit receipt at validation close."""

        receipt = self.receipt()
        writes = ",".join(
            f"{item.plugin_id}:{item.operation}:{item.relative_path}:{item.sha256}"
            for item in receipt.writes
        )
        external = ",".join(
            f"{item.kind}:{item.target}:{item.outcome}"
            for item in receipt.external_effects
        )
        return "|".join(
            (
                self.topology_identity(),
                f"revision:{self._composition_revision}",
                f"required:{','.join(receipt.required_pending)}",
                f"optional:{','.join(receipt.optional_pending)}",
                f"degraded:{','.join(receipt.required_degraded)}",
                f"incidents:{receipt.incident_sequence}",
                f"incident_overflowed:{receipt.incident_overflowed}",
                f"writes:{writes}",
                f"external:{external}",
            )
        )

    async def _mount(
        self,
        *,
        parent: Fiber,
        plugin: Plugin | PluginApply,
        name: str | None,
        inject: Iterable[ServiceKey[object]] | None,
        required_for_readiness: bool,
        runtime: PluginRuntime | None,
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
        if any(fiber.name == resolved_name for fiber in self._fibers.values()):
            raise CompositionError(
                "DUPLICATE_PLUGIN",
                f"同一拓扑不能重复挂载插件: {resolved_name}",
            )

        # 2. Parent ownership is visible before publication observers run.
        static_active = getattr(plugin, "static_active", True)
        if not isinstance(static_active, bool):
            raise TypeError("插件 static_active 必须是 bool")
        fiber = Fiber(
            root=self,
            fiber_id=self._next_fiber_id,
            name=resolved_name,
            apply=apply,
            dependencies=dependencies,
            parent=parent,
            required_for_readiness=required_for_readiness,
            runtime=runtime,
            plugin_module=(
                module
                if isinstance((module := getattr(plugin, "module", None)), ModuleType)
                else parent.plugin_module
            ),
            static_active=static_active,
        )
        self._next_fiber_id += 1
        parent.children.append(fiber)
        self._fibers[fiber.fiber_id] = fiber
        self._bump_composition_revision()
        try:
            await self._notify_mount(fiber)
        except BaseException:
            await fiber.dispose()
            raise
        if parent.state in {FiberState.UNLOADING, FiberState.DISPOSED}:
            await fiber.dispose()
            return fiber
        try:
            await fiber.reconcile()
        except BaseException:
            await fiber.dispose()
            raise
        return fiber

    def _resolve_plugin(
        self,
        plugin: Plugin | PluginApply,
        *,
        name: str | None,
        inject: Iterable[ServiceKey[object]] | None,
    ) -> tuple[PluginApply, str, tuple[ServiceKey[object], ...]]:
        if callable(plugin) and not hasattr(plugin, "apply"):
            apply = cast(PluginApply, plugin)
        else:
            candidate = getattr(plugin, "apply", None)
            if not callable(candidate):
                raise TypeError("插件必须是 callable 或提供 apply(ctx)")
            apply = cast(PluginApply, candidate)
        resolved_name = name or str(getattr(plugin, "name", "")).strip()
        resolved_name = resolved_name or getattr(apply, "__name__", "plugin")
        raw_dependencies = inject
        if raw_dependencies is None:
            raw_dependencies = getattr(plugin, "inject", ())
        dependencies = tuple(cast(Iterable[ServiceKey[object]], raw_dependencies))
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f"插件依赖重复: {resolved_name}")
        return apply, resolved_name, dependencies

    def _register_provider(
        self,
        key: ServiceKey[object],
        value: object,
        owner: Fiber,
    ) -> None:
        existing = self._providers.get(key)
        if existing is not None:
            raise CompositionError(
                "DUPLICATE_SERVICE",
                f"Service {key.name} 已由 {existing.owner.name} 提供",
            )
        self._providers[key] = _Provider(
            key=key,
            value=value,
            owner=owner,
            revision=self._next_provider_revision,
        )
        self._next_provider_revision += 1
        self._bump_composition_revision()

    def _set_static_active(self, owner: Fiber, active: bool) -> None:
        if not isinstance(active, bool):
            raise TypeError("static active 状态必须是 bool")
        if owner.static_active == active:
            return
        owner.static_active = active
        self._bump_composition_revision()

    async def _remove_provider(
        self,
        key: ServiceKey[object],
        owner: Fiber,
    ) -> None:
        provider = self._providers.get(key)
        if provider is None:
            return
        if provider.owner is not owner:
            raise CompositionError(
                "SERVICE_OWNER_MISMATCH",
                f"{owner.name} 不能移除 {provider.owner.name} 的 Service {key.name}",
            )
        del self._providers[key]
        self._bump_composition_revision()
        await self._reconcile_dependents((key,), exclude=owner)

    def _active_provider(self, key: ServiceKey[object]) -> _Provider | None:
        provider = self._providers.get(key)
        if provider is None or provider.owner.state != FiberState.ACTIVE:
            return None
        return provider

    def _dependency_snapshot(
        self,
        dependencies: tuple[ServiceKey[object], ...],
    ) -> dict[ServiceKey[object], _Provider] | None:
        providers: dict[ServiceKey[object], _Provider] = {}
        for key in dependencies:
            provider = self._active_provider(key)
            if provider is None:
                return None
            providers[key] = provider
        return providers

    @staticmethod
    def _provider_epoch(
        providers: Mapping[ServiceKey[object], _Provider] | None,
    ) -> tuple[tuple[str, int], ...] | None:
        if providers is None:
            return None
        return tuple(
            sorted((key.name, provider.revision) for key, provider in providers.items())
        )

    def _provider_epoch_if_active(
        self,
        dependencies: tuple[ServiceKey[object], ...],
    ) -> tuple[tuple[str, int], ...] | None:
        return self._provider_epoch(self._dependency_snapshot(dependencies))

    async def _owner_became_active(self, owner: Fiber) -> None:
        keys = tuple(
            key for key, provider in self._providers.items() if provider.owner is owner
        )
        await self._reconcile_dependents(keys, exclude=owner)

    async def _owner_became_inactive(self, owner: Fiber) -> None:
        keys = tuple(
            key for key, provider in self._providers.items() if provider.owner is owner
        )
        await self._reconcile_dependents(keys, exclude=owner)

    async def _reconcile_dependents(
        self,
        keys: tuple[ServiceKey[object], ...],
        *,
        exclude: Fiber,
    ) -> None:
        if not keys:
            return
        affected = [
            fiber
            for fiber in tuple(self._fibers.values())
            if fiber is not exclude
            and fiber.state != FiberState.DISPOSED
            and any(key in fiber._activation_dependencies for key in keys)
        ]
        if affected:
            results = await asyncio.gather(
                *(fiber.reconcile() for fiber in affected),
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise BaseExceptionGroup("依赖 Fiber 协调失败", errors)

    async def _notify_mount(self, fiber: Fiber) -> None:
        for observer in tuple(self._mount_observers):
            result = observer(fiber)
            if inspect.isawaitable(result):
                await result

    async def _notify_status(self, fiber: Fiber) -> None:
        await self._notify_contained(self._status_observers, fiber)

    async def _notify_disposed(self, fiber: Fiber) -> None:
        await self._notify_contained(self._dispose_observers, fiber)

    async def _notify_contained(
        self,
        observers: list[FiberObserver],
        fiber: Fiber,
    ) -> None:
        for observer in tuple(observers):
            try:
                result = observer(fiber)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError as error:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                self._record_error(fiber, error)
            except Exception as error:
                self._record_error(fiber, error)

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
                owner=entry.owner.name,
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
                    owner=fiber.name,
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
        self._incident_counts[fiber.name] = self._incident_counts.get(fiber.name, 0) + 1
        incident = IncidentView(
            sequence=self._incident_sequence,
            owner=fiber.name,
            kind=kind,
            message=message,
            error_type=error_type,
        )
        limit = self._candidate_incident_limit
        if limit is not None and len(self._recent_incidents) >= limit:
            self._incident_overflowed = True
            return incident
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
            name=fiber.name,
            state=fiber.state,
            required_for_readiness=fiber.required_for_readiness,
            missing_services=fiber.missing_services,
            error=(
                None
                if fiber.error is None
                else f"{type(fiber.error).__name__}: {fiber.error}"
            ),
        )

    @staticmethod
    def _add_observer(
        observers: list[FiberObserver],
        observer: FiberObserver,
    ) -> Callable[[], None]:
        observers.append(observer)

        def remove() -> None:
            if observer in observers:
                observers.remove(observer)

        return remove


def _error_message(error: BaseException) -> str:
    """把任意插件异常转换成不会再次失败的 Incident 文本。"""

    try:
        message = str(error)
    except BaseException:
        return f"<unprintable {type(error).__name__}>"
    return message or type(error).__name__


async def _await_critical(task: asyncio.Task[None]) -> None:
    """Finish lifecycle cleanup before propagating caller cancellation."""

    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
