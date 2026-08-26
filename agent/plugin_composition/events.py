from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, ContextManager, Generic, TypeVar, cast

from agent.plugin_composition.diagnostics import (
    PluginOperation,
    plugin_entrypoint,
    start_plugin_entrypoint,
)
from agent.plugin_composition.model import CompositionError, FiberState

if TYPE_CHECKING:
    from agent.plugin_composition.context import Fiber

P = TypeVar("P")
R = TypeVar("R")


def _validate_event_name(name: str) -> None:
    if not name or name.strip() != name:
        raise ValueError("事件名称必须是非空且无首尾空白的字符串")


@dataclass(frozen=True, slots=True)
class EmitEventKey(Generic[P]):
    name: str

    def __post_init__(self) -> None:
        _validate_event_name(self.name)


@dataclass(frozen=True, slots=True)
class SerialEventKey(Generic[P, R]):
    name: str
    bail_type: type[R] | None = None
    bail_contract: str | None = None

    def __post_init__(self) -> None:
        _validate_event_name(self.name)
        if (self.bail_type is None) != (self.bail_contract is None):
            raise ValueError("串行 Bail type 与 contract 必须同时声明")
        if self.bail_contract is not None and (
            not self.bail_contract
            or self.bail_contract.strip() != self.bail_contract
        ):
            raise ValueError("串行 Bail contract 必须是非空且无首尾空白的字符串")


@dataclass(frozen=True, slots=True)
class ParallelEventKey(Generic[P]):
    name: str

    def __post_init__(self) -> None:
        _validate_event_name(self.name)


@dataclass(frozen=True, slots=True)
class TransformEventKey(Generic[P]):
    name: str
    payload_type: type[P]
    payload_contract: str

    def __post_init__(self) -> None:
        _validate_event_name(self.name)
        if (
            not self.payload_contract
            or self.payload_contract.strip() != self.payload_contract
        ):
            raise ValueError("变换 payload contract 必须是非空且无首尾空白的字符串")


@dataclass(frozen=True, slots=True)
class ObserveEventKey(Generic[P]):
    name: str

    def __post_init__(self) -> None:
        _validate_event_name(self.name)


@dataclass(frozen=True, slots=True)
class Bail(Generic[R]):
    value: R


EventKey = (
    EmitEventKey[object]
    | SerialEventKey[object, object]
    | ParallelEventKey[object]
    | TransformEventKey[object]
    | ObserveEventKey[object]
)
EventListener = Callable[[object], object]
ListenerFailureHandler = Callable[["Fiber", str, BaseException], None]


@dataclass(frozen=True, slots=True)
class _Listener:
    owner: "Fiber"
    callback: EventListener


class EventRegistry:
    """Own typed listeners and execute one frozen listener list per dispatch."""

    def __init__(
        self,
        on_structure_changed: Callable[[], None],
        on_listener_failure: ListenerFailureHandler,
    ) -> None:
        self._listeners: dict[EventKey, list[_Listener]] = {}
        self._contracts: dict[str, EventKey] = {}
        self._on_structure_changed = on_structure_changed
        self._on_listener_failure = on_listener_failure

    def register(
        self,
        owner: "Fiber",
        key: EventKey,
        callback: EventListener,
    ) -> Callable[[], None]:
        """Validate and publish one listener until its owning Effect closes."""

        # 1. Reject malformed plugin input before publishing any Root state.
        if not callable(callback):
            raise CompositionError(
                "INVALID_EVENT_LISTENER",
                f"事件 {key.name} 的 listener 必须可调用",
            )

        # 2. One event name has one dispatch contract for the whole Root.
        existing_contract = self._contracts.get(key.name)
        if existing_contract is not None and existing_contract != key:
            raise CompositionError(
                "EVENT_MODE_CONFLICT",
                f"事件 {key.name} 已声明为 {type(existing_contract).__name__}",
            )
        if isinstance(key, EmitEventKey) and _is_async_callable(callback):
            raise CompositionError(
                "ASYNC_LISTENER_ON_EMIT",
                f"同步事件 {key.name} 不能注册异步 listener",
            )
        if isinstance(key, ParallelEventKey) and not _is_async_callable(callback):
            raise CompositionError(
                "SYNC_LISTENER_ON_PARALLEL",
                f"并发事件 {key.name} 只能注册异步 listener",
            )

        # 3. Registration order is the only listener order contract.
        self._contracts[key.name] = key
        listener = _Listener(owner=owner, callback=callback)
        listeners = self._listeners.setdefault(key, [])
        listeners.append(listener)
        self._on_structure_changed()

        def remove() -> None:
            current = self._listeners.get(key)
            if current is None or listener not in current:
                return
            current.remove(listener)
            self._on_structure_changed()
            if current:
                return
            del self._listeners[key]
            _ = self._contracts.pop(key.name, None)

        return remove

    def emit(self, key: EmitEventKey[P], payload: P) -> None:
        for listener in self._active_listeners(cast(EventKey, key)):
            with _listener_boundary(listener, "emit", key.name):
                result = listener.callback(payload)
                if inspect.isawaitable(result):
                    _close_unexpected_awaitable(result)
                    raise CompositionError(
                        "ASYNC_RESULT_FROM_EMIT",
                        f"同步事件 {key.name} 的 listener 返回了 awaitable",
                    )

    async def serial(
        self,
        key: SerialEventKey[P, R],
        payload: P,
    ) -> Bail[R] | None:
        for listener in self._active_listeners(cast(EventKey, key)):
            try:
                with _listener_boundary(listener, "serial", key.name):
                    result = listener.callback(payload)
                    if inspect.isawaitable(result):
                        result = await result
                    if result is None:
                        continue
                    if isinstance(result, Bail):
                        bail = cast(Bail[object], result)
                        if key.bail_type is not None and not isinstance(
                            bail.value,
                            key.bail_type,
                        ):
                            raise CompositionError(
                                "INVALID_SERIAL_BAIL",
                                f"串行事件 {key.name} 的 Bail value 必须符合 "
                                f"{key.bail_contract}",
                            )
                        return cast(Bail[R], bail)
                    raise CompositionError(
                        "INVALID_SERIAL_RESULT",
                        f"串行事件 {key.name} 的 listener 只能返回 None 或 Bail",
                    )
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                self._on_listener_failure(listener.owner, "serial_failure", error)
                raise
            except BaseException as error:
                self._on_listener_failure(listener.owner, "serial_failure", error)
                raise
        return None

    async def parallel(self, key: ParallelEventKey[P], payload: P) -> None:
        listeners = self._active_listeners(cast(EventKey, key))
        tasks = [
            asyncio.create_task(
                _run_parallel_listener(listener, key.name, payload),
                name=f"plugin-event:{key.name}:{listener.owner.name}",
            )
            for listener in listeners
        ]
        if not tasks:
            return
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError as cancellation:
            for task in tasks:
                _ = task.cancel()
            await _drain_tasks(tasks)
            raise cancellation
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup(f"并发事件失败: {key.name}", errors)

    async def transform(self, key: TransformEventKey[P], payload: P) -> P:
        """按注册顺序组合显式同类型变换。"""

        current = payload
        for listener in self._active_listeners(cast(EventKey, key)):
            try:
                with _listener_boundary(listener, "transform", key.name):
                    result = listener.callback(current)
                    if inspect.isawaitable(result):
                        result = await result
                    if (
                        result is None
                        or isinstance(result, Bail)
                        or not isinstance(result, key.payload_type)
                    ):
                        raise CompositionError(
                            "INVALID_TRANSFORM_RESULT",
                            f"变换事件 {key.name} 的 listener 必须返回 "
                            f"{key.payload_type.__name__}",
                        )
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                self._on_listener_failure(listener.owner, "transform_failure", error)
                raise
            except BaseException as error:
                self._on_listener_failure(listener.owner, "transform_failure", error)
                raise
            current = cast(P, result)
        return current

    async def observe(self, key: ObserveEventKey[P], payload: P) -> None:
        """调用全部 observer，并把各自失败隔离为 Incident。"""

        # 1. 冻结并调用完整 observer 列表，不让异步 body 改变后续调用顺序。
        awaitables: list[tuple[_Listener, object, PluginOperation | None]] = []
        try:
            for listener in self._active_listeners(cast(EventKey, key)):
                operation = _start_listener_operation(
                    listener,
                    "observe",
                    key.name,
                )
                try:
                    if operation is None:
                        result = listener.callback(payload)
                    else:
                        with operation.bind():
                            result = listener.callback(payload)
                except asyncio.CancelledError as error:
                    if operation is not None:
                        operation.finish(error)
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    self._on_listener_failure(
                        listener.owner,
                        "observer_failure",
                        error,
                    )
                    continue
                except Exception as error:
                    if operation is not None:
                        operation.finish(error)
                    self._on_listener_failure(
                        listener.owner,
                        "observer_failure",
                        error,
                    )
                    continue
                except BaseException as error:
                    if operation is not None:
                        operation.finish(error)
                    raise
                if inspect.isawaitable(result):
                    awaitables.append((listener, result, operation))
                elif operation is not None:
                    operation.finish()
        except BaseException as terminal:
            for listener, error in _close_unstarted_observers(
                [(listener, result) for listener, result, _ in awaitables]
            ):
                self._on_listener_failure(
                    listener.owner,
                    "observer_cleanup_failure",
                    error,
                )
            for _, _, operation in awaitables:
                if operation is not None:
                    operation.finish(terminal)
            raise

        # 2. 全部 callback 已调用后再统一启动并等待异步 observer。
        pending = [
            (
                listener,
                asyncio.create_task(
                    _capture_observer_failure(result, operation),
                    name=f"plugin-observer:{key.name}:{listener.owner.name}",
                ),
            )
            for listener, result, operation in awaitables
        ]
        if not pending:
            return
        task_owners = {task: listener for listener, task in pending}
        remaining = set(task_owners)
        try:
            while remaining:
                done, remaining = await asyncio.wait(
                    remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    error = task.result()
                    if error is None:
                        continue
                    if not isinstance(error, (Exception, asyncio.CancelledError)):
                        await _cancel_observer_tasks(list(remaining))
                        raise error
                    self._on_listener_failure(
                        task_owners[task].owner,
                        "observer_failure",
                        error,
                    )
        except asyncio.CancelledError as cancellation:
            await _cancel_observer_tasks(list(remaining))
            raise cancellation

    def registrations(self) -> tuple[str, ...]:
        return tuple(
            f"{_event_descriptor(key)}:{listener.owner.name}"
            for key, listeners in self._listeners.items()
            for listener in listeners
        )

    def _active_listeners(self, key: EventKey) -> tuple[_Listener, ...]:
        return tuple(
            listener
            for listener in self._listeners.get(key, ())
            if listener.owner.state == FiberState.ACTIVE
        )


async def _run_parallel_listener(
    listener: _Listener,
    event_name: str,
    payload: object,
) -> None:
    with _listener_boundary(listener, "parallel", event_name):
        result = listener.callback(payload)
        if not inspect.isawaitable(result):
            raise CompositionError(
                "SYNC_RESULT_FROM_PARALLEL",
                "并发事件 listener 没有返回 awaitable",
            )
        _ = await result


async def _capture_observer_failure(
    result: object,
    operation: PluginOperation | None,
) -> BaseException | None:
    assert inspect.isawaitable(result)
    try:
        if operation is None:
            _ = await result
        else:
            with operation.bind():
                _ = await result
    except BaseException as error:
        if operation is not None:
            operation.finish(error)
        return error
    if operation is not None:
        operation.finish()
    return None


def _listener_boundary(
    listener: _Listener,
    mode: str,
    event_name: str,
) -> ContextManager[object]:
    runtime = listener.owner.runtime
    if runtime is None:
        return nullcontext()
    return plugin_entrypoint(
        plugin_id=runtime.plugin_id,
        generation_id=runtime.generation_id,
        fiber=listener.owner.name,
        operation=f"event.{mode}",
        entrypoint=event_name,
    )


def _start_listener_operation(
    listener: _Listener,
    mode: str,
    event_name: str,
) -> PluginOperation | None:
    runtime = listener.owner.runtime
    if runtime is None:
        return None
    return start_plugin_entrypoint(
        plugin_id=runtime.plugin_id,
        generation_id=runtime.generation_id,
        fiber=listener.owner.name,
        operation=f"event.{mode}",
        entrypoint=event_name,
    )


async def _cancel_observer_tasks(
    tasks: list[asyncio.Task[BaseException | None]],
) -> None:
    for task in tasks:
        if not task.done():
            _ = task.cancel()
    if not tasks:
        return
    drain = asyncio.gather(*tasks, return_exceptions=True)
    while not drain.done():
        try:
            _ = await asyncio.shield(drain)
        except asyncio.CancelledError:
            continue
    _ = drain.result()


async def _drain_tasks(tasks: list[asyncio.Task[None]]) -> None:
    """Drain cancelled listeners despite repeated caller cancellation."""

    drain = asyncio.gather(*tasks, return_exceptions=True)
    while not drain.done():
        try:
            _ = await asyncio.shield(drain)
        except asyncio.CancelledError:
            continue
    _ = drain.result()


def _is_async_callable(callback: EventListener) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


def _close_unexpected_awaitable(result: object) -> None:
    if inspect.iscoroutine(result):
        result.close()
    elif isinstance(result, asyncio.Future):
        _ = result.cancel()


def _close_unstarted_observers(
    observers: list[tuple[_Listener, object]],
) -> list[tuple[_Listener, BaseException]]:
    failures: list[tuple[_Listener, BaseException]] = []
    for listener, result in observers:
        try:
            if isinstance(result, asyncio.Future):
                _ = result.cancel()
                continue
            close = getattr(result, "close", None)
            if callable(close):
                _ = close()
        except BaseException as error:
            failures.append((listener, error))
    return failures


def _event_descriptor(key: EventKey) -> str:
    if isinstance(key, EmitEventKey):
        return f"emit:{key.name}"
    if isinstance(key, SerialEventKey):
        suffix = (
            f"[bail={key.bail_contract}]"
            if key.bail_contract is not None
            else ""
        )
        return f"serial:{key.name}{suffix}"
    if isinstance(key, ParallelEventKey):
        return f"parallel:{key.name}"
    if isinstance(key, TransformEventKey):
        return f"transform:{key.name}[{key.payload_contract}]"
    return f"observe:{key.name}"
