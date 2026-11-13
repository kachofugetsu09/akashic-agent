from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.model import CompositionError, FiberState

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context, Fiber

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
            not self.bail_contract or self.bail_contract.strip() != self.bail_contract
        ):
            raise ValueError("串行 Bail contract 必须是非空且无首尾空白的字符串")


@dataclass(frozen=True, slots=True)
class Bail(Generic[R]):
    value: R


EventKey = EmitEventKey[object] | SerialEventKey[object, object]
EventListener = Callable[[object], object]
ListenerFailureHandler = Callable[["Fiber", str, BaseException], None]


@dataclass(frozen=True, slots=True)
class _Listener:
    owner: "Fiber"
    callback: EventListener
    context: "Context"


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

        # 3. Registration order is the only listener order contract.
        self._contracts[key.name] = key
        listener = _Listener(owner=owner, callback=callback, context=owner.context)
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

    def emit(
        self,
        key: EmitEventKey[P],
        payload: P,
        *,
        plugin_ids: Collection[str] | None = None,
    ) -> None:
        for listener in self._active_listeners(
            cast(EventKey, key), plugin_ids=plugin_ids
        ):
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
        *,
        plugin_ids: Collection[str] | None = None,
    ) -> Bail[R] | None:
        return await self._run_serial(
            self._active_listeners(cast(EventKey, key), plugin_ids=plugin_ids),
            key,
            payload,
        )

    async def serial_for_owner(
        self,
        owner: "Fiber",
        key: SerialEventKey[P, R],
        payload: P,
    ) -> Bail[R] | None:
        """Core-only：只派发给准确 owner Fiber 的 listener。

        生命周期 dispatch 必须覆盖 LOADING/UNLOADING 中的 owner 自身，
        不走 `_active_listeners` 的 ACTIVE 过滤；顺序、Bail 校验与
        listener 失败语义与 `serial` 完全一致。
        """

        listeners = [
            listener
            for listener in self._listeners.get(cast(EventKey, key), ())
            if listener.owner is owner
        ]
        return await self._run_serial(listeners, key, payload)

    async def _run_serial(
        self,
        listeners: Iterable[_Listener],
        key: SerialEventKey[P, R],
        payload: P,
    ) -> Bail[R] | None:
        for listener in listeners:
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

    def registrations(
        self,
        *,
        plugin_ids: Collection[str] | None = None,
    ) -> tuple[str, ...]:
        """按稳定 descriptor 顺序冻结注册组。"""

        groups = sorted(
            self.registration_groups(plugin_ids=plugin_ids),
            key=lambda item: item[0],
        )
        return tuple(
            f"{descriptor}:{owner}"
            for descriptor, owners in groups
            for owner in owners
        )

    def registration_groups(
        self,
        *,
        plugin_ids: Collection[str] | None = None,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Group registrations by event while preserving Root insertion order."""

        return tuple(
            (
                _event_descriptor(key),
                tuple(
                    listener.owner.path
                    for listener in listeners
                    if plugin_ids is None
                    or (
                        listener.owner.runtime is not None
                        and listener.owner.runtime.plugin_id in plugin_ids
                    )
                ),
            )
            for key, listeners in self._listeners.items()
            if any(
                plugin_ids is None
                or (
                    listener.owner.runtime is not None
                    and listener.owner.runtime.plugin_id in plugin_ids
                )
                for listener in listeners
            )
        )

    def _active_listeners(
        self,
        key: EventKey,
        *,
        plugin_ids: Collection[str] | None = None,
    ) -> tuple[_Listener, ...]:
        return tuple(
            listener
            for listener in self._listeners.get(key, ())
            if listener.owner.state == FiberState.ACTIVE
            if plugin_ids is None
            or (
                listener.owner.runtime is not None
                and listener.owner.runtime.plugin_id in plugin_ids
            )
        )


@contextmanager
def _listener_boundary(
    listener: _Listener,
    mode: str,
    event_name: str,
) -> Iterator[object]:
    """回调始终使用注册时的 Context，不借换代后的 activation。"""
    runtime = listener.owner.runtime
    boundary = nullcontext() if runtime is None else plugin_entrypoint(
        plugin_id=runtime.plugin_id,
        generation_id=runtime.generation_id,
        fiber=listener.owner.path,
        operation=f"event.{mode}",
        entrypoint=event_name,
    )
    with listener.context.call_scope(), boundary:
        yield


def _is_async_callable(callback: EventListener) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


def _close_unexpected_awaitable(result: object) -> None:
    if inspect.iscoroutine(result):
        result.close()
    elif isinstance(result, asyncio.Future):
        _ = result.cancel()


def _event_descriptor(key: EventKey) -> str:
    if isinstance(key, EmitEventKey):
        return f"emit:{key.name}"
    suffix = f"[bail={key.bail_contract}]" if key.bail_contract is not None else ""
    return f"serial:{key.name}{suffix}"
