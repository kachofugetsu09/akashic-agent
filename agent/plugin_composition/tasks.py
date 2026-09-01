from __future__ import annotations

# pyright: reportPrivateUsage=false

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

from agent.plugin_composition.model import ServiceKey

R = TypeVar("R")
CancelCall = Callable[[], Awaitable[None] | None]
TaskCall = Callable[[], Awaitable[R]]


class TaskLease(Protocol):
    """The exact Root lease transferred to one claimed task."""

    @property
    def active(self) -> bool: ...

    async def run(self, call: TaskCall[R]) -> R: ...

    async def release(self) -> None: ...


@dataclass(slots=True)
class _TaskRecord:
    service_key: ServiceKey[object]
    scope_key: Hashable
    task_key: Hashable
    lease: TaskLease
    cancel: CancelCall
    task: asyncio.Task[object] | None = None
    cancel_requested: bool = False


class TaskWait(Generic[R]):
    """Wait for one task without taking its lease or cancel right."""

    __slots__ = ("_task",)

    def __init__(self, task: asyncio.Task[object]) -> None:
        self._task = task

    async def wait(self) -> R:
        return cast(R, await asyncio.shield(self._task))


class TaskStart:
    """Claim work for one Core-bound ServiceKey."""

    __slots__ = ("_control", "_service_key")

    def __init__(
        self,
        control: TaskControl,
        service_key: ServiceKey[object],
    ) -> None:
        self._control = control
        self._service_key = service_key

    def claim(
        self,
        scope_key: Hashable,
        task_key: Hashable,
        lease: TaskLease,
        run: TaskCall[R],
        cancel: CancelCall,
    ) -> TaskWait[R]:
        """Transfer one exact lease and start its task atomically."""

        return self._control._claim(
            self._service_key,
            scope_key,
            task_key,
            lease,
            run,
            cancel,
        )


class TaskCancel:
    """Cancel by opaque key without exposing task state or creation."""

    __slots__ = ("_control", "_service_key")

    def __init__(
        self,
        control: TaskControl,
        service_key: ServiceKey[object],
    ) -> None:
        self._control = control
        self._service_key = service_key

    async def cancel(self, task_key: Hashable) -> bool:
        return await self._control._cancel(self._service_key, task_key)


class TaskControl:
    """Own process-wide task claims, cancellation, and lease release."""

    def __init__(self) -> None:
        self._by_scope: dict[tuple[ServiceKey[object], Hashable], _TaskRecord] = {}
        self._by_task: dict[tuple[ServiceKey[object], Hashable], _TaskRecord] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def bind_start(self, service_key: ServiceKey[object]) -> TaskStart:
        return TaskStart(self, service_key)

    def bind_cancel(self, service_key: ServiceKey[object]) -> TaskCancel:
        return TaskCancel(self, service_key)

    def _claim(
        self,
        service_key: ServiceKey[object],
        scope_key: Hashable,
        task_key: Hashable,
        lease: TaskLease,
        run: TaskCall[R],
        cancel: CancelCall,
    ) -> TaskWait[R]:
        if self._closed:
            raise RuntimeError("Task control 已关闭")
        if not lease.active:
            raise RuntimeError("Task lease 已失效")
        _ = hash(scope_key)
        _ = hash(task_key)
        scope_id = (service_key, scope_key)
        task_id = (service_key, task_key)
        if scope_id in self._by_scope:
            raise RuntimeError("Task scope 已有 owner")
        if task_id in self._by_task:
            raise RuntimeError("Task key 已有 owner")

        _ = asyncio.get_running_loop()
        record = _TaskRecord(
            service_key=service_key,
            scope_key=scope_key,
            task_key=task_key,
            lease=lease,
            cancel=cancel,
        )
        self._by_scope[scope_id] = record
        self._by_task[task_id] = record
        try:
            task = asyncio.create_task(
                self._run_task(record, cast(TaskCall[object], run)),
                name=f"plugin-task:{service_key.name}",
            )
        except BaseException:
            _ = self._by_scope.pop(scope_id, None)
            _ = self._by_task.pop(task_id, None)
            raise
        record.task = task
        task.add_done_callback(_read_result)
        return TaskWait(task)

    async def _cancel(
        self,
        service_key: ServiceKey[object],
        task_key: Hashable,
    ) -> bool:
        record = self._by_task.get((service_key, task_key))
        if record is None or record.cancel_requested:
            return False
        record.cancel_requested = True
        try:
            result = record.cancel()
            if inspect.isawaitable(result):
                cancel_task = asyncio.ensure_future(result)
                await _wait_done(cancel_task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            record.cancel_requested = False
            raise
        return True

    async def _run_task(
        self,
        record: _TaskRecord,
        run: TaskCall[object],
    ) -> object:
        result: object | None = None
        task_error: BaseException | None = None
        try:
            result = await record.lease.run(run)
        except BaseException as error:
            task_error = error

        release_error: BaseException | None = None
        if record.lease.active:
            try:
                await _release_critical(record.lease)
            except BaseException as error:
                release_error = error
        self._remove(record)

        if task_error is not None and release_error is not None:
            raise BaseExceptionGroup(
                "Task 与 lease release 同时失败",
                [task_error, release_error],
            )
        if task_error is not None:
            raise task_error
        if release_error is not None:
            raise release_error
        return result

    def _remove(self, record: _TaskRecord) -> None:
        scope_id = (record.service_key, record.scope_key)
        task_id = (record.service_key, record.task_key)
        if self._by_scope.get(scope_id) is record:
            del self._by_scope[scope_id]
        if self._by_task.get(task_id) is record:
            del self._by_task[task_id]

    async def aclose(self) -> None:
        """Finish one shared close before passing caller cancellation back."""

        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                self._close(),
                name="task-control-close",
            )
        await _wait_done(self._close_task)

    async def _close(self) -> None:
        records = tuple(self._by_task.values())
        errors: list[BaseException] = []
        for record in records:
            try:
                _ = await self._cancel(record.service_key, record.task_key)
            except BaseException as error:
                errors.append(error)
        tasks = tuple(record.task for record in records if record.task is not None)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors.extend(
                result for result in results if isinstance(result, BaseException)
            )
        if errors:
            raise BaseExceptionGroup("Task control 关闭失败", errors)


async def _release_critical(lease: TaskLease) -> None:
    task = asyncio.create_task(lease.release(), name="task-lease-release")
    cancelled = False
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    if cancelled:
        raise asyncio.CancelledError


async def _wait_done(task: asyncio.Future[None]) -> None:
    cancelled = False
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    if cancelled:
        raise asyncio.CancelledError


def _read_result(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        _ = task.exception()


__all__ = [
    "TaskCancel",
    "TaskLease",
    "TaskStart",
    "TaskWait",
]
