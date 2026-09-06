from __future__ import annotations

# 同模块内的窄句柄共享生命周期状态，不向插件公开 registry 写入。
# pyright: reportPrivateUsage=false

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Generator, Awaitable, Callable, Hashable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import TypeVar, Protocol

from agent.plugin_composition.context import Context, RuntimeScope
from uuid import uuid4

from agent.plugin_composition.model import ServiceKey

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


class TaskBusy(RuntimeError):
    """相同 key 的旧任务尚未排空，不能开始新任务。"""


class StaleTask(RuntimeError):
    """短命 handle 已不属于当前活动任务。"""


class Task:
    """一次短命工作及其资源；不拥有 Message、逻辑 Turn 或持久执行身份。"""

    def __init__(
        self, operation: Callable[[Task], Awaitable[object]], admitted: asyncio.Event
    ):
        self.handle = uuid4().hex
        self._active = True
        self._cancel_requested = False
        self._running = False
        from agent.plugins.snapshot import get_current_runtime_lease

        lease = get_current_runtime_lease()
        self._scope = None if lease is None else RuntimeScope(lease.fork())
        self._cleanup: list[Callable[[], None]] = []
        self._task = asyncio.create_task(self._run(operation, admitted))

    @property
    def active(self) -> bool:
        return self._active

    @property
    def done(self) -> bool:
        return self._task.done()

    def on_close(self, cleanup: Callable[[], None]) -> None:
        """注册写入权等同步撤销动作，取消接纳时立即失效。"""
        if not self._active:
            raise StaleTask("任务已结束，不能登记资源")
        self._cleanup.append(cleanup)

    def _close(self) -> None:
        if not self._active:
            return
        self._active = False
        failures: list[Exception] = []
        for cleanup in reversed(self._cleanup):
            try:
                cleanup()
            except Exception as exc:
                failures.append(exc)
        self._cleanup.clear()
        if failures:
            raise ExceptionGroup("任务资源撤销失败", failures)

    def cancel(self) -> None:
        """先撤销提交权，再通知协作取消；外部效果由各 owner 如实结算。"""
        if self._cancel_requested:
            return
        self._cancel_requested = True
        try:
            self._close()
        finally:
            # 尚未运行的 coroutine 也要进入 finally，归还接纳时已取得的 generation lease。
            if self._running:
                _ = self._task.cancel()

    async def join(self) -> object:
        """等待真正排空；等待者取消不再次取消已开始结算的工作。"""
        return await asyncio.shield(self._task)

    async def _run(
        self, operation: Callable[[Task], Awaitable[object]], admitted: asyncio.Event
    ) -> object:
        self._running = True
        try:
            if self._cancel_requested:
                raise asyncio.CancelledError
            # eager task factory 也必须等同步准入成功，才能运行用户代码。
            _ = await admitted.wait()
            if self._scope is not None:
                await self._scope.__aenter__()
            return await operation(self)
        finally:
            try:
                self._close()
            finally:
                if self._scope is not None:
                    await self._scope.close()


class TaskSlot:
    """一个 key 在本次同步准入中的视图，离开回调后失效。"""

    def __init__(self, owner: Tasks, key: Hashable):
        self._owner = owner
        self._key = key
        self._active = True
        self._started: Task | None = None
        self._admitted = asyncio.Event()

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("Task 准入已结束")

    @property
    def current(self) -> Task | None:
        self._check_active()
        task = self._owner._tasks.get(self._key)
        return task if task is not None and not task.done else None

    def require(self, handle: str) -> Task:
        task = self.current
        if task is None or task.handle != handle or not task.active:
            raise StaleTask("handle 不属于当前活动任务")
        return task

    def start(self, operation: Callable[[Task], Awaitable[object]]) -> Task:
        self._check_active()
        if self.current is not None:
            raise TaskBusy("旧任务尚未排空")
        task = Task(operation, self._admitted)
        self._started = task
        self._owner._tasks[self._key] = task
        task._task.add_done_callback(lambda _: self._owner._release(self._key, task))
        return task


@dataclass
class _Group:
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    references: int = 0
    activity: int = 0
    exclusive: bool = False
    waiting: list[tuple[object, bool]] = field(default_factory=lambda: list[tuple[object, bool]]())

    def notify(self) -> None:
        previous = self.changed
        self.changed = asyncio.Event()
        previous.set()


class Tasks:
    """按通用 key 串行准入，保护启动、控制接纳和 effect start 的同一顺序。"""

    def __init__(self):
        self._tasks: dict[Hashable, Task] = {}
        self._groups: dict[Hashable, _Group] = {}
        self._groups_drained = asyncio.Event()
        self._groups_drained.set()
        self._closed = False

    def _release(self, key: Hashable, task: Task) -> None:
        if self._tasks.get(key) is task:
            del self._tasks[key]
        # 独立启动的工作也必须报告失败；join 仍会收到原异常。
        if not task._task.cancelled():
            error = task._task.exception()
            if error is not None:
                logger.error("Task 失败 handle=%s", task.handle, exc_info=error)

    async def admit(self, key: Hashable, callback: Callable[[TaskSlot], _T]) -> _T:
        """回调只做同步准入；长操作在 Task 中运行，不能持锁跨 I/O。"""
        # 成功准入和回调不跨 await；拒绝只在回调退出后等待自己的清理。
        if self._closed:
            raise RuntimeError("Task 服务已关闭")
        slot = TaskSlot(self, key)
        try:
            result = callback(slot)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Task 准入回调必须同步")
            slot._admitted.set()
            return result
        except BaseException as failure:
            if slot._started is not None:
                # 拒绝在回调退出后排空；调用方收到错误时不会留下占 key 的幽灵任务。
                task = slot._started
                cleanup_failures: list[BaseException] = []
                try:
                    task.cancel()
                except Exception as cleanup_failure:
                    cleanup_failures.append(cleanup_failure)
                caller = asyncio.current_task()
                cancellations = 0 if caller is None else caller.cancelling()
                interrupted = False
                while not task.done:
                    try:
                        _ = await task.join()
                    except asyncio.CancelledError:
                        interrupted = interrupted or (
                            caller is not None and caller.cancelling() > cancellations
                        )
                    except BaseException as cleanup_failure:
                        cleanup_failures.append(cleanup_failure)
                self._release(key, task)
                if cleanup_failures:
                    raise BaseExceptionGroup(
                        "Task 准入及撤销失败", [failure, *cleanup_failures]
                    ) from None
                if interrupted:
                    raise asyncio.CancelledError from failure
            raise
        finally:
            slot._active = False

    def _group(self, key: Hashable) -> _Group:
        if self._closed:
            raise RuntimeError("Task 服务已关闭")
        group = self._groups.setdefault(key, _Group())
        group.references += 1
        self._groups_drained.clear()
        return group

    def _release_group(self, key: Hashable, group: _Group) -> None:
        group.references -= 1
        if not group.references:
            del self._groups[key]
            if not self._groups:
                self._groups_drained.set()

    @contextmanager
    def activity(self, key: Hashable) -> Generator[None]:
        """标记一段活动；接纳与撤销不跨 await，新活动不等待已开始的排他工作。"""
        group = self._group(key)
        group.activity += 1
        group.notify()
        try:
            yield
        finally:
            group.activity -= 1
            group.notify()
            self._release_group(key, group)

    async def wait_idle(self, key: Hashable) -> None:
        """等待当前活动结束，不占用排他权，也不预留稍后的开始时机。"""
        group = self._group(key)
        try:
            while True:
                changed = group.changed
                if self._closed:
                    raise RuntimeError("Task 服务已关闭")
                if not group.activity:
                    return
                _ = await changed.wait()
        finally:
            self._release_group(key, group)

    @asynccontextmanager
    async def exclusive(self, key: Hashable, *, idle: bool = False) -> AsyncGenerator[None]:
        """按 FIFO 接纳排他工作；空闲检查与取得排他权共用同一同步准入段。"""
        group = self._group(key)
        request = (object(), idle)
        group.waiting.append(request)
        entered = False
        try:
            while True:
                changed = group.changed
                if self._closed:
                    raise RuntimeError("Task 服务已关闭")
                eligible = next((item for item in group.waiting if not item[1] or not group.activity), None)
                # 与 admit 相同：状态检查、变更和撤权都不跨 await。
                if not group.exclusive and eligible is request:
                    group.waiting.remove(request)
                    group.exclusive = True
                    entered = True
                    break
                _ = await changed.wait()
            yield
        finally:
            if entered:
                group.exclusive = False
            else:
                group.waiting.remove(request)
            group.notify()
            self._release_group(key, group)

    async def close(self) -> None:
        """停止准入，撤销并等待全部任务，保持真实失败可见。"""
        self._closed = True
        for group in self._groups.values():
            group.notify()
        tasks = tuple(self._tasks.values())
        failures: list[Exception] = []
        for task in tasks:
            try:
                task.cancel()
            except Exception as exc:
                failures.append(exc)
        results = await asyncio.gather(
            *(task.join() for task in tasks), return_exceptions=True
        )
        failures.extend(result for result in results if isinstance(result, Exception))
        self._tasks.clear()
        _ = await self._groups_drained.wait()
        if failures:
            raise ExceptionGroup("Task 关闭失败", failures)


class TaskAdmission(Protocol):
    async def admit(self, key: Hashable, callback: Callable[[TaskSlot], _T]) -> _T: ...

    def activity(self, key: Hashable) -> AbstractContextManager[None]: ...

    async def wait_idle(self, key: Hashable) -> None: ...

    def exclusive(self, key: Hashable, *, idle: bool = False) -> AbstractAsyncContextManager[None]: ...


class PluginTasks:
    """Core 按真实插件 owner 保留 Task 服务，热更新不会丢失同 owner 的活动工作。"""

    def __init__(self, *, formal: bool = True):
        self._formal = formal
        self._owners: dict[str, Tasks] = {}
        self._closed = False

    def start(self) -> None:
        if self._closed and self._owners:
            raise RuntimeError("旧插件 Task 尚未排空")
        self._closed = False

    def open(self, ctx: Context) -> TaskAdmission:
        if not self._formal or self._closed:
            raise RuntimeError("当前不能接纳正式 Task")
        owner = ctx.require_runtime_owner(TASKS, self)
        if owner not in self._owners:
            self._owners[owner] = Tasks()
        return self._owners[owner]

    async def close(self) -> None:
        """关闭属于本 runtime 的全部 owner，逐一排空并保留真实错误。"""
        self._closed = True
        results = await asyncio.gather(
            *(tasks.close() for tasks in self._owners.values()), return_exceptions=True
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("插件 Task 关闭失败", failures)
        self._owners.clear()


TASKS = ServiceKey[PluginTasks]("core.tasks")
