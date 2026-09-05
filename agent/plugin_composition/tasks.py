from __future__ import annotations

# 同模块内的窄句柄共享生命周期状态，不向插件公开 registry 写入。
# pyright: reportPrivateUsage=false

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Hashable
from typing import TypeVar
from uuid import uuid4

from agent.plugin_composition.model import ServiceKey

_T = TypeVar("_T")


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
            _ = self._task.cancel()

    async def join(self) -> object:
        """等待真正排空；等待者取消不再次取消已开始结算的工作。"""
        return await asyncio.shield(self._task)

    async def _run(
        self, operation: Callable[[Task], Awaitable[object]], admitted: asyncio.Event
    ) -> object:
        try:
            # eager task factory 也必须等同步准入成功，才能运行用户代码。
            _ = await admitted.wait()
            return await operation(self)
        finally:
            self._close()


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


class Tasks:
    """按通用 key 串行准入，保护启动、控制接纳和 effect start 的同一顺序。"""

    def __init__(self):
        self._tasks: dict[Hashable, Task] = {}
        self._closed = False

    def _release(self, key: Hashable, task: Task) -> None:
        if self._tasks.get(key) is task:
            del self._tasks[key]

    async def admit(self, key: Hashable, callback: Callable[[TaskSlot], _T]) -> _T:
        """回调只做同步准入；长操作在 Task 中运行，不能持锁跨 I/O。"""
        # 本段没有 await；同一事件循环内，准入不会与其他任务交错。
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
                try:
                    slot._started.cancel()
                except Exception as cleanup_failure:
                    raise BaseExceptionGroup(
                        "Task 准入及撤销失败", [failure, cleanup_failure]
                    ) from None
            raise
        finally:
            slot._active = False

    async def close(self) -> None:
        """停止准入，撤销并等待全部任务，保持真实失败可见。"""
        self._closed = True
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
        if failures:
            raise ExceptionGroup("Task 关闭失败", failures)


TASKS = ServiceKey[Tasks]("core.tasks")
