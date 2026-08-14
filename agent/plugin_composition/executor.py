from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from agent.plugin_composition.model import CompositionError, ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.context import Context

T = TypeVar("T", covariant=True)
_worker_state = threading.local()


@dataclass(frozen=True, slots=True)
class SyncTask(Generic[T]):
    name: str
    call: Callable[[], T]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("SyncTask.name 必须是非空且无首尾空白的字符串")
        if not callable(self.call):
            raise TypeError("SyncTask.call 必须可调用")


class ExecutorService:
    """Run explicit synchronous tasks in one bounded, lifecycle-owned pool."""

    name = "executor-service"
    inject: tuple[ServiceKey[object], ...] = ()

    def __init__(self, *, max_workers: int = 4) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers 必须大于 0")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="plugin-sync",
        )
        self._inflight: set[Future[object]] = set()
        self._inflight_lock = threading.Lock()
        self._closed = False

    async def apply(self, ctx: "Context") -> None:
        # 1. Provider removal runs before pool shutdown during reverse cleanup.
        _ = await ctx.effect(lambda: self.aclose, label="executor:pool")
        _ = await ctx.provide(EXECUTOR_SERVICE, self)

    async def parallel_sync(
        self,
        tasks: Iterable[SyncTask[T]],
    ) -> tuple[T, ...]:
        """Run pure synchronous tasks concurrently and preserve result order."""

        # 1. Submit the complete bounded batch before awaiting any result.
        if self._closed:
            raise CompositionError("EXECUTOR_CLOSED", "同步执行服务已经关闭")
        batch = tuple(tasks)
        futures = [self._submit(task) for task in batch]
        join_task = asyncio.create_task(
            _join_futures(futures),
            name="plugin-sync-join",
        )

        # 2. Caller cancellation stops queued work and joins running threads.
        try:
            results = await asyncio.shield(join_task)
        except asyncio.CancelledError as cancellation:
            for future in futures:
                _ = future.cancel()
            _ = await _finish_join(join_task)
            raise cancellation
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("同步并发任务失败", errors)
        return tuple(cast(T, result) for result in results)

    async def aclose(self) -> None:
        """Reject new work, cancel queued calls, and join running calls."""

        if self._closed:
            return
        self._closed = True
        with self._inflight_lock:
            inflight = tuple(self._inflight)
        for future in inflight:
            _ = future.cancel()
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    def _submit(self, task: SyncTask[T]) -> Future[object]:
        future = cast(
            Future[object],
            self._executor.submit(_run_sync_task, cast(SyncTask[object], task)),
        )
        with self._inflight_lock:
            self._inflight.add(future)
        future.add_done_callback(self._remove_inflight)
        return future

    def _remove_inflight(self, future: Future[object]) -> None:
        with self._inflight_lock:
            self._inflight.discard(future)


EXECUTOR_SERVICE = ServiceKey[ExecutorService]("executor")


def reject_executor_context_access() -> None:
    if getattr(_worker_state, "active", False):
        raise CompositionError(
            "CONTEXT_IN_SYNC_WORKER",
            "同步并发工作线程不能访问 Context 或 Fiber",
        )


def _run_sync_task(task: SyncTask[object]) -> object:
    _worker_state.active = True
    try:
        return task.call()
    finally:
        _worker_state.active = False


async def _join_futures(
    futures: list[Future[object]],
) -> list[object | BaseException]:
    wrapped = [asyncio.wrap_future(future) for future in futures]
    return await asyncio.gather(*wrapped, return_exceptions=True)


async def _finish_join(
    task: asyncio.Task[list[object | BaseException]],
) -> list[object | BaseException]:
    """Join worker futures despite repeated caller cancellation."""

    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
