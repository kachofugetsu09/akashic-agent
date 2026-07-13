from __future__ import annotations

import asyncio
import inspect
import logging
import subprocess
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from bus.event_bus import EventBus, EventSubscription, Handler

logger = logging.getLogger(__name__)

T = TypeVar("T")
Cleanup = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True)
class CleanupFailure:
    resource: str
    error: str


class PluginScope:
    def __init__(self, plugin_id: str) -> None:
        self.plugin_id = plugin_id
        self._cleanups: list[tuple[str, Cleanup]] = []
        self._closed = False

    @property
    def resource_count(self) -> int:
        return len(self._cleanups)

    @property
    def closed(self) -> bool:
        return self._closed

    def defer(self, resource: str, cleanup: Cleanup) -> None:
        self._ensure_open()
        if not callable(cleanup):
            raise TypeError(f"插件清理动作不可调用: {self.plugin_id}:{resource}")
        self._cleanups.append((resource, cleanup))

    def subscribe(
        self,
        event_bus: EventBus,
        event_type: type[T],
        handler: Handler[T],
    ) -> EventSubscription:
        self._ensure_open()
        subscription = event_bus.on(event_type, handler)
        self.defer(
            f"event:{event_type.__name__}",
            subscription.close,
        )
        return subscription

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        if self._closed:
            coroutine.close()
            self._ensure_open()
        task = asyncio.create_task(coroutine, name=name)

        def report_failure(completed: asyncio.Task[T]) -> None:
            if completed.cancelled():
                return
            error = completed.exception()
            if error is None:
                return
            logger.error(
                "插件作用域任务异常: plugin=%s task=%s",
                self.plugin_id,
                completed.get_name(),
                exc_info=(type(error), error, error.__traceback__),
            )

        task.add_done_callback(report_failure)

        async def cancel() -> None:
            if not task.done():
                _ = task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return

        self.defer(f"task:{name or task.get_name()}", cancel)
        return task

    def track_async_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        name: str,
        timeout: float = 5,
    ) -> None:
        async def terminate() -> None:
            """终止异步进程并在竞态或超时时完成有限等待。"""

            # 1. 进程已退出时仍 wait 一次完成 transport 收尾
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    _ = await asyncio.wait_for(process.wait(), timeout=timeout)
                    return
            else:
                _ = await asyncio.wait_for(process.wait(), timeout=timeout)
                return

            # 2. 先等待优雅退出
            try:
                _ = await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                # 3. 超时后强杀，并保持二次等待有界
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                _ = await asyncio.wait_for(process.wait(), timeout=timeout)

        self.defer(f"process:{name}", terminate)

    def track_process(
        self,
        process: subprocess.Popen[Any],
        *,
        name: str,
        timeout: float = 5,
    ) -> None:
        async def terminate() -> None:
            """终止同步进程并在竞态或超时时完成有限等待。"""

            # 1. 进程已退出时 poll 已完成回收
            if process.poll() is not None:
                return
            try:
                process.terminate()
            except ProcessLookupError:
                if process.poll() is None:
                    raise

            # 2. 先等待优雅退出
            try:
                _ = await asyncio.to_thread(process.wait, timeout)
            except subprocess.TimeoutExpired:
                # 3. 超时后强杀，并保持二次等待有界
                try:
                    process.kill()
                except ProcessLookupError:
                    if process.poll() is None:
                        raise
                _ = await asyncio.to_thread(process.wait, timeout)

        self.defer(f"process:{name}", terminate)

    async def aclose(self) -> list[CleanupFailure]:
        """按逆序完成全部资源清理，并在末尾恢复外部取消。"""

        # 1. 关闭入口只消费一次，后续调用保持幂等
        if self._closed:
            return []
        self._closed = True
        failures: list[CleanupFailure] = []
        current = asyncio.current_task()
        externally_cancelled = current is not None and current.cancelling() > 0

        # 2. 每个 cleanup 脱离调用方取消，保证当前资源完成后再处理下一个
        while self._cleanups:
            resource, cleanup = self._cleanups.pop()

            async def run_cleanup() -> None:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result

            cleanup_task = asyncio.create_task(
                run_cleanup(),
                name=f"plugin_cleanup:{self.plugin_id}:{resource}",
            )
            while not cleanup_task.done():
                try:
                    _ = await asyncio.wait({cleanup_task})
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling() > 0:
                        externally_cancelled = True
                    continue
            try:
                await cleanup_task
            except (asyncio.CancelledError, Exception) as error:
                error_text = str(error) or type(error).__name__
                failure = CleanupFailure(resource=resource, error=error_text)
                failures.append(failure)
                logger.warning(
                    "插件资源清理失败: plugin=%s resource=%s error=%s",
                    self.plugin_id,
                    resource,
                    error_text,
                )
            current = asyncio.current_task()
            externally_cancelled = externally_cancelled or (
                current is not None and current.cancelling() > 0
            )

        # 3. 所有资源处理完后才恢复原始取消语义
        if externally_cancelled:
            raise asyncio.CancelledError
        return failures

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"插件作用域已关闭: {self.plugin_id}")


class ScopedEventBus:
    def __init__(
        self,
        event_bus: EventBus,
        scope: PluginScope,
        *,
        staged: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._scope = scope
        self._active = not staged
        self._snapshot_managed = staged
        self._pending: list[_StagedEventSubscription[Any]] = []

    def on(
        self,
        event_type: type[T],
        handler: Handler[T],
    ) -> EventSubscription | _StagedEventSubscription[T]:
        if self._snapshot_managed:
            if self._active:
                raise RuntimeError("插件事件订阅只能在 initialize 中注册")
            subscription = _StagedEventSubscription(
                self._event_bus,
                event_type,
                handler,
            )
            self._pending.append(subscription)
            self._scope.defer(
                f"event:{event_type.__name__}",
                subscription.close,
            )
            return subscription
        return self._scope.subscribe(self._event_bus, event_type, handler)

    def publish(self) -> None:
        if not self._snapshot_managed:
            return
        self._active = True

    def staged_handlers(self) -> tuple[tuple[type[object], Handler[object]], ...]:
        return tuple(
            (subscription.event_type, subscription.dispatch)
            for subscription in self._pending
            if subscription.active
        )

    def activate(self) -> None:
        if self._active:
            return
        self._active = True
        for subscription in self._pending:
            subscription.activate()
        self._pending.clear()

    async def emit(self, event: T) -> T:
        self._ensure_active()
        return await self._event_bus.emit(event)

    async def observe(self, event: object) -> None:
        self._ensure_active()
        await self._event_bus.observe(event)

    async def fanout(self, event: object) -> None:
        self._ensure_active()
        await self._event_bus.fanout(event)

    def enqueue(self, event: object) -> None:
        self._ensure_active()
        self._event_bus.enqueue(event)

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("候选插件尚未发布，不能发送事件")


class _StagedEventSubscription(Generic[T]):
    def __init__(
        self,
        event_bus: EventBus,
        event_type: type[T],
        handler: Handler[T],
    ) -> None:
        self._event_bus = event_bus
        self._event_type = event_type
        self._handler = handler
        self._subscription: EventSubscription | None = None
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def event_type(self) -> type[object]:
        return self._event_type

    async def dispatch(self, event: object) -> object | None:
        if not self._active:
            return None
        result = self._handler(cast(T, event))
        if inspect.isawaitable(result):
            return await result
        return result

    def activate(self) -> None:
        if not self._active or self._subscription is not None:
            return
        self._subscription = self._event_bus.on(self._event_type, self._handler)

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._subscription is not None:
            self._subscription.close()
