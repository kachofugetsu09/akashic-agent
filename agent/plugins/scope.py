from __future__ import annotations

import asyncio
import inspect
import logging
import subprocess
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

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

    def defer(self, resource: str, cleanup: Cleanup) -> None:
        self._ensure_open()
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
            if process.returncode is None:
                process.terminate()
            try:
                _ = await asyncio.wait_for(process.wait(), timeout=timeout)
            except TimeoutError:
                process.kill()
                _ = await process.wait()

        self.defer(f"process:{name}", terminate)

    def track_process(
        self,
        process: subprocess.Popen[Any],
        *,
        name: str,
        timeout: float = 5,
    ) -> None:
        async def terminate() -> None:
            if process.poll() is not None:
                return
            process.terminate()
            try:
                _ = await asyncio.to_thread(process.wait, timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                _ = await asyncio.to_thread(process.wait)

        self.defer(f"process:{name}", terminate)

    async def aclose(self) -> list[CleanupFailure]:
        if self._closed:
            return []
        self._closed = True
        failures: list[CleanupFailure] = []
        externally_cancelled = False
        while self._cleanups:
            resource, cleanup = self._cleanups.pop()
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except (asyncio.CancelledError, Exception) as error:
                failure = CleanupFailure(resource=resource, error=str(error))
                failures.append(failure)
                if isinstance(error, asyncio.CancelledError):
                    task = asyncio.current_task()
                    externally_cancelled = task is not None and task.cancelling() > 0
                logger.warning(
                    "插件资源清理失败: plugin=%s resource=%s error=%s",
                    self.plugin_id,
                    resource,
                    error,
                )
        if externally_cancelled:
            raise asyncio.CancelledError
        return failures

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"插件作用域已关闭: {self.plugin_id}")


class ScopedEventBus:
    def __init__(self, event_bus: EventBus, scope: PluginScope) -> None:
        self._event_bus = event_bus
        self._scope = scope

    def on(
        self,
        event_type: type[T],
        handler: Handler[T],
    ) -> EventSubscription:
        return self._scope.subscribe(self._event_bus, event_type, handler)

    async def emit(self, event: T) -> T:
        return await self._event_bus.emit(event)

    async def observe(self, event: object) -> None:
        await self._event_bus.observe(event)

    async def fanout(self, event: object) -> None:
        await self._event_bus.fanout(event)

    def enqueue(self, event: object) -> None:
        self._event_bus.enqueue(event)
