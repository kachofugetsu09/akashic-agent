from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, TypeVar

from agent.plugin_composition.diagnostics import plugin_entrypoint
from bus.event_bus import EventBus, EventSubscription, Handler

logger = logging.getLogger(__name__)

T = TypeVar("T")
Cleanup = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True)
class CleanupFailure:
    resource: str
    error: str


# 插件资源接口：现有插件通过 context 或直接使用 scope 登记订阅、任务和 cleanup。
# 迁移插件前不得绕过逆序清理、聚合失败和清理完成后再恢复取消的语义。
class PluginScope:
    def __init__(
        self,
        plugin_id: str,
        *,
        generation_id: str = "",
        diagnostic_plugin_id: str = "",
    ) -> None:
        self.plugin_id = plugin_id
        self.generation_id = generation_id
        self._diagnostic_plugin_id = diagnostic_plugin_id or plugin_id
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
                boundary = (
                    nullcontext()
                    if not self.generation_id
                    else plugin_entrypoint(
                        plugin_id=self._diagnostic_plugin_id,
                        generation_id=self.generation_id,
                        fiber=self.plugin_id,
                        operation="lifecycle.cleanup",
                    )
                )
                with boundary:
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
