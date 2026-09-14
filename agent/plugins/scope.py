from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from agent.plugin_composition.diagnostics import plugin_entrypoint
Cleanup = Callable[[], Awaitable[None] | None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupFailure:
    resource: str
    error: str


# Core generation host 的资源作用域。V3 插件只使用 Context/Fiber/Effect；这个对象不属于
# 公开插件 API。资源按取得顺序形成依赖，失败时保留更早资源。
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
        self._close_task: asyncio.Task[list[CleanupFailure]] | None = None

    @property
    def closed(self) -> bool:
        return self._close_task is not None and not self._cleanups

    @property
    def accepting_resources(self) -> bool:
        """开始关闭后不能再次用于装配，即使仍有未释放资源。"""
        return self._close_task is None

    def defer(self, resource: str, cleanup: Cleanup) -> None:
        self._ensure_open()
        if not callable(cleanup):
            raise TypeError(f"插件清理动作不可调用: {self.plugin_id}:{resource}")
        self._cleanups.append((resource, cleanup))

    async def aclose(self) -> list[CleanupFailure]:
        """并发关闭加入同一操作；失败后的显式调用重试剩余资源。"""

        if asyncio.current_task() is self._close_task:
            raise RuntimeError("cleanup 不能等待其所属作用域关闭")

        # 1. 创建任务即关闭登记入口，成功释放的资源不再重放。
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(
                self._close(), name=f"plugin_scope_cleanup:{self.plugin_id}"
            )
        task = self._close_task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        failures = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return list(failures)

    async def _close(self) -> list[CleanupFailure]:
        """逆序释放资源，失败处停止以保留其依赖。"""

        while self._cleanups:
            resource, cleanup = self._cleanups[-1]
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
            try:
                with boundary:
                    result = cleanup()
                    if inspect.isawaitable(result):
                        await result
            except (asyncio.CancelledError, Exception) as error:
                error_text = str(error) or type(error).__name__
                logger.warning(
                    "插件资源清理失败: plugin=%s resource=%s error=%s",
                    self.plugin_id,
                    resource,
                    error_text,
                )
                return [CleanupFailure(resource=resource, error=error_text)]
            self._cleanups.pop()
        return []

    def _ensure_open(self) -> None:
        if not self.accepting_resources:
            raise RuntimeError(f"插件作用域已关闭: {self.plugin_id}")
