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
# 公开插件 API。Host cleanup 保持逆序、聚合失败，并在清理完成后恢复调用方取消。
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
