from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext

from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.model import CompositionError

Cleanup = Callable[[], object]
EffectSetup = Callable[[], object]


class Effect:
    """Own setup output and make concurrent disposal join one cleanup."""

    def __init__(
        self,
        *,
        label: str,
        remove_from_owner: Callable[[Effect], None],
        plugin_id: str = "",
        generation_id: str = "",
        fiber: str = "",
        lifecycle_binder: Callable[[], AbstractContextManager[object]] | None = None,
        close_guard: Callable[[], object] | None = None,
    ) -> None:
        self.label = label
        self._remove_from_owner = remove_from_owner
        self._plugin_id = plugin_id
        self._generation_id = generation_id
        self._fiber = fiber
        # Core-only：owner Fiber 在创建时固定当次 Context 的窄 binder；
        # cleanup 的实际执行 Task 用它建立 (Context, Task) 生命周期借用。
        self._lifecycle_binder = lifecycle_binder
        # Core-only：在切换到 cleanup Task 前，由原 caller 做同步自等待检查。
        self._close_guard = close_guard
        self._cleanup: Cleanup | None = None
        self._ready = asyncio.Event()
        self._setup_task: asyncio.Task[object] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self, setup: EffectSetup) -> Effect:
        """Run setup after ownership is visible and roll it back on failure."""

        # 1. Capture setup ownership before user code can re-enter disposal.
        self._setup_task = asyncio.current_task()
        try:
            result = setup()
            if inspect.isawaitable(result):
                result = await result
            if result is not None and not callable(result):
                raise TypeError("effect setup 必须返回一个 cleanup 或 None")
            self._cleanup = result
        except BaseException:
            self._closed = True
            self._remove_from_owner(self)
            raise
        finally:
            self._ready.set()

        # 2. A reentrant disposer may already be waiting for setup to settle.
        if self._close_task is not None:
            await _join_cleanup(self._close_task)
        return self

    async def aclose(self) -> None:
        """Dispose once; concurrent callers await the same cleanup task."""

        if self._closed:
            return
        current = asyncio.current_task()
        if current is self._setup_task and not self._ready.is_set():
            raise CompositionError(
                "REENTRANT_EFFECT_WAIT",
                "effect setup 不能同步等待其 owner 完成卸载",
            )
        if self._close_guard is not None:
            self._close_guard()
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(
                self._close(),
                name=f"plugin-effect-close:{self.label}",
            )
        await _join_cleanup(self._close_task)

    async def _close(self) -> None:
        # 1. Setup may still be producing cleanup functions.
        _ = await self._ready.wait()
        if self._closed:
            return

        # 2. 关闭成功后才解除责任，失败保留同一句柄供显式重试。
        if self._cleanup is not None:
            boundary = (
                nullcontext()
                if not self._plugin_id
                else plugin_entrypoint(
                    plugin_id=self._plugin_id,
                    generation_id=self._generation_id,
                    fiber=self._fiber,
                    operation="lifecycle.cleanup",
                )
            )
            with boundary:
                lifecycle = (
                    nullcontext()
                    if self._lifecycle_binder is None
                    else self._lifecycle_binder()
                )
                with lifecycle:
                    result = self._cleanup()
                    if inspect.isawaitable(result):
                        await result
        self._cleanup = None
        self._closed = True
        self._remove_from_owner(self)


async def _join_cleanup(task: asyncio.Task[None]) -> None:
    """等待同一关闭操作，重复取消也不能中断资源 owner。"""

    if task is asyncio.current_task():
        raise CompositionError(
            "REENTRANT_CLEANUP_WAIT", "cleanup 不能等待自身关闭完成",
        )
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError
