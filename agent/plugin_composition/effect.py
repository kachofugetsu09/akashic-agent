from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from contextlib import nullcontext
from typing import cast

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
    ) -> None:
        self.label = label
        self._remove_from_owner = remove_from_owner
        self._plugin_id = plugin_id
        self._generation_id = generation_id
        self._fiber = fiber
        self._cleanups: list[Cleanup] = []
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
            await self._collect_result(result)
        except BaseException as setup_error:
            cleanup_errors = await self._run_cleanups()
            self._closed = True
            self._remove_from_owner(self)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "effect setup 与 rollback 同时失败",
                    [setup_error, *cleanup_errors],
                )
            raise
        finally:
            self._ready.set()

        # 2. A reentrant disposer may already be waiting for setup to settle.
        if self._close_task is not None:
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                await self._close_task
                raise
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
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close(),
                name=f"plugin-effect-close:{self.label}",
            )
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise

    async def _close(self) -> None:
        # 1. Setup may still be producing cleanup functions.
        _ = await self._ready.wait()
        if self._closed:
            return

        # 2. All collected cleanup is attempted in reverse order.
        errors = await self._run_cleanups()
        self._closed = True
        self._remove_from_owner(self)
        if errors:
            raise BaseExceptionGroup(f"effect cleanup 失败: {self.label}", errors)

    async def _collect_result(self, result: object) -> None:
        if inspect.isawaitable(result):
            await self._collect_result(await result)
            return
        if result is None:
            return
        if callable(result):
            self._cleanups.append(result)
            return
        if isinstance(result, AsyncIterable):
            async for item in cast(AsyncIterable[object], result):
                await self._collect_result(item)
            return
        if isinstance(result, Iterable) and not isinstance(
            result,
            (str, bytes, bytearray, dict),
        ):
            for item in cast(Iterable[object], result):
                await self._collect_result(item)
            return
        result_type = type(cast(object, result)).__name__
        raise TypeError(f"effect setup 返回了不支持的类型: {result_type}")

    async def _run_cleanups(self) -> list[BaseException]:
        errors: list[BaseException] = []
        while self._cleanups:
            cleanup = self._cleanups.pop()
            try:
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
                    result = cleanup()
                    if isinstance(result, Awaitable):
                        await result
            except BaseException as error:
                errors.append(error)
        return errors
