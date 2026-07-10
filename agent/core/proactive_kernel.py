from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable

from proactive_v2.frame import new_proactive_frame
from proactive_v2.frame import ProactiveTickResult
from proactive_v2.lifecycle import ProactiveLifecycleBuilder, ProactiveLifecycleSpec

logger = logging.getLogger(__name__)


class ProactiveKernel:
    def __init__(
        self,
        modules: Iterable[object],
        *,
        lifecycle: ProactiveLifecycleSpec,
        initial_slots_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._lifecycle = ProactiveLifecycleBuilder().build(lifecycle, modules)
        self._initial_slots_fn = initial_slots_fn
        self._last_result: ProactiveTickResult | None = None

    async def start(self) -> None:
        await self._lifecycle.start()

    async def stop(self) -> None:
        await self._lifecycle.stop()

    async def run_tick(self, session_key: str) -> float | None:
        result = await self.run_tick_result(session_key)
        return result.base_score if result is not None else None

    async def run_tick_result(
        self,
        session_key: str,
    ) -> ProactiveTickResult | None:
        initial_slots = (
            self._initial_slots_fn(session_key)
            if self._initial_slots_fn is not None
            else None
        )
        frame = await self._lifecycle.run(new_proactive_frame(session_key, initial_slots))
        self._last_result = frame.output
        return self._last_result

    @property
    def last_result(self) -> ProactiveTickResult | None:
        return self._last_result

    def inspect(self) -> str:
        return self._lifecycle.inspect()
