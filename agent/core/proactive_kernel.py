from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Callable

from proactive_v2.frame import new_proactive_frame
from proactive_v2.phases import ProactivePhaseRunner

logger = logging.getLogger(__name__)


class ProactiveKernel:
    def __init__(
        self,
        modules: Iterable[object],
        *,
        initial_slots_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._runner = ProactivePhaseRunner(modules)
        self._initial_slots_fn = initial_slots_fn

    async def start(self) -> None:
        for module in self._all_modules():
            starter = getattr(module, "start", None)
            if starter is not None:
                await starter()

    async def stop(self) -> None:
        for module in reversed(self._all_modules()):
            stopper = getattr(module, "stop", None)
            if stopper is not None:
                await stopper()

    async def run_tick(self, session_key: str) -> float | None:
        initial_slots = (
            self._initial_slots_fn(session_key)
            if self._initial_slots_fn is not None
            else None
        )
        frame = await self._runner.run(new_proactive_frame(session_key, initial_slots))
        if frame.output is None:
            return None
        return frame.output.base_score

    def inspect(self) -> str:
        return self._runner.inspect()

    def _all_modules(self) -> list[Any]:
        modules: list[Any] = []
        for phase_modules in self._runner.modules_by_phase.values():
            modules.extend(phase_modules)
        return modules
