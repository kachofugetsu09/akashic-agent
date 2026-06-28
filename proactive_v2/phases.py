from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from agent.lifecycle.phase import inspect_phase, topo_sort_modules
from proactive_v2.frame import ProactiveFrame


PROACTIVE_PHASES: tuple[str, ...] = (
    "proactive.tick",
    "proactive.gate",
    "proactive.source",
    "proactive.drift",
    "proactive.prompt",
    "proactive.judge",
    "proactive.resolve",
    "proactive.deliver",
    "proactive.schedule",
)


class ProactivePhaseRunner:
    def __init__(self, modules: Iterable[object]) -> None:
        grouped: dict[str, list[object]] = defaultdict(list)
        for module in modules:
            phase = getattr(module, "phase", None)
            if not isinstance(phase, str) or not phase:
                raise RuntimeError(f"Proactive 模块缺少 phase 声明: {type(module).__name__}")
            if phase not in PROACTIVE_PHASES:
                raise RuntimeError(f"未知 proactive phase: {phase}")
            grouped[phase].append(module)
        self._modules_by_phase = {
            phase: topo_sort_modules(grouped.get(phase, []))
            for phase in PROACTIVE_PHASES
        }

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        for phase in PROACTIVE_PHASES:
            for module in self._modules_by_phase[phase]:
                runner = getattr(module, "run")
                frame = await runner(frame)
        return frame

    def inspect(self) -> str:
        sections: list[str] = []
        for phase in PROACTIVE_PHASES:
            modules = self._modules_by_phase[phase]
            if not modules:
                continue
            sections.append(f"[{phase}]\n{inspect_phase(modules)}")
        return "\n\n".join(sections)

    @property
    def modules_by_phase(self) -> dict[str, list[object]]:
        return {phase: list(modules) for phase, modules in self._modules_by_phase.items()}
