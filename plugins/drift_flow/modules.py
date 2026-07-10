from __future__ import annotations

from plugins.default_proactive.runtime import ProactiveFlowRuntime, get_run_state
from proactive_v2.frame import ProactiveFrame


class DriftFlowModule:
    slot = "drift.flow"
    requires = ("route:selected",)
    produces = ("proposal:drift",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        await self._runtime.drift(state)
        frame.slots["proposal:drift"] = state.feed if state.ctx.drift_entered else None
        return frame


def build_drift_flow_modules(runtime: ProactiveFlowRuntime) -> list[object]:
    return [DriftFlowModule(runtime)]
