from __future__ import annotations

from plugins.default_proactive.runtime import ProactiveFlowRuntime, get_run_state
from proactive_v2.frame import ProactiveFrame


class ProactivePrepareModule:
    slot = "proactive.flow.prepare"
    requires = (
        "route:selected",
        "proposal:drift",
        "prompt:sections:collected",
    )
    produces = ("candidate:batch",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        self._runtime.prepare_proactive(state)
        frame.slots["candidate:batch"] = state.feed
        return frame


class ProactiveJudgeModule:
    slot = "proactive.flow.judge"
    requires = ("route:selected", "candidate:batch")
    produces = ("proposal:proactive",)

    def __init__(self, runtime: ProactiveFlowRuntime) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = get_run_state(frame)
        await self._runtime.judge(state)
        frame.slots["proposal:proactive"] = state.ctx.terminal_action
        return frame


def build_proactive_flow_modules(runtime: ProactiveFlowRuntime) -> list[object]:
    return [ProactivePrepareModule(runtime), ProactiveJudgeModule(runtime)]
