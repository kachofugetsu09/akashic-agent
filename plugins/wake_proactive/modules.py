from __future__ import annotations

from proactive_v2.frame import ProactiveFrame, ProactiveTickResult


_STATE_SLOT = "wake:run_state"


class WakeStartModule:
    slot = "wake.start"
    produces = (_STATE_SLOT,)

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        frame.slots[_STATE_SLOT] = self._runtime.begin(frame)  # type: ignore[attr-defined]
        return frame

    async def stop(self) -> None:
        self._runtime.close()  # type: ignore[attr-defined]


class WakeIngestModule:
    slot = "wake.ingest"
    requires = (_STATE_SLOT,)
    produces = ("wake:reservoir",)

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = frame.slots[_STATE_SLOT]
        await self._runtime.ingest(state)  # type: ignore[attr-defined]
        frame.slots["wake:reservoir"] = state
        return frame


class WakeContentDecisionModule:
    slot = "wake.content.decide"
    requires = ("wake:reservoir",)
    produces = ("wake:content_result",)

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = frame.slots[_STATE_SLOT]
        state.content_completed = await self._runtime.decide_content(state)  # type: ignore[attr-defined]
        frame.slots["wake:content_result"] = state
        return frame


class WakeDriftDecisionModule:
    slot = "wake.drift.decide"
    requires = ("wake:content_result",)
    produces = ("wake:result",)

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = frame.slots[_STATE_SLOT]
        if not state.content_completed:
            await self._runtime.decide_drift(state)  # type: ignore[attr-defined]
        frame.output = ProactiveTickResult(base_score=state.base_score)
        frame.slots["wake:result"] = state
        return frame


class WakeScheduleModule:
    slot = "wake.schedule"
    requires = ("wake:result",)
    produces = ("run:next_wakeup",)

    def __init__(self, runtime: object) -> None:
        self._runtime = runtime

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        state = frame.slots[_STATE_SLOT]
        interval = self._runtime.next_interval(state)  # type: ignore[attr-defined]
        if frame.output is None:
            frame.output = ProactiveTickResult(base_score=state.base_score)
        frame.output.next_interval_seconds = interval
        frame.slots["run:next_wakeup"] = interval
        return frame


def build_wake_runtime_modules(runtime: object) -> list[object]:
    return [
        WakeStartModule(runtime),
        WakeIngestModule(runtime),
        WakeScheduleModule(runtime),
    ]


def build_wake_content_modules(runtime: object) -> list[object]:
    return [WakeContentDecisionModule(runtime)]


def build_wake_drift_modules(runtime: object) -> list[object]:
    return [WakeDriftDecisionModule(runtime)]
