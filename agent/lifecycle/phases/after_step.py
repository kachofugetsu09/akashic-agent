from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from bus.event_bus import EventBus
from agent.lifecycle.phase import PhaseFrame, PhaseModule
from agent.lifecycle.types import AfterStepCtx


@dataclass
class AfterStepFrame(PhaseFrame[AfterStepCtx, AfterStepCtx]):
    pass


AfterStepModules: TypeAlias = list[PhaseModule[AfterStepFrame]]


_CTX_SLOT = "step:ctx"


class _CopyInputToCtxModule:
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        frame.slots[_CTX_SLOT] = frame.input
        return frame


class _FanoutAfterStepCtxModule:
    requires = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        ctx = cast(AfterStepCtx, frame.slots[_CTX_SLOT])
        await self._bus.fanout(ctx)
        return frame


class _ReturnAfterStepCtxModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        frame.output = cast(AfterStepCtx, frame.slots[_CTX_SLOT])
        return frame


def default_after_step_modules(bus: EventBus) -> AfterStepModules:
    return [
        _CopyInputToCtxModule(),
        _FanoutAfterStepCtxModule(bus),
        _ReturnAfterStepCtxModule(),
    ]
