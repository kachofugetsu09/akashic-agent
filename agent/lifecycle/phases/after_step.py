from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias, cast

from bus.event_bus import EventBus
from agent.lifecycle.phase import PhaseFrame, PhaseModule, collect_prefixed_slots
from agent.lifecycle.types import AfterStepCtx


@dataclass
class AfterStepFrame(PhaseFrame[AfterStepCtx, AfterStepCtx]):
    pass


AfterStepModules: TypeAlias = list[PhaseModule[AfterStepFrame]]


_CTX_SLOT = "step:ctx"
_TELEMETRY_PREFIX = "step:telemetry:"
_COLLECTED_TELEMETRY_SLOT = "step:telemetry_collected"


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


class _CollectAfterStepExportSlotsModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        ctx = cast(AfterStepCtx, frame.slots[_CTX_SLOT])
        collected = set(cast(set[str], frame.slots.get(_COLLECTED_TELEMETRY_SLOT, set())))
        exports = collect_prefixed_slots(frame.slots, _TELEMETRY_PREFIX)
        # after_fanout 可补充 telemetry，但不能覆盖 fanout handler 已看到的同名值。
        new_exports = {
            key: value
            for key, value in exports.items()
            if key not in collected
        }
        extra_metadata = dict(ctx.extra_metadata)
        extra_metadata.update(new_exports)
        frame.slots[_CTX_SLOT] = replace(ctx, extra_metadata=extra_metadata)
        frame.slots[_COLLECTED_TELEMETRY_SLOT] = collected | set(new_exports)
        return frame


class _ReturnAfterStepCtxModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        frame.output = cast(AfterStepCtx, frame.slots[_CTX_SLOT])
        return frame


def default_after_step_modules(
    bus: EventBus,
    plugin_modules_before_fanout: AfterStepModules | None = None,
    plugin_modules_after_fanout: AfterStepModules | None = None,
) -> AfterStepModules:
    before_fanout = plugin_modules_before_fanout or []
    after_fanout = plugin_modules_after_fanout or []
    return [
        _CopyInputToCtxModule(),
        *before_fanout,
        # collect 两次：fanout 前给 handler 读，fanout 后把 after_fanout 的补充带回返回 ctx。
        _CollectAfterStepExportSlotsModule(),
        _FanoutAfterStepCtxModule(bus),
        *after_fanout,
        _CollectAfterStepExportSlotsModule(),
        _ReturnAfterStepCtxModule(),
    ]
