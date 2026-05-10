from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias, cast

from bus.event_bus import EventBus
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    collect_prefixed_slots,
    topo_sort_modules,
)
from agent.lifecycle.types import AfterStepCtx


@dataclass
class AfterStepFrame(PhaseFrame[AfterStepCtx, AfterStepCtx]):
    pass


AfterStepModules: TypeAlias = list[PhaseModule[AfterStepFrame]]


_CTX_SLOT = "step:ctx"
_TELEMETRY_PREFIX = "step:telemetry:"
_COLLECTED_TELEMETRY_SLOT = "step:telemetry_collected"
_EARLY_STOP_REASON_SLOT = "step:early_stop_reason"


class _CopyInputToCtxModule:
    slot = "after_step.copy_input"
    requires: tuple[str, ...] = ()
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        frame.slots[_CTX_SLOT] = frame.input
        return frame


class _FanoutAfterStepCtxModule:
    slot = "after_step.fanout"
    requires = ("after_step.collect_pre", _CTX_SLOT)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        ctx = cast(AfterStepCtx, frame.slots[_CTX_SLOT])
        await self._bus.fanout(ctx)
        return frame


class _CollectAfterStepExportSlotsModule:
    produces = (_CTX_SLOT,)

    # 同一个模块类要在 fanout 前后各实例化一次，所以 slot / requires 由实例注入。
    def __init__(self, *, slot: str, requires: tuple[str, ...]) -> None:
        self.slot = slot
        self.requires = requires

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
        early_stop_reason = frame.slots.get(_EARLY_STOP_REASON_SLOT)
        if isinstance(early_stop_reason, str) and early_stop_reason.strip():
            frame.slots[_CTX_SLOT] = replace(
                ctx,
                early_stop=True,
                early_stop_reason=early_stop_reason.strip(),
                extra_metadata=extra_metadata,
            )
        else:
            frame.slots[_CTX_SLOT] = replace(ctx, extra_metadata=extra_metadata)
        frame.slots[_COLLECTED_TELEMETRY_SLOT] = collected | set(new_exports)
        return frame


class _ReturnAfterStepCtxModule:
    slot = "after_step.return"
    requires = ("after_step.collect_post", _CTX_SLOT)

    async def run(self, frame: AfterStepFrame) -> AfterStepFrame:
        frame.output = cast(AfterStepCtx, frame.slots[_CTX_SLOT])
        return frame


def default_after_step_modules(
    bus: EventBus,
    plugin_modules: AfterStepModules | None = None,
) -> AfterStepModules:
    builtins: AfterStepModules = [
        _CopyInputToCtxModule(),
        # collect 两次：fanout 前给 handler 读，fanout 后把 after_fanout 的补充带回返回 ctx。
        _CollectAfterStepExportSlotsModule(
            slot="after_step.collect_pre",
            requires=("after_step.copy_input", _CTX_SLOT),
        ),
        _FanoutAfterStepCtxModule(bus),
        _CollectAfterStepExportSlotsModule(
            slot="after_step.collect_post",
            requires=("after_step.fanout", _CTX_SLOT),
        ),
        _ReturnAfterStepCtxModule(),
    ]
    return cast(
        AfterStepModules,
        topo_sort_modules(builtins + list(plugin_modules or [])),
    )
