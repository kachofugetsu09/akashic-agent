from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from agent.core.passive_support import (
    build_context_hint_message,
    estimate_messages_tokens,
)
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    append_string_exports,
    collect_prefixed_slots,
    topo_sort_modules,
)
from agent.lifecycle.types import BeforeStepCtx, BeforeStepInput
from bus.event_bus import EventBus


@dataclass
class BeforeStepFrame(PhaseFrame[BeforeStepInput, BeforeStepCtx]):
    pass


BeforeStepModules: TypeAlias = list[PhaseModule[BeforeStepFrame]]


_CTX_SLOT = "step:ctx"
_EXTRA_HINT_PREFIX = "step:extra_hint:"
# slot suffix 统一用 abort_reply；step 内部映射为 early_stop，只终止当前 tool loop。
_ABORT_REPLY_SLOT = "step:abort_reply"


class _BuildBeforeStepCtxModule:
    slot = "before_step.build_ctx"
    requires: tuple[str, ...] = ()
    produces = (_CTX_SLOT,)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        input = frame.input
        frame.slots[_CTX_SLOT] = BeforeStepCtx(
            session_key=input.session_key,
            channel=input.channel,
            chat_id=input.chat_id,
            iteration=input.iteration,
            input_tokens_estimate=estimate_messages_tokens(input.messages),
            visible_tool_names=(
                frozenset(input.visible_names)
                if input.visible_names is not None
                else None
            ),
        )
        return frame


class _EmitBeforeStepCtxModule:
    slot = "before_step.emit"
    requires = ("before_step.build_ctx", _CTX_SLOT)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        ctx = cast(BeforeStepCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _InjectHintsModule:
    slot = "before_step.inject_hints"
    requires = ("before_step.collect_exports", _CTX_SLOT)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        ctx = cast(BeforeStepCtx, frame.slots[_CTX_SLOT])
        if ctx.extra_hints:
            frame.input.messages.append(
                build_context_hint_message(
                    "plugin_hints",
                    "\n".join(ctx.extra_hints),
                )
            )
        return frame


class _CollectBeforeStepExportSlotsModule:
    slot = "before_step.collect_exports"
    requires = ("before_step.emit", _CTX_SLOT)
    produces = (_CTX_SLOT,)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        ctx = cast(BeforeStepCtx, frame.slots[_CTX_SLOT])
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _EXTRA_HINT_PREFIX),
        )
        early_stop_reply = frame.slots.get(_ABORT_REPLY_SLOT)
        if isinstance(early_stop_reply, str) and early_stop_reply:
            ctx.early_stop = True
            ctx.early_stop_reply = early_stop_reply
        return frame


class _ReturnBeforeStepCtxModule:
    slot = "before_step.return"
    requires = ("before_step.inject_hints", _CTX_SLOT)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        frame.output = cast(BeforeStepCtx, frame.slots[_CTX_SLOT])
        return frame


def default_before_step_modules(
    bus: EventBus,
    plugin_modules: BeforeStepModules | None = None,
) -> BeforeStepModules:
    builtins: BeforeStepModules = [
        _BuildBeforeStepCtxModule(),
        _EmitBeforeStepCtxModule(bus),
        _CollectBeforeStepExportSlotsModule(),
        _InjectHintsModule(),
        _ReturnBeforeStepCtxModule(),
    ]
    return cast(
        BeforeStepModules,
        topo_sort_modules(builtins + list(plugin_modules or [])),
    )
