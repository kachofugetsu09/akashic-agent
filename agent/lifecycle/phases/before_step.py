from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

from agent.core.passive_support import (
    build_context_hint_message,
    estimate_messages_tokens,
)
from agent.lifecycle.phase import Phase, PhaseFrame, PhaseModule
from agent.lifecycle.types import BeforeStepCtx, BeforeStepInput
from bus.event_bus import EventBus


@dataclass
class BeforeStepFrame(PhaseFrame[BeforeStepInput, BeforeStepCtx]):
    pass


BeforeStepModules: TypeAlias = list[PhaseModule[BeforeStepFrame]]


_CTX_SLOT = "step:ctx"


class _BuildBeforeStepCtxModule:
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
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        ctx = cast(BeforeStepCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _InjectHintsModule:
    requires = (_CTX_SLOT,)

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


class _ReturnBeforeStepCtxModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: BeforeStepFrame) -> BeforeStepFrame:
        frame.output = cast(BeforeStepCtx, frame.slots[_CTX_SLOT])
        return frame


def default_before_step_modules(bus: EventBus) -> BeforeStepModules:
    return [
        _BuildBeforeStepCtxModule(),
        _EmitBeforeStepCtxModule(bus),
        _InjectHintsModule(),
        _ReturnBeforeStepCtxModule(),
    ]


class BeforeStepPhase(Phase[BeforeStepInput, BeforeStepCtx, BeforeStepFrame]):
    def __init__(self, bus: EventBus) -> None:
        super().__init__(default_before_step_modules(bus))

    def _build_frame(self, input: BeforeStepInput) -> BeforeStepFrame:
        return BeforeStepFrame(input=input)
