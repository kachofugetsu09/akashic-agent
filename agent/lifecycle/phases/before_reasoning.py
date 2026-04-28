from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from agent.core.passive_support import predict_current_user_source_ref
from agent.core.types import ContextRequest
from agent.lifecycle.phase import Phase, PhaseFrame, PhaseModule
from agent.lifecycle.types import BeforeReasoningCtx, BeforeReasoningInput
from bus.event_bus import EventBus

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.tools.registry import ToolRegistry
    from session.manager import SessionManager


@dataclass
class BeforeReasoningFrame(PhaseFrame[BeforeReasoningInput, BeforeReasoningCtx]):
    pass


BeforeReasoningModules: TypeAlias = list[PhaseModule[BeforeReasoningFrame]]


_CTX_SLOT = "reasoning:ctx"


class _SyncToolContextModule:
    def __init__(
        self,
        tools: ToolRegistry,
        session_manager: SessionManager,
    ) -> None:
        self._tools = tools
        self._session_manager = session_manager

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        state = frame.input.state
        before_turn = frame.input.before_turn
        if state.session is None:
            raise RuntimeError("BeforeReasoning requires TurnState.session")
        self._tools.set_context(
            channel=before_turn.channel,
            chat_id=before_turn.chat_id,
            current_user_source_ref=predict_current_user_source_ref(
                session_manager=self._session_manager,
                session=state.session,
            ),
        )
        return frame


class _BuildBeforeReasoningCtxModule:
    produces = (_CTX_SLOT,)

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        before_turn = frame.input.before_turn
        frame.slots[_CTX_SLOT] = BeforeReasoningCtx(
            session_key=before_turn.session_key,
            channel=before_turn.channel,
            chat_id=before_turn.chat_id,
            content=before_turn.content,
            timestamp=before_turn.timestamp,
            skill_names=list(before_turn.skill_names),
            retrieved_memory_block=before_turn.retrieved_memory_block,
        )
        return frame


class _EmitBeforeReasoningCtxModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        ctx = cast(BeforeReasoningCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _PromptWarmupModule:
    requires = (_CTX_SLOT,)

    def __init__(self, context: ContextBuilder) -> None:
        self._context = context

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        ctx = cast(BeforeReasoningCtx, frame.slots[_CTX_SLOT])
        _ = self._context.render(
            ContextRequest(
                history=[],
                current_message="",
                skill_names=ctx.skill_names,
                channel=ctx.channel,
                chat_id=ctx.chat_id,
                message_timestamp=ctx.timestamp,
                retrieved_memory_block=ctx.retrieved_memory_block,
            )
        )
        return frame


class _ReturnBeforeReasoningCtxModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: BeforeReasoningFrame) -> BeforeReasoningFrame:
        frame.output = cast(BeforeReasoningCtx, frame.slots[_CTX_SLOT])
        return frame


def default_before_reasoning_modules(
    bus: EventBus,
    tools: ToolRegistry,
    session_manager: SessionManager,
    context: ContextBuilder,
) -> BeforeReasoningModules:
    return [
        _SyncToolContextModule(tools, session_manager),
        _BuildBeforeReasoningCtxModule(),
        _EmitBeforeReasoningCtxModule(bus),
        _PromptWarmupModule(context),
        _ReturnBeforeReasoningCtxModule(),
    ]


class BeforeReasoningPhase(
    Phase[BeforeReasoningInput, BeforeReasoningCtx, BeforeReasoningFrame]
):
    def __init__(
        self,
        bus: EventBus,
        tools: ToolRegistry,
        session_manager: SessionManager,
        context: ContextBuilder,
    ) -> None:
        super().__init__(
            default_before_reasoning_modules(bus, tools, session_manager, context)
        )

    def _build_frame(self, input: BeforeReasoningInput) -> BeforeReasoningFrame:
        return BeforeReasoningFrame(input=input)
