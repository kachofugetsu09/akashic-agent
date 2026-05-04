from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from agent.core.passive_support import update_session_runtime_metadata
from agent.core.response_parser import parse_response
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    append_string_exports,
    collect_prefixed_slots,
)
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterReasoningInput,
    AfterReasoningResult,
)
from bus.event_bus import EventBus
from bus.events import OutboundMessage

if TYPE_CHECKING:
    from agent.looping.ports import SessionServices
    from session.manager import Session

logger = logging.getLogger(__name__)


@dataclass
class AfterReasoningFrame(PhaseFrame[AfterReasoningInput, AfterReasoningResult]):
    pass


AfterReasoningModules: TypeAlias = list[PhaseModule[AfterReasoningFrame]]


_CTX_SLOT = "reasoning:ctx"
_OUTBOUND_SLOT = "reasoning:outbound"
_PERSIST_USER_PREFIX = "persist:user:"
_PERSIST_ASSISTANT_PREFIX = "persist:assistant:"
_OUTBOUND_METADATA_PREFIX = "outbound:metadata:"
_OUTBOUND_MEDIA_PREFIX = "outbound:media:"
_ASSISTANT_FIXED_FIELDS = {"tools_used", "tool_chain", "reasoning_content"}
_USER_FIXED_FIELDS = {"media"}


class _BuildAfterReasoningCtxModule:
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        input = frame.input
        msg = input.state.msg
        turn_result = input.turn_result
        raw_reply = turn_result.reply
        if raw_reply is None:
            raw_reply = "I've completed processing but have no response to give."
        tool_chain = cast(list[dict[str, object]], turn_result.tool_chain)
        parsed = parse_response(raw_reply, tool_chain=tool_chain)
        frame.slots[_CTX_SLOT] = AfterReasoningCtx(
            session_key=input.state.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            reply=parsed.clean_text,
            response_metadata=parsed.metadata,
            tools_used=tuple(turn_result.tools_used),
            tool_chain=tuple(tool_chain),
            thinking=turn_result.thinking,
            streamed=turn_result.streamed,
            context_retry=dict(turn_result.context_retry),
            outbound_metadata={
                **(msg.metadata or {}),
                **input.state.extra_metadata,
                "tools_used": list(turn_result.tools_used),
                "tool_chain": list(tool_chain),
                "context_retry": dict(turn_result.context_retry),
                "streamed_reply": turn_result.streamed,
            },
        )
        return frame


class _EmitAfterReasoningCtxModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _PersistUserMessageModule:
    requires = (_CTX_SLOT,)

    def __init__(self, session_services: SessionServices) -> None:
        self._session_services = session_services

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        state = frame.input.state
        msg = state.msg
        raw_session = state.session
        if raw_session is None:
            raise RuntimeError("AfterReasoning requires TurnState.session")
        session = cast("Session", raw_session)
        omit_user_turn = bool((msg.metadata or {}).get("omit_user_turn"))
        if omit_user_turn:
            return frame
        if self._session_services.presence:
            self._session_services.presence.record_user_message(session.key)
        user_kwargs: dict[str, object] = {}
        llm_user_content = ctx.context_retry.get("llm_user_content")
        if isinstance(llm_user_content, (str, list)):
            user_kwargs["llm_user_content"] = llm_user_content
        llm_context_frame = ctx.context_retry.get("llm_context_frame")
        if isinstance(llm_context_frame, str) and llm_context_frame.strip():
            user_kwargs["llm_context_frame"] = llm_context_frame
        user_kwargs.update(_collect_persist_user_slots(frame.slots))
        session.add_message(
            "user",
            msg.content,
            media=msg.media if msg.media else None,
            **user_kwargs,
        )
        return frame


class _PersistAssistantMessageModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        raw_session = frame.input.state.session
        if raw_session is None:
            raise RuntimeError("AfterReasoning requires TurnState.session")
        session = cast("Session", raw_session)
        assistant_kwargs: dict[str, Any] = {
            "tools_used": list(ctx.tools_used) if ctx.tools_used else None,
            "tool_chain": list(ctx.tool_chain) if ctx.tool_chain else None,
        }
        if ctx.thinking is not None:
            assistant_kwargs["reasoning_content"] = ctx.thinking
        assistant_kwargs.update(_collect_persist_assistant_slots(frame.slots))
        session.add_message("assistant", ctx.reply, **assistant_kwargs)
        return frame


class _UpdateSessionMetadataModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        raw_session = frame.input.state.session
        if raw_session is None:
            raise RuntimeError("AfterReasoning requires TurnState.session")
        session = cast("Session", raw_session)
        update_session_runtime_metadata(
            session,
            tools_used=list(ctx.tools_used),
            tool_chain=list(ctx.tool_chain),
        )
        return frame


class _AppendMessagesModule:
    def __init__(self, session_services: SessionServices) -> None:
        self._session_services = session_services

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        state = frame.input.state
        raw_session = state.session
        if raw_session is None:
            raise RuntimeError("AfterReasoning requires TurnState.session")
        session = cast("Session", raw_session)
        persist_count = 1 if bool((state.msg.metadata or {}).get("omit_user_turn")) else 2
        await self._session_services.session_manager.append_messages(
            session,
            cast(list[dict[str, Any]], session.messages[-persist_count:]),
        )
        return frame


class _BuildOutboundMessageModule:
    requires = (_CTX_SLOT,)
    produces = (_OUTBOUND_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        metadata = dict(ctx.outbound_metadata)
        metadata.update(collect_prefixed_slots(frame.slots, _OUTBOUND_METADATA_PREFIX))
        media = list(ctx.media)
        _append_media(media, collect_prefixed_slots(frame.slots, _OUTBOUND_MEDIA_PREFIX))
        frame.slots[_OUTBOUND_SLOT] = OutboundMessage(
            channel=ctx.channel,
            chat_id=ctx.chat_id,
            content=ctx.reply,
            thinking=ctx.thinking,
            media=media,
            metadata=metadata,
        )
        return frame


class _ReturnAfterReasoningResultModule:
    requires = (_CTX_SLOT, _OUTBOUND_SLOT)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        frame.output = AfterReasoningResult(
            ctx=cast(AfterReasoningCtx, frame.slots[_CTX_SLOT]),
            outbound=cast(OutboundMessage, frame.slots[_OUTBOUND_SLOT]),
        )
        return frame


def default_after_reasoning_modules(
    bus: EventBus,
    session_services: SessionServices,
    plugin_modules_before_emit: AfterReasoningModules | None = None,
    plugin_modules_before_persist: AfterReasoningModules | None = None,
) -> AfterReasoningModules:
    before_emit = plugin_modules_before_emit or []
    before_persist = plugin_modules_before_persist or []
    return [
        _BuildAfterReasoningCtxModule(),
        *before_emit,
        _EmitAfterReasoningCtxModule(bus),
        *before_persist,
        _PersistUserMessageModule(session_services),
        _PersistAssistantMessageModule(),
        _UpdateSessionMetadataModule(),
        _AppendMessagesModule(session_services),
        _BuildOutboundMessageModule(),
        _ReturnAfterReasoningResultModule(),
    ]


def _collect_persist_assistant_slots(slots: dict[str, object]) -> dict[str, object]:
    return collect_prefixed_slots(
        slots,
        _PERSIST_ASSISTANT_PREFIX,
        reserved=_ASSISTANT_FIXED_FIELDS,
    )


def _collect_persist_user_slots(slots: dict[str, object]) -> dict[str, object]:
    return collect_prefixed_slots(
        slots,
        _PERSIST_USER_PREFIX,
        reserved=_USER_FIXED_FIELDS,
    )


def _append_media(target: list[str], exports: dict[str, object]) -> None:
    append_string_exports(target, exports)
