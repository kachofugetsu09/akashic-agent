from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from agent.core.passive_support import update_session_runtime_metadata
from agent.control.ports import TurnInputSource, TurnUserInput
from agent.core.response_parser import parse_response
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    append_string_exports,
    collect_prefixed_slots,
    topo_sort_modules,
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
_PERSISTED_USER_SLOT = "reasoning:persisted_user"
_PERSISTED_ASSISTANT_SLOT = "reasoning:persisted_assistant"
_PERSIST_USER_PREFIX = "persist:user:"
_PERSIST_ASSISTANT_PREFIX = "persist:assistant:"
_OUTBOUND_METADATA_PREFIX = "outbound:metadata:"
_OUTBOUND_MEDIA_PREFIX = "outbound:media:"
_ASSISTANT_FIXED_FIELDS = {
    "tools_used",
    "tool_chain",
    "reasoning_content",
    "model_state",
    "react_compaction",
}
_USER_FIXED_FIELDS = {
    "media",
    "timestamp",
    "client_message_id",
    "reply_to_message_id",
    "reply_role",
    "reply_preview",
}


class _BuildAfterReasoningCtxModule:
    slot = "after_reasoning.build_ctx"
    requires: tuple[str, ...] = ()
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        input = frame.input
        msg = input.state.msg
        turn_result = input.turn_result
        inbound_metadata = dict(msg.metadata or {})
        inbound_metadata.pop("mobile_attention", None)
        inbound_metadata.pop("_control_turn_input_source", None)
        inbound_metadata.pop("_control_attempt_replay", None)
        inbound_metadata.pop("_control_prior_tool_chain", None)
        inbound_metadata.pop("_control_prior_input_count", None)
        inbound_metadata.pop("_control_execution_turn_id", None)
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
            media=list(turn_result.media),
            thinking=turn_result.thinking,
            streamed=turn_result.streamed,
            context_retry=dict(turn_result.context_retry),
            outbound_metadata={
                **inbound_metadata,
                **input.state.extra_metadata,
                "tools_used": list(turn_result.tools_used),
                "tool_chain": list(tool_chain),
                "context_retry": dict(turn_result.context_retry),
                "streamed_reply": turn_result.streamed,
                **(
                    {"mobile_attention": turn_result.mobile_attention}
                    if turn_result.mobile_attention is not None
                    else {}
                ),
            },
        )
        return frame


class _EmitAfterReasoningCtxModule:
    slot = "after_reasoning.emit"
    requires = ("after_reasoning.build_ctx", _CTX_SLOT)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _PersistUserMessageModule:
    slot = "after_reasoning.persist_user"
    requires = ("after_reasoning.emit", _CTX_SLOT)
    produces = (_PERSISTED_USER_SLOT,)

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
        if not state.persistence.persist_user:
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
        shared_user_kwargs = _collect_persist_user_slots(frame.slots)
        control_turn_id = str(msg.metadata.get("control_turn_id") or "")
        persisted_users: list[dict[str, Any]] = []
        for index, turn_input in enumerate(_turn_user_inputs(msg)):
            input_kwargs = dict(user_kwargs) if index == 0 else {}
            if index == 0:
                input_kwargs.update(shared_user_kwargs)
            input_kwargs["timestamp"] = turn_input.timestamp.isoformat()
            input_kwargs["turn_input_ordinal"] = turn_input.ordinal
            if control_turn_id:
                input_kwargs["control_turn_id"] = control_turn_id
            if turn_input.metadata.get("skip_post_memory") is True:
                input_kwargs["skip_post_memory"] = True
            for field in (
                "client_message_id",
                "client_created_at",
                "reply_to_message_id",
                "reply_role",
                "reply_preview",
            ):
                value = turn_input.metadata.get(field)
                if isinstance(value, str) and value:
                    input_kwargs[
                        "timestamp" if field == "client_created_at" else field
                    ] = value
            display_content = turn_input.metadata.get("display_content")
            persisted_users.append(
                session.add_message(
                    "user",
                    display_content
                    if isinstance(display_content, str)
                    else turn_input.content,
                    media=list(turn_input.media) if turn_input.media else None,
                    **input_kwargs,
                )
            )
        frame.slots[_PERSISTED_USER_SLOT] = persisted_users
        return frame


class _PersistAssistantMessageModule:
    slot = "after_reasoning.persist_asst"
    requires = ("after_reasoning.persist_user", _CTX_SLOT)
    produces = (_PERSISTED_ASSISTANT_SLOT,)

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
        turn_duration_ms = frame.input.state.extra_metadata.get("turn_duration_ms")
        if isinstance(turn_duration_ms, int):
            assistant_kwargs["turn_duration_ms"] = turn_duration_ms
        if ctx.thinking is not None:
            assistant_kwargs["reasoning_content"] = ctx.thinking
        if frame.input.turn_result.model_state is not None:
            assistant_kwargs["model_state"] = frame.input.turn_result.model_state
        if frame.input.turn_result.react_compaction is not None:
            assistant_kwargs["react_compaction"] = dict(
                frame.input.turn_result.react_compaction
            )
        assistant_kwargs.update(_collect_persist_assistant_slots(frame.slots))
        turn_inputs = _turn_user_inputs(frame.input.state.msg)
        control_turn_id = str(
            frame.input.state.msg.metadata.get("control_turn_id") or ""
        )
        if control_turn_id and frame.input.state.persistence.persist_user:
            assistant_kwargs["control_turn_id"] = control_turn_id
            assistant_kwargs["turn_terminal"] = True
            assistant_kwargs["turn_input_count"] = len(turn_inputs)
        if any(
            turn_input.metadata.get("skip_post_memory") is True
            for turn_input in turn_inputs
        ):
            assistant_kwargs["skip_post_memory"] = True
        if frame.input.state.persistence.persist_assistant:
            media = list(ctx.media)
            _append_media(
                media,
                collect_prefixed_slots(frame.slots, _OUTBOUND_MEDIA_PREFIX),
            )
            frame.slots[_PERSISTED_ASSISTANT_SLOT] = session.add_message(
                "assistant",
                ctx.reply,
                media=media if media else None,
                **assistant_kwargs,
            )
        return frame


class _UpdateSessionMetadataModule:
    slot = "after_reasoning.update_meta"
    requires = ("after_reasoning.persist_asst", _CTX_SLOT)

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
    slot = "after_reasoning.append_messages"
    requires = ("after_reasoning.update_meta",)

    def __init__(self, session_services: SessionServices) -> None:
        self._session_services = session_services

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        state = frame.input.state
        raw_session = state.session
        if raw_session is None:
            raise RuntimeError("AfterReasoning requires TurnState.session")
        session = cast("Session", raw_session)
        messages: list[dict[str, Any]] = []
        if state.persistence.persist_user:
            messages.extend(
                cast(list[dict[str, Any]], frame.slots[_PERSISTED_USER_SLOT])
            )
        if state.persistence.persist_assistant:
            messages.append(
                cast(dict[str, Any], frame.slots[_PERSISTED_ASSISTANT_SLOT])
            )
        if not messages:
            return frame
        await self._session_services.session_manager.append_messages(
            session,
            messages,
        )
        return frame


class _BuildOutboundMessageModule:
    slot = "after_reasoning.build_outbound"
    requires = ("after_reasoning.append_messages", _CTX_SLOT)
    produces = (_OUTBOUND_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        metadata = dict(ctx.outbound_metadata)
        metadata.update(collect_prefixed_slots(frame.slots, _OUTBOUND_METADATA_PREFIX))
        if frame.input.state.persistence.persist_user:
            persisted_users = cast(
                list[dict[str, object]],
                frame.slots[_PERSISTED_USER_SLOT],
            )
            if not persisted_users:
                raise RuntimeError("本轮 user 消息列表为空")
            persisted_user = persisted_users[0]
            persisted_user_ids = [
                str(item["id"])
                for item in persisted_users
                if isinstance(item.get("id"), str) and item["id"]
            ]
            if len(persisted_user_ids) == len(persisted_users):
                metadata["persisted_user_message_ids"] = persisted_user_ids
            elif ctx.channel == "mobile":
                raise RuntimeError("本轮 user 消息缺少稳定 ID")
            raw_user_message_id = persisted_user.get("id")
            raw_client_message_id = persisted_user.get("client_message_id")
            if isinstance(raw_user_message_id, str) and raw_user_message_id:
                metadata["persisted_user_message_id"] = raw_user_message_id
            elif ctx.channel == "mobile":
                raise RuntimeError("本轮 mobile user 消息缺少稳定 ID")
            if isinstance(raw_client_message_id, str) and raw_client_message_id:
                metadata["client_message_id"] = raw_client_message_id
            elif ctx.channel == "mobile":
                raise RuntimeError("本轮 mobile user 消息缺少客户端 ID")
        media = list(ctx.media)
        _append_media(media, collect_prefixed_slots(frame.slots, _OUTBOUND_MEDIA_PREFIX))
        session_message_id: str | None = None
        if frame.input.state.persistence.persist_assistant:
            persisted = cast(
                dict[str, object],
                frame.slots[_PERSISTED_ASSISTANT_SLOT],
            )
            raw_message_id = persisted.get("id")
            if isinstance(raw_message_id, str) and raw_message_id:
                session_message_id = raw_message_id
            elif ctx.channel == "mobile":
                raise RuntimeError("本轮 assistant 消息缺少稳定 ID")
        frame.slots[_OUTBOUND_SLOT] = OutboundMessage(
            channel=ctx.channel,
            chat_id=ctx.chat_id,
            content=ctx.reply,
            thinking=ctx.thinking,
            media=media,
            metadata=metadata,
            session_message_id=session_message_id,
        )
        return frame


def _turn_user_inputs(msg: object) -> tuple[TurnUserInput, ...]:
    """读取 sealed control source；普通内部 turn 投影为单条输入。"""

    metadata = getattr(msg, "metadata", None) or {}
    raw_source = metadata.get("_control_turn_input_source")
    if raw_source is not None:
        source = cast(TurnInputSource, raw_source)
        inputs = source.consumed_inputs()
        if not inputs:
            raise RuntimeError("sealed control turn 缺少用户输入")
        return inputs
    return (
        TurnUserInput(
            item_id="direct",
            ordinal=0,
            content=str(getattr(msg, "content")),
            media=tuple(getattr(msg, "media", ()) or ()),
            metadata=dict(metadata),
            timestamp=getattr(msg, "timestamp"),
        ),
    )


class _ReturnAfterReasoningResultModule:
    slot = "after_reasoning.return"
    requires = ("after_reasoning.build_outbound", _CTX_SLOT, _OUTBOUND_SLOT)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        frame.output = AfterReasoningResult(
            ctx=cast(AfterReasoningCtx, frame.slots[_CTX_SLOT]),
            outbound=cast(OutboundMessage, frame.slots[_OUTBOUND_SLOT]),
        )
        return frame


def default_after_reasoning_modules(
    bus: EventBus,
    session_services: SessionServices,
    plugin_modules: AfterReasoningModules | None = None,
) -> AfterReasoningModules:
    builtins: AfterReasoningModules = [
        _BuildAfterReasoningCtxModule(),
        _EmitAfterReasoningCtxModule(bus),
        _PersistUserMessageModule(session_services),
        _PersistAssistantMessageModule(),
        _UpdateSessionMetadataModule(),
        _AppendMessagesModule(session_services),
        _BuildOutboundMessageModule(),
        _ReturnAfterReasoningResultModule(),
    ]
    return cast(
        AfterReasoningModules,
        topo_sort_modules(builtins + list(plugin_modules or [])),
    )


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
