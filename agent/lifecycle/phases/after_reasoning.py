from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from agent.control.context import running_turn_id
from agent.core.passive_support import build_session_runtime_metadata
from agent.control.ports import InputLock, TurnUserInput
from agent.core.response_parser import parse_response
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    append_string_exports,
    collect_prefixed_slots,
    topo_sort_modules,
)
from agent.lifecycle.composition import (
    AFTER_REASONING_CLEANUP_EVENT,
    AFTER_REASONING_PREPROCESS_EVENT,
    run_composition_lifecycle,
)
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterReasoningInput,
    TurnSnapshot,
)
from agent.plugin_composition.channels import AttachmentRef
from agent.turn_effects import (
    TURN_EFFECTS_KEY,
    PostCommitEffect,
    post_commit_effect,
    set_post_commit_effect,
)
from bus.event_bus import EventBus
from bus.events import OutboundMessage
from core.common.diagnostic_log import turn_milestone
from core.error_context import current_client_message_id, current_session_key

if TYPE_CHECKING:
    from agent.looping.ports import SessionServices
    from session.manager import Session

logger = logging.getLogger(__name__)


def _milestone(
    logger: logging.Logger,
    event: str,
    *,
    duration_ms: float | None = None,
    counts: str = "",
    outcome: str = "",
    level: int = logging.INFO,
) -> None:
    """打一个 turn 尾里程碑；身份统一从 contextvar 读取，字段全部走 turn_milestone。"""

    turn_milestone(
        logger,
        event,
        session_id=current_session_key.get() or "",
        turn_id=running_turn_id.get(),
        client_message_id=current_client_message_id.get(),
        duration_ms=duration_ms,
        counts=counts,
        outcome=outcome,
        level=level,
    )


@dataclass
class AfterReasoningFrame(PhaseFrame[AfterReasoningInput, TurnSnapshot]):
    pass


AfterReasoningModules: TypeAlias = list[PhaseModule[AfterReasoningFrame]]


_CTX_SLOT = "reasoning:ctx"
_OUTBOUND_SLOT = "reasoning:outbound"
_PERSISTED_USER_SLOT = "reasoning:persisted_user"
_PERSISTED_ASSISTANT_SLOT = "reasoning:persisted_assistant"
_SEALED_ASSISTANT_METADATA_SLOT = "reasoning:assistant_metadata"
_PENDING_SESSION_METADATA_SLOT = "reasoning:session_metadata"
_ASSISTANT_ATTACHMENT_REFS_SLOT = "reasoning:assistant_attachment_refs"
_OUTBOUND_METADATA_PREFIX = "outbound:metadata:"
_OUTBOUND_MEDIA_PREFIX = "outbound:media:"
_ASSISTANT_FIXED_FIELDS = {
    "tools_used",
    "tool_chain",
    "reasoning_content",
    "model_state",
}
_ASSISTANT_MESSAGE_FIELDS = {
    "role",
    "content",
    "timestamp",
    "media",
    "id",
    "seq",
}
_ASSISTANT_CORE_METADATA_FIELDS = {
    "turn_duration_ms",
    "control_turn_id",
    "turn_terminal",
    "turn_input_count",
    TURN_EFFECTS_KEY,
    "attachment_ids",
}
_RETIRED_ASSISTANT_FIELDS = frozenset({"react_compaction", "skip_post_memory"})
_ASSISTANT_FORBIDDEN_PLUGIN_FIELDS = frozenset(
    _ASSISTANT_FIXED_FIELDS
    | _ASSISTANT_MESSAGE_FIELDS
    | _ASSISTANT_CORE_METADATA_FIELDS
    | _RETIRED_ASSISTANT_FIELDS
)
_USER_FIXED_FIELDS = {
    "media",
    "timestamp",
    "client_message_id",
    "reply_to_message_id",
    "reply_role",
    "reply_preview",
}
_USER_CORE_METADATA_FIELDS = frozenset(
    _USER_FIXED_FIELDS
    | {
        "role",
        "content",
        "id",
        "seq",
        "turn_input_ordinal",
        "control_turn_id",
        "skip_post_memory",
        TURN_EFFECTS_KEY,
        "attachment_ids",
        "display_content",
        "client_created_at",
        "llm_user_content",
        "llm_context_frame",
    }
)


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
        await run_composition_lifecycle(AFTER_REASONING_PREPROCESS_EVENT, ctx)
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _RunCompositionAfterReasoningCleanupModule:
    slot = "after_reasoning.composition_cleanup"
    requires = ("after_reasoning.emit", _CTX_SLOT)
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        await run_composition_lifecycle(AFTER_REASONING_CLEANUP_EVENT, ctx)
        return frame


class _ImportAssistantAttachmentsModule:
    slot = "after_reasoning.import_attachments"
    requires = ("after_reasoning.composition_cleanup", _CTX_SLOT)
    produces = (_ASSISTANT_ATTACHMENT_REFS_SLOT,)

    def __init__(self, session_services: SessionServices) -> None:
        self._session_services = session_services

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        """Import outbound media before any Session message mutation."""

        # 1. 冻结 Core 与 v3 lifecycle 共同产生的媒体列表。
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        media = list(ctx.media)
        _append_media(
            media,
            collect_prefixed_slots(frame.slots, _OUTBOUND_MEDIA_PREFIX),
        )
        if not media:
            frame.slots[_ASSISTANT_ATTACHMENT_REFS_SLOT] = ()
            return frame

        # 2. 先完成 artifact transaction，随后 message binding 才能同批提交。
        importer = self._session_services.outbound_attachment_importer
        if importer is None:
            raise RuntimeError("outbound attachment importer 尚未绑定")
        refs = await importer.import_media(tuple(media))
        if any(not isinstance(ref, AttachmentRef) for ref in refs):
            raise TypeError("outbound attachment importer 返回值无效")
        frame.slots[_ASSISTANT_ATTACHMENT_REFS_SLOT] = refs
        return frame


class _PersistUserMessageModule:
    slot = "after_reasoning.persist_user"
    requires = ("after_reasoning.import_attachments", _CTX_SLOT)
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
        shared_user_kwargs = _collect_persist_user_metadata(ctx)
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
            effect = post_commit_effect(turn_input.metadata)
            if effect is PostCommitEffect.SUPPRESS:
                set_post_commit_effect(input_kwargs, effect)
            for field in (
                "client_message_id",
                "client_created_at",
                "reply_to_message_id",
                "reply_role",
                "reply_preview",
            ):
                value = turn_input.metadata.get(field)
                if isinstance(value, str) and value:
                    input_kwargs[field] = value
            attachment_ids = turn_input.metadata.get("attachment_ids")
            if attachment_ids is not None:
                if not isinstance(attachment_ids, list) or not all(
                    isinstance(item, str) and item for item in attachment_ids
                ):
                    raise ValueError("turn input attachment_ids 必须是非空字符串数组")
                input_kwargs["attachment_ids"] = list(attachment_ids)
            display_content = turn_input.metadata.get("display_content")
            persisted_users.append(
                _pending_message(
                    "user",
                    (
                        display_content
                        if isinstance(display_content, str)
                        else turn_input.content
                    ),
                    media=(
                        None
                        if attachment_ids is not None
                        else (list(turn_input.media) if turn_input.media else None)
                    ),
                    **input_kwargs,
                )
            )
        frame.slots[_PERSISTED_USER_SLOT] = persisted_users
        return frame


class _PersistAssistantMessageModule:
    slot = "after_reasoning.persist_asst"
    requires = (
        "after_reasoning.persist_user",
        _CTX_SLOT,
        _SEALED_ASSISTANT_METADATA_SLOT,
        _ASSISTANT_ATTACHMENT_REFS_SLOT,
    )
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
        assistant_kwargs.update(
            cast(dict[str, object], frame.slots[_SEALED_ASSISTANT_METADATA_SLOT])
        )
        turn_inputs = _turn_user_inputs(frame.input.state.msg)
        control_turn_id = str(
            frame.input.state.msg.metadata.get("control_turn_id") or ""
        )
        if control_turn_id and frame.input.state.persistence.persist_user:
            assistant_kwargs["control_turn_id"] = control_turn_id
            assistant_kwargs["turn_terminal"] = True
            assistant_kwargs["turn_input_count"] = len(turn_inputs)
        if any(
            post_commit_effect(turn_input.metadata) is PostCommitEffect.SUPPRESS
            for turn_input in turn_inputs
        ):
            set_post_commit_effect(assistant_kwargs, PostCommitEffect.SUPPRESS)
        if frame.input.state.persistence.persist_assistant:
            refs = cast(
                tuple[AttachmentRef, ...],
                frame.slots[_ASSISTANT_ATTACHMENT_REFS_SLOT],
            )
            if refs:
                assistant_kwargs["attachment_ids"] = [ref.artifact_id for ref in refs]
            frame.slots[_PERSISTED_ASSISTANT_SLOT] = _pending_message(
                "assistant",
                ctx.reply,
                **assistant_kwargs,
            )
        return frame


class _UpdateSessionMetadataModule:
    slot = "after_reasoning.update_meta"
    requires = ("after_reasoning.persist_asst", _CTX_SLOT)
    produces = (_PENDING_SESSION_METADATA_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        raw_session = frame.input.state.session
        if raw_session is None:
            raise RuntimeError("AfterReasoning requires TurnState.session")
        session = cast("Session", raw_session)
        frame.slots[_PENDING_SESSION_METADATA_SLOT] = build_session_runtime_metadata(
            session.metadata,
            tools_used=list(ctx.tools_used),
            tool_chain=list(ctx.tool_chain),
        )
        return frame


class _AppendMessagesModule:
    slot = "after_reasoning.append_messages"
    requires = ("after_reasoning.update_meta", _PENDING_SESSION_METADATA_SLOT)

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
        _milestone(logger, "after_reasoning.append.start")
        append_started = perf_counter()
        try:
            await self._session_services.session_manager.append_messages(
                session,
                messages,
                metadata=cast(
                    dict[str, Any],
                    frame.slots[_PENDING_SESSION_METADATA_SLOT],
                ),
            )
        except asyncio.CancelledError:
            _milestone(
                logger,
                "after_reasoning.append.cancelled",
                duration_ms=(perf_counter() - append_started) * 1000,
                outcome="cancelled",
                level=logging.WARNING,
            )
            raise
        except Exception:
            _milestone(
                logger,
                "after_reasoning.append.error",
                duration_ms=(perf_counter() - append_started) * 1000,
                outcome="error",
                level=logging.ERROR,
            )
            raise
        attached = {id(message) for message in session.messages}
        session.messages.extend(
            message for message in messages if id(message) not in attached
        )
        session.metadata = dict(
            cast(dict[str, Any], frame.slots[_PENDING_SESSION_METADATA_SLOT])
        )
        _milestone(
            logger,
            "after_reasoning.append.done",
            duration_ms=(perf_counter() - append_started) * 1000,
            outcome="done",
        )
        return frame


class _BuildOutboundMessageModule:
    slot = "after_reasoning.build_outbound"
    requires = (
        "after_reasoning.append_messages",
        _CTX_SLOT,
        _ASSISTANT_ATTACHMENT_REFS_SLOT,
    )
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
            persisted_user = persisted_users[-1]
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
        attachment_refs = cast(
            tuple[AttachmentRef, ...],
            frame.slots[_ASSISTANT_ATTACHMENT_REFS_SLOT],
        )
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
            attachment_refs=attachment_refs,
            metadata=metadata,
            session_message_id=session_message_id,
            control_turn_id=running_turn_id.get(),
        )
        return frame


def _turn_user_inputs(msg: object) -> tuple[TurnUserInput, ...]:
    """读取已锁定的 control source；普通内部 turn 投影为单条输入。"""

    metadata = getattr(msg, "metadata", None) or {}
    raw_source = metadata.get("_control_turn_input_source")
    if raw_source is not None:
        source = cast(InputLock, raw_source)
        inputs = source.used_inputs()
        if not inputs:
            raise RuntimeError("locked control turn 缺少用户输入")
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


class _BuildTurnSnapshotModule:
    slot = "after_reasoning.return"
    requires = ("after_reasoning.build_outbound", _CTX_SLOT, _OUTBOUND_SLOT)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        frame.output = TurnSnapshot(
            state=frame.input.state,
            outbound=cast(OutboundMessage, frame.slots[_OUTBOUND_SLOT]),
            ctx=cast(AfterReasoningCtx, frame.slots[_CTX_SLOT]),
        )
        return frame


def default_after_reasoning_modules(
    bus: EventBus,
    session_services: SessionServices,
    plugin_modules: AfterReasoningModules | None = None,
) -> AfterReasoningModules:
    legacy_modules = list(plugin_modules or [])
    builtins: AfterReasoningModules = [
        _BuildAfterReasoningCtxModule(),
        _EmitAfterReasoningCtxModule(bus),
        _RunCompositionAfterReasoningCleanupModule(),
        _ImportAssistantAttachmentsModule(session_services),
        _PersistUserMessageModule(session_services),
        _SealAssistantMetadataModule(),
        _PersistAssistantMessageModule(),
        _UpdateSessionMetadataModule(),
        _AppendMessagesModule(session_services),
        _BuildOutboundMessageModule(),
        _BuildTurnSnapshotModule(),
    ]
    return cast(
        AfterReasoningModules,
        topo_sort_modules(builtins + legacy_modules),
    )


class _SealAssistantMetadataModule:
    slot = "after_reasoning.seal_metadata"
    requires = ("after_reasoning.persist_user", _CTX_SLOT)
    produces = (_SEALED_ASSISTANT_METADATA_SLOT,)

    async def run(self, frame: AfterReasoningFrame) -> AfterReasoningFrame:
        ctx = cast(AfterReasoningCtx, frame.slots[_CTX_SLOT])
        frame.slots[_SEALED_ASSISTANT_METADATA_SLOT] = (
            _collect_persist_assistant_metadata(ctx)
        )
        return frame


def _pending_message(
    role: str,
    content: str,
    media: list[str] | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """构造尚未挂入 Session 的 append-only message。"""

    message: dict[str, object] = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(UTC).isoformat(),
        **kwargs,
    }
    if media:
        message["media"] = list(media)
    return message


def _collect_persist_assistant_metadata(
    ctx: AfterReasoningCtx,
) -> dict[str, object]:
    """校验并冻结 v3 assistant metadata。"""

    metadata = dict(ctx.persist_assistant_metadata)
    forbidden = set(metadata) & _ASSISTANT_FORBIDDEN_PLUGIN_FIELDS
    if forbidden:
        fields = ", ".join(sorted(forbidden))
        raise ValueError(f"assistant plugin metadata 字段不可写: {fields}")
    return metadata


def _collect_persist_user_metadata(
    ctx: AfterReasoningCtx,
) -> dict[str, object]:
    """校验并冻结 v3 user metadata。"""

    metadata = dict(ctx.persist_user_metadata)
    forbidden = set(metadata) & _USER_CORE_METADATA_FIELDS
    if forbidden:
        fields = ", ".join(sorted(forbidden))
        raise ValueError(f"user plugin metadata 字段不可写: {fields}")
    return metadata


def _append_media(target: list[str], exports: dict[str, object]) -> None:
    append_string_exports(target, exports)
