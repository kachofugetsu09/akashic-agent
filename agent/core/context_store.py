from __future__ import annotations

import copy
import inspect
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.core.response_parser import ResponseMetadata
from agent.core.types import (
    ChatMessage,
    ContextBundle,
    HistoryMessage,
    to_tool_call_groups,
)
from agent.postturn.protocol import PostTurnEvent
from agent.prompting import is_context_frame
from agent.retrieval.protocol import RetrievalRequest
from agent.turns.outbound import OutboundDispatch
from bus.event_bus import EventBus
from bus.events import OutboundMessage
from bus.events_lifecycle import BeforeDispatch, TurnCompleted, TurnPersisted

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.runtime_support import SessionLike
    from agent.looping.ports import ObservabilityServices, SessionServices
    from agent.memes.decorator import MemeDecorator
    from agent.postturn.protocol import PostTurnPipeline
    from agent.retrieval.protocol import MemoryRetrievalPipeline
    from agent.turns.outbound import OutboundPort
    from bus.events import InboundMessage

logger = logging.getLogger("agent.core.context_store")


@runtime_checkable
class _SideEffect(Protocol):
    def run(self) -> object: ...


class ContextStore(ABC):
    """
    ┌──────────────────────────────────────┐
    │ ContextStore                         │
    ├──────────────────────────────────────┤
    │ 1. 读取 session history              │
    │ 2. 调 retrieval pipeline             │
    │ 3. 收 skill mentions                 │
    │ 4. 输出 ContextBundle                │
    └──────────────────────────────────────┘
    """

    @abstractmethod
    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        """准备本轮对话需要的上下文。"""

    @abstractmethod
    async def commit(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        reply: str,
        response_metadata: ResponseMetadata,
        tools_used: list[str],
        tool_chain: list[dict],
        thinking: str | None,
        streamed_reply: bool,
        retrieval_raw: object | None,
        context_retry: dict[str, object],
        post_turn_actions: list[object] | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        """提交本轮被动 turn，并返回最终出站消息。"""


class DefaultContextStore(ContextStore):
    def __init__(
        self,
        *,
        retrieval: "MemoryRetrievalPipeline",
        context: "ContextBuilder",
        history_window: int = 500,
        session: "SessionServices | None" = None,
        trace: "ObservabilityServices | None" = None,
        post_turn: "PostTurnPipeline | None" = None,
        outbound: "OutboundPort | None" = None,
        meme_decorator: "MemeDecorator | None" = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._context = context
        self._history_window = max(1, int(history_window))
        self._session = session
        self._trace = trace
        self._post_turn = post_turn
        self._outbound = outbound
        self._meme_decorator = meme_decorator
        self._event_bus = event_bus

    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        # 1. 先读取 session history，并转换成 retrieval pipeline 需要的结构。
        raw_history = [
            item
            for item in session.get_history()
            if not _is_llm_context_frame(item)
        ]
        history_messages = _to_history_messages(raw_history)

        # 2. 再执行 retrieval，保持当前 pipeline 行为不变。
        retrieval_result = await self._retrieval.retrieve(
            RetrievalRequest(
                message=msg.content,
                session_key=session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                history=history_messages,
                session_metadata=(
                    session.metadata if isinstance(session.metadata, dict) else {}
                ),
                timestamp=msg.timestamp,
            )
        )

        # 3. 最后补齐 ContextBundle，把主链正式字段直接收进显式合同。
        skill_mentions = _collect_skill_mentions(
            msg.content,
            self._context.skills.list_skills(filter_unavailable=False),
        )
        return ContextBundle(
            history=_to_chat_messages(raw_history),
            memory_blocks=[retrieval_result.block] if retrieval_result.block else [],
            skill_mentions=skill_mentions,
            retrieved_memory_block=retrieval_result.block or "",
            retrieval_trace_raw=(
                retrieval_result.trace.raw
                if retrieval_result.trace is not None
                else None
            ),
            retrieval_metadata=dict(retrieval_result.metadata or {}),
            history_messages=history_messages,
        )

    async def commit(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        reply: str,
        response_metadata: ResponseMetadata,
        tools_used: list[str],
        tool_chain: list[dict],
        thinking: str | None,
        streamed_reply: bool,
        retrieval_raw: object | None,
        context_retry: dict[str, object],
        post_turn_actions: list[object] | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        if self._session is None or self._post_turn is None or self._outbound is None:
            raise RuntimeError("ContextStore.commit requires session/post_turn/outbound")

        cited_memory_ids = list(response_metadata.cited_memory_ids)

        # 1. 先做 meme decorate，并准备最终回复文本。
        final_content = reply
        meme_media: list[str] = []
        meme_tag = response_metadata.meme_tag
        if self._meme_decorator is not None:
            decorated = self._meme_decorator.decorate(
                final_content,
                meme_tag=meme_tag,
            )
            final_content = decorated.content
            meme_media = decorated.media
            meme_tag = decorated.tag

        # 2. 再把 user/assistant 两条消息持久化到 session。
        session = self._session.session_manager.get_or_create(session_key)
        omit_user_turn = bool((msg.metadata or {}).get("omit_user_turn"))
        if not omit_user_turn:
            if self._session.presence:
                self._session.presence.record_user_message(session.key)
            user_kwargs: dict[str, object] = {}
            llm_user_content = context_retry.get("llm_user_content")
            if isinstance(llm_user_content, (str, list)):
                user_kwargs["llm_user_content"] = llm_user_content
            llm_context_frame = context_retry.get("llm_context_frame")
            if isinstance(llm_context_frame, str) and llm_context_frame.strip():
                user_kwargs["llm_context_frame"] = llm_context_frame
            session.add_message(
                "user",
                msg.content,
                media=msg.media if msg.media else None,
                **user_kwargs,
            )
        _assistant_kwargs: dict = {
            "tools_used": tools_used if tools_used else None,
            "tool_chain": tool_chain if tool_chain else None,
        }
        if thinking is not None:
            _assistant_kwargs["reasoning_content"] = thinking
        if cited_memory_ids:
            _assistant_kwargs["cited_memory_ids"] = cited_memory_ids
        session.add_message("assistant", final_content, **_assistant_kwargs)
        _update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        persist_count = 1 if omit_user_turn else 2
        await self._session.session_manager.append_messages(
            session,
            session.messages[-persist_count:],
        )
        post_reply_budget = _build_post_reply_context_budget(
            context=self._context,
            history=session.get_history(max_messages=self._history_window),
            history_window=self._history_window,
        )
        react_stats = _extract_react_stats(context_retry)
        if self._event_bus is not None:
            await self._event_bus.observe(
                TurnPersisted(
                    session_key=session_key,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    user_message=None if omit_user_turn else msg.content,
                    assistant_response=final_content,
                    tools_used=list(tools_used),
                    thinking=thinking,
                    raw_reply=response_metadata.raw_text,
                    meme_tag=meme_tag,
                    meme_media_count=len(meme_media),
                    tool_chain=copy.deepcopy(tool_chain),
                    retrieval_raw=retrieval_raw,
                    post_reply_budget=dict(post_reply_budget),
                    react_stats=dict(react_stats),
                )
            )
        _log_post_reply_context_budget(
            session_key=session_key,
            budget=post_reply_budget,
        )
        _log_react_context_budget(session_key=session_key, react_stats=react_stats)

        # 3. 安排 post_turn。
        self._post_turn.schedule(
            PostTurnEvent(
                session_key=session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                user_message=msg.content,
                assistant_response=final_content,
                tools_used=tools_used,
                tool_chain=to_tool_call_groups(tool_chain),
                session=session,
                timestamp=msg.timestamp,
                extra=(
                    {"skip_post_memory": True}
                    if (msg.metadata or {}).get("skip_post_memory")
                    else {}
                ),
            )
        )
        await _run_effects(post_turn_actions or [])

        # 4. 发出成功完成事件，再给出站消息留出最后干预点。
        if self._event_bus is not None:
            await self._event_bus.observe(
                TurnCompleted(
                    session_key=session_key,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    reply=final_content,
                    tools_used=list(tools_used),
                    thinking=thinking,
                )
            )
        dispatch_event = BeforeDispatch(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            thinking=thinking,
            media=list(meme_media),
            metadata={
                **(msg.metadata or {}),
                "tools_used": tools_used,
                "tool_chain": tool_chain,
                "context_retry": context_retry,
                "streamed_reply": streamed_reply,
            },
        )
        if self._event_bus is not None:
            dispatch_event = await self._event_bus.emit(dispatch_event)

        # 5. 最后构造 outbound，并按需 dispatch。
        outbound = OutboundMessage(
            channel=dispatch_event.channel,
            chat_id=dispatch_event.chat_id,
            content=dispatch_event.content,
            thinking=dispatch_event.thinking,
            media=list(dispatch_event.media),
            metadata=dict(dispatch_event.metadata),
        )
        if dispatch_outbound:
            await self._outbound.dispatch(
                OutboundDispatch(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=outbound.content,
                    thinking=outbound.thinking,
                    metadata=outbound.metadata,
                    media=outbound.media,
                )
            )
        return outbound


def _collect_skill_mentions(content: str, skills: list[dict]) -> list[str]:
    raw_names = re.findall(r"\$([a-zA-Z0-9_-]+)", content)
    if not raw_names:
        return []
    available = {s["name"] for s in skills if isinstance(s.get("name"), str)}
    seen: set[str] = set()
    result: list[str] = []
    for name in raw_names:
        if name in available and name not in seen:
            seen.add(name)
            result.append(name)
    if result:
        logger.info("检测到 $skill 提及，直接注入完整内容: %s", result)
    return result


def _to_chat_messages(messages: list[dict]) -> list[ChatMessage]:
    return [
        ChatMessage(
            role=str(msg.get("role", "") or ""),
            content=str(msg.get("content", "") or ""),
        )
        for msg in messages
    ]


def _to_history_messages(messages: list[dict]) -> list[HistoryMessage]:
    out: list[HistoryMessage] = []
    for msg in messages:
        role = str(msg.get("role", "") or "")
        content = str(msg.get("content", "") or "")
        tools_used = [
            str(tool_name)
            for tool_name in (msg.get("tools_used") or [])
            if isinstance(tool_name, str)
        ]
        out.append(
            HistoryMessage(
                role=role,
                content=content,
                tools_used=tools_used,
                tool_chain=to_tool_call_groups(msg.get("tool_chain") or []),
            )
        )
    return out


def _is_llm_context_frame(message: dict) -> bool:
    content = message.get("content")
    return isinstance(content, str) and is_context_frame(content)


def _build_post_reply_context_budget(
    *,
    context: "ContextBuilder",
    history: list[dict],
    history_window: int,
) -> dict[str, int]:
    history_stats = _estimate_history_budget(history)
    debug_breakdown = getattr(context, "last_debug_breakdown", []) or []
    prompt_tokens = sum(
        int(getattr(item, "est_tokens", 0) or 0)
        for item in debug_breakdown
    )
    return {
        "history_window": history_window,
        "history_messages": history_stats["messages"],
        "history_chars": history_stats["chars"],
        "history_tokens": history_stats["tokens"],
        "prompt_tokens": prompt_tokens,
        "next_turn_baseline_tokens": history_stats["tokens"] + prompt_tokens,
    }


def _log_post_reply_context_budget(
    *,
    session_key: str,
    budget: dict[str, int],
) -> None:
    logger.info(
        "post_reply_context: session_key=%s history_window=%d history_messages=%d history_chars=%d history_tokens~=%d prompt_tokens~=%d next_turn_baseline_tokens~=%d",
        session_key,
        budget["history_window"],
        budget["history_messages"],
        budget["history_chars"],
        budget["history_tokens"],
        budget["prompt_tokens"],
        budget["next_turn_baseline_tokens"],
    )


def _extract_react_stats(context_retry: dict[str, object]) -> dict[str, int]:
    raw = context_retry.get("react_stats")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key in (
        "iteration_count",
        "turn_input_sum_tokens",
        "turn_input_peak_tokens",
        "final_call_input_tokens",
        "cache_prompt_tokens",
        "cache_hit_tokens",
    ):
        value = raw.get(key)
        if value is None:
            continue
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _log_react_context_budget(
    *,
    session_key: str,
    react_stats: dict[str, int],
) -> None:
    if not react_stats:
        return
    logger.info(
        "react_context: session_key=%s iteration_count=%d turn_input_sum_tokens~=%d turn_input_peak_tokens~=%d final_call_input_tokens~=%d cache_hit=%d/%d",
        session_key,
        react_stats.get("iteration_count", 0),
        react_stats.get("turn_input_sum_tokens", 0),
        react_stats.get("turn_input_peak_tokens", 0),
        react_stats.get("final_call_input_tokens", 0),
        react_stats.get("cache_hit_tokens", 0),
        react_stats.get("cache_prompt_tokens", 0),
    )


def _estimate_history_budget(history: list[dict]) -> dict[str, int]:
    if not history:
        return {"messages": 0, "chars": 0, "tokens": 0}
    payload = json.dumps(history, ensure_ascii=False)
    chars = len(payload)
    return {
        "messages": len(history),
        "chars": chars,
        "tokens": max(1, chars // 3),
    }


async def _run_effects(effects: list[object]) -> None:
    for effect in effects:
        if not isinstance(effect, _SideEffect):
            continue
        try:
            maybe = effect.run()
            if inspect.isawaitable(maybe):
                await maybe
        except Exception as e:
            logger.warning("turn side effect failed: %s", e)




# ── Session metadata helpers (moved from agent/looping/memory_gate.py) ────────
# agent/looping/memory_gate.py re-exports this for backward compat.


def _update_session_runtime_metadata(
    session: object,
    *,
    tools_used: list[str],
    tool_chain: list[dict],
) -> None:
    from datetime import datetime

    md = session.metadata if isinstance(session.metadata, dict) else {}  # type: ignore[union-attr]
    call_count = sum(
        len(group.get("calls") or [])
        for group in tool_chain
        if isinstance(group, dict)
    )

    md["last_turn_tool_calls_count"] = call_count
    md["last_turn_ts"] = datetime.now().astimezone().isoformat()
    session.metadata = md  # type: ignore[union-attr]
