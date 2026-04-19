from __future__ import annotations

import inspect
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.core.types import (
    ChatMessage,
    ContextBundle,
    HistoryMessage,
    to_tool_call_groups,
)
from agent.postturn.protocol import PostTurnEvent
from agent.retrieval.protocol import RetrievalRequest
from agent.turns.outbound import OutboundDispatch
from bus.events import OutboundMessage

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
class _ObserveWriter(Protocol):
    def emit(self, event: object) -> None: ...


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
    ) -> None:
        self._retrieval = retrieval
        self._context = context
        self._history_window = max(1, int(history_window))
        self._session = session
        self._trace = trace
        self._post_turn = post_turn
        self._outbound = outbound
        self._meme_decorator = meme_decorator

    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        # 1. 先读取 session history，并转换成 retrieval pipeline 需要的结构。
        raw_history = session.get_history()
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
        tools_used: list[str],
        tool_chain: list[dict],
        thinking: str | None,
        streamed_reply: bool,
        retrieval_raw: object | None,
        context_retry: dict[str, object],
        post_turn_actions: list[object] | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        if (
            self._session is None
            or self._trace is None
            or self._post_turn is None
            or self._outbound is None
        ):
            raise RuntimeError("ContextStore.commit requires session/trace/post_turn/outbound")

        # 0. 从 reply 里剥离隐式引用行 §cited:[id1,id2]§，存入 cited_memory_ids。
        reply, cited_memory_ids = _extract_cited_ids(reply)
        if not cited_memory_ids:
            cited_memory_ids = _extract_cited_ids_from_tool_chain(tool_chain)

        # 1. 先做 meme decorate，并准备最终回复文本。
        final_content = reply
        meme_media: list[str] = []
        meme_tag: str | None = None
        if self._meme_decorator is not None:
            decorated = self._meme_decorator.decorate(final_content)
            final_content = decorated.content
            meme_media = decorated.media
            meme_tag = decorated.tag

        # 2. 再把 user/assistant 两条消息持久化到 session。
        session = self._session.session_manager.get_or_create(session_key)
        omit_user_turn = bool((msg.metadata or {}).get("omit_user_turn"))
        if not omit_user_turn:
            if self._session.presence:
                self._session.presence.record_user_message(session.key)
            session.add_message("user", msg.content, media=msg.media if msg.media else None)
        _assistant_kwargs: dict = {
            "tools_used": tools_used if tools_used else None,
            "tool_chain": tool_chain if tool_chain else None,
        }
        if cited_memory_ids:
            _assistant_kwargs["cited_memory_ids"] = cited_memory_ids
        session.add_message("assistant", final_content, **_assistant_kwargs)
        _update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        persist_count = 1 if omit_user_turn else 2
        await self._session.session_manager.append_messages(session, session.messages[-persist_count:])
        post_reply_budget = _build_post_reply_context_budget(
            context=self._context,
            history=session.get_history(max_messages=self._history_window),
            history_window=self._history_window,
        )
        _log_post_reply_context_budget(session_key=session_key, budget=post_reply_budget)
        react_stats = _extract_react_stats(context_retry)
        _log_react_context_budget(session_key=session_key, react_stats=react_stats)

        # 3. 发 observe trace，并安排 post_turn。
        _emit_observe_traces(
            trace=self._trace,
            session_key=session_key,
            msg=msg,
            final_content=final_content,
            raw_content=reply,
            meme_tag=meme_tag,
            meme_media_count=len(meme_media),
            tool_chain=tool_chain,
            retrieval_raw=retrieval_raw,
            post_reply_budget=post_reply_budget,
            react_stats=react_stats,
        )
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

        # 4. 最后构造 outbound，并按需 dispatch。
        outbound = OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            thinking=thinking,
            media=meme_media,
            metadata={
                **(msg.metadata or {}),
                "tools_used": tools_used,
                "tool_chain": tool_chain,
                "context_retry": context_retry,
                "streamed_reply": streamed_reply,
            },
        )
        if dispatch_outbound:
            await self._outbound.dispatch(
                OutboundDispatch(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=outbound.content,
                    thinking=outbound.thinking,
                    metadata=outbound.metadata,
                    media=meme_media,
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


def _emit_observe_traces(
    *,
    trace: "ObservabilityServices",
    session_key: str,
    msg: "InboundMessage",
    final_content: str,
    raw_content: str,
    meme_tag: str | None,
    meme_media_count: int,
    tool_chain: list[dict],
    retrieval_raw: object | None,
    post_reply_budget: dict[str, int],
    react_stats: dict[str, int],
) -> None:
    writer = trace.observe_writer
    if writer is None:
        return
    observe_writer = writer if isinstance(writer, _ObserveWriter) else None
    if observe_writer is None:
        return
    from core.observe.events import TurnTrace as TurnTraceEvent

    tool_calls = [
        {
            "name": call.get("name", ""),
            "args": str(call.get("arguments", ""))[:300],
            "result": str(call.get("result", ""))[:500],
        }
        for group in tool_chain
        for call in (group.get("calls") or [])
    ]

    def _slim_chain(chain: list[dict]) -> list[dict]:
        out = []
        for group in chain:
            text = str(group.get("text") or "")
            calls = [
                {
                    "name": c.get("name", ""),
                    "args": str(c.get("arguments", ""))[:800],
                    "result": str(c.get("result", ""))[:1200],
                }
                for c in (group.get("calls") or [])
            ]
            out.append({"text": text, "calls": calls})
        return out

    tool_chain_json = (
        json.dumps(_slim_chain(tool_chain), ensure_ascii=False) if tool_chain else None
    )

    observe_writer.emit(
        TurnTraceEvent(
            source="agent",
            session_key=session_key,
            user_msg=msg.content,
            llm_output=final_content,
            raw_llm_output=raw_content,
            meme_tag=meme_tag,
            meme_media_count=meme_media_count,
            tool_calls=tool_calls,
            tool_chain_json=tool_chain_json,
            history_window=post_reply_budget["history_window"],
            history_messages=post_reply_budget["history_messages"],
            history_chars=post_reply_budget["history_chars"],
            history_tokens=post_reply_budget["history_tokens"],
            prompt_tokens=post_reply_budget["prompt_tokens"],
            next_turn_baseline_tokens=post_reply_budget["next_turn_baseline_tokens"],
            react_iteration_count=react_stats.get("iteration_count"),
            react_input_sum_tokens=react_stats.get("turn_input_sum_tokens"),
            react_input_peak_tokens=react_stats.get("turn_input_peak_tokens"),
            react_final_input_tokens=react_stats.get("final_call_input_tokens"),
        )
    )
    if retrieval_raw is not None:
        observe_writer.emit(retrieval_raw)


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
        "react_context: session_key=%s iteration_count=%d turn_input_sum_tokens~=%d turn_input_peak_tokens~=%d final_call_input_tokens~=%d",
        session_key,
        react_stats.get("iteration_count", 0),
        react_stats.get("turn_input_sum_tokens", 0),
        react_stats.get("turn_input_peak_tokens", 0),
        react_stats.get("final_call_input_tokens", 0),
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


# ── Citation extraction ────────────────────────────────────────────────────────

_CITED_RE = re.compile(r"(?:\n|\r\n)?§cited:\[([A-Za-z0-9_,\-\s]+)\]§\s*$")


def _extract_cited_ids(response: str) -> tuple[str, list[str]]:
    """从回复正文中剥离隐式引用行 §cited:[id1,id2]§。

    返回 (干净回复, cited_id列表)。若无引用行则返回原文和空列表。
    """
    match = _CITED_RE.search(response)
    if not match:
        return response, []
    raw = match.group(1)
    ids = [i.strip() for i in raw.split(",") if i.strip()]
    clean = response[: match.start()].rstrip()
    return clean, ids


def _extract_cited_ids_from_tool_chain(tool_chain: list[dict]) -> list[str]:
    """从工具结果里兜底提取 cited ids。

    目标场景：模型调用了 recall_memory，但忘了在最终回复里输出 §cited:[...]§。
    这时至少把工具结果里明确返回的 cited_item_ids 落库，避免本轮引用信息彻底丢失。
    """
    cited: list[str] = []
    seen: set[str] = set()
    for group in tool_chain:
        if not isinstance(group, dict):
            continue
        for call in group.get("calls") or []:
            if not isinstance(call, dict):
                continue
            if str(call.get("name", "") or "") != "recall_memory":
                continue
            raw_result = str(call.get("result", "") or "").strip()
            if not raw_result:
                continue
            try:
                data = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            raw_ids = data.get("cited_item_ids")
            if not isinstance(raw_ids, list):
                items = data.get("items")
                if isinstance(items, list):
                    raw_ids = [item.get("id") for item in items if isinstance(item, dict)]
                else:
                    raw_ids = []
            for raw_id in raw_ids:
                item_id = str(raw_id or "").strip()
                if item_id and item_id not in seen:
                    seen.add(item_id)
                    cited.append(item_id)
    return cited


# ── Session metadata helpers (moved from agent/looping/memory_gate.py) ────────
# agent/looping/memory_gate.py re-exports these for backward compat.


def _extract_task_tools(tools_used: list[str]) -> list[str]:
    return [name for name in tools_used if name in {"update_now"}]


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

    turn_task_tools = _extract_task_tools(tools_used)
    turns = md.get("_task_tools_turns")
    if not isinstance(turns, list):
        turns = []
    turns.append(turn_task_tools)
    turns = turns[-2:]

    flat_recent: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        if not isinstance(turn, list):
            continue
        for name in turn:
            if isinstance(name, str) and name not in seen:
                seen.add(name)
                flat_recent.append(name)

    md["last_turn_tool_calls_count"] = call_count
    md["recent_task_tools"] = flat_recent
    md["last_turn_had_task_tool"] = bool(turn_task_tools)
    md["last_turn_ts"] = datetime.now().astimezone().isoformat()
    md["_task_tools_turns"] = turns
    session.metadata = md  # type: ignore[union-attr]
