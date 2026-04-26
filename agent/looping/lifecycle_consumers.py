from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from bus.events_lifecycle import TurnCommitted

if TYPE_CHECKING:
    from agent.looping.consolidation import ConsolidationService
    from agent.looping.ports import ObservabilityServices
    from bus.event_bus import EventBus
    from session.manager import SessionManager

logger = logging.getLogger("agent.loop.lifecycle")


@runtime_checkable
class _ObserveWriter(Protocol):
    def emit(self, event: object) -> None: ...


def register_post_turn_consumers(
    *,
    event_bus: "EventBus",
    consolidation: "ConsolidationService",
    session_manager: "SessionManager",
) -> None:
    queues: dict[str, deque[str]] = {}
    tasks: dict[str, asyncio.Task[None]] = {}

    def _enqueue_recent_context(event: TurnCommitted) -> None:
        queue = queues.setdefault(event.session_key, deque())
        queue.append(event.session_key)
        running = event.session_key in tasks
        source = (
            "proactive"
            if bool((event.extra or {}).get("skip_observe_trace"))
            else "passive"
        )
        logger.info(
            "[_enqueue_recent_context] recent_context 已入队 source=%s session=%s pending=%d running=%s",
            source,
            event.session_key,
            len(queue),
            running,
        )
        if running:
            return
        task = asyncio.create_task(
            _run_recent_context_queue(event.session_key),
            name=f"recent_context:{event.session_key}",
        )
        tasks[event.session_key] = task
        task.add_done_callback(lambda t: _on_recent_context_done(t, event.session_key))

    async def _run_recent_context_queue(session_key: str) -> None:
        try:
            while True:
                queue = queues.get(session_key)
                if not queue:
                    return
                _ = queue.popleft()
                logger.info(
                    "[_run_recent_context_queue] recent_context 刷新开始 session=%s remaining=%d",
                    session_key,
                    len(queue),
                )
                session = session_manager.get_or_create(session_key)
                await consolidation.refresh_recent_turns(session=session)
                logger.info(
                    "[_run_recent_context_queue] recent_context 刷新完成 session=%s remaining=%d",
                    session_key,
                    len(queue),
                )
        finally:
            _ = tasks.pop(session_key, None)
            queue = queues.get(session_key)
            if queue:
                task = asyncio.create_task(
                    _run_recent_context_queue(session_key),
                    name=f"recent_context:{session_key}",
                )
                tasks[session_key] = task
                task.add_done_callback(lambda t: _on_recent_context_done(t, session_key))
            else:
                _ = queues.pop(session_key, None)

    def _on_recent_context_done(task: asyncio.Task[None], key: str) -> None:
        if task.cancelled():
            logger.info("recent_context refresh cancelled: %s", key)
            return
        try:
            exc = task.exception()
        except Exception as e:
            logger.warning(
                "recent_context refresh inspect failed session=%s err=%s",
                key,
                e,
            )
            return
        if exc is not None:
            logger.warning("recent_context refresh failed: session=%s err=%s", key, exc)

    event_bus.on(TurnCommitted, _enqueue_recent_context)


def register_observe_trace_consumers(
    *,
    event_bus: "EventBus",
    trace: "ObservabilityServices",
) -> None:
    writer = trace.observe_writer
    if not isinstance(writer, _ObserveWriter):
        return

    def _observe_turn_committed(event: TurnCommitted) -> None:
        if bool((event.extra or {}).get("skip_observe_trace")):
            logger.info(
                "[_observe_turn_committed] turn_trace 跳过 session=%s reason=skip_observe_trace",
                event.session_key,
            )
            return
        _emit_turn_trace(writer, event)

    event_bus.on(TurnCommitted, _observe_turn_committed)


def _emit_turn_trace(writer: _ObserveWriter, event: TurnCommitted) -> None:
    from core.observe.events import TurnTrace as TurnTraceEvent

    post_reply_budget = event.post_reply_budget
    react_stats = event.react_stats
    tool_chain = event.tool_chain_raw
    tool_chain_json = (
        json.dumps(_slim_tool_chain(tool_chain), ensure_ascii=False)
        if tool_chain
        else None
    )
    writer.emit(
        TurnTraceEvent(
            source="agent",
            session_key=event.session_key,
            user_msg=event.persisted_user_message,
            llm_output=event.assistant_response,
            raw_llm_output=event.raw_reply,
            meme_tag=event.meme_tag,
            meme_media_count=event.meme_media_count,
            tool_calls=_slim_tool_calls(tool_chain),
            tool_chain_json=tool_chain_json,
            history_window=post_reply_budget.get("history_window"),
            history_messages=post_reply_budget.get("history_messages"),
            history_chars=post_reply_budget.get("history_chars"),
            history_tokens=post_reply_budget.get("history_tokens"),
            prompt_tokens=post_reply_budget.get("prompt_tokens"),
            next_turn_baseline_tokens=post_reply_budget.get(
                "next_turn_baseline_tokens"
            ),
            react_iteration_count=react_stats.get("iteration_count"),
            react_input_sum_tokens=react_stats.get("turn_input_sum_tokens"),
            react_input_peak_tokens=react_stats.get("turn_input_peak_tokens"),
            react_final_input_tokens=react_stats.get("final_call_input_tokens"),
            react_cache_prompt_tokens=react_stats.get("cache_prompt_tokens"),
            react_cache_hit_tokens=react_stats.get("cache_hit_tokens"),
        )
    )
    if event.retrieval_raw is not None:
        writer.emit(event.retrieval_raw)
    logger.info(
        "[_emit_turn_trace] turn_trace 已入队 session=%s tool_calls=%d retrieval=%s",
        event.session_key,
        len(_slim_tool_calls(tool_chain)),
        event.retrieval_raw is not None,
    )


def _slim_tool_calls(tool_chain: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {
            "name": str(call.get("name", "")),
            "args": str(call.get("arguments", ""))[:300],
            "result": str(call.get("result", ""))[:500],
        }
        for group in tool_chain
        for call in _group_calls(group)
    ]


def _slim_tool_chain(tool_chain: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "text": str(group.get("text") or ""),
            "calls": [
                {
                    "name": str(call.get("name", "")),
                    "args": str(call.get("arguments", ""))[:800],
                    "result": str(call.get("result", ""))[:1200],
                }
                for call in _group_calls(group)
            ],
        }
        for group in tool_chain
    ]


def _group_calls(group: dict[str, object]) -> list[dict[str, object]]:
    calls = group.get("calls")
    if not isinstance(calls, list):
        return []
    raw_calls = cast(list[object], calls)
    out: list[dict[str, object]] = []
    for call in raw_calls:
        if isinstance(call, Mapping):
            mapping = cast(Mapping[object, object], call)
            out.append(
                {
                    str(key): value
                    for key, value in mapping.items()
                    if isinstance(key, str)
                }
            )
    return out
