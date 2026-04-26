from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryIngestRequest, MemoryScope

if TYPE_CHECKING:
    from agent.core.types import ToolCallGroup
    from agent.looping.consolidation import ConsolidationService
    from agent.looping.ports import ObservabilityServices, TurnScheduler
    from bus.event_bus import EventBus
    from core.memory.engine import MemoryEngine
    from session.manager import SessionManager

logger = logging.getLogger("agent.loop.lifecycle")


@runtime_checkable
class _ObserveWriter(Protocol):
    def emit(self, event: object) -> None: ...


def register_turn_committed_consumers(
    *,
    event_bus: "EventBus",
    consolidation: "ConsolidationService",
    session_manager: "SessionManager",
    scheduler: "TurnScheduler",
    memory_engine: "MemoryEngine | None",
) -> None:
    recent_queues: dict[str, deque[str]] = {}
    recent_tasks: dict[str, asyncio.Task[None]] = {}
    memory_queues: dict[str, deque[TurnCommitted]] = {}
    memory_tasks: dict[str, asyncio.Task[None]] = {}
    memory_failures = 0

    def _schedule_consolidation(event: TurnCommitted) -> None:
        session = session_manager.get_or_create(event.session_key)
        scheduler.schedule_consolidation(session, event.session_key)

    def _enqueue_post_memory(event: TurnCommitted) -> None:
        if bool((event.extra or {}).get("skip_post_memory")):
            return
        if memory_engine is None:
            return
        queue = memory_queues.setdefault(event.session_key, deque())
        queue.append(event)
        if event.session_key in memory_tasks:
            return
        task = asyncio.create_task(
            _run_post_memory_queue(event.session_key),
            name=f"post_mem_queue:{event.session_key}",
        )
        memory_tasks[event.session_key] = task
        task.add_done_callback(lambda t: _on_post_memory_done(t, event.session_key))

    async def _run_post_memory_queue(session_key: str) -> None:
        nonlocal memory_failures
        try:
            while True:
                queue = memory_queues.get(session_key)
                if not queue:
                    return
                event = queue.popleft()
                try:
                    await _ingest_post_memory(event)
                except Exception as e:
                    memory_failures += 1
                    logger.warning(
                        "post_mem failed session=%s failures=%d err=%s",
                        session_key,
                        memory_failures,
                        e,
                    )
        finally:
            _ = memory_tasks.pop(session_key, None)
            queue = memory_queues.get(session_key)
            if queue:
                task = asyncio.create_task(
                    _run_post_memory_queue(session_key),
                    name=f"post_mem_queue:{session_key}",
                )
                memory_tasks[session_key] = task
                task.add_done_callback(lambda t: _on_post_memory_done(t, session_key))
            else:
                _ = memory_queues.pop(session_key, None)

    async def _ingest_post_memory(event: TurnCommitted) -> None:
        if memory_engine is None:
            return
        source_ref = f"{event.session_key}@post_response"
        _ = await memory_engine.ingest(
            MemoryIngestRequest(
                content={
                    "user_message": event.input_message,
                    "assistant_response": event.assistant_response,
                    "tool_chain": [
                        _tool_group_to_dict(group) for group in event.tool_call_groups
                    ],
                    "source_ref": source_ref,
                },
                source_kind="conversation_turn",
                scope=MemoryScope(
                    session_key=event.session_key,
                    channel=event.channel,
                    chat_id=event.chat_id,
                ),
                metadata={"source_ref": source_ref},
            )
        )

    def _on_post_memory_done(task: asyncio.Task[None], key: str) -> None:
        nonlocal memory_failures
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.info("post_mem cancelled: %s", key)
            return
        except Exception as e:
            memory_failures += 1
            logger.warning(
                "post_mem inspect failed session=%s failures=%d err=%s",
                key,
                memory_failures,
                e,
            )
            return
        if exc is not None:
            memory_failures += 1
            logger.warning(
                "post_mem failed session=%s failures=%d err=%s",
                key,
                memory_failures,
                exc,
            )

    def _enqueue_recent_context(event: TurnCommitted) -> None:
        queue = recent_queues.setdefault(event.session_key, deque())
        queue.append(event.session_key)
        running = event.session_key in recent_tasks
        logger.info(
            "[_enqueue_recent_context] recent_context 已入队 session=%s pending=%d running=%s",
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
        recent_tasks[event.session_key] = task
        task.add_done_callback(lambda t: _on_recent_context_done(t, event.session_key))

    async def _run_recent_context_queue(session_key: str) -> None:
        try:
            while True:
                queue = recent_queues.get(session_key)
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
            _ = recent_tasks.pop(session_key, None)
            queue = recent_queues.get(session_key)
            if queue:
                task = asyncio.create_task(
                    _run_recent_context_queue(session_key),
                    name=f"recent_context:{session_key}",
                )
                recent_tasks[session_key] = task
                task.add_done_callback(lambda t: _on_recent_context_done(t, session_key))
            else:
                _ = recent_queues.pop(session_key, None)

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

    event_bus.on(TurnCommitted, _schedule_consolidation)
    event_bus.on(TurnCommitted, _enqueue_post_memory)
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


def _tool_group_to_dict(group: "ToolCallGroup") -> dict[str, object]:
    return {
        "text": group.text,
        "calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "result": call.result,
            }
            for call in group.calls
        ],
    }
