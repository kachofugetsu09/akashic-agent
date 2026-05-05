from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryIngestRequest, MemoryScope, RefreshRecentTurnsRequest

if TYPE_CHECKING:
    from agent.core.types import ToolCallGroup
    from agent.looping.ports import TurnScheduler
    from bus.event_bus import EventBus
    from core.memory.engine import MemoryEngine
    from session.manager import SessionManager

logger = logging.getLogger("agent.loop.lifecycle")


def register_turn_committed_consumers(
    *,
    event_bus: "EventBus",
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
        if memory_engine is None:
            return
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
                await memory_engine.refresh_recent_turns(
                    RefreshRecentTurnsRequest(session=session)
                )
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
