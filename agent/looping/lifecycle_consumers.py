from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

from bus.events_lifecycle import TurnCommitted
from core.memory.engine import RefreshRecentTurnsRequest

if TYPE_CHECKING:
    from agent.looping.ports import TurnScheduler
    from bus.event_bus import EventBus
    from core.memory.engine import MemoryEngine
    from session.manager import SessionManager

logger = logging.getLogger("agent.loop.lifecycle")


# 注册 TurnCommitted 后的后台消费者：自然压缩和 recent context 刷新。
def register_turn_committed_consumers(
    *,
    event_bus: "EventBus",
    session_manager: "SessionManager",
    scheduler: "TurnScheduler",
    memory_engine: "MemoryEngine | None",
) -> None:
    recent_queues: dict[str, deque[str]] = {}
    recent_tasks: dict[str, asyncio.Task[None]] = {}

    # 自然压缩只交给 scheduler 判断阈值，consumer 不直接跑 LLM。
    def _schedule_consolidation(event: TurnCommitted) -> None:
        session = session_manager.get_or_create(event.session_key)
        scheduler.schedule_consolidation(session, event.session_key)

    # recent context 每轮都刷新 recent turns，但按 session 串行。
    def _enqueue_recent_context(event: TurnCommitted) -> None:
        if memory_engine is None:
            return

        # 1. 同一个 session 的刷新合并进队列，避免并发覆盖 RECENT_CONTEXT.md。
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

        # 2. 当前没有运行任务时，启动后台队列消费者。
        task = asyncio.create_task(
            _run_recent_context_queue(event.session_key),
            name=f"recent_context:{event.session_key}",
        )
        recent_tasks[event.session_key] = task
        task.add_done_callback(lambda t: _on_recent_context_done(t, event.session_key))

    async def _run_recent_context_queue(session_key: str) -> None:
        engine = memory_engine
        if engine is None:
            return
        try:
            # 1. 串行消费同 session 的 recent context 刷新请求。
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
                await engine.refresh_recent_turns(
                    RefreshRecentTurnsRequest(session=session)
                )
                logger.info(
                    "[_run_recent_context_queue] recent_context 刷新完成 session=%s remaining=%d",
                    session_key,
                    len(queue),
                )
        finally:
            # 2. 任务结束后若队列又有新请求，重新拉起消费者。
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
    event_bus.on(TurnCommitted, _enqueue_recent_context)
