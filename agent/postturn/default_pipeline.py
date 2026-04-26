from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import TYPE_CHECKING

from agent.core.types import ToolCallGroup
from agent.looping.ports import TurnScheduler
from agent.postturn.protocol import PostTurnEvent, PostTurnPipeline
from core.memory.engine import MemoryIngestRequest, MemoryScope

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine
    from bus.event_bus import EventBus

logger = logging.getLogger("agent.postturn")


class DefaultPostTurnPipeline(PostTurnPipeline):
    def __init__(
        self,
        scheduler: TurnScheduler,
        engine: "MemoryEngine | None",
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._scheduler = scheduler
        self._engine = engine
        _ = event_bus
        self._failures: int = 0
        self._post_mem_queues: dict[str, deque[PostTurnEvent]] = {}
        self._post_mem_tasks: dict[str, asyncio.Task[None]] = {}

    def schedule(self, event: PostTurnEvent) -> None:
        # 1. 回复一落库就先尝试挂起 consolidation；是否真的执行由 scheduler 决定。
        self._scheduler.schedule_consolidation(event.session, event.session_key)
        if bool((event.extra or {}).get("skip_post_memory")):
            return
        if self._engine is None:
            return
        queue = self._post_mem_queues.setdefault(event.session_key, deque())
        queue.append(event)
        if event.session_key in self._post_mem_tasks:
            return
        task = asyncio.create_task(
            self._run_post_memory_queue(event.session_key),
            name=f"post_mem_queue:{event.session_key}",
        )
        self._post_mem_tasks[event.session_key] = task
        task.add_done_callback(lambda t: self._on_done(t, event.session_key))

    async def _run_post_memory_queue(self, session_key: str) -> None:
        try:
            while True:
                queue = self._post_mem_queues.get(session_key)
                if not queue:
                    return
                event = queue.popleft()
                try:
                    await self._ingest_event(event)
                except Exception as e:
                    self._failures += 1
                    logger.warning(
                        "post_mem failed session=%s failures=%d err=%s",
                        session_key, self._failures, e,
                    )
        finally:
            _ = self._post_mem_tasks.pop(session_key, None)
            queue = self._post_mem_queues.get(session_key)
            if queue:
                task = asyncio.create_task(
                    self._run_post_memory_queue(session_key),
                    name=f"post_mem_queue:{session_key}",
                )
                self._post_mem_tasks[session_key] = task
                task.add_done_callback(lambda t: self._on_done(t, session_key))
            else:
                _ = self._post_mem_queues.pop(session_key, None)

    async def _ingest_event(self, event: PostTurnEvent) -> None:
        if self._engine is None:
            return
        tool_chain_raw: list[dict[str, object]] = [
            _tool_group_to_dict(g) for g in event.tool_chain
        ]
        source_ref = f"{event.session_key}@post_response"
        _ = await self._engine.ingest(
            MemoryIngestRequest(
                content={
                    "user_message": event.user_message,
                    "assistant_response": event.assistant_response,
                    "tool_chain": tool_chain_raw,
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

    def _on_done(self, task: asyncio.Task[None], key: str) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.info("post_mem cancelled: %s", key)
            return
        except Exception as e:
            self._failures += 1
            logger.warning(
                "post_mem inspect failed session=%s failures=%d err=%s",
                key, self._failures, e,
            )
            return
        if exc is not None:
            self._failures += 1
            logger.warning(
                "post_mem failed session=%s failures=%d err=%s",
                key, self._failures, exc,
            )


def _tool_group_to_dict(group: ToolCallGroup) -> dict[str, object]:
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
