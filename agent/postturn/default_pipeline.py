from __future__ import annotations

import asyncio
import logging

from agent.looping.turn_types import ToolCallGroup

logger = logging.getLogger("agent.postturn")
from agent.looping.ports import TurnScheduler
from agent.postturn.protocol import PostTurnEvent, PostTurnPipeline
from memory2.post_response_worker import PostResponseMemoryWorker

class DefaultPostTurnPipeline(PostTurnPipeline):
    def __init__(
        self,
        scheduler: TurnScheduler,
        post_mem_worker: PostResponseMemoryWorker | None,
    ) -> None:
        self._scheduler = scheduler
        self._post_mem_worker = post_mem_worker
        self._failures: int = 0

    def schedule(self, event: PostTurnEvent) -> None:
        # 1. 回复一落库就先尝试挂起 consolidation；是否真的执行由 scheduler 决定。
        self._scheduler.schedule_consolidation(event.session, event.session_key)
        if not self._post_mem_worker:
            return
        if bool((event.extra or {}).get("skip_post_memory")):
            return
        # 2. post-response memory 是另一条并行后台链，和 consolidation 分开跑。
        tool_chain_raw = [_tool_group_to_dict(g) for g in event.tool_chain]
        task = asyncio.create_task(
            self._post_mem_worker.run(
                user_msg=event.user_message,
                agent_response=event.assistant_response,
                tool_chain=tool_chain_raw,
                source_ref=f"{event.session_key}@post_response",
                session_key=event.session_key,
            ),
            name=f"post_mem:{event.session_key}",
        )
        task.add_done_callback(lambda t: self._on_done(t, event.session_key))

    def _on_done(self, task: asyncio.Task, key: str) -> None:
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


def _tool_group_to_dict(group: ToolCallGroup) -> dict:
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
