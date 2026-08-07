from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from agent.control.errors import RuntimeClosedError, ThreadBusyError
from agent.control.models import TurnRequest, TurnStatus
from agent.control.runtime import ConversationRuntime, TurnHandle
from agent.looping.core import AgentLoop
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus

logger = logging.getLogger(__name__)


class PassiveMessageWorker:
    """把渠道入站消息转换为 ConversationRuntime turn。"""

    def __init__(
        self, bus: MessageBus, runtime: ConversationRuntime, legacy_loop: AgentLoop
    ) -> None:
        self._bus = bus
        self._runtime = runtime
        self._legacy_loop = legacy_loop
        self._running = False
        self._lane_queues: dict[str, asyncio.Queue[InboundMessage | object]] = {}
        self._lane_tasks: dict[str, asyncio.Task[None]] = {}
        self._result_tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        self._running = True
        try:
            # 1. 仅重放有界 durable handoff 页，lane 准入由 MessageBus 统一持有。
            await self._bus.recover_durable_inbounds()
            while self._running:
                try:
                    item = await asyncio.wait_for(
                        self._bus.consume_inbound(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                self._enqueue(item)
        finally:
            self._running = False
            for task in tuple(self._lane_tasks.values()):
                task.cancel()
            if self._lane_tasks:
                await asyncio.gather(
                    *tuple(self._lane_tasks.values()),
                    return_exceptions=True,
                )
            self._lane_tasks.clear()
            self._lane_queues.clear()
            for task in tuple(self._result_tasks):
                task.cancel()
            if self._result_tasks:
                await asyncio.gather(*tuple(self._result_tasks), return_exceptions=True)
            self._result_tasks.clear()

    def _enqueue(self, item: object) -> None:
        key = cast(Any, item).session_key
        queue = self._lane_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(item)
        task = self._lane_tasks.get(key)
        if task is None or task.done():
            self._lane_tasks[key] = asyncio.create_task(
                self._run_lane(key, queue),
                name=f"passive-lane:{key}",
            )

    async def _run_lane(
        self,
        key: str,
        queue: asyncio.Queue[InboundMessage | object],
    ) -> None:
        """串行执行单 thread 队列，并隔离单条消息失败。"""

        while True:
            item = await queue.get()
            try:
                if isinstance(item, InboundMessage):
                    await self._admit_message(item)
                else:
                    await self._legacy_loop._run_inbound_turn(cast(Any, item))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("passive lane message failed thread=%s", key)
            if queue.empty():
                task = asyncio.current_task()
                if self._lane_tasks.get(key) is task:
                    self._lane_tasks.pop(key)
                    self._lane_queues.pop(key)
                return

    async def _run_message(self, item: InboundMessage) -> None:
        """兼容直接调用：等待由本条消息新建的 turn 完成。"""

        result_task = await self._admit_message(item)
        if result_task is not None:
            await result_task

    async def _admit_message(
        self,
        item: InboundMessage,
    ) -> asyncio.Task[None] | None:
        """快速准入渠道消息，并把唯一终态发送职责交给新 turn owner。"""

        # 1. canonical 用户消息证明该 handoff 已进入过 turn。
        if self._is_recovered_mobile_duplicate(item):
            await self._bus.complete_inbound(item)
            return
        if item.channel == "mobile" and item.session_admission_id is None:
            _, item.session_admission_id = (
                self._legacy_loop.session_manager.admit_existing(item.session_key)
            )
        transferred = False
        try:
            # 2. 渠道信息只作为 executor 所需的受控 metadata，不改变 thread identity。
            request = TurnRequest(
                item.session_key,
                item.content,
                {
                    "channel": item.channel,
                    "chatId": item.chat_id,
                    "sender": item.sender,
                    "media": list(item.media),
                    "inputTimestamp": item.timestamp.isoformat(),
                    "inboundMetadata": dict(item.metadata),
                },
            )
            while True:
                try:
                    handle = await self._runtime.start_turn(request)
                except ThreadBusyError:
                    await self._runtime.wait_thread_available(item.session_key)
                    continue
                except RuntimeClosedError:
                    await self._bus.publish_outbound(
                        OutboundMessage(
                            channel=item.channel,
                            chat_id=item.chat_id,
                            content="服务正在重启，请稍后重发这条消息。",
                        )
                    )
                    return
                break
            task = asyncio.create_task(
                self._finish_message(item, handle),
                name=f"passive-result:{handle.id}",
            )
            self._result_tasks.add(task)
            task.add_done_callback(self._result_tasks.discard)
            transferred = True
            return task
        finally:
            if not transferred:
                await self._complete_message(item)

    async def _finish_message(self, item: InboundMessage, handle: TurnHandle) -> None:
        """等待新 turn 的唯一终态并发送一次 outbound。"""

        try:
            result = await handle.result()
            if result.status is TurnStatus.COMPLETED:
                assistant = next(
                    entry
                    for entry in reversed(result.items)
                    if entry.kind.value == "assistantMessage"
                )
                data = assistant.data
                outbound = OutboundMessage(
                    channel=item.channel,
                    chat_id=item.chat_id,
                    content=result.final_response or "",
                    thinking=cast(str | None, data.get("thinking")),
                    reply_to=cast(str | None, data.get("replyTo")),
                    media=list(cast(list[str], data.get("media", []))),
                    metadata=dict(cast(dict[str, Any], data.get("metadata", {}))),
                    control_turn_id=handle.id,
                    session_message_id=cast(str | None, data.get("sessionMessageId")),
                )
            elif result.status is TurnStatus.FAILED:
                outbound = OutboundMessage(
                    channel=item.channel,
                    chat_id=item.chat_id,
                    content="处理消息时出错，请稍后再试。",
                )
            else:
                return
            await self._bus.publish_outbound(outbound)
        finally:
            await self._complete_message(item)

    async def _complete_message(self, item: InboundMessage) -> None:
        """确认 durable inbound 并释放 mobile session admission。"""

        try:
            await self._bus.complete_inbound(item)
        finally:
            if item.session_admission_id is not None:
                self._legacy_loop.session_manager.release_admission(
                    item.session_admission_id
                )

    def _is_recovered_mobile_duplicate(self, item: InboundMessage) -> bool:
        """识别 canonical 用户消息已存在的移动 handoff，避免重复 turn。"""

        if item.channel != "mobile" or item.handoff_id is None:
            return False
        client_message_id = item.metadata.get("client_message_id")
        if not isinstance(client_message_id, str) or not client_message_id:
            return False
        message = (
            self._legacy_loop.session_manager.control_store.get_message_by_client_id(
                item.session_key,
                client_message_id,
            )
        )
        return (
            message is not None
            or self._legacy_loop.session_manager.control_store.has_turn_user_input_by_client_id(
                item.session_key,
                client_message_id,
            )
        )

    def stop(self) -> None:
        self._running = False
