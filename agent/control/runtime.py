from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast

from agent.control.errors import (
    ControlExecutionError,
    RuntimeClosedError,
    SlowConsumerError,
    ThreadBusyError,
    TurnNotFoundError,
)
from agent.control.events import TurnEvent
from agent.control.ids import new_item_id, new_turn_id
from agent.control.models import (
    TurnError,
    TurnItem,
    TurnItemKind,
    TurnRecord,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from agent.control.ports import ControlExecutionResult, TurnExecutor
from agent.restart import RestartCoordinator
from session.store import SessionStore
from agent.looping.interrupt import InterruptResult

logger = logging.getLogger(__name__)
_STREAM_END = object()
StreamValue = TurnEvent | BaseException | object


class TurnHandle:
    """持有一个 turn 的结果、事件流和精确中断入口。"""

    def __init__(self, runtime: ConversationRuntime, thread_id: str, turn_id: str) -> None:
        self._runtime = runtime
        self.thread_id = thread_id
        self.id = turn_id

    def record(self) -> dict[str, object]:
        return self._runtime.read_turn(self.thread_id, self.id).to_dict()

    async def result(self) -> TurnResult:
        return await self._runtime.wait_result(self.thread_id, self.id)

    def events(self) -> AsyncIterator[TurnEvent]:
        return self._runtime.subscribe(self.thread_id, self.id)

    async def interrupt(self) -> TurnRecord:
        return await self._runtime.interrupt_turn(self.thread_id, self.id)


class ConversationRuntime:
    """拥有 turn 排队、执行、中断、事件和持久状态。"""

    def __init__(
        self,
        store: SessionStore,
        executor: TurnExecutor,
        *,
        subscriber_queue_size: int = 256,
        restart_coordinator: RestartCoordinator | None = None,
    ) -> None:
        if subscriber_queue_size < 2:
            raise ValueError("subscriber_queue_size 必须至少为 2")
        self._store = store
        self._executor = executor
        self._subscriber_queue_size = subscriber_queue_size
        self._admission = asyncio.Lock()
        self._active_by_thread: dict[str, str] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, asyncio.Future[TurnResult]] = {}
        self._history: dict[str, list[TurnEvent]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[StreamValue]]] = {}
        self._interrupt_requested: set[str] = set()
        self._thread_idle: dict[str, asyncio.Event] = {}
        self._closed = False
        self._accepting_turns = True
        self._restart_owner_turn_id: str | None = None
        self._restart_coordinator = restart_coordinator

    async def start_turn(self, request: TurnRequest) -> TurnHandle:
        """持久化 queued turn 并立即返回可操作句柄。"""

        # 1. 在唯一 owner 处拒绝同 thread 并发。
        if self._closed or not self._accepting_turns:
            raise RuntimeClosedError("conversation runtime is shutting down")
        if request.thread_id in self._active_by_thread:
            raise ThreadBusyError(f"thread 已有 active turn: {request.thread_id}")

        # 2. 先持久化 handle，再让后台任务推进状态。
        turn_id = new_turn_id()
        user_item = TurnItem(
            TurnItemKind.USER_MESSAGE,
            new_item_id(),
            {"content": request.input},
        )
        record = self._store.create_turn(
            TurnRecord(
                id=turn_id,
                thread_id=request.thread_id,
                status=TurnStatus.QUEUED,
                input=request.input,
                metadata=dict(request.metadata),
                items=[user_item],
                usage=None,
                error=None,
                created_at=datetime.now(UTC),
            )
        )
        self._active_by_thread[request.thread_id] = turn_id
        self._thread_idle[request.thread_id] = asyncio.Event()
        loop = asyncio.get_running_loop()
        self._results[turn_id] = loop.create_future()
        self._history[turn_id] = []
        self._subscribers[turn_id] = set()
        self._publish(TurnEvent.create("turn/queued", request.thread_id, turn_id, turn=record.to_dict()))
        self._publish(
            TurnEvent.create(
                "item/started",
                request.thread_id,
                turn_id,
                item=user_item.to_dict(),
            )
        )
        self._publish(
            TurnEvent.create(
                "item/completed",
                request.thread_id,
                turn_id,
                item=user_item.to_dict(),
            )
        )
        task = asyncio.create_task(self._run(request, turn_id), name=f"conversation-turn:{turn_id}")
        self._tasks[turn_id] = task
        return TurnHandle(self, request.thread_id, turn_id)

    async def _run(self, request: TurnRequest, turn_id: str) -> None:
        """在全局 admission 内执行 turn，并保证只写一个终态。"""

        terminal: TurnRecord | None = None
        fatal_error: BaseException | None = None
        observed_items: dict[str, TurnItem] = {}
        open_item_ids: dict[str, None] = {}

        def close_observed_items(status: TurnStatus) -> list[TurnItem]:
            """闭合并返回本轮已经实时发布的全部 item。"""

            for item_id in tuple(open_item_ids):
                item = observed_items[item_id]
                closed = TurnItem(
                    item.kind,
                    item.id,
                    {**item.data, "status": status.value},
                )
                observed_items[item_id] = closed
                open_item_ids.pop(item_id)
                self._publish(
                    TurnEvent.create(
                        "item/completed",
                        request.thread_id,
                        turn_id,
                        item=closed.to_dict(),
                    )
                )
            return list(observed_items.values())

        try:
            # 1. 当前 v1 保留全局串行，但 queued 状态真实可见。
            async with self._admission:
                record = self._store.transition_turn(
                    turn_id,
                    expected_status=TurnStatus.QUEUED,
                    status=TurnStatus.IN_PROGRESS,
                    thread_id=request.thread_id,
                )
                self._publish(TurnEvent.create("turn/started", request.thread_id, turn_id, turn=record.to_dict()))

                # 2. 核心执行不依赖 transport；成功结果进入正式 assistant item。
                execution_request = TurnRequest(
                    request.thread_id,
                    request.input,
                    {**request.metadata, "turnId": turn_id},
                )
                live_item_ids: set[str] = set()

                def publish_item(method: str, item: TurnItem) -> None:
                    live_item_ids.add(item.id)
                    if method == "item/started":
                        if item.id in observed_items:
                            raise ValueError(f"item 重复 started: {item.id}")
                        observed_items[item.id] = item
                        open_item_ids[item.id] = None
                    elif method == "item/completed":
                        if item.id not in open_item_ids:
                            raise ValueError(f"item 未 started 即 completed: {item.id}")
                        observed_items[item.id] = item
                        open_item_ids.pop(item.id)
                    else:
                        raise ValueError(f"未知 control item event: {method}")
                    self._publish(
                        TurnEvent.create(
                            method,
                            request.thread_id,
                            turn_id,
                            item=item.to_dict(),
                        )
                    )

                execution_request.metadata["_controlItemEvent"] = publish_item
                execution = await self._executor(execution_request)
                if open_item_ids:
                    raise RuntimeError(
                        f"executor 返回时仍有未闭合 item: {sorted(open_item_ids)}"
                    )
                if isinstance(execution, str):
                    execution = ControlExecutionResult(execution)
                for item in execution.items:
                    if item.id in live_item_ids:
                        continue
                    self._publish(
                        TurnEvent.create(
                            "item/started",
                            request.thread_id,
                            turn_id,
                            item=item.to_dict(),
                        )
                    )
                    self._publish(
                        TurnEvent.create(
                            "item/completed",
                            request.thread_id,
                            turn_id,
                            item=item.to_dict(),
                        )
                    )
                assistant_item = TurnItem(
                    TurnItemKind.ASSISTANT_MESSAGE,
                    new_item_id(),
                    {"content": execution.response, **execution.assistant_data},
                )
                self._publish(
                    TurnEvent.create(
                        "item/started",
                        request.thread_id,
                        turn_id,
                        item=assistant_item.to_dict(),
                    )
                )
                deltas = execution.deltas or [execution.response]
                for sequence, delta in enumerate(deltas):
                    self._publish(
                        TurnEvent.create(
                            "item/assistantMessage/delta",
                            request.thread_id,
                            turn_id,
                            itemId=assistant_item.id,
                            delta=delta,
                            sequence=sequence,
                        )
                    )
                self._publish(
                    TurnEvent.create(
                        "item/completed",
                        request.thread_id,
                        turn_id,
                        item=assistant_item.to_dict(),
                    )
                )
                items = [*record.items, *execution.items, assistant_item]
                terminal = self._store.transition_turn(
                    turn_id,
                    expected_status=TurnStatus.IN_PROGRESS,
                    status=TurnStatus.COMPLETED,
                    thread_id=request.thread_id,
                    items=items,
                    final_response=execution.response,
                    usage=execution.usage,
                )
        except asyncio.CancelledError:
            current = self._store.read_turn(turn_id)
            if current is not None and current.status.is_terminal:
                terminal = current
            elif current is not None:
                status = (
                    TurnStatus.INTERRUPTED
                    if current.status is TurnStatus.IN_PROGRESS
                    and turn_id in self._interrupt_requested
                    else TurnStatus.CANCELLED
                )
                items = [*current.items, *close_observed_items(status)]
                terminal = self._store.transition_turn(
                    turn_id,
                    expected_status=current.status,
                    status=status,
                    thread_id=request.thread_id,
                    items=items,
                )
        except Exception as exc:
            logger.exception("conversation turn failed thread=%s turn=%s", request.thread_id, turn_id)
            current = self._store.read_turn(turn_id)
            if current is not None and current.status is TurnStatus.IN_PROGRESS:
                items = [
                    *current.items,
                    *close_observed_items(TurnStatus.FAILED),
                ]
                terminal = self._store.transition_turn(
                    turn_id,
                    expected_status=current.status,
                    status=TurnStatus.FAILED,
                    thread_id=request.thread_id,
                    items=items,
                    error=TurnError(
                        type=(
                            exc.error_type
                            if isinstance(exc, ControlExecutionError)
                            else type(exc).__name__
                        ),
                        message=str(exc),
                        retryable=bool(getattr(exc, "retryable", False)),
                    ),
                )
            else:
                fatal_error = exc
        finally:
            # 3. terminal 是唯一结束通知；结果 future 与 active owner 一起收束。
            future = self._results[turn_id]
            if terminal is not None:
                if self._restart_coordinator is not None:
                    self._restart_coordinator.mark_turn_terminal(
                        turn_id,
                        terminal.status.value,
                    )
                event = TurnEvent.create("turn/completed", request.thread_id, turn_id, turn=terminal.to_dict())
                self._publish(event)
                if not future.done():
                    future.set_result(TurnResult.from_record(terminal))
                self._finish_streams(turn_id)
            else:
                error = fatal_error or RuntimeError(f"turn 未建立终态: {turn_id}")
                if not future.done():
                    future.set_exception(error)
                self._fail_streams(turn_id, error)
            _ = self._active_by_thread.pop(request.thread_id, None)
            idle = self._thread_idle.pop(request.thread_id, None)
            if idle is not None:
                idle.set()
            _ = self._tasks.pop(turn_id, None)
            self._interrupt_requested.discard(turn_id)

    def _publish(self, event: TurnEvent) -> None:
        self._history[event.turn_id].append(event)
        for queue in tuple(self._subscribers[event.turn_id]):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                while not queue.empty():
                    _ = queue.get_nowait()
                queue.put_nowait(SlowConsumerError(f"turn event consumer too slow: {event.turn_id}"))
                self._subscribers[event.turn_id].discard(queue)

    def _finish_streams(self, turn_id: str) -> None:
        for queue in tuple(self._subscribers[turn_id]):
            try:
                queue.put_nowait(_STREAM_END)
            except asyncio.QueueFull:
                while not queue.empty():
                    _ = queue.get_nowait()
                queue.put_nowait(SlowConsumerError(f"turn event consumer too slow: {turn_id}"))

    def _fail_streams(self, turn_id: str, error: BaseException) -> None:
        for queue in tuple(self._subscribers[turn_id]):
            while not queue.empty():
                _ = queue.get_nowait()
            queue.put_nowait(error)

    async def subscribe(self, thread_id: str, turn_id: str) -> AsyncIterator[TurnEvent]:
        record = self.read_turn(thread_id, turn_id)
        queue: asyncio.Queue[StreamValue] = asyncio.Queue(self._subscriber_queue_size)
        history = self._history.get(turn_id)
        if history is None:
            if record.status.is_terminal:
                return
            raise TurnNotFoundError(f"turn 不在当前 runtime: {thread_id}/{turn_id}")
        for event in history:
            queue.put_nowait(event)
        if record.status.is_terminal:
            queue.put_nowait(_STREAM_END)
        else:
            self._subscribers[turn_id].add(queue)
        try:
            while True:
                value = await queue.get()
                if value is _STREAM_END:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield cast(TurnEvent, value)
        finally:
            self._subscribers.get(turn_id, set()).discard(queue)

    def read_turn(self, thread_id: str, turn_id: str) -> TurnRecord:
        record = self._store.read_turn(turn_id)
        if record is None or record.thread_id != thread_id:
            raise TurnNotFoundError(f"turn 不存在: {thread_id}/{turn_id}")
        return record

    def is_thread_active(self, thread_id: str) -> bool:
        return thread_id in self._active_by_thread

    def quiesce_for_restart(self, caller_turn_id: str) -> None:
        """仅在 caller 是唯一 turn 时冻结新的 turn 准入。"""

        # 1. caller 必须是当前 runtime 唯一已经持久化的 turn。
        active_turns = set(self._tasks)
        if caller_turn_id not in active_turns:
            raise RuntimeClosedError(f"restart caller turn 不在当前 runtime: {caller_turn_id}")
        others = active_turns - {caller_turn_id}
        if others:
            raise RuntimeClosedError(
                f"仍有其他 turn 等待或执行，拒绝重启: {sorted(others)}"
            )
        if not self._accepting_turns:
            if self._restart_owner_turn_id == caller_turn_id:
                return
            raise RuntimeClosedError("conversation runtime 已在排空")

        # 2. 不获取全局 admission，避免 caller 在工具执行中自锁。
        self._accepting_turns = False
        self._restart_owner_turn_id = caller_turn_id

    def resume_after_restart_cancel(self, caller_turn_id: str) -> None:
        """只允许原 restart owner 在提交前恢复准入。"""

        if self._restart_owner_turn_id != caller_turn_id:
            raise RuntimeError(
                f"restart admission owner 不匹配: {caller_turn_id}"
            )
        self._restart_owner_turn_id = None
        if not self._closed:
            self._accepting_turns = True

    async def wait_thread_available(self, thread_id: str) -> None:
        """等待当前 thread owner 释放，不获取新的 owner。"""

        while event := self._thread_idle.get(thread_id):
            await event.wait()

    async def wait_result(self, thread_id: str, turn_id: str) -> TurnResult:
        record = self.read_turn(thread_id, turn_id)
        if record.status.is_terminal:
            return TurnResult.from_record(record)
        future = self._results.get(turn_id)
        if future is None:
            raise TurnNotFoundError(f"turn 不在当前 runtime: {thread_id}/{turn_id}")
        return await asyncio.shield(future)

    async def interrupt_turn(self, thread_id: str, turn_id: str) -> TurnRecord:
        record = self.read_turn(thread_id, turn_id)
        if record.status.is_terminal:
            return record
        if self._active_by_thread.get(thread_id) != turn_id:
            raise TurnNotFoundError(f"active turn 不匹配: {thread_id}/{turn_id}")
        if record.status is TurnStatus.QUEUED:
            # 1. 先让已启动 task 自行收束；启动前取消则由 owner 补交 cancelled。
            task = self._tasks[turn_id]
            task.cancel()
            future = self._results[turn_id]
            _ = await asyncio.gather(task, return_exceptions=True)
            if future.done():
                return self.read_turn(thread_id, turn_id)
            terminal = self._store.transition_turn(
                turn_id,
                expected_status=TurnStatus.QUEUED,
                status=TurnStatus.CANCELLED,
                thread_id=thread_id,
            )
            self._publish(TurnEvent.create("turn/completed", thread_id, turn_id, turn=terminal.to_dict()))
            future.set_result(TurnResult.from_record(terminal))
            self._finish_streams(turn_id)
            _ = self._active_by_thread.pop(thread_id, None)
            idle = self._thread_idle.pop(thread_id, None)
            if idle is not None:
                idle.set()
            _ = self._tasks.pop(turn_id, None)
            return terminal

        # 2. in-progress task 自己在取消处理器中提交 interrupted。
        self._interrupt_requested.add(turn_id)
        task = self._tasks[turn_id]
        task.cancel()
        await asyncio.shield(self._results[turn_id])
        return self.read_turn(thread_id, turn_id)

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult:
        """为现有 channel 命令提供 session 定位的同步 adapter。"""
        turn_id = self._active_by_thread.get(session_key)
        if turn_id is None:
            return InterruptResult("idle", session_key, "当前没有正在执行的任务。")
        _ = asyncio.create_task(
            self.interrupt_turn(session_key, turn_id),
            name=f"channel-interrupt:{turn_id}",
        )
        return InterruptResult(
            "interrupted",
            session_key,
            "本轮已中断。你可以继续补充要求，我会接着这件事处理。",
        )

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
