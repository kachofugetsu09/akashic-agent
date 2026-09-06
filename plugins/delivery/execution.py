from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Hashable
from contextlib import AbstractContextManager, AsyncExitStack
from typing import cast

from agent.plugin_composition.tasks import Task, TaskAdmission, TaskSlot
from session.log import MessageCatalog, MessageReader, MessageWriter
from session.message import Body, Message

from .api import OpenSender, Receipt, Sink
from .records import Delivery, DeliveryRecords, Selection, delivery_key


class Deliveries:
    """独立发送已 prepared 的消息；重试只使用原绑定、地址和幂等键。"""

    def __init__(
        self, records: DeliveryRecords, catalog: MessageCatalog,
        tasks: TaskAdmission, open_sender: OpenSender, *, task_key: Hashable,
    ):
        self._records = records
        self._catalog = catalog
        self._tasks = tasks
        self._open_sender = open_sender
        self._task_key = task_key

    def prepare(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...], *, passive: bool = False) -> Selection:
        return self._records.prepare(reader, message, sinks, passive=passive)

    def publish(self, writer: MessageWriter, message_id: str, body: Body,
                sinks: tuple[Sink, ...], *, passive: bool = False) -> tuple[Message, Selection]:
        return self._records.publish(writer, message_id, body, sinks, passive=passive)

    def consume(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...] | None, *, passive: bool = False) -> Selection | None:
        selected = self._records.consume(reader, message, sinks, passive=passive)
        return selected if selected is not None and selected.recovery_owner == self._records.recovery_owner else None

    def cursor(self, session_id: str) -> int:
        return self._records.cursor(session_id)

    def selection(self, message_id: str) -> Selection | None:
        return self._records.selection(message_id)

    def add(self, message_id: str, sink: Sink) -> None:
        self._records.add(message_id, sink)

    def destination(self, message_id: str, sink: str) -> Sink:
        return self._records.read(message_id, sink)[1].sink

    def receipt(self, message_id: str, sink: str) -> Receipt | None:
        return self._records.read(message_id, sink)[1].receipt

    def pending(self) -> tuple[tuple[str, str], ...]:
        return self._records.pending()

    def activity(self, channel: str, address: str) -> AbstractContextManager[None]:
        """目标的被动工作从回复接纳持续到实际发送或查询结算。"""
        return self._tasks.activity((self._task_key, "target", channel, address))

    async def wait_idle(self, channel: str, address: str) -> None:
        """推理前等待当前被动工作；发送仍在实际边界重新检查。"""
        await self._tasks.wait_idle((self._task_key, "target", channel, address))

    async def _start(self, message_id: str, sink: str, before_start: Callable[[], str | None] | None = None) -> tuple[Task, bool]:
        """同步接纳时先占被动活动；实际 I/O 的活动范围继续覆盖取消后的清理。"""
        selected = self._records.check_owner(message_id)
        _, delivery = self._records.read(message_id, sink)

        def admit(slot: TaskSlot) -> tuple[Task, bool]:
            if slot.current is not None:
                return slot.current, False
            hold = self.activity(delivery.sink.name, delivery.sink.address)
            claimed = selected.passive and delivery.phase not in {"delivered", "rejected"}
            if claimed:
                _ = hold.__enter__()

            async def run(task: Task) -> Receipt:
                return await self._send(task, message_id, sink, before_start)

            try:
                task = slot.start(run)
                if claimed:
                    def release() -> None:
                        _ = hold.__exit__(None, None, None)
                    task.on_close(release)
            except BaseException:
                if claimed:
                    _ = hold.__exit__(None, None, None)
                raise
            return task, True

        return await self._tasks.admit((self._task_key, message_id, sink), admit)

    async def start(self, message_id: str, sink: str, *, before_start: Callable[[], str | None] | None = None) -> Task:
        """独立启动原效果并返回等待句柄；后来的回复取消不取得它的撤销权。"""
        task, _ = await self._start(message_id, sink, before_start)
        return task

    async def send(self, message_id: str, sink: str, *, before_start: Callable[[], str | None] | None = None) -> Receipt:
        """发送并等待；重复等待者不取得实际发送任务的取消权。"""
        task, owned = await self._start(message_id, sink, before_start)
        try:
            return cast(Receipt, await task.join())
        except asyncio.CancelledError:
            if owned:
                task.cancel()
                await _drain(task)
            raise

    async def retry(self, message_id: str, sink: str) -> Receipt:
        """显式重试被拒绝的原效果；不改消息、目的地、绑定或幂等键。"""
        def rearm(slot: TaskSlot) -> Task | None:
            record, delivery = self._records.read(message_id, sink)
            if delivery.phase != "rejected":
                return None
            # 旧 Task 可能已读取 rejected 但仍在关闭 scope；先排空才能重新接纳。
            if slot.current is not None:
                return slot.current
            _ = self._records.save(message_id, record, Delivery(sink=delivery.sink, phase="prepared"))
            return None

        while True:
            previous = await self._tasks.admit((self._task_key, message_id, sink), rearm)
            if previous is None:
                return await self.send(message_id, sink)
            try:
                _ = await previous.join()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise

    async def cancel_prepared(self, message_id: str, sink: str, reason: str) -> bool:
        """明确撤回尚未开始的发送；关闭进程的 Task 取消不冒充此业务决定。"""
        receipt = Receipt(status="rejected", error=reason)

        def cancel(slot: TaskSlot) -> tuple[bool, Task | None]:
            record, delivery = self._records.read(message_id, sink)
            if delivery.phase != "prepared":
                return False, None
            _ = self._records.save(message_id, record, Delivery(
                sink=delivery.sink, phase="rejected", receipt=receipt,
            ))
            current = slot.current
            if current is not None:
                current.cancel()
            return True, current

        cancelled, task = await self._tasks.admit((self._task_key, message_id, sink), cancel)
        if task is not None:
            await _drain(task)
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                raise asyncio.CancelledError
        return cancelled

    async def _send(self, task: Task, message_id: str, sink: str, before_start: Callable[[], str | None] | None) -> Receipt:
        """先读耐久事实，再恢复未知效果；确认即将发送后才提交 started。"""
        record, delivery = self._records.read(message_id, sink)
        if delivery.phase in {"delivered", "rejected"}:
            assert delivery.receipt is not None
            return delivery.receipt
        selection = self._records.selection(message_id)
        if selection is None:
            raise ValueError("发送缺少已提交的消息选择")
        message = self._catalog.reader(selection.session_id).get(message_id)
        if message is None:
            raise ValueError("发送引用的原消息缺失")
        key = delivery_key(message_id, sink)
        target = (self._task_key, "target", delivery.sink.name, delivery.sink.address)
        async with AsyncExitStack() as scope:
            if selection.passive:
                _ = scope.enter_context(self.activity(delivery.sink.name, delivery.sink.address))
            _ = await scope.enter_async_context(self._tasks.exclusive(target, idle=not selection.passive))
            # 等待期间允许明确撤回 prepared；旧快照不能覆盖撤回事实。
            if not task.active:
                raise asyncio.CancelledError
            record, delivery = self._records.read(message_id, sink)
            if delivery.phase in {"delivered", "rejected"}:
                assert delivery.receipt is not None
                return delivery.receipt
            sender = await scope.enter_async_context(self._open_sender(delivery.sink.binding_id))
            # 1. started/unknown 都表示可能已发出；没有幂等保证便只查询。
            if delivery.phase in {"started", "unknown"}:
                found = await sender.query(key, delivery.sink.address)
                if found is not None:
                    found = Receipt.model_validate(found.model_dump())
                if found is not None and found.status != "unknown":
                    _ = self._records.save(message_id, record, Delivery(
                        sink=delivery.sink, phase=found.status, receipt=found,
                    ))
                    return found
                if not sender.idempotent:
                    result = found or delivery.receipt or Receipt(status="unknown", error="原发送缺少可确认回执")
                    _ = self._records.save(message_id, record, Delivery(
                        sink=delivery.sink, phase="unknown", receipt=result,
                    ))
                    return result

            # 2. 同一事件循环的撤权与 start 不跨 await；随后异常不冒称 rejected。
            if not task.active:
                raise asyncio.CancelledError
            # 业务只可拒绝确定尚未开始的效果；未知效果仍按原幂等协议恢复。
            if delivery.phase == "prepared" and before_start is not None:
                reason = before_start()
                if reason is not None:
                    if inspect.iscoroutine(reason):
                        reason.close()
                    if not isinstance(reason, str) or not reason:
                        raise TypeError("发送前检查必须同步返回 None 或非空拒绝原因")
                    result = Receipt(status="rejected", error=reason)
                    _ = self._records.save(message_id, record, Delivery(
                        sink=delivery.sink, phase="rejected", receipt=result,
                    ))
                    return result
            record = self._records.save(message_id, record, Delivery(sink=delivery.sink, phase="started"))
            try:
                response = await sender.send(key, delivery.sink.address, message)
                # 不同归档拥有独立 Python 类型；按本版本公开 schema 接纳跨插件回执。
                result = Receipt.model_validate(response.model_dump())
            except BaseException:
                _ = self._records.save(message_id, record, Delivery(
                    sink=delivery.sink, phase="unknown",
                    receipt=Receipt(status="unknown", error="发送已开始但未取得可确认回执"),
                ))
                raise
            _ = self._records.save(message_id, record, Delivery(
                sink=delivery.sink, phase=result.status, receipt=result,
            ))
            return result


async def _drain(task: Task) -> None:
    """重复取消等待者不能打断原发送的回执提交与资源关闭。"""
    while True:
        try:
            _ = await task.join()
            return
        except asyncio.CancelledError:
            if task.done:
                return
