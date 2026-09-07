from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from agent.plugin_composition import Context
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS, Task, TaskSlot
from agent.plugin_composition.timers import TIMERS
from plugins.content.api import check_artifact
from plugins.content.plugin import check_text
from plugins.delivery.api import Sink
from plugins.delivery.plugin import DELIVERY
from plugins.delivery.senders import DELIVERY_SENDERS
from agent.plugin_composition.bindings import BINDINGS
from session.log import MessageReader, SessionAttributes
from session.message import ContentPart, Input, Message, Output

from .schedule import LatencyTracker, compute_actual_trigger
from .store import Fire, JobStore, aware, fire_key
from .tools import drain

logger = logging.getLogger(__name__)
Program = Callable[[Task, MessageReader], Awaitable[Message]]


class SchedulerRuntime:
    """调度文件拥有触发事实；同一个 Task 路径恢复消息、发送与周期结算。"""

    def __init__(self, ctx: Context, store: JobStore, program: Program,
                 *, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self._ctx = ctx
        self.store = store
        self._program = program
        self._now = now
        self.tracker = LatencyTracker()
        self._active: dict[str, Task] = {}
        self._attempted: set[tuple[str, str]] = set()

    async def follow(self) -> None:
        """单次 Timer 等待不保留 generation scope；文件变化也唤醒旧归档提交的新任务。"""
        ctx = self._ctx
        async with ctx.runtime_scope():
            state = self.store.recover(self._now())
        stamp: tuple[int, int, int] | None | object = object()
        cancelled: tuple[Fire, ...] = ()
        try:
            while True:
                async with ctx.runtime_scope():
                    now = aware(self._now())
                    try:
                        stat = self.store.path.stat()
                        current = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                    except FileNotFoundError:
                        current = None
                    if current != stamp:
                        state = self.store.read()
                        stamp = current
                        # 1. 文件变化才重读历史回执；恢复与正常触发共用 fire key。
                        for fire in state.fires.values():
                            if fire.status in {"pending", "cancelled"} and (fire.key, fire.status) not in self._attempted:
                                _ = await self.start(fire)
                        cancelled = tuple(fire for fire in state.fires.values() if fire.status == "cancelled"
                                          and (fire.key, fire.status) not in self._attempted)
                    # 取消可能先遇到仍在排空的旧 Task；完成后再接纳清理。
                    for fire in cancelled:
                        if (fire.status == "cancelled" and (fire.key, fire.status) not in self._attempted
                                and fire.key not in self._active):
                            _ = await self.start(fire)
                    cancelled = tuple(fire for fire in cancelled if (fire.key, fire.status) not in self._attempted)
                    for job in state.jobs.values():
                        deadline = compute_actual_trigger(job.fire_at, job.tier, self.tracker)
                        if job.enabled and aware(deadline) <= now and (fire_key(job), "pending") not in self._attempted:
                            fire = self.store.start_fire(job)
                            if fire is not None and (fire.key, fire.status) not in self._attempted:
                                _ = await self.start(fire)
                    # 2. 归档工具可在另一代修改同一文件，短等待让这些变化及时可见。
                    deadline = now + timedelta(milliseconds=250)
                    for job in state.jobs.values():
                        target = aware(compute_actual_trigger(job.fire_at, job.tier, self.tracker))
                        if job.enabled and target > now:
                            deadline = min(deadline, target)
                handle = ctx.require(TIMERS).schedule(deadline)
                try:
                    _ = await handle.result()
                finally:
                    _ = await handle.cancel()
                    await handle.cleanup()
        finally:
            pending = tuple(self._active.values())
            for task in pending:
                task.cancel()
            for task in pending:
                await drain(task)

    async def start(self, fire: Fire) -> Task:
        """先同步接纳真正的 fire Task，再由它重读取消事实与原恢复材料。"""
        async def run(task: Task) -> None:
            try:
                await self._fire(task, fire.key)
            except Exception:
                # 未处理的磁盘或绑定错误不伪造业务终态，也不在本进程无界重试。
                logger.exception("调度触发未结算，保留恢复事实 fire=%s", fire.key)
            finally:
                _ = self._active.pop(fire.key, None)

        def admit(slot: TaskSlot) -> tuple[Task, bool]:
            return (slot.current, False) if slot.current is not None else (slot.start(run), True)

        task, started = await self._ctx.require(TASKS).open(self._ctx).admit(("fire", fire.key), admit)
        if started:
            self._active[fire.key] = task
            self._attempted.add((fire.key, fire.status))
        return task

    async def _fire(self, task: Task, key: str) -> None:
        """只有 pending 可以推进；prepared 撤回与未知发送沿 Delivery 原回执处理。"""
        ctx = self._ctx
        fire = self.store.read().fires[key]
        delivery = ctx.require(DELIVERY).open(ctx)
        selected = delivery.selection(fire.notification_id)
        if fire.status == "cancelled":
            if selected is not None:
                for sink in selected.sinks:
                    _ = await delivery.cancel_prepared(fire.notification_id, sink, fire.error or "任务已取消")
            return
        if fire.status != "pending":
            return

        # 1. 通知已经保存便直接恢复发送，不再次运行模型或工具。
        catalog = ctx.require(MESSAGE_CATALOG)
        target = catalog.reader(f"{fire.job.channel}:{fire.job.chat_id}")
        notification = target.get(fire.notification_id)
        if notification is None:
            await delivery.wait_idle(fire.job.channel, fire.job.chat_id)
            parts = await self._content(task, fire)
            if not task.active:
                raise asyncio.CancelledError
            if not parts:
                self.store.settle(key, "failed", now=self._now(), error="调度任务没有可发送的最终内容")
                return
            writer = ctx.require(MESSAGE_WRITERS).bind(
                ctx, author="scheduler", source="scheduler", body_types=(Output,),
                content={"text": check_text, "artifact_ref": check_artifact},
            )(target.session_id)
            task.on_close(writer.expire)
            binding = ctx.require(DELIVERY_SENDERS).bind(fire.job.channel, ctx.require(BINDINGS))
            notification, selected = delivery.publish(writer, fire.notification_id, Output(parts, "complete"), (Sink(
                name=fire.job.channel, binding_id=binding, address=fire.job.chat_id,
            ),))

        # 2. Scheduler 只恢复自己原定的非空选路；空集合不能冒称通知已送达。
        if selected is None:
            raise ValueError("调度通知缺少原发送选择")
        selected = delivery.prepare(target, notification, ())
        if not selected.sinks:
            raise ValueError("调度通知缺少原发送目的地")
        try:
            receipts = [await delivery.send(notification.message_id, sink) for sink in selected.sinks]
        except asyncio.CancelledError:
            # 领域取消已先落盘；单纯 shutdown 只保留 prepared/unknown，不冒充撤回。
            cancelled = self.store.read().fires[key]
            if cancelled.status == "cancelled":
                for sink in selected.sinks:
                    _ = await delivery.cancel_prepared(notification.message_id, sink, cancelled.error or "任务已取消")
            raise
        if all(receipt.status == "delivered" for receipt in receipts):
            self.store.settle(key, "delivered", now=self._now())
        elif any(receipt.status == "rejected" for receipt in receipts):
            self.store.settle(key, "failed", now=self._now(), error="调度通知被拒绝")
        # unknown 不是失败或送达，保持原 fire 以便下次启动查询原发送。

    async def _content(self, task: Task, fire: Fire) -> tuple[ContentPart, ...]:
        """每次触发有独立内部 Session；已保存的完整输出足以恢复最终通知。"""
        ctx = self._ctx
        _ = ctx.require(SESSION_ADMISSION).ensure(ctx, fire.session_id,
            SessionAttributes(visibility="internal", learning="excluded"))
        if fire.job.tier == "instant":
            assert fire.job.message is not None
            return (ContentPart("text", fire.job.message),)
        reader = ctx.require(MESSAGE_CATALOG).reader(fire.session_id)
        finished = [message for message in reader.snapshot() if isinstance(message.body, Output)
                    and message.body.finish != "continue"]
        if finished:
            output = finished[-1]
        else:
            assert fire.job.prompt is not None
            writer = ctx.require(MESSAGE_WRITERS).bind(
                ctx, author="scheduler", source="scheduler", body_types=(Input,),
                content={"text": check_text},
            )(fire.session_id)
            task.on_close(writer.expire)
            _ = writer.append("scheduler-input:" + fire.key, Input((ContentPart("text", fire.job.prompt),)))
            started = time.monotonic()
            output = await self._program(task, reader)
            self.tracker.record(time.monotonic() - started)
        if not isinstance(output.body, Output):
            raise TypeError("调度程序必须返回 Output")
        if output.body.finish != "complete":
            return ()
        return tuple(part for part in output.body.parts if isinstance(part, ContentPart)
                     and (part.kind == "artifact_ref" or part.kind == "text"
                          and isinstance(part.value, str) and part.value.strip()))
