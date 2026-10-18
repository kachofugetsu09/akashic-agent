from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime, timedelta

from agent.plugin_composition import Context
from plugins.ledger.contract import MESSAGE_CATALOG, MESSAGE_WRITERS
from agent.plugin_composition.tasks import TASKS, Task, TaskSlot
from plugins.timer.contract import TIMERS
from plugins.sources.contract import SourceGuard
from plugins.models.contract import ModelError
from .inputs import CONTENT, CONVERSATION_COMPLETE, DELIVERY, DELIVERY_SENDERS
from plugins.ledger.contract import BINDINGS
from plugins.ledger.contract import MessageReader, MessageWriter
from plugins.ledger.contract import ContentPart, Control, Input, Message, Output

from .store import Fire, JobStore, aware, fire_key
from .tools import drain
from core.common.file_io import run_file_io

logger = logging.getLogger(__name__)
# 在目标会话里以本次触发的独占来源跑一轮回复。
Program = Callable[[Task, MessageReader, SourceGuard, Fire], Awaitable[Message]]


class SchedulerRuntime:
    """调度文件拥有触发事实；同一个 Task 路径恢复消息、发送与周期结算。"""

    def __init__(self, ctx: Context, store: JobStore, program: Program,
                 *, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self._ctx = ctx
        self.store = store
        self._program = program
        self._now = now
        self._active: dict[str, Task] = {}
        self._attempted: set[tuple[str, str]] = set()

    async def follow(self) -> None:
        """单次 Timer 等待不保留 generation scope；文件变化也唤醒旧归档提交的新任务。"""
        ctx = self._ctx
        async with ctx.runtime_scope():
            state = await self.store.recover(self._now())
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
                        state = await run_file_io(self.store.read)
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
                        if job.enabled and aware(job.fire_at) <= now and (fire_key(job), "pending") not in self._attempted:
                            fire = await self.store.start_fire(job)
                            if fire is not None and (fire.key, fire.status) not in self._attempted:
                                _ = await self.start(fire)
                    # 2. 归档工具可在另一代修改同一文件，短等待让这些变化及时可见。
                    deadline = now + timedelta(milliseconds=250)
                    for job in state.jobs.values():
                        if job.enabled and aware(job.fire_at) > now:
                            deadline = min(deadline, aware(job.fire_at))
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
                logger.exception("调度触发失败，原消息与发送回执已保留 fire=%s", fire.key)
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
        fire = (await run_file_io(self.store.read)).fires[key]
        delivery = ctx.require(DELIVERY).open(ctx)
        target = ctx.require(MESSAGE_CATALOG).reader(fire.job.session_id)
        if fire.status == "cancelled":
            final = await run_file_io(lambda: _final(target, fire))
            # The old fire has drained before this task starts. Close its round so
            # unfinished tool calls cannot block later session compaction.
            if final is None:
                control_id = "scheduler-cancelled:" + fire.key
                if target.get(control_id) is None:
                    head = target.head(source=fire.source)
                    if head >= 0:
                        writer = ctx.require(MESSAGE_WRITERS).bind(
                            ctx, author="scheduler", source=fire.source, body_types=(Control,), content={},
                        )(target.session_id)
                        task.on_close(writer.expire)
                        await writer.append_async(control_id, Control("abandon", head, fire.error),
                                                  expected_source_head=head)
            selected = None if final is None else delivery.selection(final.message_id)
            if final is not None and selected is not None:
                for sink in selected.sinks:
                    _ = await delivery.cancel_prepared(final.message_id, sink, fire.error or "任务已取消")
            return
        if fire.status != "pending":
            return

        # 1. 已有最终消息便直接恢复发送，不再次运行模型或工具；执行任务的目标会话已不存在时明确失败。
        catalog = ctx.require(MESSAGE_CATALOG)
        if fire.job.action == "run" and not await run_file_io(lambda: catalog.exists(target.session_id)):
            await self.store.settle(key, "failed", now=self._now(), error="目标会话不存在")
            return
        final = await run_file_io(lambda: _final(target, fire))
        if final is None:
            # 写入授权属于 scheduler 自己的 Task；会话执行位里只使用已绑定的 writer。
            writer = ctx.require(MESSAGE_WRITERS).bind(
                ctx, author="scheduler", source=fire.source,
                body_types=(Output,) if fire.job.action == "notify" else (Input, Control),
                content={"text": ctx.require(CONTENT).check_text},
            )(target.session_id)
            task.on_close(writer.expire)
            final = await (writer.append_async(fire.notification_id, Output((ContentPart("text", fire.job.text),), "complete"))
                           if fire.job.action == "notify"
                           else ctx.require(CONVERSATION_COMPLETE)(target.session_id, self._run(fire, writer)))
        if not task.active:
            raise asyncio.CancelledError
        if not isinstance(final.body, Output) or final.body.finish != "complete":
            error = final.body.reason if isinstance(final.body, Control) else "调度任务没有可发送的最终内容"
            await self.store.settle(key, "failed", now=self._now(), error=error)
            return

        # 2. 投递原消息；恢复时沿用已固定的选路，空集合不能冒称已送达。
        sinks: tuple[dict[str, object], ...] = ()
        if delivery.selection(final.message_id) is None:
            binding = ctx.require(DELIVERY_SENDERS).bind(fire.job.channel, ctx.require(BINDINGS))
            sinks = ({"name": fire.job.channel, "binding_id": binding, "address": fire.job.chat_id},)
        selected = await delivery.prepare_async(target, final, sinks)
        if not selected.sinks:
            raise ValueError("调度通知缺少原发送目的地")
        try:
            receipts = [await delivery.send(final.message_id, sink) for sink in selected.sinks]
        except asyncio.CancelledError:
            # 领域取消已先落盘；单纯 shutdown 只保留 prepared/started，不冒充撤回。
            cancelled = (await run_file_io(self.store.read)).fires[key]
            if cancelled.status == "cancelled":
                for sink in selected.sinks:
                    _ = await delivery.cancel_prepared(final.message_id, sink, cancelled.error or "任务已取消")
            raise
        except Exception:
            # Sender 已保存失败回执时，本次触发也结束；原异常继续向上传播。
            saved = [delivery.receipt(final.message_id, sink) for sink in selected.sinks]
            failures = [receipt for receipt in saved if receipt is not None and receipt.status == "failed"]
            if failures:
                await self.store.settle(key, "failed", now=self._now(),
                                  error="; ".join(receipt.error or receipt.status for receipt in failures))
            raise
        if all(receipt.status == "delivered" for receipt in receipts):
            await self.store.settle(key, "delivered", now=self._now())
        else:
            errors = "; ".join(receipt.error or receipt.status for receipt in receipts if receipt.status != "delivered")
            await self.store.settle(key, "failed", now=self._now(), error=errors)

    def _run(self, fire: Fire, writer: MessageWriter) -> Callable[[Task, MessageReader, SourceGuard], Awaitable[Message]]:
        """主对话空闲后才执行；用户插话会撤权，之后从同一 Input 与已结算工具继续。"""
        async def run(task: Task, reader: MessageReader, check_admission: SourceGuard) -> Message:
            final = await run_file_io(lambda: _final(reader, fire))
            if final is not None:
                return final
            if reader.get(fire.input_id) is None:
                label = f"「{fire.job.name}」" if fire.job.name else ""
                text = f"⏰ 定时任务{label}自动触发（不是用户发言）：\n{fire.job.text}"
                _ = await writer.append_async(fire.input_id, Input((ContentPart("text", text),)))
            try:
                return await self._program(task, reader, check_admission, fire)
            except (RuntimeError, TimeoutError) as error:
                if ModelError.read(error) is None:
                    raise
                # Save a model failure before settling the fire, so restart cannot repeat it.
                check_admission()
                return await writer.append_async("scheduler-failure:" + fire.key,
                    Control("failure", reader.head(source=fire.source), str(error)))
        return run


def _final(reader: MessageReader, fire: Fire) -> Message | None:
    """读取原文提醒、已保存的模型失败，或本来源最后一条结束的 Output。"""
    notification = reader.get(fire.notification_id) or reader.get("scheduler-failure:" + fire.key)
    if notification is not None:
        return notification

    def last(messages: Iterable[Message]) -> Message | None:
        found = None
        for message in messages:
            if isinstance(message.body, Output) and message.body.finish != "continue":
                found = message
        return found
    return reader.scan(last, source=fire.source)
