from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import cast
from uuid import uuid4

from agent.plugin_composition.messages import (
    MessageConflict,
    MessageReader,
    MessageWriter,
    OwnerTransaction,
)
from agent.plugin_composition.tasks import RestartGate, Task, TaskAdmission, TaskSlot
from agent.plugin_contracts import Control, Input, Message, Output

logger = logging.getLogger(__name__)
Changed = Callable[[MessageReader, str], None]


def check_source(
    task: Task, reader: MessageReader, source: str, through_seq: int, *,
    transaction: OwnerTransaction | None = None,
) -> None:
    """首次效果与来源提交在同一 SQL 顺序内核对，不依赖取消通知到达。"""
    changed = (
        reader.source_changed(source, through_seq) if transaction is None
        else transaction.source_changed(reader, source, through_seq)
    )
    if not task.active or changed:
        raise asyncio.CancelledError


class SourceSession:
    """一个已获授权来源的接纳与控制；活动任务短命，重启只重读日志。"""

    @staticmethod
    def needs_reply(messages: Sequence[Message] | MessageReader, source: str) -> bool:
        """来源从输入和控制事实决定是否唤醒；不依赖逻辑 Turn 或消费 cursor。"""
        # 最近 Input 之前的控制和终结只能覆盖更早的 seq，不影响本次唤醒。
        if isinstance(messages, MessageReader):
            with messages.read_snapshot():
                head = messages.head()
                latest = messages.latest_input(source, through_seq=head)
                if latest is None:
                    return False
                messages = (latest, *messages.snapshot(after_seq=latest.seq, through_seq=head))
        boundary = -1
        latest_input = -1
        paused_through = -1
        for message in messages:
            if message.source != source:
                continue
            body = message.body
            if isinstance(body, Input):
                latest_input = message.seq
            elif isinstance(body, Output) and body.finish != "continue":
                boundary = message.seq
            elif isinstance(body, Control):
                if body.action == "abandon":
                    boundary = max(boundary, body.through_seq)
                elif body.action in {"pause", "failure"}:
                    paused_through = max(paused_through, body.through_seq)
                elif body.action == "resume" and body.through_seq >= paused_through:
                    paused_through = -1
        return latest_input > max(boundary, paused_through)

    def __init__(
        self,
        *,
        reader: MessageReader,
        inputs: MessageWriter,
        controls: MessageWriter,
        tasks: TaskAdmission,
        changed: Changed | None = None,
        restart_gate: RestartGate | None = None,
    ):
        if inputs.session_id != reader.session_id or (
            controls.session_id, controls.source
        ) != (reader.session_id, inputs.source):
            raise ValueError("来源的 reader 与 writer 必须属于同一 Session/source")
        self._source = inputs.source
        self._reader = reader
        self._inputs = inputs
        self._controls = controls
        self._tasks = tasks
        self._on_changed = changed
        self._restart_gate = restart_gate
        self._key = (reader.session_id, self._source)

    def _changed(self, message: Message) -> Message:
        """提交后仍在同步准入段通知可选回复消费者，后续发送不能抢过已接纳输入。"""
        if self._on_changed is not None:
            with self._reader.read_snapshot():
                self._on_changed(self._reader, self._source)
        return message

    def _committed(self, slot: TaskSlot, message: Message, created: bool) -> None:
        """提交通知失败也必须撤权；重放不重复通知或取消。"""
        if not created:
            return
        try:
            _ = self._changed(message)
        finally:
            current = slot.current
            if current is not None and current.active:
                if isinstance(message.body, Control) and message.body.action == "abandon":
                    current.supersede()
                elif not isinstance(message.body, Control) or message.body.action != "resume":
                    current.cancel()

    async def accept(self, message_id: str, body: Input) -> Message:
        """先持久接纳，再使旧回复失效；ACK 不等待回复或旧工具排空。"""
        async def admit(slot: TaskSlot) -> Message:
            with self._reader.read_snapshot():
                existing = self._reader.get(message_id)
            if existing is None and self._restart_gate is not None:
                self._restart_gate.check_open()
            return await self._inputs.append_async(
                message_id, body, on_commit=lambda message, created: self._committed(slot, message, created),
            )

        return await self._tasks.admit_async(self._key, admit)

    async def control(
        self,
        message_id: str,
        body: Control,
        *,
        expected_head: int,
        handle: str | None,
    ) -> Message:
        """原子接纳控制并撤权；abandon 不等待物理清理，其余停止仍排空。"""
        async def admit(slot: TaskSlot) -> tuple[Message, Task | None]:
            with self._reader.read_snapshot():
                existing = self._reader.get(message_id)
            if existing is not None:
                message = await self._controls.append_async(
                    message_id, body, on_commit=lambda message, created: self._committed(slot, message, created),
                )
                current = slot.current
                pending = current if current is not None and not current.active else None
                return message, pending if body.action != "resume" else None
            with self._reader.read_snapshot():
                target = self._reader.read(
                    after_seq=body.through_seq - 1, through_seq=body.through_seq, limit=1
                )
                if not target or target[0].source != self._source:
                    raise MessageConflict("控制前缀必须指向已接纳的同来源消息")
                if body.action == "abandon" and any(
                    item.source == self._source
                    and (
                        isinstance(item.body, Output)
                        and item.body.finish != "continue"
                        and item.seq >= body.through_seq
                        or isinstance(item.body, Control)
                        and item.body.action == "abandon"
                        and item.body.through_seq >= body.through_seq
                    )
                    for item in self._reader.snapshot()
                ):
                    raise MessageConflict("不能放弃已经关闭的前缀")
                current = slot.current
                if handle is not None:
                    current = slot.require(handle)
                elif current is not None and current.active:
                    raise MessageConflict("控制活动来源需要当前 handle")
            message = await self._controls.append_async(
                message_id, body, expected_source_head=expected_head,
                on_commit=lambda message, created: self._committed(slot, message, created),
            )
            return message, current if body.action != "resume" else None

        message, pending = await self._tasks.admit_async(self._key, admit)
        if pending is not None and body.action != "abandon":
            try:
                _ = await pending.join()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise
        return message

    async def pause(self, message_id: str) -> Message:
        """停止当前来源；目标选择、pause 提交和撤权在同一准入回调内排序。"""
        async def admit(slot: TaskSlot) -> tuple[Message, Task | None]:
            with self._reader.read_snapshot():
                existing = self._reader.get(message_id)
                head = self._reader.head(source=self._source)
            current = slot.current
            if existing is not None:
                if not isinstance(existing.body, Control) or existing.body.action != "pause":
                    raise MessageConflict("停止身份已被其他消息使用")
                message = await self._controls.append_async(
                    message_id, existing.body,
                    on_commit=lambda message, created: self._committed(slot, message, created),
                )
                return message, current if current is not None and not current.active else None
            if head < 0:
                raise MessageConflict("当前来源没有可暂停的消息")
            if current is not None and current.active:
                _ = slot.require(current.handle)
            message = await self._controls.append_async(
                message_id, Control("pause", head), expected_source_head=head,
                on_commit=lambda message, created: self._committed(slot, message, created),
            )
            return message, current

        message, pending = await self._tasks.admit_async(self._key, admit)
        if pending is not None:
            try:
                _ = await pending.join()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise
        return message

    async def resume(self, message_id: str, input_id: str) -> Message:
        """显式重试恢复原输入，不追加副本；只能恢复最新的失败或暂停前缀。"""
        async def admit(slot: TaskSlot) -> Message:
            with self._reader.read_snapshot():
                target = self._reader.get(input_id)
                if target is None or target.source != self._source or not isinstance(target.body, Input):
                    raise MessageConflict("重试目标不是当前来源的 Input")
                existing = self._reader.get(message_id)
                if existing is not None:
                    if not isinstance(existing.body, Control) or existing.body.action != "resume":
                        raise MessageConflict("重试身份已被其他消息使用")
                    latest = self._reader.latest_input(self._source, through_seq=existing.body.through_seq)
                    if latest is None or latest.message_id != input_id:
                        raise MessageConflict("重试身份已用于另一条输入")
                    body = existing.body
                    expected_head = None
                else:
                    # 1. 准入回调内核对当前日志与活动 handle，不存在检查后的写入窗口。
                    through = self._reader.head()
                    latest = self._reader.latest_input(self._source, through_seq=through)
                    if latest is None or latest.message_id != input_id:
                        raise MessageConflict("只能重试本来源的最新输入")
                    messages = self._reader.scan(tuple, after_seq=target.seq, through_seq=through, source=self._source)
                    if any(
                        isinstance(m.body, Output) and m.body.finish != "continue"
                        or isinstance(m.body, Control) and m.body.action == "abandon"
                        and m.body.through_seq >= target.seq
                        for m in messages
                    ):
                        raise MessageConflict("已关闭的输入不能重试")
                    control = self._reader.latest_control(self._source, through_seq=through)
                    if control is None or cast(Control, control.body).action not in {"failure", "pause"}:
                        raise MessageConflict("输入没有等待恢复的失败或暂停")
                    if slot.current is not None and slot.current.active:
                        raise MessageConflict("不能重试仍在运行的来源")

                    if self._restart_gate is not None:
                        self._restart_gate.check_open()

                    # resume 只记录恢复意图；未知效果仍由 Tool owner 拒绝重跑。
                    expected_head = messages[-1].seq if messages else target.seq
                    body = Control("resume", expected_head)
            return await self._controls.append_async(
                message_id, body, expected_source_head=expected_head,
                on_commit=lambda message, created: self._committed(slot, message, created),
            )

        return await self._tasks.admit_async(self._key, admit)

    async def complete(self, program: Callable[[Task, MessageReader], Awaitable[Message]]) -> Message:
        """在主回复空闲后处理材料；新输入可以撤权，只重试被抢占的本次程序。"""
        # 1. 用户输入优先；等待旧 Task 不取得其取消权。
        async for _ in self._reader.follow():
            while True:
                def admit(slot: TaskSlot) -> tuple[Task | None, bool]:
                    if slot.current is not None:
                        return slot.current, False
                    if self.needs_reply(self._reader, self._source):
                        return None, False
                    task = slot.start(lambda task: program(task, self._reader))
                    with self._reader.read_snapshot():
                        task.boundary_hint = self._reader.head(source=self._source)
                    return task, True

                task, owned = await self._tasks.admit(self._key, admit)
                if task is None:
                    break
                try:
                    result = await task.join()
                except asyncio.CancelledError:
                    caller = asyncio.current_task()
                    if caller is not None and caller.cancelling():
                        if owned:
                            task.cancel()
                            while not task.done:
                                try:
                                    _ = await task.join()
                                except asyncio.CancelledError:
                                    continue
                        raise
                    # 新输入已同步撤权；先让用户回复运行，再从原消息继续。
                except Exception:
                    if owned:
                        raise
                    logger.warning("等待中的主回复失败，继续核对输入状态", exc_info=True)
                else:
                    if owned:
                        return cast(Message, result)
        raise RuntimeError("Session 订阅在回传完成前结束")

    def _boundary_committed(self, task: Task) -> bool:
        """残留任务负责的区间是否已有持久终态；只有确认边界才允许 lane 让位。"""
        hint = task.boundary_hint
        if not isinstance(hint, int) or hint < 0:
            return False
        with self._reader.read_snapshot():
            return any(
                message.source == self._source and (
                    isinstance(message.body, Output) and message.body.finish != "continue"
                    or isinstance(message.body, Control)
                )
                for message in self._reader.snapshot(after_seq=hint)
            )

    async def start(
        self,
        program: Callable[[Task, MessageReader, str], Awaitable[object]],
    ) -> Task | None:
        """已提交边界的旧工作不阻塞新接纳；物理清理由 Task owner 独立排空。"""
        # 1. 活动任务仍持有提交权；已撤权任务只保留资源，业务边界已在日志中关闭。
        current = await self._tasks.admit(self._key, lambda slot: slot.current)
        if current is not None and current.active:
            return current
        if (
            current is not None
            and not current.superseded
            and not self._boundary_committed(current)
        ):
            # 普通 Input/pause 的残留先真实排空再让位；已提交终态的不等待物理清理。
            try:
                _ = await current.join()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise
            except Exception:
                logger.warning("残留回复任务排空失败", exc_info=True)

        # 2. 日志判定与 Task 创建间没有 await，不增加持久 active/attempt 状态。
        def admit(slot: TaskSlot) -> Task | None:
            residual = slot.current
            if residual is not None and not residual.superseded:
                if not self._boundary_committed(residual):
                    return residual
                # 旧工作负责的区间已提交持久终态；物理清理转入残留集合。
                residual.supersede()
            if not self.needs_reply(self._reader, self._source):
                return None

            if self._restart_gate is not None and not self._restart_gate.accepting:
                return None
            permit = None if self._restart_gate is None else self._restart_gate.acquire()

            async def run(task: Task) -> object:
                try:
                    return await program(task, self._reader, self._source)
                except Exception as error:
                    # 只有仍持有本来源的任务能记录 failure；旧草稿错误只向上报告。
                    async def failed(slot: TaskSlot) -> None:
                        if slot.current is task and task.active:
                            with self._reader.read_snapshot():
                                head = self._reader.head(source=self._source)
                            _ = await self._controls.append_async(
                                uuid4().hex, Control("failure", head, str(error)),
                                expected_source_head=head,
                                on_commit=lambda message, created: self._changed(message) if created else None,
                            )
                    await self._tasks.admit_async(self._key, failed)
                    raise

            try:
                task = slot.start(
                    run,
                    child_permit=None if permit is None else permit.child,
                )
            except BaseException:
                if permit is not None:
                    permit.release()
                raise
            # 记录接纳时的来源边界；只有本任务之后的持久终态才允许 lane 让位。
            with self._reader.read_snapshot():
                task.boundary_hint = self._reader.head(source=self._source)
            if permit is not None:
                task.on_done(permit.release)
            return task

        return await self._tasks.admit(self._key, admit)

    async def record_failure(self, error: BaseException, *, boundary: int | None = None) -> None:
        """为无持久进展的失败补记 failure Control；只重试保存，不重新执行程序。"""
        async def admit(slot: TaskSlot) -> None:
            with self._reader.read_snapshot():
                if not self.needs_reply(self._reader, self._source):
                    return
                head = self._reader.head(source=self._source)
            if boundary is not None and boundary < 0:
                raise ValueError("failure 回执不能绑定伪造的负边界")
            through = head if boundary is None else min(boundary, head)
            _ = await self._controls.append_async(
                uuid4().hex, Control("failure", through, str(error)), expected_source_head=head,
                on_commit=lambda message, created: self._changed(message) if created else None,
            )

        await self._tasks.admit_async(self._key, admit)

    async def wait_capacity(self) -> None:
        """等待 Task 残留额度释放；容量等待不构成无进展故障。"""
        await self._tasks.wait_capacity()
