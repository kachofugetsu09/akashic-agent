from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from core.common.file_io import run_file_io
from typing import cast
from functools import partial
from uuid import uuid4

from agent.plugin_composition.messages import (
    MessageConflict,
    SourceHeadConflict,
    MessageReader,
    MessageWriter,
    OwnerTransaction,
)
from agent.plugin_composition.tasks import RestartGate, Task, TaskAdmission, TaskSlot
from agent.plugin_contracts import Control, Input, Message, Output
from agent.plugin_contracts.sources import CompletionProgram

logger = logging.getLogger(__name__)
Changed = Callable[[MessageReader, str, bool], None]


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


@dataclass(slots=True)
class _ReplyState:
    """本次只读前缀的来源判定；提交后即释放，不保存第二份业务事实。"""
    head: int = -1
    latest_input: int = -1
    boundary: int = -1
    paused_through: int = -1

    @property
    def pending(self) -> bool:
        return self.latest_input > max(self.boundary, self.paused_through)

    def add(self, message: Message) -> None:
        self.head = message.seq
        body = message.body
        if isinstance(body, Input):
            self.latest_input = message.seq
        elif isinstance(body, Output) and body.finish != "continue":
            self.boundary = message.seq
        elif isinstance(body, Control):
            self.add_control(body)

    def add_control(self, body: Control) -> None:
        """Keep ordered pause/resume and abandon rules independent of message content."""
        if body.action == "abandon":
            self.boundary = max(self.boundary, body.through_seq)
        elif body.action in {"pause", "failure"}:
            self.paused_through = max(self.paused_through, body.through_seq)
        elif body.action == "resume" and body.through_seq >= self.paused_through:
            self.paused_through = -1


def _read_state(messages: Sequence[Message] | MessageReader, source: str) -> _ReplyState:
    """来源规则在固定只读前缀内归约，只把少量判定数据带回 loop。"""
    state = _ReplyState()
    if isinstance(messages, MessageReader):
        state.head = messages.head(source=source)
        latest = messages.latest_input_seq(source, through_seq=state.head)
        if latest is None:
            return state
        state.latest_input = latest
        boundary = messages.latest_finished_output_seq(source, after_seq=latest, through_seq=state.head)
        state.boundary = -1 if boundary is None else boundary
        def consume(rows):
            for _, control in rows:
                state.add_control(control)
        messages.scan_controls(consume, after_seq=latest, through_seq=state.head, source=source)
    else:
        for message in messages:
            if message.source == source:
                state.add(message)
    return state


def _read_boundary(reader: MessageReader, source: str, hint: object) -> bool:
    """边界只来自 hint 后的持久终态或 Control，与当前待回复判定共用快照。"""
    if not isinstance(hint, int) or hint < 0:
        return False
    return reader.scan(lambda rows: any(
        isinstance(message.body, Output) and message.body.finish != "continue"
        or isinstance(message.body, Control)
        for message in rows
    ), after_seq=hint, source=source)


class SourceSession:
    """一个已获授权来源的接纳与控制；活动任务短命，重启只重读日志。"""

    @staticmethod
    async def needs_reply(messages: Sequence[Message] | MessageReader, source: str) -> bool:
        """异步读取来源规则；worker 不借 Context、Task 或任何写入能力。"""
        state = (
            await messages.read_async(lambda reader: _read_state(reader, source))
            if isinstance(messages, MessageReader)
            else await run_file_io(lambda: _read_state(messages, source))
        )
        return state.pending

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

    def _changed(self, message: Message, pending: bool) -> None:
        """提交收据在原 loop 同步通知，监听者无需重新扫描正文才能占位。"""
        if self._on_changed is not None:
            self._on_changed(self._reader, self._source, pending)

    def _committed(self, slot: TaskSlot, message: Message, created: bool, pending: bool) -> None:
        """提交通知失败也必须撤权；重放不重复通知或取消。"""
        if not created:
            return
        try:
            _ = self._changed(message, pending)
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
            existing = await self._reader.read_async(lambda reader: reader.get(message_id) is not None)
            if not existing and self._restart_gate is not None:
                self._restart_gate.check_open()
            return await self._inputs.append_async(
                message_id, body,
                on_commit=lambda message, created: self._committed(slot, message, created, True),
            )
        return await self._tasks.admit_async(self._key, admit)

    async def control(
        self, message_id: str, body: Control, *, expected_head: int, handle: str | None,
    ) -> Message:
        """原子接纳控制并撤权；abandon 不等待物理清理，其余停止仍排空。"""
        def read(reader: MessageReader) -> _ReplyState | None:
            if reader.get(message_id) is not None:
                return None
            target = reader.read(after_seq=body.through_seq - 1, through_seq=body.through_seq, limit=1)
            if not target or target[0].source != self._source:
                raise MessageConflict("控制前缀必须指向已接纳的同来源消息")
            if body.action == "abandon" and reader.scan(
                lambda rows: any(
                    isinstance(item.body, Output) and item.body.finish != "continue" and item.seq >= body.through_seq
                    or isinstance(item.body, Control) and item.body.action == "abandon" and item.body.through_seq >= body.through_seq
                    for item in rows
                ), source=self._source,
            ):
                raise MessageConflict("不能放弃已经关闭的前缀")
            return _read_state(reader, self._source)

        async def admit(slot: TaskSlot) -> tuple[Message, Task | None]:
            state = await self._reader.read_async(read)
            if state is None:
                message = await self._controls.append_async(
                    message_id, body, on_commit=lambda message, created: self._committed(slot, message, created, False),
                )
                current = slot.current
                pending = current if current is not None and not current.active else None
                return message, pending if body.action != "resume" else None
            if state.head != expected_head:
                raise SourceHeadConflict("控制判定的来源前缀与显式 head 不同")
            current = slot.current
            if handle is not None:
                current = slot.require(handle)
            elif current is not None and current.active:
                raise MessageConflict("控制活动来源需要当前 handle")

            def committed(message: Message, created: bool) -> None:
                if created:
                    state.add(message)
                self._committed(slot, message, created, state.pending)
            message = await self._controls.append_async(
                message_id, body, expected_source_head=expected_head, on_commit=committed,
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
        """停止当前来源；只读判定与条件提交沿同一准入排序。"""
        def read(reader: MessageReader) -> tuple[Control | None, int]:
            existing = reader.get(message_id)
            if existing is not None:
                if not isinstance(existing.body, Control) or existing.body.action != "pause":
                    raise MessageConflict("停止身份已被其他消息使用")
                return existing.body, reader.head(source=self._source)
            return None, reader.head(source=self._source)

        async def admit(slot: TaskSlot) -> tuple[Message, Task | None]:
            while True:
                existing, head = await self._reader.read_async(read)
                current = slot.current
                if existing is not None:
                    message = await self._controls.append_async(
                        message_id, existing, on_commit=lambda message, created: self._committed(slot, message, created, False),
                    )
                    return message, current if current is not None and not current.active else None
                if head < 0:
                    raise MessageConflict("当前来源没有可暂停的消息")
                if current is not None and current.active:
                    _ = slot.require(current.handle)
                try:
                    message = await self._controls.append_async(
                        message_id, Control("pause", head), expected_source_head=head,
                        on_commit=lambda message, created: self._committed(slot, message, created, False),
                    )
                except SourceHeadConflict:
                    continue
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
        """显式重试恢复原输入；只读 worker 不改变最新输入、控制与 CAS 规则。"""
        def read(reader: MessageReader) -> tuple[Control, int | None]:
            target = reader.get(input_id)
            if target is None or target.source != self._source or not isinstance(target.body, Input):
                raise MessageConflict("重试目标不是当前来源的 Input")
            existing = reader.get(message_id)
            if existing is not None:
                if not isinstance(existing.body, Control) or existing.body.action != "resume":
                    raise MessageConflict("重试身份已被其他消息使用")
                latest = reader.latest_input(self._source, through_seq=existing.body.through_seq)
                if latest is None or latest.message_id != input_id:
                    raise MessageConflict("重试身份已用于另一条输入")
                return existing.body, None
            through = reader.head()
            latest = reader.latest_input(self._source, through_seq=through)
            if latest is None or latest.message_id != input_id:
                raise MessageConflict("只能重试本来源的最新输入")
            closed = reader.scan(
                lambda rows: any(
                    isinstance(item.body, Output) and item.body.finish != "continue"
                    or isinstance(item.body, Control) and item.body.action == "abandon" and item.body.through_seq >= target.seq
                    for item in rows
                ), after_seq=target.seq, through_seq=through, source=self._source,
            )
            if closed:
                raise MessageConflict("已关闭的输入不能重试")
            control = reader.latest_control(self._source, through_seq=through)
            if control is None or cast(Control, control.body).action not in {"failure", "pause"}:
                raise MessageConflict("输入没有等待恢复的失败或暂停")
            head = reader.head(source=self._source)
            return Control("resume", head), head

        async def admit(slot: TaskSlot) -> Message:
            body, head = await self._reader.read_async(read)
            if head is not None:
                if slot.current is not None and slot.current.active:
                    raise MessageConflict("不能重试仍在运行的来源")
                if self._restart_gate is not None:
                    self._restart_gate.check_open()
            return await self._controls.append_async(
                message_id, body, expected_source_head=head,
                on_commit=lambda message, created: self._committed(slot, message, created, True),
            )
        return await self._tasks.admit_async(self._key, admit)

    async def complete(self, program: CompletionProgram) -> Message:
        """在主回复空闲后处理材料；新输入可以撤权，只重试被抢占的本次程序。"""
        # 1. 用户输入优先；等待旧 Task 不取得其取消权。
        async with aclosing(self._reader.follow_heads()) as changes:
            async for _ in changes:
                while True:
                    async def admit(slot: TaskSlot) -> tuple[Task | None, bool]:
                        while True:
                            if slot.current is not None:
                                return slot.current, False
                            state = await self._reader.read_async(lambda reader: _read_state(reader, self._source))
                            if self._reader.head(source=self._source) != state.head:
                                continue
                            if slot.current is not None:
                                return slot.current, False
                            if state.pending:
                                return None, False
                            task = slot.start(lambda task: program(
                                task, self._reader,
                                partial(check_source, task, self._reader, self._source, state.head),
                            ))
                            task.boundary_hint = state.head
                            return task, True

                    task, owned = await self._tasks.admit_async(self._key, admit)
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

    async def _boundary_committed(self, task: Task) -> bool:
        """hint 在原 loop 固定，worker 只读取原消息，不借 Task。"""
        hint = task.boundary_hint
        return await self._reader.read_async(lambda reader: _read_boundary(reader, self._source, hint))

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
            and not await self._boundary_committed(current)
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

        # 2. 读取可以等待；回到 loop 后核对 head 与 exact Task，再同步创建。
        async def admit(slot: TaskSlot) -> Task | None:
            while True:
                residual = slot.current
                hint = None if residual is None else residual.boundary_hint
                state, committed = await self._reader.read_async(lambda reader: (
                    _read_state(reader, self._source), _read_boundary(reader, self._source, hint),
                ))
                if slot.current is not residual or self._reader.head(source=self._source) != state.head:
                    continue
                if residual is not None and not residual.superseded:
                    if not committed:
                        return residual
                    residual.supersede()
                if not state.pending:
                    return None
                break

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
                            head = await self._reader.read_async(lambda reader: reader.head(source=self._source))
                            if slot.current is not task or not task.active:
                                return
                            _ = await self._controls.append_async(
                                uuid4().hex, Control("failure", head, str(error)),
                                expected_source_head=head,
                                on_commit=lambda message, created: self._changed(message, False) if created else None,
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
            task.boundary_hint = state.head
            if permit is not None:
                task.on_done(permit.release)
            return task

        return await self._tasks.admit_async(self._key, admit)

    async def record_failure(self, error: BaseException, *, boundary: int | None = None) -> None:
        """补记 failure；固定前缀判定后仍由原 source-head 条件提交。"""
        async def admit(slot: TaskSlot) -> None:
            state = await self._reader.read_async(lambda reader: _read_state(reader, self._source))
            if not state.pending:
                return
            if boundary is not None and boundary < 0:
                raise ValueError("failure 回执不能绑定伪造的负边界")
            through = state.head if boundary is None else min(boundary, state.head)
            def committed(message: Message, created: bool) -> None:
                if created:
                    state.add(message)
                    self._changed(message, state.pending)
            _ = await self._controls.append_async(
                uuid4().hex, Control("failure", through, str(error)), expected_source_head=state.head,
                on_commit=committed,
            )
        await self._tasks.admit_async(self._key, admit)

    async def wait_capacity(self) -> None:
        """等待 Task 残留额度释放；容量等待不构成无进展故障。"""
        await self._tasks.wait_capacity()
