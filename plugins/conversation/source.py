from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from uuid import uuid4
from typing import cast

from agent.model_runtime.session_selection import SessionModelSelection, write_session_model_selection
from agent.plugin_composition.tasks import Task, TaskAdmission, TaskSlot
from session.log import MessageConflict, MessageReader, MessageWriter
from session.message import Body, Control, Input, Message, Output


logger = logging.getLogger(__name__)
Changed = Callable[[MessageReader, str], None]


def update_selection(body: Body) -> Mapping[str, object | None]:
    """只从本次已验证 Input 生成选择变化，与正文同事务保存。"""
    if not isinstance(body, Input):
        raise TypeError("会话选择只能随 Input 更新")
    parts = [part for part in body.parts if part.kind == "model.selection"]
    if not parts:
        return {}
    if len(parts) != 1:
        raise ValueError("一个 Input 只能包含一次模型选择")
    value = cast(Mapping[str, str | None], parts[0].value)
    selected: dict[str, object] = {}
    write_session_model_selection(selected, SessionModelSelection(
        value["model_id"] or "", value["reasoning_effort"] or "",
    ))
    return {"model_selection": selected.get("model_selection"), "model_runtime_override": None}


def needs_reply(messages: Sequence[Message], source: str) -> bool:
    """来源从输入和控制事实决定是否唤醒；不依赖逻辑 Turn 或消费 cursor。"""
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


class Conversation:
    """一个已获授权来源的接纳与控制；活动任务短命，重启只重读日志。"""

    def __init__(
        self,
        *,
        reader: MessageReader,
        inputs: MessageWriter,
        controls: MessageWriter,
        tasks: TaskAdmission,
        changed: Changed | None = None,
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
        self._key = (reader.session_id, self._source)

    def _changed(self, message: Message) -> Message:
        """提交后仍在同步准入段通知可选回复消费者，后续发送不能抢过已接纳输入。"""
        if self._on_changed is not None:
            self._on_changed(self._reader, self._source)
        return message

    async def accept(self, message_id: str, body: Input) -> Message:
        """先持久接纳，再使旧回复失效；ACK 不等待回复或旧工具排空。"""
        def admit(slot: TaskSlot) -> Message:
            existing = self._reader.get(message_id)
            message = self._inputs.append(message_id, body)
            if existing is not None:
                return message
            _ = self._changed(message)
            current = slot.current
            if current is not None and current.active:
                current.cancel()
            return message

        return await self._tasks.admit(self._key, admit)

    async def control(
        self,
        message_id: str,
        body: Control,
        *,
        expected_head: int,
        handle: str | None,
    ) -> Message:
        """原子接纳控制并撤权；停止返回前等待已开始的工作真实排空。"""
        def admit(slot: TaskSlot) -> tuple[Message, Task | None]:
            if self._reader.get(message_id) is not None:
                message = self._controls.append(message_id, body)
                current = slot.current
                pending = current if current is not None and not current.active else None
                return message, pending if body.action != "resume" else None
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
            message = self._changed(self._controls.append(
                message_id, body, expected_source_head=expected_head
            ))
            if current is not None and body.action != "resume":
                current.cancel()
            return message, current if body.action != "resume" else None

        message, pending = await self._tasks.admit(self._key, admit)
        if pending is not None:
            try:
                _ = await pending.join()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise
        return message

    async def pause(self, message_id: str) -> Message:
        """停止当前来源；目标选择、pause 提交和撤权在同一准入回调内排序。"""
        def admit(slot: TaskSlot) -> tuple[Message, Task | None]:
            existing = self._reader.get(message_id)
            current = slot.current
            if existing is not None:
                if not isinstance(existing.body, Control) or existing.body.action != "pause":
                    raise MessageConflict("停止身份已被其他消息使用")
                return self._controls.append(message_id, existing.body), (
                    current if current is not None and not current.active else None
                )
            head = self._reader.head(source=self._source)
            if head < 0:
                raise MessageConflict("当前来源没有可暂停的消息")
            if current is not None and current.active:
                _ = slot.require(current.handle)
            message = self._changed(self._controls.append(
                message_id, Control("pause", head), expected_source_head=head,
            ))
            if current is not None:
                current.cancel()
            return message, current

        message, pending = await self._tasks.admit(self._key, admit)
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
        def admit(slot: TaskSlot) -> Message:
            target = self._reader.get(input_id)
            if target is None or target.source != self._source or not isinstance(target.body, Input):
                raise MessageConflict("重试目标不是当前来源的 Input")
            existing = self._reader.get(message_id)
            if existing is not None:
                if not isinstance(existing.body, Control) or existing.body.action != "resume":
                    raise MessageConflict("重试身份已被其他消息使用")
                prefix = self._reader.snapshot(through_seq=existing.body.through_seq)
                inputs = [m for m in prefix if m.source == self._source and isinstance(m.body, Input)]
                if not inputs or inputs[-1].message_id != input_id:
                    raise MessageConflict("重试身份已用于另一条输入")
                return self._controls.append(message_id, existing.body)

            # 1. 准入回调内核对当前日志与活动 handle，不存在检查后的写入窗口。
            messages = [m for m in self._reader.snapshot() if m.source == self._source]
            inputs = [m for m in messages if isinstance(m.body, Input)]
            if inputs[-1].message_id != input_id:
                raise MessageConflict("只能重试本来源的最新输入")
            if any(
                isinstance(m.body, Output) and m.body.finish != "continue"
                or isinstance(m.body, Control) and m.body.action == "abandon"
                and m.body.through_seq >= target.seq
                for m in messages if m.seq > target.seq
            ):
                raise MessageConflict("已关闭的输入不能重试")
            controls = [m.body for m in messages if isinstance(m.body, Control)]
            if not controls or controls[-1].action not in {"failure", "pause"}:
                raise MessageConflict("输入没有等待恢复的失败或暂停")
            if slot.current is not None and slot.current.active:
                raise MessageConflict("不能重试仍在运行的来源")

            # 2. resume 只记录恢复意图；未知外部效果仍由 Tool owner 拒绝自动重跑。
            head = messages[-1].seq
            return self._changed(self._controls.append(
                message_id, Control("resume", head), expected_source_head=head,
            ))

        return await self._tasks.admit(self._key, admit)

    async def complete(self, program: Callable[[Task, MessageReader], Awaitable[Message]]) -> Message:
        """在主回复空闲后处理材料；新输入可以撤权，只重试被抢占的本次程序。"""
        # 1. 用户输入优先；等待旧 Task 不取得其取消权。
        async for _ in self._reader.follow():
            while True:
                def admit(slot: TaskSlot) -> tuple[Task | None, bool]:
                    if slot.current is not None:
                        return slot.current, False
                    if needs_reply(self._reader.snapshot(), self._source):
                        return None, False
                    return slot.start(lambda task: program(task, self._reader)), True

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

    async def start(
        self, program: Callable[[Task, MessageReader, str], Awaitable[object]]
    ) -> Task | None:
        """等待旧工作真实排空后重读事实；多个唤醒共用同一个活动任务。"""
        # 1. 已取消的任务仍持有资源；只有完成 join 才能接纳替代者。
        current = await self._tasks.admit(self._key, lambda slot: slot.current)
        if current is not None:
            if current.active:
                return current
            try:
                _ = await current.join()
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                if caller is not None and caller.cancelling():
                    raise
            except Exception:
                logger.warning("已撤权的旧回复在排空时失败", exc_info=True)

        # 2. 日志判定与 Task 创建间没有 await，不增加持久 active/attempt 状态。
        def admit(slot: TaskSlot) -> Task | None:
            if slot.current is not None:
                return slot.current
            if not needs_reply(self._reader.snapshot(), self._source):
                return None

            async def run(task: Task) -> object:
                try:
                    return await program(task, self._reader, self._source)
                except Exception as error:
                    # 只有仍持有本来源的任务能记录 failure；旧草稿错误只向上报告。
                    def failed(slot: TaskSlot) -> None:
                        if slot.current is task and task.active:
                            head = self._reader.head(source=self._source)
                            _ = self._changed(self._controls.append(
                                uuid4().hex,
                                Control("failure", head, str(error)),
                                expected_source_head=head,
                            ))
                    await self._tasks.admit(self._key, failed)
                    raise

            return slot.start(run)

        return await self._tasks.admit(self._key, admit)
