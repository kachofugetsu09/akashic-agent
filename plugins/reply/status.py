from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.models import StreamCallback
from agent.plugin_composition.tasks import Task
from plugins.react.plugin import Preview


@dataclass(frozen=True, slots=True)
class ReplyPreview:
    message_id: str
    text: str = ""
    thinking: str = ""
    call_record_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReplyActivity:
    session_id: str
    source: str
    handle: str
    active: bool
    preview: ReplyPreview | None = None


class ReplyRead:
    """只读当前回复与预览，不持有取消、写消息或执行模型的能力。"""

    def __init__(self, state: ReplyState):
        self._state = state

    def snapshot(self, session_id: str) -> tuple[ReplyActivity, ...]:
        return self._state.snapshot(session_id)

    async def follow(self, session_id: str) -> AsyncGenerator[tuple[ReplyActivity, ...], None]:
        """订阅当前快照；慢读者合并通知，重连不重放旧 token。"""
        while True:
            changed = self._state.changed
            yield self._state.snapshot(session_id)
            if self._state.closed:
                return
            _ = await changed.wait()


class ReplyState:
    """默认回复组合拥有短命 scope 与草稿；消息提交权仍属于日志 writer。"""

    def __init__(self):
        self._items: dict[str, ReplyActivity] = {}
        self.changed = asyncio.Event()
        self.closed = False
        self.read = ReplyRead(self)

    def snapshot(self, session_id: str) -> tuple[ReplyActivity, ...]:
        return tuple(item for item in self._items.values() if item.session_id == session_id)

    def close(self) -> None:
        """卸载只能在真实工作排空后结束只读订阅，不抹去仍运行的状态。"""
        if self._items:
            raise RuntimeError("回复尚未排空，不能关闭状态读取")
        self.closed = True
        self._notify()

    def _notify(self) -> None:
        changed = self.changed
        self.changed = asyncio.Event()
        changed.set()

    @contextmanager
    def open(self, task: Task, session_id: str, source: str) -> Generator[Preview]:
        """从程序开始到真实排空保存活动；撤权同步撤掉草稿。"""
        if self.closed:
            raise RuntimeError("回复状态已关闭")
        if task.handle in self._items:
            raise RuntimeError("回复 scope 已登记")
        self._items[task.handle] = ReplyActivity(session_id, source, task.handle, True)

        def revoked() -> None:
            item = self._items.get(task.handle)
            if item is not None:
                self._items[task.handle] = replace(item, active=False, preview=None)
                self._notify()

        task.on_close(revoked)
        self._notify()

        @contextmanager
        def preview(message_id: str) -> Generator[StreamCallback]:
            if not task.active:
                raise asyncio.CancelledError
            item = self._items[task.handle]
            if item.preview is not None:
                raise RuntimeError("同一回复 scope 不能同时生成两条草稿")
            self._items[task.handle] = replace(item, preview=ReplyPreview(message_id))
            self._notify()

            async def delta(value: dict[str, str]) -> None:
                if not task.active:
                    raise asyncio.CancelledError
                current = self._items[task.handle]
                draft = current.preview
                if draft is None or draft.message_id != message_id:
                    raise RuntimeError("预览回调已离开原消息 scope")
                call_id = value.get("call_record_id", draft.call_record_id)
                if draft.call_record_id is not None and call_id != draft.call_record_id:
                    raise RuntimeError("同一草稿的模型调用 ID 已改变")
                self._items[task.handle] = replace(current, preview=replace(
                    draft, text=draft.text + value.get("content_delta", ""),
                    thinking=draft.thinking + value.get("thinking_delta", ""),
                    call_record_id=call_id,
                ))
                self._notify()

            try:
                yield delta
            finally:
                # cancel 已同步撤掉草稿；排空中的 provider 不能重新发布它。
                self._items[task.handle] = replace(self._items[task.handle], preview=None)
                self._notify()

        try:
            yield preview
        finally:
            del self._items[task.handle]
            self._notify()


REPLY_STATUS = ServiceKey[ReplyRead]("reply.status.v1")
