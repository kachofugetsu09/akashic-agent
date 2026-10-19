"""生成中的插话：回复进行时先暂存，回复任务在下一步模型请求之前把它写成本来源输入。"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

from agent.plugin_composition.tasks import Task, independent_task_context
from plugins.channels.contract import ChannelInboundMessage
from plugins.ledger.contract import Input, Message
from plugins.sources.contract import GuardedSourceSession as SourceSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Staged:
    message_id: str
    message: ChannelInboundMessage


class Steers:
    """暂存只在内存里：回合结束或停止时转成普通输入开启下一轮；重启时由前端退回输入框，不另存第二份事实。"""

    def __init__(
        self, *,
        open_session: Callable[[str], SourceSession],
        accept: Callable[[str, str, ChannelInboundMessage], Awaitable[Message]],
        body: Callable[[ChannelInboundMessage], Input],
        scope: Callable[[], AbstractAsyncContextManager[object]],
    ):
        self._open = open_session
        self._scope = scope
        self._accept = accept
        self._body = body
        self._staged: dict[str, list[_Staged]] = {}
        self._watched: dict[str, Task] = {}
        # 只有回复程序登记过的任务会在步间注入；后台回调等其他程序不接纳插话。
        self._bound: weakref.WeakSet[Task] = weakref.WeakSet()
        self._queue: asyncio.Queue[tuple[str, Task]] = asyncio.Queue()

    # 插件任务持有全部等待者；卸载时一并取消。
    async def run(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            while True:
                session_id, task = await self._queue.get()
                _ = tasks.create_task(
                    self._settle_logged(session_id, task), name=f"steer-settle:{session_id}",
                    context=independent_task_context(),
                )

    # 一个会话的收尾失败只丢下它自己的插话，不能带走其他会话的等待者。
    async def _settle_logged(self, session_id: str, task: Task) -> None:
        try:
            await self._settle(session_id, task)
        except Exception:
            logger.exception("插话收尾失败 session=%s", session_id)

    def bind(self, task: Task, session_id: str) -> Callable[[], Awaitable[None]]:
        self._bound.add(task)
        return lambda: self.inject(task, session_id)

    async def stage(self, session_id: str, message_id: str, message: ChannelInboundMessage) -> bool:
        task = await self._open(session_id).active_task()
        if task is None or task not in self._bound:
            return False
        items = self._staged.setdefault(session_id, [])
        if all(item.message_id != message_id for item in items):
            items.append(_Staged(message_id, message))
        if self._watched.get(session_id) is not task:
            self._watched[session_id] = task
            self._queue.put_nowait((session_id, task))
        return True

    def withdraw(self, session_id: str, message_id: str) -> bool:
        items = self._staged.get(session_id, [])
        for index, item in enumerate(items):
            if item.message_id == message_id:
                del items[index]
                return True
        return False

    async def inject(self, task: Task, session_id: str) -> None:
        items = self._staged.pop(session_id, [])
        session = self._open(session_id)
        for index, item in enumerate(items):
            try:
                _ = await session.steer(item.message_id, self._body(item.message), task)
            except Exception:
                # 任务已撤权或来源暂不接纳：剩下的交还暂存，由回合结局决定去向。
                logger.info("插话未能在本步写入 session=%s", session_id, exc_info=True)
                self._staged.setdefault(session_id, [])[:0] = items[index:]
                return
            except BaseException:
                # 取消时同样交还未写入的余项，再继续传播取消。
                self._staged.setdefault(session_id, [])[:0] = items[index:]
                raise

    # 回复任务结束后：另一轮已在进行就跟随它；否则（含用户停止）转成普通输入开启下一轮。
    async def _settle(self, session_id: str, task: Task) -> None:
        # 1. 等原任务真实结束；自己被取消时照常退出。
        try:
            _ = await task.join()
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller is not None and caller.cancelling():
                raise
        except Exception:
            pass
        if self._watched.get(session_id) is task:
            del self._watched[session_id]
        if not self._staged.get(session_id):
            return
        async with self._scope():
            # 2. 新一轮已经开始：插话留给它的下一步。
            current = await self._open(session_id).active_task()
            if current is not None:
                self._watched[session_id] = current
                self._queue.put_nowait((session_id, current))
                return
            # 3. 本轮结束或被停止：插话直接作为下一轮的输入。
            for item in self._staged.pop(session_id, []):
                try:
                    _ = await self._accept(session_id, item.message_id, item.message)
                except Exception:
                    logger.exception("插话转为普通输入失败 session=%s", session_id)
