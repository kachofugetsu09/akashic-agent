"""一次等待与其终态回执由 timer 实例持有。"""
from __future__ import annotations

import asyncio
import secrets
from datetime import UTC, datetime

from plugins.timer.contract import TimerHandle, TimerReceipt, TimerStatus


class AsyncioOneShotTimer:
    """提供一次等待，并在 Effect 结束时排空全部登记。"""

    def __init__(self) -> None:
        self._handles: set[_AsyncioTimerHandle] = set()
        self._closed = False

    def schedule(self, deadline: datetime) -> TimerHandle:
        if self._closed:
            raise RuntimeError("timer 已关闭")
        if deadline.tzinfo is None:
            raise ValueError("Timer deadline 必须包含时区")
        handle = _AsyncioTimerHandle(deadline.astimezone(UTC))
        self._handles.add(handle)
        handle.settled.add_done_callback(lambda _: self._handles.remove(handle))
        return handle

    async def close(self) -> None:
        """关闭后不再接纳登记；每个未完成等待都产生 cancelled 回执。"""
        self._closed = True
        await asyncio.gather(*(handle.cleanup() for handle in tuple(self._handles)))


class _AsyncioTimerHandle:
    """持有独立等待；调用方取消不改变其它调用方看到的终态。"""

    def __init__(self, deadline: datetime) -> None:
        self._id = "timer:" + secrets.token_hex(16)
        self._deadline = deadline
        self.settled: asyncio.Future[TimerReceipt] = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._wait(), name=self._id)
        self._task.add_done_callback(self._settle)

    @property
    def id(self) -> str:
        return self._id

    async def result(self) -> TimerReceipt:
        return await asyncio.shield(self.settled)

    async def cancel(self) -> TimerReceipt:
        if not self._task.done():
            self._task.cancel()
        return await self.result()

    async def cleanup(self) -> None:
        await self.cancel()

    async def _wait(self) -> None:
        delay = max(0.0, (self._deadline - datetime.now(UTC)).total_seconds())
        await asyncio.sleep(delay)

    def _settle(self, task: asyncio.Task[None]) -> None:
        """done 回调也覆盖 coroutine 首次运行前的取消。"""
        # 1. 不把异常误报成 fired，也不让调用方取消污染共享回执。
        if not task.cancelled() and (error := task.exception()) is not None:
            self.settled.set_exception(error)
            return
        # 2. 只生成一次事实，result/cancel/cleanup 都读取同一回执。
        self.settled.set_result(TimerReceipt(
            timer_id=self._id,
            deadline=self._deadline,
            settled_at=datetime.now(UTC),
            status=TimerStatus.CANCELLED if task.cancelled() else TimerStatus.FIRED,
        ))
