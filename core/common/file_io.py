"""有界文件线程；取消必须等待实际磁盘工作结束。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


@dataclass
class _FileIoState:
    slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))
    users: int = 0


_FILE_IO_SLOTS: dict[asyncio.AbstractEventLoop, _FileIoState] = {}


async def run_file_io(fn: Callable[[], T]) -> T:
    """最多四个磁盘操作并行；取消后仍等物理工作结束才归还锁与 owner。"""
    # 1. 等待名额时可以取消；线程启动后不能把取消当作工作已结束。
    loop = asyncio.get_running_loop()
    state = _FILE_IO_SLOTS.setdefault(loop, _FileIoState())
    state.users += 1
    try:
        async with state.slots:
            work = asyncio.create_task(asyncio.to_thread(fn))
            cancelled: asyncio.CancelledError | None = None
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError as exc:
                    cancelled = exc
                except Exception:
                    # 实际错误由 result 取回；同时发生取消时保留两种失败。
                    break
            # 2. 到这里线程已结束，外层才可以释放文件锁和 manager operation。
            try:
                result = work.result()
            except Exception as exc:
                if cancelled is not None:
                    raise BaseExceptionGroup("文件操作取消且物理工作失败", [cancelled, exc]) from None
                raise
            if cancelled is not None:
                raise cancelled
            return result
    finally:
        state.users -= 1
        if state.users == 0:
            del _FILE_IO_SLOTS[loop]


