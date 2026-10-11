"""合并只读消息水位、已读水位与回复活动，向一个 WebSocket 发布侧栏摘要。"""

import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing


async def follow_session_activity(
    heads: AsyncGenerator[Mapping[str, int], None],
    active: AsyncGenerator[frozenset[str] | None, None],
    *, prefix: str,
    reads: AsyncGenerator[Mapping[str, int], None] | None = None,
) -> AsyncGenerator[dict[str, object], None]:
    """先取得各份基线，再发布变化；慢读者合并唤醒，水位始终来自已提交消息。"""
    changed = asyncio.Event()
    current_heads: Mapping[str, int] | None = None
    current_active: frozenset[str] | None = None
    current_reads: Mapping[str, int] | None = None
    reply_ready = False

    # 1. 两个 owner 独立跟随；生成器关闭或连接取消时共同排空。
    async def read_heads() -> None:
        nonlocal current_heads
        async with aclosing(heads):
            async for value in heads:
                current_heads = value
                changed.set()
        raise RuntimeError("会话日志订阅已结束，需要重新连接")

    async def read_active() -> None:
        nonlocal current_active, reply_ready
        async with aclosing(active):
            async for value in active:
                current_active, reply_ready = value, True
                changed.set()
        if current_active is not None:
            raise RuntimeError("回复状态订阅已结束，需要重新连接")

    async def read_marks() -> None:
        nonlocal current_reads
        if reads is None:
            return
        async with aclosing(reads):
            async for value in reads:
                current_reads = value
                changed.set()
        raise RuntimeError("已读水位订阅已结束，需要重新连接")

    previous_heads: Mapping[str, int] | None = None
    previous_active: frozenset[str] | None = None
    previous_reads: Mapping[str, int] = {}
    async with asyncio.TaskGroup() as tasks:
        head_task = tasks.create_task(read_heads())
        active_task = tasks.create_task(read_active())
        reads_task = tasks.create_task(read_marks())
        try:
            while True:
                await changed.wait()
                changed.clear()
                if current_heads is None or not reply_ready or (reads is not None and current_reads is None):
                    continue
                running = None if current_active is None else frozenset(
                    key for key in current_active if key in current_heads and key.startswith(prefix))
                visible_reads = {key: seq for key, seq in (current_reads or {}).items() if key in current_heads}
                if (previous_heads is not None and current_heads == previous_heads and running == previous_active
                        and visible_reads == previous_reads):
                    continue
                # 2. 初次连接发快照；之后只发变化的水位和删除身份，活动集合很小。
                snapshot = previous_heads is None
                updates = {key: seq for key, seq in current_heads.items()
                           if snapshot or previous_heads is not None and previous_heads.get(key) != seq}
                read_updates = {key: seq for key, seq in visible_reads.items()
                                if snapshot or previous_reads.get(key) != seq}
                removed = [] if previous_heads is None else sorted(previous_heads.keys() - current_heads.keys())
                previous_heads, previous_active, previous_reads = current_heads, running, visible_reads
                yield {"version": 1, "snapshot": snapshot, "available": running is not None,
                       "active": sorted(running or ()), "heads": updates, "removed": removed,
                       "read_available": reads is not None, "read": read_updates}
        except GeneratorExit:
            return
        finally:
            head_task.cancel()
            active_task.cancel()
            _ = reads_task.cancel()
