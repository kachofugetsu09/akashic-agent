from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent.plugin_composition import Context
from agent.plugin_composition.tasks import Task
from plugins.sources.plugin import Source, Sources
from session.log import MessageCatalog, MessageReader

logger = logging.getLogger(__name__)
Program = Callable[[Task, MessageReader, str], Awaitable[object]]


@dataclass(slots=True)
class _Wake:
    changed: bool = True


async def follow(
    ctx: Context, catalog: MessageCatalog,
    sources: Sources, program: Program,
) -> None:
    """从日志追赶可回复来源；空闲不保留 scope，不保存 cursor 或回复队列。"""
    active: dict[tuple[str, str], _Wake] = {}

    async def drive(session_id: str, source: Source, wake: _Wake) -> None:
        """每个 Session 独立排空旧工作；并发通知只要求再次读取日志。"""
        task: Task | None = None
        try:
            while wake.changed:
                wake.changed = False
                async with ctx.runtime_scope():
                    task = await source.open(session_id).start(program)
                if task is None:
                    continue
                try:
                    _ = await task.join()
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                except Exception:
                    # Source 已记录可见 failure；失败的 Session 不阻塞其他 Session。
                    logger.warning("回复程序失败，保留日志等待新输入或控制", exc_info=True)
                task = None
                wake.changed = True
        finally:
            del active[(session_id, source.name)]
            if task is not None:
                task.cancel()
                # 卸载只撤销新决策；已开始的工具仍必须真实结算并归还资源。
                while not task.done:
                    try:
                        _ = await task.join()
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        logger.warning("停止回复时已开始的工作结算失败", exc_info=True)

    # TaskGroup 把接纳失败交给所属 Fiber；正常程序失败由来源逐 Session 记录。
    previous: dict[str, int] = {}
    async with asyncio.TaskGroup() as group:
        async for heads in catalog.follow():
            changed = [key for key, head in heads.items() if previous.get(key) != head]
            previous = dict(heads)
            for session_id in changed:
                present = {message.source for message in catalog.reader(session_id).snapshot()}
                for source in sources.entries():
                    if source.name not in present:
                        continue
                    key = (session_id, source.name)
                    wake = active.get(key)
                    if wake is None:
                        wake = _Wake()
                        active[key] = wake
                        _ = group.create_task(drive(session_id, source, wake))
                    else:
                        wake.changed = True
