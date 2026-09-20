from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent.plugin_composition import Context
from agent.plugin_composition.tasks import Task, TaskCapacity
from agent.plugin_composition.tasks import RestartGate
from typing import Protocol, cast
from agent.plugin_composition.messages import MessageCatalog, MessageReader
from agent.plugin_contracts import Control, Output

logger = logging.getLogger(__name__)
Program = Callable[[Task, MessageReader, str], Awaitable[object]]

_FAULT_BACKOFF = 0.2


class SourceSession(Protocol):
    async def start(self, program: Program) -> Task | None: ...
    def needs_reply(self, reader: MessageReader, source: str) -> bool: ...
    async def record_failure(self, error: BaseException, *, boundary: int | None = None) -> None: ...
    async def wait_capacity(self) -> None: ...


class Source(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def open(self) -> Callable[[str], SourceSession]: ...


class Sources(Protocol):
    def entries(self) -> tuple[Source, ...]: ...
    def needs_reply(self, reader: MessageReader, source: str) -> bool: ...


@dataclass(slots=True)
class _Wake:
    changed: bool = True
    event: asyncio.Event = field(default_factory=asyncio.Event)


async def follow(
    ctx: Context, catalog: MessageCatalog,
    sources: Sources, program: Program, restart_gate: RestartGate | None = None,
    *, poll_interval: float = 15.0,
) -> None:
    """从日志追赶可回复来源；空闲不保留 scope，不保存 cursor 或回复队列。"""
    active: dict[tuple[str, str], _Wake] = {}
    fault: list[BaseException] = []
    faulted = asyncio.Event()

    def fail(error: BaseException) -> None:
        fault.append(error)
        faulted.set()

    async def drive(session_id: str, source: Source, wake: _Wake) -> None:
        """每个 Session 独立排空旧工作；并发通知只要求再次读取日志。"""
        task: Task | None = None
        task_head = -1
        session: SourceSession | None = None
        pending: tuple[int, BaseException] | None = None

        def head() -> int:
            try:
                return int(catalog.reader(session_id).head(source=source.name))
            except AttributeError:
                # 测试替身可以不提供 head；此时没有可判定的持久边界。
                return -1
            except Exception:
                logger.warning("持久 head 读取失败", exc_info=True)
                return -1

        def stalled() -> bool:
            """来源是否已把失败持久停摆；只有持久边界才允许静默退出。"""
            if session is None:
                return False
            try:
                return not bool(session.needs_reply(catalog.reader(session_id), source.name))
            except AttributeError:
                return False
            except Exception:
                return False

        def boundary_committed() -> bool:
            """旧任务负责的区间已提交持久终态；lane 不必等它的物理清理。"""
            if task_head < 0:
                return False
            try:
                for message in catalog.reader(session_id).snapshot(
                    after_seq=task_head
                ):
                    if message.source != source.name:
                        continue
                    if isinstance(message.body, Output) and message.body.finish != "continue":
                        return True
                    if isinstance(message.body, Control):
                        return True
            except Exception:
                logger.warning("持久边界核对失败", exc_info=True)
            return False

        try:
            while True:
                if pending is not None:
                    # 停摆回执绑定原边界身份；head 变化不直接清除，只重试保存。
                    boundary, error = pending
                    if session is None:
                        fail(RuntimeError(f"回复驱动没有持久进展 (no progress): {error}"))
                        return
                    try:
                        await session.record_failure(error, boundary=boundary)
                    except AttributeError:
                        fail(RuntimeError(f"回复驱动没有持久进展 (no progress): {error}"))
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning("停摆回执保存失败，退避后重试", exc_info=True)
                        await asyncio.sleep(_FAULT_BACKOFF)
                        continue
                    pending = None
                    wake.changed = True
                    continue
                if task is None:
                    if not wake.changed:
                        return
                    wake.changed = False
                    try:
                        async with ctx.runtime_scope():
                            session = source.open(session_id)
                            task = await session.start(program)
                    except asyncio.CancelledError:
                        raise
                    except TaskCapacity:
                        # 容量等待是明确的等待原因，不走无进展故障路径。
                        try:
                            if session is not None:
                                await session.wait_capacity()
                            else:
                                await asyncio.sleep(_FAULT_BACKOFF)
                        except AttributeError:
                            await asyncio.sleep(_FAULT_BACKOFF)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.warning("Task 容量等待失败", exc_info=True)
                        wake.changed = True
                        continue
                    except Exception:
                        # 接纳故障不重复执行程序；同 head 不再驱动。
                        logger.warning("来源接纳失败，等待新的持久事实", exc_info=True)
                        return
                    if task is None:
                        # 来源在准入段判定不需要回复；等待下一条持久事实。
                        if restart_gate is not None and not restart_gate.accepting:
                            await restart_gate.wait_until_open()
                            wake.changed = True
                        continue
                    task_head = head()
                    continue
                joined = asyncio.ensure_future(task.join())
                woken = asyncio.ensure_future(wake.event.wait())
                done, _ = await asyncio.wait(
                    (joined, woken), return_when=asyncio.FIRST_COMPLETED
                )
                woken.cancel()
                _ = await asyncio.gather(woken, return_exceptions=True)
                if joined in done:
                    try:
                        _ = joined.result()
                    except asyncio.CancelledError:
                        current = asyncio.current_task()
                        if current is not None and current.cancelling():
                            raise
                    except Exception as error:
                        if stalled():
                            # 来源已持久停摆；失败的 Session 不阻塞其他 Session。
                            logger.warning("回复程序失败，保留日志等待新输入或控制", exc_info=True)
                        else:
                            try:
                                boundary = head()
                            except Exception:
                                boundary = -1
                            pending = (boundary, error)
                        task = None
                        continue
                    task = None
                    wake.changed = True
                    continue
                # 新持久事实先于排空到达：旧业务边界已提交时才提前释放 lane。
                joined.cancel()
                _ = await asyncio.gather(joined, return_exceptions=True)
                wake.event.clear()
                if boundary_committed():
                    task = None
                    continue
        finally:
            active.pop((session_id, source.name), None)
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

    try:
        stream = catalog.follow(poll_interval=poll_interval)
    except TypeError:
        stream = catalog.follow()

    # TaskGroup 把接纳失败交给所属 Fiber；正常程序失败由来源逐 Session 记录。
    # 停摆故障在 group 退出后原样抛出，不把领域错误包进 ExceptionGroup。
    previous: dict[str, int] = {}
    stop = False
    async with asyncio.TaskGroup() as group:
        aiter = stream.__aiter__()
        next_heads: asyncio.Task | None = None
        try:
            while not stop:
                next_heads = asyncio.ensure_future(aiter.__anext__())
                fault_wait = asyncio.ensure_future(faulted.wait())
                done, _ = await asyncio.wait(
                    (next_heads, fault_wait), return_when=asyncio.FIRST_COMPLETED
                )
                fault_wait.cancel()
                await asyncio.gather(fault_wait, return_exceptions=True)
                if faulted.is_set() or next_heads.cancelled():
                    next_heads.cancel()
                    await asyncio.gather(next_heads, return_exceptions=True)
                    next_heads = None
                    stop = True
                    continue
                try:
                    heads = next_heads.result()
                except StopAsyncIteration:
                    return
                changed = {
                    key for key, head in heads.items() if previous.get(key) != head
                }
                previous = dict(heads)
                # needs_reply 是 Sources 的可选核对；缺失时只靠持久 head 变化驱动。
                needs = getattr(sources, "needs_reply", None)
                for session_id in heads:
                    try:
                        present = catalog.reader(session_id).source_names()
                    except Exception:
                        # 单项读取故障只隔离该 Session，不波及其他 lane。
                        logger.warning("Session 来源读取失败 session=%s", session_id, exc_info=True)
                        continue
                    for source in sources.entries():
                        if source.name not in present:
                            continue
                        key = (session_id, source.name)
                        due = session_id in changed
                        if not due and key not in active and needs is not None:
                            # 轮询重扫时 head 未变也可能仍欠回复；持久事实决定驱动。
                            try:
                                due = bool(
                                    needs(catalog.reader(session_id), source.name)
                                )
                            except Exception:
                                logger.warning(
                                    "回复需求核对失败 session=%s source=%s",
                                    session_id, source.name, exc_info=True,
                                )
                                due = False
                        if not due:
                            continue
                        wake = active.get(key)
                        if wake is None:
                            wake = _Wake()
                            active[key] = wake
                            _ = group.create_task(drive(session_id, source, wake))
                        else:
                            wake.changed = True
                            wake.event.set()
                # 每次轮询都让存活 lane 重新核对持久边界与容量，丢失的进程内唤醒不致命。
                for wake in active.values():
                    wake.event.set()
        finally:
            if next_heads is not None and not next_heads.done():
                next_heads.cancel()
                await asyncio.gather(next_heads, return_exceptions=True)
            await asyncio.shield(stream.aclose())
    if fault:
        raise fault[0]
    raise asyncio.CancelledError
