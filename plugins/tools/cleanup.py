"""工具收尾归 tools：划定本程序的工具调用并排空，再交给各工具注册时声明的释放入口。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from agent.plugin_composition import Context
from agent.plugin_composition.tasks import Task, TaskAdmission, TaskSlot
from plugins.ledger.contract import BINDINGS, Bindings, CallRef, Control, Message, MessageReader, Output, ToolCall
from plugins.tools.contract import TOOLS, ToolRelease


@dataclass(frozen=True, slots=True)
class Release:
    """入口时固定的释放入口与其插件 owner；收尾不重新选择 provider。"""

    owner: str
    release: ToolRelease


def _abandon_before(messages: tuple[Message, ...], source: str, through_seq: int) -> str | None:
    """只从已持久化边界区分新旧资源集合，暂停和新输入不改变 owner 分区。"""
    return next((message.message_id for message in reversed(messages)
                 if message.source == source and message.seq <= through_seq
                 and isinstance(message.body, Control) and message.body.action == "abandon"), None)


# 只释放入口时已声明 release 且 owner 一致的调用；失败报告事故，不中断其他调用的释放。
async def _release(ctx: Context, bindings: Bindings, releases: Mapping[str, Release], identity: str,
                   session_id: str, source: str, boundary: str | None) -> None:
    try:
        metadata = bindings.describe(identity, TOOLS)
        description = cast(Mapping[str, object], metadata["tool"])
        entry = releases.get(str(description["name"]))
        if entry is None or description.get("owner", entry.owner) != entry.owner:
            return
        failures = await entry.release(cast(Mapping[str, object], metadata["state"]), session_id, source, boundary)
        if failures:
            _ = ctx.report_incident("tool_cleanup_failed", f"{identity}: {failures}")
    except Exception as error:
        # SH-002: 此处是独立清理边界，失败保留真实 owner 并明确报告。
        _ = ctx.report_incident("tool_cleanup_failed", f"{identity}: {type(error).__name__}: {error}")


@asynccontextmanager
async def tool_cleanup(
    ctx: Context, tasks: TaskAdmission, releases: Mapping[str, Release],
    reader: MessageReader, source: str, from_seq: int, *,
    task: Task | None = None, drain: Callable[[tuple[CallRef, ...]], Awaitable[None]] | None = None,
) -> AsyncGenerator[None]:
    """放弃可停止等待；独立 Task 保留旧工具资源清理、generation 与重启许可。"""
    async with ctx.runtime_scope():
        try:
            yield
        finally:
            # 1. 放弃后到达的新输入与调用不属于旧程序的清理范围。
            # 调用与放弃判定只涉及 from_seq 起的前缀；更早历史只贡献最后一个
            # abandon Control 的 owner 分区，逐个点查 Control 即可，不做全量快照。
            messages = reader.snapshot(after_seq=from_seq - 1)
            through = from_seq - 1
            while through >= 0:
                control = reader.latest_control(source, through_seq=through)
                if control is None:
                    break
                if control.body.action == "abandon":
                    messages = (control, *messages)
                    break
                through = control.seq - 1
            abandoned = next((message for message in messages
                              if message.source == source and isinstance(message.body, Control)
                              and message.body.action == "abandon" and message.body.through_seq >= from_seq), None)
            end = reader.head() if abandoned is None else cast(Control, abandoned.body).through_seq
            calls = tuple(
                (CallRef(message.message_id, index), part.binding_id,
                 _abandon_before(messages, source, message.seq))
                for message in messages
                if message.source == source and from_seq <= message.seq <= end and isinstance(message.body, Output)
                for index, part in enumerate(message.body.parts) if isinstance(part, ToolCall)
            )
            if calls:
                async def cleanup() -> None:
                    try:
                        if drain is not None:
                            await drain(tuple(ref for ref, _, _ in calls))
                    finally:
                        bindings = ctx.require(BINDINGS)
                        for identity, boundary in dict.fromkeys((identity, boundary) for _, identity, boundary in calls):
                            await _release(ctx, bindings, releases, identity, reader.session_id, source, boundary)

                async def run(_task: Task | None) -> None:
                    """宿主停止也先排空真实清理，不能把撤权当作进程已退出。"""
                    scope = ctx.capture_runtime_scope()
                    async def scoped_cleanup() -> None:
                        async with scope:
                            await cleanup()
                    operation = scoped_cleanup()
                    try:
                        work = asyncio.create_task(operation)
                    except BaseException:
                        operation.close()
                        await scope.close()
                        raise
                    try:
                        await asyncio.shield(work)
                    except asyncio.CancelledError:
                        while not work.done():
                            try:
                                await asyncio.shield(work)
                            except asyncio.CancelledError:
                                continue
                        if not work.cancelled():
                            work.result()
                        raise

                # 3. Task 自己固定当前 scope；源程序结束不会释放它的资源或 permit。
                def admit(slot: TaskSlot) -> Task:
                    permit = task.child_permit() if task is not None and task.has_external_permit else None
                    try:
                        owned = slot.start(run)
                    except BaseException:
                        if permit is not None:
                            permit.release()
                        raise
                    if permit is not None:
                        owned.on_done(permit.release)
                    return owned

                if task is None:
                    await run(None)
                else:
                    owned = await tasks.admit(
                        ("tool-cleanup", reader.session_id, source, from_seq, end), admit,
                    )
                    if abandoned is None:
                        await _wait_cleanup(owned, reader, source, from_seq)


async def _wait_cleanup(task: Task, reader: MessageReader, source: str, from_seq: int) -> None:
    """普通停止排空；若随后明确放弃，只释放等待者，原清理 Task 继续持有资源。"""
    async def abandoned() -> None:
        async for message in reader.follow(after_seq=from_seq - 1):
            if (message.source == source and isinstance(message.body, Control)
                    and message.body.action == "abandon" and message.body.through_seq >= from_seq):
                return
        raise RuntimeError("工具清理订阅提前结束")

    joined = asyncio.create_task(task.join())
    stopped = asyncio.create_task(abandoned())
    cancellation: asyncio.CancelledError | None = None
    try:
        while True:
            try:
                done, _ = await asyncio.wait((joined, stopped), return_when=asyncio.FIRST_COMPLETED)
                break
            except asyncio.CancelledError as error:
                cancellation = error
        if joined in done:
            _ = joined.result()
        else:
            stopped.result()
        if cancellation is not None:
            raise cancellation
    finally:
        _ = joined.cancel()
        _ = stopped.cancel()
        _ = await asyncio.gather(joined, stopped, return_exceptions=True)
