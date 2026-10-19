"""工具收尾归 tools：划定本程序的工具调用并排空，再交给各工具注册时声明的释放入口。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import cast

from agent.plugin_composition import CompositionError, Context
from agent.plugin_composition.context import RuntimeScope
from agent.plugin_composition.tasks import Task, TaskAdmission, TaskSlot
from plugins.ledger.contract import BINDINGS, Bindings, CallRef, Control, Message, MessageReader, Output, ToolCall
from plugins.tools.contract import TOOLS, ToolRelease

Drain = Callable[[tuple[CallRef, ...]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Release:
    """入口时固定的释放入口、其插件 owner 与 Context；收尾不重新选择 provider。"""

    owner: str
    release: ToolRelease
    context: Context
    captures: bool


@dataclass(frozen=True, slots=True)
class _Plan:
    calls: tuple[tuple[CallRef, str, str | None], ...]
    abandoned: bool
    end: int


def _abandon_before(messages: tuple[Message, ...], source: str, through_seq: int) -> str | None:
    """只从已持久化边界区分新旧资源集合，暂停和新输入不改变 owner 分区。"""
    return next((message.message_id for message in reversed(messages)
                 if message.source == source and message.seq <= through_seq
                 and isinstance(message.body, Control) and message.body.action == "abandon"), None)


# 入口固定各 release owner 的调用许可：回合期间 owner 卸载会等待收尾，清理 Task 接手后持有到释放完成。
async def _retain_owners(releases: Mapping[str, Release]) -> list[RuntimeScope]:
    owners = {entry.owner: entry.context for entry in releases.values()}
    scopes: list[RuntimeScope] = []
    for context in owners.values():
        try:
            async with context.runtime_scope():
                scopes.append(context.capture_runtime_scope())
        except CompositionError:
            # owner 已被替换或停止接纳：本回合打不开它的工具，也就没有要它释放的调用。
            continue
    return scopes


# 放弃后到达的新输入与调用不属于旧程序；更早历史只贡献最后一个 abandon 的 owner 分区。
def _plan(reader: MessageReader, source: str, from_seq: int) -> _Plan:
    messages = reader.snapshot(after_seq=from_seq - 1)
    through = from_seq - 1
    while through >= 0:
        control = reader.latest_control(source, through_seq=through)
        if control is None:
            break
        if isinstance(control.body, Control) and control.body.action == "abandon":
            messages = (control, *messages)
            break
        through = control.seq - 1
    abandoned = next((message for message in messages
                      if message.source == source and isinstance(message.body, Control)
                      and message.body.action == "abandon" and message.body.through_seq >= from_seq), None)
    end = reader.head() if abandoned is None else cast(Control, abandoned.body).through_seq
    calls = tuple(
        (CallRef(message.message_id, index), part.binding_id, _abandon_before(messages, source, message.seq))
        for message in messages
        if message.source == source and from_seq <= message.seq <= end and isinstance(message.body, Output)
        for index, part in enumerate(message.body.parts) if isinstance(part, ToolCall)
    )
    return _Plan(calls, abandoned is not None, end)


# 只释放入口时已声明 release 且 owner 一致的调用；state 合同与 ToolCatalog.open 相同。
async def _release(ctx: Context, bindings: Bindings, releases: Mapping[str, Release], identity: str,
                   session_id: str, source: str, boundary: str | None) -> None:
    try:
        metadata = bindings.describe(identity, TOOLS)
        description = cast(Mapping[str, object], metadata["tool"])
        entry = releases.get(str(description["name"]))
        if entry is None or description.get("owner", entry.owner) != entry.owner:
            return
        state: object = metadata.get("state", {}) if entry.captures else {}
        if not isinstance(state, Mapping) or entry.captures and "state" not in metadata:
            raise ValueError("工具 binding state 缺失或不是 JSON 对象")
        failures = await entry.release(cast(Mapping[str, object], state), session_id, source, boundary)
        if failures:
            _ = ctx.report_incident("tool_cleanup_failed", f"{identity}: {failures}")
    except Exception as error:
        # SH-002: 此处是独立清理边界，失败保留真实 owner 并明确报告；不中断其他调用的释放。
        _ = ctx.report_incident("tool_cleanup_failed", f"{identity}: {type(error).__name__}: {error}")


@asynccontextmanager
async def tool_cleanup(
    ctx: Context, tasks: TaskAdmission, releases: Mapping[str, Release],
    reader: MessageReader, source: str, from_seq: int, *,
    task: Task | None = None, drain: Drain | None = None,
) -> AsyncGenerator[None]:
    """放弃可停止等待；独立 Task 保留旧工具资源清理、generation 与重启许可。"""
    async with ctx.runtime_scope():
        owners = await _retain_owners(releases)
        try:
            yield
        finally:
            try:
                plan = _plan(reader, source, from_seq)
                if plan.calls:
                    cleanup = _Cleanup(ctx, owners, releases, reader, source, plan.calls, drain)
                    owners = []
                    try:
                        await _settle(tasks, cleanup, reader, source, from_seq, plan, task)
                    except BaseException:
                        cleanup.discard()
                        raise
            finally:
                for scope in owners:
                    scope.discard()


class _Cleanup:
    """一次收尾：先排空本程序的调用，再逐个交给原 owner 释放；tools 与 owner 的许可都持有到结束。"""

    def __init__(self, ctx: Context, owners: list[RuntimeScope], releases: Mapping[str, Release],
                 reader: MessageReader, source: str, calls: tuple[tuple[CallRef, str, str | None], ...],
                 drain: Drain | None):
        self._ctx = ctx
        self._owners = owners
        self._releases = releases
        self._reader = reader
        self._source = source
        self._calls = calls
        self._drain = drain
        self._started = False

    # 清理没能启动时释放入口固定的 owner 许可，避免 owner 永远等不到排空；已启动时由 __call__ 负责。
    def discard(self) -> None:
        if not self._started:
            owners, self._owners = self._owners, []
            for held in owners:
                held.discard()

    async def __call__(self, _task: Task | None) -> None:
        """宿主停止也先排空真实清理，不能把撤权当作进程已退出。"""
        self._started = True
        scope = self._ctx.capture_runtime_scope()
        operation = self._scoped(scope)
        try:
            work = asyncio.create_task(operation)
        except BaseException:
            operation.close()
            for held in (scope, *self._owners):
                held.discard()
            raise
        await _shielded(work)

    async def _scoped(self, scope: RuntimeScope) -> None:
        async with AsyncExitStack() as stack:
            for held in (scope, *self._owners):
                _ = stack.push(held.__exit__)
            for held in (scope, *self._owners):
                _ = held.__enter__()
            await self._cleanup()

    async def _cleanup(self) -> None:
        try:
            if self._drain is not None:
                await self._drain(tuple(ref for ref, _, _ in self._calls))
        finally:
            bindings = self._ctx.require(BINDINGS)
            for identity, boundary in dict.fromkeys((identity, boundary) for _, identity, boundary in self._calls):
                await _release(self._ctx, bindings, self._releases, identity,
                               self._reader.session_id, self._source, boundary)


# 等待者被取消也不打断清理本身；清理结束后再把取消交还调用方。
async def _shielded(work: asyncio.Task[None]) -> None:
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


# 有源程序 Task 时清理进入独立 Task 并继承重启许可；普通结束等待清理，明确放弃只释放等待者。
async def _settle(tasks: TaskAdmission, run: _Cleanup, reader: MessageReader,
                  source: str, from_seq: int, plan: _Plan, task: Task | None) -> None:
    if task is None:
        await run(None)
        return

    def admit(slot: TaskSlot) -> Task:
        permit = task.child_permit() if task.has_external_permit else None
        try:
            owned = slot.start(run)
        except BaseException:
            if permit is not None:
                permit.release()
            raise
        if permit is not None:
            owned.on_done(permit.release)
        # 接纳后、首次执行前就被取消时 __call__ 不会运行，由终态回调释放入口固定的 owner 许可。
        owned.on_done(run.discard)
        return owned

    owned = await tasks.admit(("tool-cleanup", reader.session_id, source, from_seq, plan.end), admit)
    if not plan.abandoned:
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
