from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from agent.plugin_composition import Context, RuntimeScope
from agent.plugin_composition.messages import MessageCatalog, MessageReader
from agent.plugin_composition.model import CompositionError
from agent.plugin_composition.tasks import RestartGate, Task, TaskServiceClosed
from agent.plugin_contracts import Control, Input, Output
from plugins.sources.contract import (
    AsyncSource as Source,
    SourcesV5 as Sources,
    GuardedSourceSession as SourceSession,
)

logger = logging.getLogger(__name__)
Program = Callable[[Task, MessageReader, str], Awaitable[object]]


@dataclass(slots=True)
class _Wake:
    source: Source | None = None
    changed: bool = True
    event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True)
class _Monitor:
    task: asyncio.Task[None]
    finished: asyncio.Event


@dataclass(slots=True)
class _Iteration:
    """保存一次 source drive 的 owner 与物理结算事实。"""

    drive_task: asyncio.Task[Any]
    session: SourceSession | None = None
    task: Task | None = None
    task_finished: asyncio.Event = field(default_factory=asyncio.Event)
    task_joined: bool = False
    task_cancel_requested: bool = False
    task_error: BaseException | None = None
    reply_scope: RuntimeScope | None = None
    reply_entered: bool = False
    source_scope: RuntimeScope | None = None
    monitors: list[_Monitor] = field(default_factory=list)
    admission_cancel_requested: bool = False
    monitor_errors: list[BaseException] = field(default_factory=list)
    monitor_cancel: object = field(default_factory=object)
    monitor_failure_cancel: object = field(default_factory=object)
    caller_cancellations: list[asyncio.CancelledError] = field(default_factory=list)
    coalesced_caller_cancel: bool = False
    pending_errors: list[BaseException] = field(default_factory=list)
    cleanup_errors: list[BaseException] = field(default_factory=list)
    source_settlement_errors: list[BaseException] = field(default_factory=list)
    stop_drive: bool = False
    detached: bool = False

    # 消费 follower 自己发出的取消，并保持外部取消可观察。
    def consume_internal_cancel(self, error: asyncio.CancelledError) -> bool:
        if self.monitor_cancel in error.args or self.monitor_failure_cancel in error.args:
            current = asyncio.current_task()
            if current is not None and current.uncancel():
                # 同一个 await 可能只交付 marker；剩余计数只证明还有外部取消
                # 事实，不足以恢复一个 Python 从未单独交付的原始消息对象。
                self.coalesced_caller_cancel = True
            self.stop_drive = True
            return True
        return False

    # 保存真实 caller CancelledError，并清掉本次取消以继续结算。
    def save_caller_cancel(self, error: asyncio.CancelledError) -> None:
        self.coalesced_caller_cancel = False
        self.caller_cancellations.append(error)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            current.uncancel()

    # 清理阶段收到的取消：follower 自己的 marker 被消费，其余记为 caller 取消。
    def absorb_cancel(self, error: asyncio.CancelledError) -> None:
        if not self.consume_internal_cancel(error):
            self.save_caller_cancel(error)

    # 等待一次物理完成事件；清理阶段每轮使用新的 await。
    async def wait_event(self, event: asyncio.Event, *, tolerate_caller_cancel: bool) -> None:
        while not event.is_set():
            try:
                await event.wait()
            except asyncio.CancelledError as error:
                if not tolerate_caller_cancel:
                    raise
                self.absorb_cancel(error)

    # 在 on_done 后调用公开 join，区分 caller 与 Source Task 终态。
    async def retrieve_task(self, *, tolerate_caller_cancel: bool) -> None:
        task = self.task
        assert task is not None
        while not self.task_joined:
            current = asyncio.current_task()
            cancellation_count = 0 if current is None else current.cancelling()
            try:
                _ = await task.join()
            except asyncio.CancelledError as error:
                # on_done 已证明 exact Task 完成；只把本次 public join await
                # 新增的 waiter cancellation 归给 caller，不能用历史计数吞掉
                # child 的原始终态。
                if current is not None and current.cancelling() > cancellation_count:
                    if not tolerate_caller_cancel:
                        raise
                    self.absorb_cancel(error)
                    continue
                self.task_error = error
            except BaseException as error:
                self.task_error = error
            break
        self.task_joined = True

    # 等待 exact Source Task 完成，再读取其真实终态。
    async def wait_task(self) -> None:
        await self.wait_event(self.task_finished, tolerate_caller_cancel=False)
        await self.retrieve_task(tolerate_caller_cancel=False)
        if self.task_error is None:
            return
        if not isinstance(self.task_error, asyncio.CancelledError):
            logger.warning("回复程序失败，保留日志等待新输入或控制", exc_info=self.task_error)
        elif self.task_error.__cause__ is not None:
            self.source_settlement_errors.append(self.task_error)

    # 最多取消一次 exact Task，再等待 on_done 与公开 join。
    async def settle_task(self) -> None:
        task = self.task
        if task is None or self.task_joined:
            return
        if task.active and not self.task_cancel_requested:
            self.task_cancel_requested = True
            try:
                task.cancel()
            except BaseException as error:
                self.source_settlement_errors.append(error)
        await self.wait_event(self.task_finished, tolerate_caller_cancel=True)
        await self.retrieve_task(tolerate_caller_cancel=True)
        if self.task_error is not None:
            if not isinstance(self.task_error, asyncio.CancelledError):
                self.source_settlement_errors.append(self.task_error)
            elif not self.task_cancel_requested or self.task_error.__cause__ is not None:
                # 纯粹由 owner 发出的取消仍是正常结算；带真实 cleanup
                # cause 的 child CE 是 Source Task 的终态，必须原样保留。
                self.source_settlement_errors.append(self.task_error)

    # 停止一个 raw admission monitor，并取得其真实结果。
    async def stop_monitor(self, monitor: _Monitor) -> None:
        if not monitor.task.done():
            monitor.task.cancel()
        await self.wait_event(monitor.finished, tolerate_caller_cancel=True)
        try:
            monitor.task.result()
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            self.monitor_errors.append(error)

    # 关闭 captured scope，同时保留清理期间的 caller 取消。
    async def close_scope(self, scope: Any) -> None:
        try:
            await scope.close()
        except asyncio.CancelledError as error:
            self.absorb_cancel(error)
        except BaseException as error:
            self.cleanup_errors.append(error)

    # admission 关闭时只停止一次本次 drive。
    def request_admission_stop(self, cancel_marker: object) -> None:
        if self.admission_cancel_requested:
            return
        self.admission_cancel_requested = True
        self.stop_drive = True
        self.drive_task.cancel(cancel_marker)

    async def watch_admission(self, scope: Any) -> None:
        try:
            await scope.wait_admission_closed()
        except asyncio.CancelledError:
            raise
        except BaseException:
            self.request_admission_stop(self.monitor_failure_cancel)
            raise
        self.request_admission_stop(self.monitor_cancel)

    # Monitor 只等待已捕获的 admission；不继承 caller Context。
    def add_monitor(self, scope: Any) -> None:
        coroutine = self.watch_admission(scope)
        try:
            task = asyncio.create_task(coroutine, context=contextvars.Context())
        except BaseException:
            coroutine.close()
            raise
        finished = asyncio.Event()
        task.add_done_callback(lambda _task: finished.set())
        self.monitors.append(_Monitor(task, finished))

    async def run_program(self, program: Program, *args: Any) -> object:
        if self.reply_scope is None:
            raise RuntimeError("Reply scope 尚未准备")
        async with self.reply_scope:
            self.reply_entered = True
            return await program(*args)

    # 汇总本次 drive 自身的失败；Source Task 终态另行判断。
    def follower_errors(self) -> list[BaseException]:
        errors: list[BaseException] = [
            *self.pending_errors,
            *self.monitor_errors,
            *self.cleanup_errors,
            *self.caller_cancellations,
        ]
        if self.coalesced_caller_cancel:
            errors.append(
                asyncio.CancelledError("caller cancellation coalesced with admission stop")
            )
        return errors


# Enter one owner scope and report only local admission loss.
async def _enter_scope(context: Context) -> Any:
    scope = context.runtime_scope()
    try:
        await scope.__aenter__()
    except CompositionError as error:
        if error.code in {"OWNER_UNAVAILABLE", "STALE_ACTIVATION"}:
            logger.warning("回复 owner scope 不可用，结束当前 source drive: %s", error.code)
            return None
        raise
    return scope


class _Follower:
    """从日志追赶可回复来源；空闲不保留 scope，不保存 cursor 或回复队列。"""

    def __init__(
        self, ctx: Context, catalog: MessageCatalog,
        sources: Sources, program: Program, restart_gate: RestartGate | None,
    ) -> None:
        self._ctx = ctx
        self._catalog = catalog
        self._sources = sources
        self._program = program
        self._restart_gate = restart_gate
        self._active: dict[tuple[str, str], _Wake] = {}
        self._group: asyncio.TaskGroup | None = None
        self._drive_tasks: set[asyncio.Task[None]] = set()
        # 两个监听器只更新本地快照并置位；TaskGroup 仍是唯一业务 drive owner。
        self._latest_heads: dict[str, int] | None = None
        self._catalog_closed = False
        self._sources_closed = False
        self._wake_event = asyncio.Event()
        self._previous_heads: dict[str, int] = {}
        self._previous_sources: dict[str, Source] = {}
        self._observed_source_heads: dict[tuple[str, str], int] = {}

    async def run(self) -> None:
        async with asyncio.TaskGroup() as group:
            self._group = group
            catalog_task = self._create_owned(self._watch_catalog(), name="reply-catalog-watch")
            source_task = self._create_owned(self._watch_sources(), name="reply-source-watch")
            while True:
                await self._wake_event.wait()
                self._wake_event.clear()
                if self._catalog_closed or self._sources_closed:
                    break
                if self._latest_heads is not None:
                    self._dispatch_changes(self._latest_heads)
            for task in (catalog_task, source_task, *self._drive_tasks):
                if not task.done():
                    task.cancel()

    # Create one TaskGroup child and close its coroutine on handoff failure.
    def _create_owned(self, coroutine: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task[None]:
        assert self._group is not None
        try:
            return self._group.create_task(coroutine, name=name)
        except BaseException:
            coroutine.close()
            raise

    async def _watch_catalog(self) -> None:
        async for heads in self._catalog.follow():
            self._latest_heads = dict(heads)
            self._wake_event.set()
        self._catalog_closed = True
        self._wake_event.set()

    async def _watch_sources(self) -> None:
        async for _entries in self._sources.changes():
            self._wake_event.set()
        self._sources_closed = True
        self._wake_event.set()

    # Read the registration authority instead of the notification snapshot.
    def _current_source(self, name: str) -> Source | None:
        return next((item for item in self._sources.entries() if item.name == name), None)

    def _dispatch_changes(self, heads: dict[str, int]) -> None:
        current_sources = {source.name: source for source in self._sources.entries()}
        source_names = set(self._previous_sources) | set(current_sources)
        changed_source_names = {
            name for name in source_names
            if self._previous_sources.get(name) is not current_sources.get(name)
        }
        current_identity_changes = {
            name for name in changed_source_names if name in current_sources
        }

        # 1. A revoked source only updates an existing local handoff; its
        # exact Source owner still stops the in-flight Task via the scope monitor.
        for (_session_id, source_name), wake in tuple(self._active.items()):
            if source_name in changed_source_names:
                wake.source = current_sources.get(source_name)
                wake.changed = True

        changed_sessions = {
            session_id for session_id, head in heads.items()
            if self._previous_heads.get(session_id) != head
        }
        self._previous_heads = dict(heads)
        self._previous_sources = current_sources

        # 2. One scan handles changed sessions and genuine identity changes.
        session_ids = (
            heads
            if current_identity_changes
            else (session_id for session_id in heads if session_id in changed_sessions)
        )
        for session_id in session_ids:
            reader = self._catalog.reader(session_id)
            present = reader.source_names()
            session_changed = session_id in changed_sessions
            for source in current_sources.values():
                source_name = source.name
                if source_name not in present:
                    continue
                identity_changed = source_name in current_identity_changes
                if not session_changed and not identity_changed:
                    continue
                source_head = reader.head(source=source_name)
                key = (session_id, source_name)
                if identity_changed or self._observed_source_heads.get(key) != source_head:
                    self._observed_source_heads[key] = source_head
                    self._schedule(session_id, source)

    def _schedule(self, session_id: str, source: Source) -> None:
        key = (session_id, source.name)
        wake = self._active.get(key)
        if wake is not None:
            wake.source = source
            wake.changed = True
            wake.event.set()
            return
        wake = _Wake(source=source)
        self._active[key] = wake
        try:
            task = self._create_owned(
                self._drive(session_id, source.name, wake),
                name=f"reply-drive:{session_id}:{source.name}",
            )
        except BaseException:
            if self._active.get(key) is wake:
                del self._active[key]
            raise
        self._drive_tasks.add(task)
        task.add_done_callback(self._drive_tasks.discard)

    # 每个 Session 独立排空旧工作；并发通知只要求再次读取日志。
    async def _drive(self, session_id: str, source_name: str, wake: _Wake) -> None:
        try:
            while wake.changed:
                wake.changed = False
                source = wake.source
                if source is None:
                    break
                registered = self._current_source(source.name)
                if registered is not source:
                    wake.source = registered
                    if registered is None:
                        break
                    wake.changed = True
                    continue
                if not await self._drive_source(session_id, source, wake):
                    break
        finally:
            self._active.pop((session_id, source_name), None)

    # 一次 source drive；返回 False 时结束本 Session lane。
    async def _drive_source(self, session_id: str, source: Source, wake: _Wake) -> bool:
        current_drive = asyncio.current_task()
        if current_drive is None:
            raise RuntimeError("回复 follower 必须运行在 asyncio Task 中")
        state = _Iteration(drive_task=current_drive)
        admission_boundary = self._catalog.reader(session_id).head(source=source.name)
        restart_after_cleanup = False
        try:
            await self._start(state, session_id, source)
            if state.task is None and not state.stop_drive:
                restart_after_cleanup = (
                    self._restart_gate is not None and not self._restart_gate.accepting
                )
            elif state.task is not None:
                await self._await_task(state, session_id, source, wake, admission_boundary)
        except asyncio.CancelledError as error:
            if not state.consume_internal_cancel(error):
                state.save_caller_cancel(error)
                state.stop_drive = True
        except BaseException as error:
            state.pending_errors.append(error)
        finally:
            await self._release(state, session_id, source)
        await self._record_failure(state, admission_boundary)
        return await self._conclude(state, source, wake, restart_after_cleanup)

    # 1. 依次进入 reply 与 source scope，捕获后立即退出，再启动 Source Task。
    async def _start(self, state: _Iteration, session_id: str, source: Source) -> None:
        reply_scope_cm = await _enter_scope(self._ctx)
        if reply_scope_cm is None:
            state.stop_drive = True
            return
        try:
            state.reply_scope = self._ctx.capture_runtime_scope()
        finally:
            await reply_scope_cm.__aexit__(None, None, None)

        source_scope_cm = await _enter_scope(source.context)
        if source_scope_cm is None:
            state.stop_drive = True
            return
        try:
            state.source_scope = source.context.capture_runtime_scope()
            try:
                state.session = source.open(session_id)
            except TaskServiceClosed:
                state.stop_drive = True
                return
            state.add_monitor(state.reply_scope)
            state.add_monitor(state.source_scope)

            async def wrapped_program(*args: Any) -> object:
                return await state.run_program(self._program, *args)

            try:
                state.task = await state.session.start(wrapped_program)
            except TaskServiceClosed:
                state.stop_drive = True
            if state.task is not None:
                state.task.on_done(state.task_finished.set)
        finally:
            await source_scope_cm.__aexit__(None, None, None)

    # 2. 等 Task 完成；期间若日志已有终态且出现后续输入，则分离旧 Task。
    async def _await_task(
        self, state: _Iteration, session_id: str, source: Source,
        wake: _Wake, admission_boundary: int,
    ) -> None:
        assert state.task is not None
        joined = asyncio.create_task(state.wait_task())
        notified = asyncio.create_task(wake.event.wait())
        try:
            done, _ = await asyncio.wait((joined, notified), return_when=asyncio.FIRST_COMPLETED)
            if joined in done:
                await joined
            elif (
                self._still_current(wake, source)
                and await self._terminal_committed(session_id, source.name, admission_boundary)
                and self._still_current(wake, source)
            ):
                state.detached = True
                state.task.cancel()
            else:
                await joined
        finally:
            wake.event.clear()
            for waiter in (joined, notified):
                if not waiter.done():
                    waiter.cancel()
            await asyncio.gather(joined, notified, return_exceptions=True)
        wake.changed = True

    def _still_current(self, wake: _Wake, source: Source) -> bool:
        return wake.source is source and self._current_source(source.name) is source

    # 大尾部的终态/后续输入在同一只读 worker 中判定。
    async def _terminal_committed(self, session_id: str, source_name: str, boundary: int) -> bool:
        reader = self._catalog.reader(session_id)

        def read(reader: MessageReader) -> bool:
            terminal = False

            def consume(rows) -> bool:
                nonlocal terminal
                for message in rows:
                    if isinstance(message.body, Output) and message.body.finish != "continue" or isinstance(message.body, Control):
                        terminal = True
                    if terminal and isinstance(message.body, Input):
                        return True
                return False

            return reader.scan(consume, after_seq=boundary, source=source_name)

        return await reader.read_async(read)

    # 3. 停 monitor、归还 captured scope，再对 exact Source Task 做物理 join。
    async def _release(self, state: _Iteration, session_id: str, source: Source) -> None:
        for monitor in state.monitors:
            try:
                await state.stop_monitor(monitor)
            except asyncio.CancelledError as error:
                state.absorb_cancel(error)
            except BaseException as error:
                state.cleanup_errors.append(error)
        if state.reply_scope is not None and not state.reply_entered:
            await state.close_scope(state.reply_scope)
        if state.source_scope is not None:
            await state.close_scope(state.source_scope)
        # monitor 只保护 admission；captured scope 归还后，
        # exact Source Task 才进入独立的 physical join。
        if state.detached:
            self._create_owned(state.settle_task(), name=f"reply-settle:{session_id}:{source.name}")
            return
        try:
            await state.settle_task()
        except asyncio.CancelledError as error:
            state.absorb_cancel(error)
        except BaseException as error:
            state.cleanup_errors.append(error)

    # 4. 程序失败交给 Source 记录；记录成功即视为本 lane 已持久结算。
    async def _record_failure(self, state: _Iteration, admission_boundary: int) -> None:
        failure = next((error for error in state.pending_errors if isinstance(error, Exception)), None)
        if failure is None and isinstance(state.task_error, Exception):
            failure = state.task_error
        if failure is None:
            return
        if state.session is None:
            logger.warning("来源打开失败，封闭当前回复 lane", exc_info=failure)
            state.pending_errors.clear()
            state.stop_drive = True
            return
        boundary = state.task.boundary_hint if state.task is not None else admission_boundary
        if not isinstance(boundary, int):
            boundary = admission_boundary
        try:
            await state.session.record_failure(failure, boundary=boundary)
        except Exception as error:
            state.pending_errors = [RuntimeError(f"回复驱动没有持久进展 (no progress): {error}")]
        else:
            state.pending_errors.clear()
            state.stop_drive = True

    # 5. 决定 lane 是否继续：抛出 follower 失败，或按停止/重启门决定下一轮。
    async def _conclude(
        self, state: _Iteration, source: Source, wake: _Wake, restart_after_cleanup: bool,
    ) -> bool:
        follower_errors = state.follower_errors()
        if state.source_settlement_errors and not follower_errors:
            for error in state.source_settlement_errors:
                logger.warning("回复来源任务结算失败，停止当前 source drive", exc_info=error)
            state.stop_drive = True
            return False
        errors = [*follower_errors, *state.source_settlement_errors]
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("回复 owner 结算失败", errors)
        if state.stop_drive:
            return wake.source is not None and wake.source is not source
        if restart_after_cleanup:
            assert self._restart_gate is not None
            await self._restart_gate.wait_until_open()
            wake.changed = True
        return True


async def follow(
    ctx: Context, catalog: MessageCatalog,
    sources: Sources, program: Program, restart_gate: RestartGate | None = None,
) -> None:
    """从日志追赶可回复来源；空闲不保留 scope，不保存 cursor 或回复队列。"""
    await _Follower(ctx, catalog, sources, program, restart_gate).run()
