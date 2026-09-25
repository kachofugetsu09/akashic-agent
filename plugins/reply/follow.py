from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from agent.plugin_composition import Context, RuntimeScope
from agent.plugin_composition.messages import MessageCatalog, MessageReader
from agent.plugin_composition.model import CompositionError
from agent.plugin_composition.tasks import RestartGate, Task, TaskServiceClosed
from agent.plugin_contracts import Control, Input, Output
from agent.plugin_contracts.sources import (
    Source as Source,
    Sources as Sources,
    SourceSession as SourceSession,
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


async def follow(
    ctx: Context, catalog: MessageCatalog,
    sources: Sources, program: Program, restart_gate: RestartGate | None = None,
) -> None:
    """从日志追赶可回复来源；空闲不保留 scope，不保存 cursor 或回复队列。"""
    active: dict[tuple[str, str], _Wake] = {}

    async def enter_scope(context: Context):
        """Enter one owner scope and report only local admission loss."""
        scope = context.runtime_scope()
        try:
            await scope.__aenter__()
        except CompositionError as error:
            if error.code in {"OWNER_UNAVAILABLE", "STALE_ACTIVATION"}:
                logger.warning("回复 owner scope 不可用，结束当前 source drive: %s", error.code)
                return None
            raise
        return scope

    def consume_internal_cancel(state: _Iteration, error: asyncio.CancelledError) -> bool:
        """消费 follower 自己发出的取消，并保持外部取消可观察。"""
        if state.monitor_cancel in error.args or state.monitor_failure_cancel in error.args:
            current = asyncio.current_task()
            if current is not None and current.uncancel():
                # 同一个 await 可能只交付 marker；剩余计数只证明还有外部取消
                # 事实，不足以恢复一个 Python 从未单独交付的原始消息对象。
                state.coalesced_caller_cancel = True
            state.stop_drive = True
            return True
        return False

    def save_caller_cancel(state: _Iteration, error: asyncio.CancelledError) -> None:
        """保存真实 caller CancelledError，并清掉本次取消以继续结算。"""
        state.coalesced_caller_cancel = False
        state.caller_cancellations.append(error)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            current.uncancel()

    async def wait_event(
        state: _Iteration, event: asyncio.Event, *, tolerate_caller_cancel: bool,
    ) -> None:
        """等待一次物理完成事件；清理阶段每轮使用新的 await。"""
        while not event.is_set():
            try:
                await event.wait()
            except asyncio.CancelledError as error:
                if not tolerate_caller_cancel:
                    raise
                if not consume_internal_cancel(state, error):
                    save_caller_cancel(state, error)

    async def retrieve_task(state: _Iteration, *, tolerate_caller_cancel: bool) -> None:
        """在 on_done 后调用公开 join，区分 caller 与 Source Task 终态。"""
        task = state.task
        assert task is not None
        if state.task_joined:
            return
        while not state.task_joined:
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
                    if not consume_internal_cancel(state, error):
                        save_caller_cancel(state, error)
                    continue
                state.task_error = error
                break
            except BaseException as error:
                state.task_error = error
                break
            else:
                break
        state.task_joined = True

    async def wait_task(state: _Iteration) -> None:
        """等待 exact Source Task 完成，再读取其真实终态。"""
        await wait_event(state, state.task_finished, tolerate_caller_cancel=False)
        await retrieve_task(state, tolerate_caller_cancel=False)
        if state.task_error is None:
            return
        if not isinstance(state.task_error, asyncio.CancelledError):
            logger.warning("回复程序失败，保留日志等待新输入或控制", exc_info=state.task_error)
        elif state.task_error.__cause__ is not None:
            state.source_settlement_errors.append(state.task_error)

    async def settle_task(state: _Iteration) -> None:
        """最多取消一次 exact Task，再等待 on_done 与公开 join。"""
        task = state.task
        if task is None or state.task_joined:
            return
        if task.active and not state.task_cancel_requested:
            state.task_cancel_requested = True
            try:
                task.cancel()
            except BaseException as error:
                state.source_settlement_errors.append(error)
        await wait_event(state, state.task_finished, tolerate_caller_cancel=True)
        await retrieve_task(state, tolerate_caller_cancel=True)
        if state.task_error is not None:
            if not isinstance(state.task_error, asyncio.CancelledError):
                state.source_settlement_errors.append(state.task_error)
            elif (
                not state.task_cancel_requested
                or state.task_error.__cause__ is not None
            ):
                # 纯粹由 owner 发出的取消仍是正常结算；带真实 cleanup
                # cause 的 child CE 是 Source Task 的终态，必须原样保留。
                state.source_settlement_errors.append(state.task_error)

    async def stop_monitor(state: _Iteration, monitor: _Monitor) -> None:
        """停止一个 raw admission monitor，并取得其真实结果。"""
        if not monitor.task.done():
            monitor.task.cancel()
        await wait_event(state, monitor.finished, tolerate_caller_cancel=True)
        try:
            monitor.task.result()
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            state.monitor_errors.append(error)

    async def close_scope(state: _Iteration, scope: object) -> None:
        """关闭 captured scope，同时保留清理期间的 caller 取消。"""
        try:
            await scope.close()
        except asyncio.CancelledError as error:
            if not consume_internal_cancel(state, error):
                save_caller_cancel(state, error)
        except BaseException as error:
            state.cleanup_errors.append(error)

    def current_source(name: str) -> Source | None:
        """Read the registration authority instead of the notification snapshot."""
        return next((item for item in sources.entries() if item.name == name), None)

    async def drive(session_id: str, source_name: str, wake: _Wake) -> None:
        """每个 Session 独立排空旧工作；并发通知只要求再次读取日志。"""
        try:
            while wake.changed:
                wake.changed = False
                source = wake.source
                if source is None:
                    break
                registered = current_source(source.name)
                if registered is not source:
                    wake.source = registered
                    if registered is None:
                        break
                    wake.changed = True
                    continue
                state = _Iteration()
                session: SourceSession | None = None
                admission_boundary = catalog.reader(session_id).head(source=source.name)

                def terminal_committed() -> bool:
                    """A later Input may use a saved terminal while old cleanup runs."""
                    terminal = False
                    for message in catalog.reader(session_id).snapshot(after_seq=admission_boundary):
                        if message.source != source.name:
                            continue
                        if isinstance(message.body, Output) and message.body.finish != "continue":
                            terminal = True
                        if isinstance(message.body, Control):
                            terminal = True
                        if terminal and isinstance(message.body, Input):
                            return True
                    return False
                restart_after_cleanup = False
                current_drive = asyncio.current_task()
                if current_drive is None:
                    raise RuntimeError("回复 follower 必须运行在 asyncio Task 中")

                def request_admission_stop(_owner: str) -> None:
                    if state.admission_cancel_requested:
                        return
                    state.admission_cancel_requested = True
                    state.stop_drive = True
                    current_drive.cancel(state.monitor_cancel)

                async def watch(scope: object, owner: str) -> None:
                    try:
                        await scope.wait_admission_closed()
                    except asyncio.CancelledError:
                        raise
                    except BaseException:
                        if not state.admission_cancel_requested:
                            state.admission_cancel_requested = True
                            state.stop_drive = True
                            current_drive.cancel(state.monitor_failure_cancel)
                        raise
                    request_admission_stop(owner)

                def create_monitor(scope: object, owner: str) -> _Monitor:
                    coroutine = watch(scope, owner)
                    try:
                        task = asyncio.create_task(
                            coroutine, context=contextvars.Context()
                        )
                    except BaseException:
                        coroutine.close()
                        raise
                    finished = asyncio.Event()
                    task.add_done_callback(lambda _task: finished.set())
                    return _Monitor(task, finished)

                async def wrapped_program(*args):
                    if state.reply_scope is None:
                        raise RuntimeError("Reply scope 尚未准备")
                    async with state.reply_scope:
                        state.reply_entered = True
                        return await program(*args)

                try:
                    reply_scope_cm = await enter_scope(ctx)
                    if reply_scope_cm is None:
                        state.stop_drive = True
                    else:
                        try:
                            state.reply_scope = ctx.capture_runtime_scope()
                        finally:
                            await reply_scope_cm.__aexit__(None, None, None)

                    if not state.stop_drive:
                        source_scope_cm = await enter_scope(source.context)
                        if source_scope_cm is None:
                            state.stop_drive = True
                        else:
                            try:
                                state.source_scope = source.context.capture_runtime_scope()
                                try:
                                    session = source.open(session_id)
                                except TaskServiceClosed:
                                    state.stop_drive = True
                                else:
                                    # Monitor 只等待已捕获的 admission；不继承 caller Context。
                                    state.monitors.append(
                                        create_monitor(state.reply_scope, "reply")
                                    )
                                    state.monitors.append(
                                        create_monitor(state.source_scope, "source")
                                    )
                                    try:
                                        state.task = await session.start(wrapped_program)
                                    except TaskServiceClosed:
                                        state.stop_drive = True
                                    if state.task is not None:
                                        state.task.on_done(state.task_finished.set)
                            finally:
                                await source_scope_cm.__aexit__(None, None, None)

                    if state.task is None and not state.stop_drive:
                        restart_after_cleanup = (
                            restart_gate is not None and not restart_gate.accepting
                        )
                    elif state.task is not None:
                        joined = asyncio.create_task(wait_task(state))
                        notified = asyncio.create_task(wake.event.wait())
                        try:
                            done, _ = await asyncio.wait(
                                (joined, notified), return_when=asyncio.FIRST_COMPLETED,
                            )
                            if joined in done:
                                await joined
                            elif wake.source is source and current_source(source.name) is source and terminal_committed():
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
                except asyncio.CancelledError as error:
                    if not consume_internal_cancel(state, error):
                        save_caller_cancel(state, error)
                        state.stop_drive = True
                except BaseException as error:
                    state.pending_errors.append(error)
                finally:
                    for monitor in state.monitors:
                        try:
                            await stop_monitor(state, monitor)
                        except asyncio.CancelledError as error:
                            if not consume_internal_cancel(state, error):
                                save_caller_cancel(state, error)
                        except BaseException as error:
                            state.cleanup_errors.append(error)
                    if state.reply_scope is not None and not state.reply_entered:
                        await close_scope(state, state.reply_scope)
                    if state.source_scope is not None:
                        await close_scope(state, state.source_scope)
                    # monitor 只保护 admission；captured scope 归还后，
                    # exact Source Task 才进入独立的 physical join。
                    if state.detached:
                        create_owned(settle_task(state), name=f"reply-settle:{session_id}:{source.name}")
                    else:
                        try:
                            await settle_task(state)
                        except asyncio.CancelledError as error:
                            if not consume_internal_cancel(state, error):
                                save_caller_cancel(state, error)
                        except BaseException as error:
                            state.cleanup_errors.append(error)

                failure = next((error for error in state.pending_errors if isinstance(error, Exception)), None)
                if failure is None and isinstance(state.task_error, Exception):
                    failure = state.task_error
                if failure is not None:
                    reader = catalog.reader(session_id)
                    if session is None:
                        logger.warning("来源打开失败，封闭当前回复 lane", exc_info=failure)
                        state.pending_errors.clear()
                        state.stop_drive = True
                    else:
                        boundary = state.task.boundary_hint if state.task is not None else admission_boundary
                        if not isinstance(boundary, int):
                            boundary = admission_boundary
                        try:
                            await session.record_failure(failure, boundary=boundary)
                        except Exception as error:
                            state.pending_errors = [RuntimeError(f"回复驱动没有持久进展 (no progress): {error}")]
                        else:
                            state.pending_errors.clear()
                            state.stop_drive = True

                follower_errors: list[BaseException] = [
                    *state.pending_errors,
                    *state.monitor_errors,
                    *state.cleanup_errors,
                    *state.caller_cancellations,
                ]
                if state.coalesced_caller_cancel:
                    follower_errors.append(
                        asyncio.CancelledError(
                            "caller cancellation coalesced with admission stop"
                        )
                    )

                if state.source_settlement_errors and not follower_errors:
                    for error in state.source_settlement_errors:
                        logger.warning(
                            "回复来源任务结算失败，停止当前 source drive",
                            exc_info=error,
                        )
                    state.stop_drive = True
                    break

                errors = [*follower_errors, *state.source_settlement_errors]
                if errors:
                    if len(errors) == 1:
                        raise errors[0]
                    raise BaseExceptionGroup("回复 owner 结算失败", errors)
                if state.stop_drive:
                    if wake.source is not None and wake.source is not source:
                        continue
                    break
                if restart_after_cleanup:
                    assert restart_gate is not None
                    await restart_gate.wait_until_open()
                    wake.changed = True
        finally:
            active.pop((session_id, source_name), None)

    # 两个监听器只更新本地快照并置位；TaskGroup 仍是唯一业务 drive owner。
    latest_heads: dict[str, int] | None = None
    catalog_closed = False
    sources_closed = False
    wake_event = asyncio.Event()

    async def watch_catalog() -> None:
        nonlocal latest_heads, catalog_closed
        async for heads in catalog.follow():
            latest_heads = dict(heads)
            wake_event.set()
        catalog_closed = True
        wake_event.set()

    async def watch_sources() -> None:
        nonlocal sources_closed
        async for _entries in sources.changes():
            wake_event.set()
        sources_closed = True
        wake_event.set()

    previous_heads: dict[str, int] = {}
    previous_sources: dict[str, Source] = {}
    observed_source_heads: dict[tuple[str, str], int] = {}
    async with asyncio.TaskGroup() as group:
        def create_owned(coroutine, *, name):
            """Create one TaskGroup child and close its coroutine on handoff failure."""
            try:
                return group.create_task(coroutine, name=name)
            except BaseException:
                coroutine.close()
                raise

        catalog_task = create_owned(watch_catalog(), name="reply-catalog-watch")
        source_task = create_owned(watch_sources(), name="reply-source-watch")
        drive_tasks: set[asyncio.Task[None]] = set()

        def schedule(session_id: str, source: Source) -> None:
            key = (session_id, source.name)
            wake = active.get(key)
            if wake is None:
                wake = _Wake(source=source)
                active[key] = wake
                try:
                    task = create_owned(
                        drive(session_id, source.name, wake),
                        name=f"reply-drive:{session_id}:{source.name}",
                    )
                except BaseException:
                    if active.get(key) is wake:
                        del active[key]
                    raise
                drive_tasks.add(task)
                task.add_done_callback(drive_tasks.discard)
            else:
                wake.source = source
                wake.changed = True
                wake.event.set()

        while True:
            await wake_event.wait()
            wake_event.clear()
            if catalog_closed or sources_closed:
                break
            if latest_heads is None:
                continue

            heads = latest_heads
            current_sources = {
                source.name: source for source in sources.entries()
            }
            source_names = set(previous_sources) | set(current_sources)
            changed_source_names = {
                name for name in source_names
                if previous_sources.get(name) is not current_sources.get(name)
            }
            current_identity_changes = {
                name for name in changed_source_names if name in current_sources
            }

            # 1. A revoked source only updates an existing local handoff; its
            # exact Source owner still stops the in-flight Task via the scope monitor.
            for (session_id, source_name), wake in tuple(active.items()):
                if source_name not in changed_source_names:
                    continue
                wake.source = current_sources.get(source_name)
                wake.changed = True

            changed_sessions = {
                session_id for session_id, head in heads.items()
                if previous_heads.get(session_id) != head
            }
            previous_heads = dict(heads)
            previous_sources = current_sources

            # 2. One scan handles changed sessions and genuine identity changes.
            session_ids = (
                heads
                if current_identity_changes
                else (session_id for session_id in heads if session_id in changed_sessions)
            )
            for session_id in session_ids:
                reader = catalog.reader(session_id)
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
                    if identity_changed or observed_source_heads.get(key) != source_head:
                        observed_source_heads[key] = source_head
                        schedule(session_id, source)

        for task in (catalog_task, source_task, *drive_tasks):
            if not task.done():
                task.cancel()
