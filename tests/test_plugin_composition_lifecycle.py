from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import (
    Bail,
    CompositionError,
    CompositionRoot,
    EmitEventKey,
    Effect,
    FiberState,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
)
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from core.memory.events import MemoryWritten


_MEMORY_WRITTEN_EVENT = EmitEventKey[MemoryWritten]("test.memory.written")


@pytest.mark.asyncio
async def test_root_registers_real_fiber_and_reports_startup_state() -> None:
    """The mounted Fiber is an observable Root owner, not a source-text claim."""
    root = CompositionRoot("fiber-owner")
    started: list[str] = []

    async def plugin(ctx):
        started.append(ctx._fiber.name)  # pyright: ignore[reportPrivateUsage]

    try:
        await root.mount(plugin, name="owner")
        receipt = root.receipt()
        assert started == ["owner"]
        assert len(receipt.fibers) == 1
        assert receipt.fibers[0].state.value == "active"
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_failed_consumer_cleanup_keeps_provider_until_explicit_retry():
    """消费者关闭失败后，原资源及依赖仍由同一 Root 持有。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("cleanup-retry")
    service = ServiceKey[list[str]]("test.cleanup-resource")
    events: list[str] = []
    attempts = 0

    async def provider(ctx):
        await ctx.effect(lambda: lambda: events.append("provider-close"))
        await ctx.provide(service, events)

    async def consumer(ctx):
        resource = ctx.require(service)

        def close():
            nonlocal attempts
            attempts += 1
            resource.append("consumer-close")
            if attempts == 1:
                raise OSError("resource still open")

        await ctx.effect(lambda: close)

    await root.mount(provider, name="provider")
    await root.mount(consumer, name="consumer", inject=(service,))
    with pytest.raises(BaseExceptionGroup):
        await root.dispose()
    assert events == ["consumer-close"]

    await root.dispose()
    assert events == ["consumer-close", "consumer-close", "provider-close"]
    await root.dispose()
    assert events == ["consumer-close", "consumer-close", "provider-close"]


@pytest.mark.asyncio
async def test_effect_cleanup_cannot_wait_for_its_own_close():
    """错误的自等待明确失败并保留 owner，修正后可显式重试。"""
    from agent.plugin_composition.effect import Effect

    owners: list[Effect] = []
    attempts = 0

    async def cleanup():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await effect.aclose()

    effect = Effect(label="connection", remove_from_owner=owners.remove)
    owners.append(effect)
    await effect.start(lambda: cleanup)
    with pytest.raises(CompositionError, match="自身关闭"):
        await effect.aclose()
    assert owners == [effect]
    await effect.aclose()
    assert owners == []
    assert attempts == 2


@pytest.mark.asyncio
async def test_effect_close_joins_concurrent_callers_despite_repeated_cancel():
    """多个关闭调用和重复取消只执行一次实际关闭。"""
    from agent.plugin_composition.effect import Effect

    entered, release = asyncio.Event(), asyncio.Event()
    closed: list[str] = []
    owners: list[Effect] = []

    async def cleanup():
        entered.set()
        await release.wait()
        closed.append("closed")

    effect = Effect(label="connection", remove_from_owner=owners.remove)
    owners.append(effect)
    await effect.start(lambda: cleanup)
    first = asyncio.create_task(effect.aclose())
    await entered.wait()
    second = asyncio.create_task(effect.aclose())
    first.cancel()
    # 事件循环回调确定第二次取消在首次取消投递之后发生。
    asyncio.get_running_loop().call_soon(first.cancel)
    asyncio.get_running_loop().call_soon(release.set)
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert closed == ["closed"]
    assert owners == []


@pytest.mark.asyncio
async def test_effect_close_guard_runs_before_create_and_join() -> None:
    """Core close guards run on each caller before one real cleanup."""

    entered, release = asyncio.Event(), asyncio.Event()
    second_guard = asyncio.Event()
    owners: list[Effect] = []
    guard_tasks: list[asyncio.Task[object] | None] = []
    cleanup_tasks: list[asyncio.Task[object] | None] = []
    first: asyncio.Task[None] | None = None
    second: asyncio.Task[None] | None = None

    def guard() -> None:
        guard_tasks.append(asyncio.current_task())
        if len(guard_tasks) == 2:
            second_guard.set()

    async def cleanup() -> None:
        cleanup_tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()

    effect = Effect(
        label="guarded-connection",
        remove_from_owner=owners.remove,
        close_guard=guard,
    )
    owners.append(effect)
    try:
        await effect.start(lambda: cleanup)
        first = asyncio.create_task(effect.aclose())
        await entered.wait()
        second = asyncio.create_task(effect.aclose())
        await second_guard.wait()
        assert guard_tasks == [first, second]
        assert cleanup_tasks[0] not in {first, second}
        assert len(cleanup_tasks) == 1
        assert not second.done()
        cleanup_task = cleanup_tasks[0]
        assert cleanup_task is not None
        assert not cleanup_task.done()
        assert owners == [effect]
        first.cancel()
        loop = asyncio.get_running_loop()
        first_cancel_delivered = asyncio.Event()
        loop.call_soon(first.cancel)
        loop.call_soon(first_cancel_delivered.set)
        await first_cancel_delivered.wait()
        second_cancel_delivered = asyncio.Event()
        loop.call_soon(second_cancel_delivered.set)
        await second_cancel_delivered.wait()
        assert first.cancelling() >= 2
        assert not first.done()
        assert not second.done()
        assert not release.is_set()
        assert cleanup_tasks == [cleanup_task]
        assert not cleanup_task.done()
        assert owners == [effect]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert cleanup_tasks == [cleanup_task]
        assert owners == []
    finally:
        release.set()
        tasks = [task for task in (first, second) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_mount_and_cleanup_failure_keep_both_errors_and_owner():
    """挂载失败不隐藏清理失败，也不允许重用未释放的名称。"""
    root = CompositionRoot("mount-cleanup")
    attempts = 0

    async def plugin(ctx):
        def close():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("still open")

        await ctx.effect(lambda: close)
        raise ValueError("initialization failed")
    with pytest.raises(BaseExceptionGroup) as caught:
        await root.mount(plugin, name="resource")
    assert isinstance(caught.value.exceptions[0], ValueError)
    assert isinstance(caught.value.exceptions[1], OSError)
    with pytest.raises(CompositionError, match="重复挂载"):
        await root.mount(plugin, name="resource")
    await root.dispose()
    assert attempts == 2
    assert root.receipt().fibers == ()


@pytest.mark.asyncio
async def test_runtime_lifecycle_bail_fails_loud(tmp_path) -> None:
    calls: list[str] = []
    root = CompositionRoot("runtime-lifecycle-bail")

    async def first(ctx) -> None:
        await ctx.on(
            RUNTIME_STARTED,
            lambda _: (calls.append("bail"), Bail("blocked"))[1],
        )

    async def second(ctx) -> None:
        await ctx.on(RUNTIME_STARTED, lambda _: calls.append("second"))

    failed = await root.mount(first, name="bailing-plugin")
    await root.mount(second, name="later-plugin")
    assert failed.state == FiberState.FAILED
    assert isinstance(failed.error, CompositionError)
    assert failed.error.code == "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED"
    assert calls == ["bail", "second"]
    assert not root.receipt().ready
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_stop_failure_remains_retryable(tmp_path) -> None:
    stop_calls: list[str] = []
    root = CompositionRoot("runtime-stop-retry")

    async def plugin(ctx) -> None:
        async def stop(_event: object) -> None:
            stop_calls.append("stop")
            if len(stop_calls) == 1:
                raise RuntimeError("fixture stop failure")

        await ctx.on(RUNTIME_STOPPING, stop)

    await root.mount(plugin, name="retrying-plugin")
    with pytest.raises(ExceptionGroup, match="Root 子作用域关闭失败") as caught:
        await root.dispose()
    assert isinstance(caught.value.exceptions[0], RuntimeError)
    assert str(caught.value.exceptions[0]) == "fixture stop failure"
    await root.dispose()
    await root.dispose()

    assert stop_calls == ["stop", "stop"]


@pytest.mark.asyncio
async def test_emit_event_listener_failure_is_fail_loud() -> None:
    root = CompositionRoot("emit-event-failure")

    def fail(_: object) -> None:
        raise RuntimeError("observe failed")

    async def plugin(ctx) -> None:
        await ctx.on(EmitEventKey[object]("test.emit.failure"), fail)

    await root.mount(plugin, name="failing-emit-plugin")
    try:
        with pytest.raises(RuntimeError, match="observe failed"):
            root.context.emit(EmitEventKey[object]("test.emit.failure"), object())
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_event_bus_does_not_bridge_into_plugin_composition() -> None:
    observed: list[MemoryWritten] = []
    root = CompositionRoot("event-bus-is-core-only")

    async def plugin(ctx) -> None:
        await ctx.on(_MEMORY_WRITTEN_EVENT, observed.append)

    await root.mount(plugin, name="composition-observer")
    bus = EventBus()
    try:
        await bus.fanout(_memory_written_event())
    finally:
        try:
            await bus.aclose()
        finally:
            await root.dispose()

    assert observed == []


@pytest.mark.asyncio
async def test_event_bus_rejects_inherited_wrong_task_binding() -> None:
    """Observer task cannot inherit a Channel lease from the caller task."""
    from types import SimpleNamespace

    from agent.plugin_composition.channels import (
        bind_channel_turn_binding,
        get_current_channel_turn_binding,
        reset_channel_turn_binding,
    )

    bus = EventBus()
    observed: list[object | None] = []
    bus.on(int, lambda _: observed.append(get_current_channel_turn_binding()))
    lease = SimpleNamespace(active=True)
    token = bind_channel_turn_binding(lease)
    try:
        assert get_current_channel_turn_binding() is lease
        await asyncio.create_task(bus.fanout(1))
    finally:
        reset_channel_turn_binding(token)
        await bus.aclose()
    assert observed == [None]


@pytest.mark.asyncio
async def test_event_bus_emits_in_order_and_propagates_handler_errors() -> None:
    bus = EventBus()
    seen: list[tuple[str, int]] = []

    async def increment(event: int) -> int:
        seen.append(("increment", event))
        return event + 1

    def double(event: int) -> int:
        seen.append(("double", event))
        return event * 2

    def fail(_: str) -> None:
        raise RuntimeError("emit failed")

    bus.on(int, increment)
    bus.on(int, double)
    bus.on(str, fail)
    try:
        assert await bus.emit(1) == 4
        assert seen == [("increment", 1), ("double", 2)]
        with pytest.raises(RuntimeError, match="emit failed"):
            await bus.emit("bad")
    finally:
        await bus.aclose()


@pytest.mark.asyncio
async def test_event_bus_observers_isolate_errors_and_cancellation(caplog) -> None:
    bus = EventBus()
    observed: list[str] = []

    async def bad(_: str) -> None:
        raise RuntimeError("observer failed")

    async def good(event: str) -> None:
        observed.append(event)

    self_cancelled = asyncio.Event()
    self_cancel_wait = asyncio.Event()

    async def self_cancel(_: bytes) -> None:
        self_cancelled.set()
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await self_cancel_wait.wait()

    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def wait_for_caller_cancel(_: float) -> None:
        entered.set()
        try:
            await release.wait()
        finally:
            finished.set()

    bus.on(str, bad)
    bus.on(str, good)
    bus.on(bytes, self_cancel)
    bus.on(float, wait_for_caller_cancel)
    caller: asyncio.Task[None] | None = None
    try:
        with caplog.at_level("ERROR", logger="bus.event_bus"):
            await bus.observe("observe")
            await bus.fanout("fanout")
        assert observed == ["observe", "fanout"]
        assert any("observer error" in record.message for record in caplog.records)

        await bus.observe(b"self-cancel")
        assert self_cancelled.is_set()

        caller = asyncio.create_task(bus.observe(1.0))
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert finished.is_set()
    finally:
        release.set()
        if caller is not None and not caller.done():
            caller.cancel()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        await bus.aclose()


@pytest.mark.asyncio
async def test_event_bus_enqueue_drains_callbacks_before_close() -> None:
    bus = EventBus()
    started = {1: asyncio.Event(), 2: asyncio.Event()}
    release = {1: asyncio.Event(), 2: asyncio.Event()}
    finished = {1: asyncio.Event(), 2: asyncio.Event()}
    seen: list[int] = []

    async def callback(event: int) -> None:
        seen.append(event)
        started[event].set()
        try:
            await release[event].wait()
        finally:
            finished[event].set()

    bus.on(int, callback)
    drain_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None
    closed = False
    try:
        bus.enqueue(1)
        drain_task = asyncio.create_task(bus.drain())
        await started[1].wait()
        assert not drain_task.done()
        release[1].set()
        await drain_task
        drain_task = None
        assert finished[1].is_set()

        bus.enqueue(2)
        close_task = asyncio.create_task(bus.aclose())
        await started[2].wait()
        assert not close_task.done()
        release[2].set()
        await close_task
        close_task = None
        closed = True
        assert finished[2].is_set()

        bus.enqueue(3)
        assert bus._observe_queue is not None
        assert bus._observe_queue.empty()
        await bus.drain()
        assert seen == [1, 2]
    finally:
        release[1].set()
        release[2].set()
        if drain_task is not None:
            drain_task.cancel()
            await asyncio.gather(drain_task, return_exceptions=True)
        if close_task is not None:
            close_task.cancel()
            await asyncio.gather(close_task, return_exceptions=True)
        if not closed:
            await bus.aclose()




def _memory_written_event() -> MemoryWritten:
    return MemoryWritten(
        session_key="session",
        channel="test",
        chat_id="chat",
        action="supersede",
        source_ref="session@post_response",
        superseded_ids=["memory-1"],
    )


def _root_failure_manager(tmp_path, *, fail_mount=False):
    """建立真实插件，其连接首次关闭失败且需要原模块和数据才能重试。"""
    source = tmp_path / "plugins" / "root_owner"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text(
        f'''
api_version = 3
name = "root_owner"
version = "1.0.0"
attempts = 0
entered = None
release = None
async def apply(ctx):
    marker = ctx.runtime.data_dir / "connection-owner"
    marker.write_text("open")
    async def cleanup():
        global attempts
        attempts += 1
        assert marker.read_text() == "open"
        if entered is not None:
            entered.set()
            await release.wait()
        if attempts == 1:
            raise OSError("connection still open")
        marker.write_text("closed")
    await ctx.effect(lambda: cleanup)
    if {fail_mount!r}:
        raise ValueError("apply failed after acquisition")
''', encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_plugin_workspace(workspace)
    return PluginManager(
        [source.parent], event_bus=EventBus(), workspace=workspace,
        installed_cache_root=tmp_path / "home",
    )


def _error_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _error_leaves(child)]
    return [error]


@pytest.mark.asyncio
async def test_failed_root_build_retries_cleanup_and_keeps_both_errors(
    tmp_path,
):
    """首次 live Root 装配失败后清理重试成功仍保留两个原始错误。"""
    manager = _root_failure_manager(tmp_path, fail_mount=True)
    with pytest.raises(BaseExceptionGroup) as caught:
        await manager.load_all()
    leaves = _error_leaves(caught.value)
    assert any(isinstance(error, ValueError) for error in leaves)
    assert any(isinstance(error, OSError) for error in leaves)
    assert manager.generation("root_owner") is None
    assert manager.live_root is None
    assert (tmp_path / "workspace/plugin-data/root_owner-builtin/connection-owner").read_text() == "closed"
    assert manager._building_roots == {}
    assert manager._draining_generations == {}
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_terminate_joins_untransferred_root_without_generations(tmp_path, monkeypatch, fail_close):
    """空组合也保留真实 Root；并发关闭和重复取消只尝试一次。"""
    manager = PluginManager(
        [], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home",
    )
    entered, release = asyncio.Event(), asyncio.Event()
    attempts = 0

    async def provide(root, generations, **kwargs):
        async def cleanup():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("still open")
            entered.set()
            await release.wait()
            if fail_close and attempts == 2:
                raise OSError("still open on explicit retry")
        await root.context.effect(lambda: cleanup)
        raise ValueError("service initialization failed")

    monkeypatch.setattr(manager, "_provide_composition_services", provide)
    with pytest.raises(BaseExceptionGroup):
        initialize_plugin_workspace(tmp_path / "workspace")
        await manager.load_all()
    assert attempts == 1
    assert len(manager._building_roots) == 1
    first = asyncio.create_task(manager.terminate_all())
    await entered.wait()
    second = asyncio.create_task(manager.terminate_all())
    first.cancel()
    asyncio.get_running_loop().call_soon(first.cancel)
    asyncio.get_running_loop().call_soon(release.set)
    if fail_close:
        with pytest.raises(asyncio.CancelledError):
            await first
        # 第二观察者直接收到实际关闭失败；已取消的第一观察者保留取消结果。
        with pytest.raises(OSError, match="still open on explicit retry"):
            await second
        assert attempts == 2
        assert len(manager._building_roots) == 1
        await manager.terminate_all()
        assert attempts == 3
    else:
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert attempts == 2
    assert manager._building_roots == {}




@pytest.mark.asyncio
async def test_pending_initial_dependency_resolves_then_frozen_teardown_closes_consumers_first():
    """初始待定依赖可激活；退出顺序按实际依赖而非挂载顺序。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("initial-resolution")
    service = ServiceKey[list[str]]("test.initial.service")
    events = []

    async def consumer(ctx):
        shared = ctx.require(service)
        events.append("consumer-start")
        await ctx.effect(lambda: lambda: shared.append("consumer-close"))

    async def provider(ctx):
        await ctx.effect(lambda: lambda: events.append("provider-close"))
        await ctx.provide(service, events)

    await root.mount(consumer, name="consumer", inject=(service,))
    assert events == []
    await root.mount(provider, name="provider")
    assert events == ["consumer-start"]
    await root.dispose()
    assert events == ["consumer-start", "consumer-close", "provider-close"]
