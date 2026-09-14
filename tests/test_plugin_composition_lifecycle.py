from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

import pytest

from agent.plugin_composition import (
    Bail,
    CompositionError,
    CompositionRoot,
    EmitEventKey,
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from bus.event_bus import EventBus
from core.memory.events import MemoryWritten


_MEMORY_WRITTEN_EVENT = EmitEventKey[MemoryWritten]("test.memory.written")


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
async def test_restart_finishes_failed_cleanup_before_acquiring_again():
    """显式重试不能覆盖上次尚未关闭的资源。"""
    root = CompositionRoot("restart-cleanup")
    events = []
    attempts = 0

    async def plugin(ctx):
        events.append("open")

        def close():
            nonlocal attempts
            attempts += 1
            events.append("close")
            if attempts == 1:
                raise OSError("still open")

        await ctx.effect(lambda: close)

    fiber = await root.mount(plugin, name="resource")
    with pytest.raises(OSError, match="still open"):
        await fiber.restart()
    assert events == ["open", "close"]
    await fiber.restart()
    assert events == ["open", "close", "close", "open"]
    await root.dispose()


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


@asynccontextmanager
async def _bound_root(root: CompositionRoot) -> AsyncIterator[None]:
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        yield
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("prepare_failure", [False, True])
async def test_stable_root_rebuild_prepares_before_resume_and_cleans_partial_failure(tmp_path, prepare_failure):
    """真实重新编译路径在关闭的当前快照中恢复活动，失败不开放接纳。"""
    from agent.plugin_composition import ServiceKey
    from session.log import MessageLog

    source = tmp_path / "plugins" / "probe"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text('''
from agent.plugin_composition import RUNTIME_STARTING, RUNTIME_STARTED, RUNTIME_STOPPING, ServiceKey
from agent.plugin_composition.tasks import TASKS
from agent.plugins.snapshot import get_current_runtime_snapshot
api_version = 3
name = "probe"
version = "1.0.0"
workspace_files = ("fail-prepare",)
async def apply(ctx):
    events, held = [], []
    def prepare(_):
        snapshot = get_current_runtime_snapshot()
        assert snapshot.composition_root.instance_token is ctx.root_instance_token
        assert not snapshot.accepting_leases
        hold = ctx.require(TASKS).open(ctx).activity("resource")
        hold.__enter__()
        held.append(hold)
        events.append("prepare")
        if ctx.workspace_file("fail-prepare").exists():
            raise ValueError("rebuild prepare failed")
    def stop(_):
        events.append("stop")
        for hold in held:
            hold.__exit__(None, None, None)
        held.clear()
    await ctx.provide(ServiceKey("probe.events"), events)
    await ctx.on(RUNTIME_STARTING, prepare)
    await ctx.on(RUNTIME_STARTED, lambda _: events.append("start"))
    await ctx.on(RUNTIME_STOPPING, stop)
''')
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log = MessageLog(workspace / "sessions.db")
    manager = PluginManager([source.parent], event_bus=EventBus(), workspace=workspace,
                            installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await manager.load_all()
        await manager.start_runtime()
        snapshot = manager.snapshot_store.pause_admission()
        assert snapshot is not None
        await manager.snapshot_store.wait_for_no_leases(snapshot)
        old_root = snapshot.composition_root
        await manager._stop_runtime_snapshot(snapshot)
        stable = next(iter(manager._active_generations.values()))
        if prepare_failure:
            (workspace / "fail-prepare").touch()
            with pytest.raises(ValueError, match="rebuild prepare failed"):
                await manager._rebuild_stable_root(stable, snapshot)
        else:
            await manager._rebuild_stable_root(stable, snapshot)
        assert snapshot.composition_root is not old_root
        assert not snapshot.accepting_leases and snapshot.lease_count == 0
        events = snapshot.composition_root.context.require(ServiceKey("probe.events"))
        if prepare_failure:
            assert events == ["prepare", "stop"]
            assert snapshot.composition_root.instance_token not in manager._runtime_starting_roots
            (workspace / "fail-prepare").unlink()
            await manager._rebuild_stable_root(stable, snapshot)
            events = snapshot.composition_root.context.require(ServiceKey("probe.events"))
        assert events == ["prepare"]
        await manager.snapshot_store.resume(snapshot)
        await manager.start_runtime()
        assert events == ["prepare", "start"]
    finally:
        async with asyncio.timeout(3):
            await manager.terminate_all()
        log.close()


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

    await root.mount(first, name="bailing-plugin")
    await root.mount(second, name="later-plugin")
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    await manager._publish_committed_snapshot(snapshot)

    with pytest.raises(CompositionError) as caught:
        await cast(Any, manager)._start_runtime_snapshot(snapshot)

    assert caught.value.code == "RUNTIME_LIFECYCLE_BAIL_NOT_ALLOWED"
    assert calls == ["bail"]
    await manager.snapshot_store.close()
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
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    await manager._publish_committed_snapshot(snapshot)
    await cast(Any, manager)._start_runtime_snapshot(snapshot)

    with pytest.raises(RuntimeError, match="fixture stop failure"):
        await cast(Any, manager)._stop_runtime_snapshot(snapshot)
    await cast(Any, manager)._stop_runtime_snapshot(snapshot)
    await cast(Any, manager)._stop_runtime_snapshot(snapshot)

    assert stop_calls == ["stop", "stop"]
    await manager.snapshot_store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_start_ignores_snapshot_replaced_before_start(
    tmp_path,
) -> None:
    """A retired Root must not start after publication replaces it."""

    calls: list[str] = []
    old_root = CompositionRoot("runtime-start-old")
    new_root = CompositionRoot("runtime-start-new")

    async def old_plugin(ctx) -> None:
        async def start(_event: object) -> None:
            async with ctx.runtime_scope():
                calls.append("old")

        await ctx.on(RUNTIME_STARTED, start)

    async def new_plugin(ctx) -> None:
        await ctx.on(RUNTIME_STARTED, lambda _: calls.append("new"))

    await old_root.mount(old_plugin, name="old-plugin")
    await new_root.mount(new_plugin, name="new-plugin")
    compiler = RuntimeSnapshotCompiler()
    old_snapshot = compiler.compile({}, composition_root=old_root)
    new_snapshot = compiler.compile({}, composition_root=new_root)
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    await manager._publish_committed_snapshot(old_snapshot)
    await manager._publish_committed_snapshot(new_snapshot)

    await cast(Any, manager)._start_runtime_snapshot(old_snapshot)
    await cast(Any, manager)._start_runtime_snapshot(new_snapshot)

    assert calls == ["new"]
    assert old_root.instance_token not in cast(Any, manager)._runtime_started_roots
    await manager.snapshot_store.close()
    await old_root.dispose()
    await new_root.dispose()


@pytest.mark.asyncio
async def test_emit_event_listener_failure_is_fail_loud() -> None:
    root = CompositionRoot("emit-event-failure")

    def fail(_: object) -> None:
        raise RuntimeError("observe failed")

    async def plugin(ctx) -> None:
        await ctx.on(EmitEventKey[object]("test.emit.failure"), fail)

    await root.mount(plugin, name="failing-emit-plugin")
    async with _bound_root(root):
        with pytest.raises(RuntimeError, match="observe failed"):
            root.context.emit(EmitEventKey[object]("test.emit.failure"), object())


@pytest.mark.asyncio
async def test_event_bus_does_not_bridge_into_plugin_composition() -> None:
    observed: list[MemoryWritten] = []
    root = CompositionRoot("event-bus-is-core-only")

    async def plugin(ctx) -> None:
        await ctx.on(_MEMORY_WRITTEN_EVENT, observed.append)

    await root.mount(plugin, name="composition-observer")
    async with _bound_root(root):
        await EventBus().fanout(_memory_written_event())

    assert observed == []


@pytest.mark.asyncio
async def test_runtime_snapshot_rejects_inherited_wrong_task_binding() -> None:
    from agent.plugins.snapshot import get_lifecycle_runtime_snapshot

    root = CompositionRoot("runtime-snapshot-wrong-task")
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        async def read_snapshot() -> object:
            return get_lifecycle_runtime_snapshot()

        task = asyncio.create_task(read_snapshot())
        with pytest.raises(CompositionError) as caught:
            await task
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()
        await root.dispose()

    assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["fanout", "enqueue"])
async def test_event_bus_rejects_inherited_wrong_task_binding(
    operation: str,
) -> None:
    root = CompositionRoot(f"event-bus-wrong-task-{operation}")

    async def plugin(ctx) -> None:
        await ctx.on(_MEMORY_WRITTEN_EVENT, lambda _: None)

    await root.mount(plugin, name="composition-observer")
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    bus = EventBus()
    bus.bind_runtime_snapshot_store(store)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        if operation == "fanout":
            task = asyncio.create_task(bus.fanout(_memory_written_event()))
        else:

            async def enqueue() -> None:
                bus.enqueue(_memory_written_event())

            task = asyncio.create_task(enqueue())
        with pytest.raises(CompositionError) as caught:
            await task
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await bus.aclose()
        await store.close()
        await root.dispose()

    assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"


def _memory_written_event() -> MemoryWritten:
    return MemoryWritten(
        session_key="session",
        channel="test",
        chat_id="chat",
        action="supersede",
        source_ref="session@post_response",
        superseded_ids=["memory-1"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "prepare", "start"])
async def test_prepublication_resources_keep_exact_scope_and_cleanup_after_start_failure(tmp_path, failure):
    from agent.plugin_composition import RUNTIME_STARTING
    from agent.plugin_composition.tasks import Tasks
    from agent.plugins.snapshot import get_current_runtime_snapshot

    root = CompositionRoot("startup-resources")
    tasks = Tasks()
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    held = []
    events = []

    async def plugin(ctx):
        def prepare(_):
            snapshot = get_current_runtime_snapshot()
            assert snapshot is not None and snapshot.composition_root is root
            assert not snapshot.accepting_leases
            with pytest.raises(RuntimeError, match="不可租用|暂停"):
                manager.snapshot_store.lease(snapshot.snapshot_id)
            hold = tasks.activity("resource")
            hold.__enter__()
            held.append(hold)
            events.append("prepare")
            if failure == "prepare":
                raise ValueError("prepare failed")

        async def start(_):
            events.append("start")
            if failure == "start":
                raise ValueError("start failed")

        def stop(_):
            events.append("stop")
            for hold in held:
                hold.__exit__(None, None, None)
            held.clear()

        await ctx.on(RUNTIME_STARTING, prepare)
        await ctx.on(RUNTIME_STARTED, start)
        await ctx.on(RUNTIME_STOPPING, stop)

    await root.mount(plugin, name="resource-owner")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    try:
        if failure == "prepare":
            with pytest.raises(ValueError, match="prepare failed"):
                await manager._publish_committed_snapshot(snapshot)
            assert manager.current_snapshot is None
            assert snapshot.lease_count == 0
            assert events == ["prepare", "stop"]
            assert root.instance_token not in manager._runtime_starting_roots
            transaction = manager.snapshot_store.pending_transaction
            assert transaction is not None
            await manager.snapshot_store.abort(transaction)
        else:
            await manager._publish_committed_snapshot(snapshot)
            assert events == ["prepare"]
            if failure == "start":
                with pytest.raises(ValueError, match="start failed"):
                    await manager.start_runtime()
                assert events == ["prepare", "start", "stop"]
                assert not held
                assert root.instance_token not in manager._runtime_starting_roots
                assert root.instance_token not in manager._runtime_started_roots
                # 已清理的 Root 不能跳过发布前准备直接重试。
                with pytest.raises(RuntimeError, match="发布前准备"):
                    await manager.start_runtime()
            else:
                await manager.start_runtime()
                # 同 Root 的快照替换不重复恢复活动或重复启动消费者。
                replacement = RuntimeSnapshotCompiler().compile({}, composition_root=root, snapshot_revision="replacement")
                await manager._publish_committed_snapshot(replacement)
                await manager.start_runtime()
                assert events == ["prepare", "start"]
                current = manager.current_snapshot
                assert current is not None
                await manager._stop_runtime_snapshot(current)
                assert events == ["prepare", "start", "stop"]
        async with asyncio.timeout(2):
            await tasks.close()
        assert not held
    finally:
        await manager.terminate_all()
        await root.dispose()


@pytest.mark.asyncio
async def test_recovery_disposes_candidate_root_before_rebuilding_same_generation(tmp_path):
    """回退时先释放候选 Root，避免 generation effect 残留到 stable 重建。"""
    source = tmp_path / "plugins" / "registry"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text(
        """
api_version = 3
name = "registry"
version = "1.0.0"
_registry = set()
async def apply(ctx):
    generation_id = ctx.runtime.generation_id
    if generation_id in _registry:
        raise RuntimeError("generation owner still registered")
    _registry.add(generation_id)
    async def cleanup():
        _registry.remove(generation_id)
    await ctx.effect(lambda: cleanup, label="registry-owner")
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = PluginManager(
        [source.parent],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "home",
    )
    try:
        await manager.load_all()
        await manager.start_runtime()
        stable_snapshot = manager.current_snapshot
        assert stable_snapshot is not None
        stable = manager.generation("registry")
        assert stable is not None
        paused = manager.snapshot_store.pause_admission()
        assert paused is stable_snapshot
        await manager.snapshot_store.wait_for_no_leases(stable_snapshot)
        await manager._stop_runtime_snapshot(stable_snapshot)
        await manager._stop_stable_root(stable, stable_snapshot)
        old_root = stable_snapshot.composition_root
        candidate = await manager._compile_generation_snapshot(
            stable, force_fresh_composition=True
        )
        candidate_root = candidate.composition_root
        assert candidate_root is not None and candidate_root is not old_root
        assert manager._building_roots == {}
        assert stable_snapshot.composition_root is old_root
        stable.runtime_snapshot = candidate

        await manager._recover_stable_root(stable, stable_snapshot)

        assert candidate_root.receipt().fibers == ()
        assert stable_snapshot.composition_root is not old_root
        root = stable_snapshot.composition_root
        assert root is not None
        assert root.receipt().ready
        module = stable.instance.module
        assert module is not None
        assert module.__dict__["_registry"] == {stable.generation_id}
    finally:
        await manager.terminate_all()


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
    return PluginManager(
        [source.parent], event_bus=EventBus(), workspace=workspace,
        installed_cache_root=tmp_path / "home",
    )


def _error_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _error_leaves(child)]
    return [error]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["batch", "generation"])
@pytest.mark.parametrize("failure", ["mount", "compile"])
async def test_failed_root_build_keeps_module_data_and_cleanup_owner(
    tmp_path, monkeypatch, entry, failure,
):
    """批次和单 generation 回滚都不能丢弃部分装配后未关闭的连接。"""
    import sys

    manager = _root_failure_manager(tmp_path, fail_mount=failure == "mount")
    if failure == "compile":
        def fail_compile(*args, **kwargs):
            raise ValueError("catalog compilation failed")
        monkeypatch.setattr(manager._snapshot_compiler, "compile", fail_compile)
    with pytest.raises(BaseExceptionGroup) as caught:
        if entry == "batch":
            await manager.load_all()
        else:
            await manager._load_one(manager.discover()[0])
    leaves = _error_leaves(caught.value)
    assert any(isinstance(error, ValueError) for error in leaves)
    assert any(isinstance(error, OSError) for error in leaves)
    [(root, generations)] = manager._building_roots.items()
    [generation] = generations
    module = sys.modules[generation.module_path]
    assert module.attempts == 1
    assert (generation.data_dir / "connection-owner").read_text() == "open"
    assert not generation.scope.closed
    assert generation.runtime_snapshot is None
    assert generation in manager._draining_generations[generation.plugin_id]

    await manager.terminate_all()

    assert module.attempts == 2
    assert root.receipt().fibers == ()
    assert manager._building_roots == {}
    assert manager._draining_generations == {}
    assert generation.module_path not in sys.modules
    assert generation.scope.closed


@pytest.mark.asyncio
async def test_cancelled_compilation_cleanup_keeps_cancel_and_real_failure(tmp_path, monkeypatch):
    """重复取消等待中的回收仍保留原始错误、取消和失败连接。"""
    import sys

    manager = _root_failure_manager(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    def fail_compile(generations, **kwargs):
        generation = generations["root_owner"]
        module = sys.modules[generation.module_path]
        module.entered, module.release = entered, release
        raise ValueError("catalog compilation failed")

    monkeypatch.setattr(manager._snapshot_compiler, "compile", fail_compile)
    task = asyncio.create_task(manager._load_one(manager.discover()[0]))
    await entered.wait()
    task.cancel()
    asyncio.get_running_loop().call_soon(task.cancel)
    asyncio.get_running_loop().call_soon(release.set)
    with pytest.raises(BaseExceptionGroup) as caught:
        await task
    leaves = _error_leaves(caught.value)
    assert any(isinstance(error, asyncio.CancelledError) for error in leaves)
    assert any(isinstance(error, ValueError) for error in leaves)
    assert any(isinstance(error, OSError) for error in leaves)
    [generations] = manager._building_roots.values()
    [generation] = generations
    module = sys.modules[generation.module_path]
    assert module.attempts == 1
    assert generation in manager._draining_generations[generation.plugin_id]
    await manager.terminate_all()
    assert module.attempts == 2
    assert manager._building_roots == {}


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
        await manager._resolve_composition_root({})
    assert attempts == 1
    assert len(manager._building_roots) == 1
    first = asyncio.create_task(manager.terminate_all())
    await entered.wait()
    second = asyncio.create_task(manager.terminate_all())
    first.cancel()
    asyncio.get_running_loop().call_soon(first.cancel)
    asyncio.get_running_loop().call_soon(release.set)
    if fail_close:
        with pytest.raises(BaseExceptionGroup) as caught:
            await first
        assert any(isinstance(error, asyncio.CancelledError) for error in _error_leaves(caught.value))
        with pytest.raises(OSError):
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
async def test_compilation_cancellation_with_successful_cleanup_stays_cancelled(tmp_path, monkeypatch):
    """清理成功后仍抛调用者取消，不伪装为候选拒绝或清理失败。"""
    import sys

    manager = _root_failure_manager(tmp_path)
    loaded = []

    def cancel_compile(generations, **kwargs):
        generation = generations["root_owner"]
        loaded.append(generation)
        sys.modules[generation.module_path].attempts = 1
        raise asyncio.CancelledError

    monkeypatch.setattr(manager._snapshot_compiler, "compile", cancel_compile)
    with pytest.raises(asyncio.CancelledError):
        await manager._load_one(manager.discover()[0])
    [generation] = loaded
    assert generation.module_path not in sys.modules
    assert generation.scope.closed
    assert manager._building_roots == {}
    assert manager._draining_generations == {}
    await manager.terminate_all()
