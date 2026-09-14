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
        await manager._close_formal_root(snapshot)
        stable = next(iter(manager._active_generations.values()))
        if prepare_failure:
            (workspace / "fail-prepare").touch()
            with pytest.raises(ValueError, match="rebuild prepare failed"):
                await manager._build_and_publish_root(
                    dict(snapshot.generations), previous=snapshot, old_channel=None,
                )
            assert manager.current_snapshot is snapshot
            assert not snapshot.accepting_leases and snapshot.lease_count == 0
            (workspace / "fail-prepare").unlink()
        replacement = await manager._build_and_publish_root(
            dict(snapshot.generations), previous=snapshot, old_channel=None,
        )
        assert replacement is manager.current_snapshot and replacement is not snapshot
        assert snapshot.composition_root is old_root
        assert replacement.generations[stable.plugin_id] is not stable
        events = replacement.composition_root.context.require(ServiceKey("probe.events"))
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
async def test_recovery_disposes_candidate_root_before_building_new_instances(tmp_path):
    """关闭候选后重建实际新实例，原 snapshot 仍描述它原来的 Root。"""
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
        old_root = stable_snapshot.composition_root
        candidate = await manager._compile_generation_snapshot(
            stable, candidate_owner=stable,
        )
        candidate_root = candidate.composition_root
        assert candidate_root is not None and candidate_root is not old_root
        assert candidate_root.frozen and old_root.frozen
        assert candidate_root in manager._building_roots
        assert stable_snapshot.composition_root is old_root
        assert stable.runtime_snapshot is stable_snapshot
        await manager._dispose_unreferenced_composition_root(candidate)
        replacement = await manager._replace_formal_root(dict(stable_snapshot.generations))

        assert candidate_root.receipt().fibers == ()
        assert stable_snapshot.composition_root is old_root
        root = replacement.composition_root
        assert root is not None
        assert root.receipt().ready
        assert root.frozen
        fresh = replacement.generations[stable.plugin_id]
        assert fresh is not stable
        module = fresh.instance.module
        assert module is not None
        assert module.__dict__["_registry"] == {fresh.generation_id}
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
    assert manager._scopes[generation.module_path] is generation.scope

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
    assert manager._scopes[generation.module_path] is generation.scope
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


@pytest.mark.asyncio
async def test_compiled_root_rejects_binding_changes_without_restarting_work():
    """编译后拒绝全部绑定入口，已有消费者继续使用原实例和激活身份。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("fixed-bindings")
    service = ServiceKey[list[str]]("test.fixed.service")
    values = ["original"]
    contexts = []
    calls = []

    async def provider(ctx):
        await ctx.provide(service, values)

    async def consumer(ctx):
        contexts.append(ctx)
        calls.append(ctx.require(service))

    provider_fiber = await root.mount(provider, name="provider")
    fiber = await root.mount(consumer, name="consumer", inject=(service,))
    context = contexts[0]
    token = context.fiber.activation_token
    missing = ServiceKey("test.new.service")
    await root.context.inject((missing,), lambda ctx: calls.append("late activation"))
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    assert snapshot.composition_root is root and root.frozen
    changes = (
        lambda: root.mount(lambda ctx: None, name="late-root-child"),
        lambda: context.mount(lambda ctx: None, name="late-child"),
        lambda: context.inject((service,), lambda ctx: None, name="late-inject"),
        lambda: context.provide(service, ["replacement"]),
        lambda: context.provide(missing, object()),
        context.fiber.dispose,
        provider_fiber.context.fiber.dispose,
    )
    for change in changes:
        with pytest.raises(CompositionError) as caught:
            await change()
        assert caught.value.code == "COMPOSITION_FROZEN"
    # 服务内部状态仍可变化；协调通知不能产生新实例。
    values.append("client retry completed")
    await fiber.reconcile()
    assert calls == [values]
    assert context.require(service) is values
    assert context.fiber.activation_token is token
    assert root.topology_view() == snapshot.composition_topology
    await root.dispose()
    assert root.frozen


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
    RuntimeSnapshotCompiler().compile({}, composition_root=root)
    await root.dispose()
    assert events == ["consumer-start", "consumer-close", "provider-close"]


@pytest.mark.asyncio
async def test_frozen_binding_removal_requires_whole_root_teardown():
    """手动移除不改变已发布组合；被拒绝的 Effect 留给整个 Root 退出。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("fixed-service-removal")
    service = ServiceKey[object]("test.removal.service")
    value = object()
    registration = await root.context.provide(service, value)
    attempts = 0
    starts, resource_closes = [], []
    resource = await root.context.effect(lambda: lambda: resource_closes.append("closed"))

    async def consumer(ctx):
        starts.append(ctx.require(service))
        def close():
            nonlocal attempts
            attempts += 1
            assert ctx.require(service) is value
            if attempts == 1:
                raise OSError("consumer connection still open")
        await ctx.effect(lambda: close)

    fiber = await root.mount(consumer, name="consumer", inject=(service,))
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    store.install(snapshot)
    topology = root.topology_view()
    try:
        for operation in (registration.aclose, fiber.context.fiber.dispose):
            with pytest.raises(CompositionError) as caught:
                await operation()
            assert caught.value.code == "COMPOSITION_FROZEN"
        assert store.current is snapshot and snapshot.state == "committed"
        assert root.context.require(service) is value
        assert root.topology_view() == topology
        assert registration in root.root_fiber.effects
        assert attempts == 0 and starts == [value]

        # 普通资源 Effect 不拥有服务绑定或挂载树，可以独立关闭。
        await resource.aclose()
        assert resource_closes == ["closed"]
        assert root.topology_view().identity == topology.identity
        assert root.context.require(service) is value
        await store.close()

        with pytest.raises(BaseExceptionGroup):
            await root.dispose()
        assert attempts == 1
        assert registration in root.root_fiber.effects
        assert service.name in root.receipt().services
        # 退出失败也不能补挂、替换服务或重新激活旧工作。
        for operation in (
            lambda: root.context.provide(service, object()),
            lambda: root.mount(lambda ctx: None, name="replacement"),
        ):
            with pytest.raises(CompositionError) as caught:
                await operation()
            assert caught.value.code == "COMPOSITION_FROZEN"
        await root.dispose()
        assert attempts == 2 and starts == [value]
        assert registration not in root.root_fiber.effects
        assert root.receipt().services == ()
        assert root.receipt().fibers == ()
        # 已完成的句柄再次关闭没有结构变化。
        await registration.aclose()
        await fiber.context.fiber.dispose()
        assert resource_closes == ["closed"]
    finally:
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_sealing_precedes_freeze_and_started_resources_and_health_remain_live():
    """封印回调完成装配，启动回调仍能取得资源，健康变化不重启绑定。"""
    from agent.plugin_composition import SNAPSHOT_SEALING, SnapshotSealing, RuntimeStarted, ServiceKey

    root = CompositionRoot("freeze-start-resources")
    service = ServiceKey[list[str]]("test.sealing.service")
    values = []
    contexts, health, tasks = [], [], []
    running = asyncio.Event()
    closed = []

    async def plugin(ctx):
        contexts.append(ctx)
        health.append(await ctx.health("connection"))
        async def seal(_):
            await ctx.provide(service, values)
        async def follow():
            running.set()
            await asyncio.Event().wait()
        async def start(_):
            assert root.frozen
            await ctx.effect(lambda: lambda: closed.append("connection"))
            tasks.append(await ctx.spawn(follow(), name="connection-follower"))
        await ctx.on(SNAPSHOT_SEALING, seal)
        await ctx.on(RUNTIME_STARTED, start)

    await root.mount(plugin, name="resource-owner")
    await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
    token = contexts[0].fiber.activation_token
    async with _bound_root(root):
        await root.context.serial(RUNTIME_STARTED, RuntimeStarted())
        await running.wait()
        health[0].degrade("connection retrying")
        assert root.receipt().required_degraded
        values.append("reconnected")
        health[0].recover()
        assert root.receipt().ready
        assert contexts[0].fiber.activation_token is token
        assert root.context.require(service) == ["reconnected"]
    assert tasks[0].done()
    assert closed == ["connection"]


@pytest.mark.asyncio
async def test_freeze_rejects_incomplete_mount_without_caching_assembly_status():
    """已有 Fiber 过渡锁覆盖 apply 的异步等待。"""
    root = CompositionRoot("freeze-during-mount")
    entered, release = asyncio.Event(), asyncio.Event()

    async def plugin(ctx):
        entered.set()
        await release.wait()

    mount = asyncio.create_task(root.mount(plugin, name="mounting"))
    await entered.wait()
    with pytest.raises(CompositionError) as caught:
        root.freeze()
    assert caught.value.code == "COMPOSITION_NOT_SETTLED"
    assert not root.frozen
    release.set()
    await mount
    root.freeze()
    await root.dispose()


@pytest.mark.asyncio
async def test_failed_compilation_does_not_freeze_root():
    """缺失依赖的组合不能冻结，初始化 owner 仍可继续装配。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("failed-compile-not-frozen")
    service = ServiceKey("test.after.failed.compile")

    async def plugin(ctx):
        ctx.require(service)

    await root.mount(plugin, name="missing-dependency", inject=(service,))
    with pytest.raises(RuntimeError, match="组合拓扑未就绪"):
        RuntimeSnapshotCompiler().compile({}, composition_root=root)
    assert not root.frozen
    await root.context.provide(service, object())
    await root.dispose()
