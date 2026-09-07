from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import subprocess
import sys
from typing import Any, AsyncIterator, cast

import pytest

from agent.plugin_composition import (
    Bail,
    CompositionError,
    CompositionRoot,
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
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
from agent.turn_events.observe import MEMORY_WRITTEN_EVENT, RETRIEVAL_COMPLETED_EVENT
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQueryResult, MemoryRecord
from core.memory.events import MemoryWritten, RetrievalCompleted
from agent.retrieval.events import build_retrieval_completed
from agent.retrieval.protocol import RetrievalRequest


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
async def apply(ctx, config):
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
async def test_after_turn_committed_event_runs_after_core_fanout() -> None:
    order: list[str] = []
    observed: list[TurnCommitted] = []
    root = CompositionRoot("after-turn-committed")

    def on_committed(event: TurnCommitted) -> None:
        order.append("composition")
        observed.append(event)

    async def plugin(ctx) -> None:
        await ctx.on(AFTER_TURN_COMMITTED, on_committed)

    await root.mount(plugin, name="observe-plugin")
    committed = _committed_event()
    bus = EventBus()
    bus.on(TurnCommitted, lambda _: order.append("event-bus"))

    async with _bound_root(root):
        await bus.fanout(committed)
        root.context.emit(AFTER_TURN_COMMITTED, committed)

    assert order == ["event-bus", "composition"]
    assert observed == [committed]


@pytest.mark.asyncio
async def test_after_turn_committed_event_propagates_listener_failure() -> None:
    root = CompositionRoot("after-turn-committed-failure")

    def fail(_: TurnCommitted) -> None:
        raise RuntimeError("observe failed")

    async def plugin(ctx) -> None:
        await ctx.on(AFTER_TURN_COMMITTED, fail)

    await root.mount(plugin, name="failing-observe-plugin")
    async with _bound_root(root):
        with pytest.raises(RuntimeError, match="observe failed"):
            root.context.emit(AFTER_TURN_COMMITTED, _committed_event())


def test_after_turn_event_contract_imports_without_phase_runtime() -> None:
    code = (
        "from agent.turn_events.after_turn import AFTER_TURN_COMMITTED; "
        "import sys; "
        "assert 'agent.lifecycle.phases.after_turn' not in sys.modules; "
        "assert AFTER_TURN_COMMITTED.name == 'turn.after_turn.committed'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_after_turn_committed_event_keeps_core_path_without_root() -> None:
    observed: list[TurnCommitted] = []
    bus = EventBus()
    bus.on(TurnCommitted, observed.append)
    event = _committed_event()
    result = await bus.fanout(event)

    assert result is None
    assert observed == [event]




@pytest.mark.asyncio
async def test_domain_observe_event_uses_bound_candidate_root() -> None:
    observed: list[str] = []
    first = CompositionRoot("domain-observe-first")
    second = CompositionRoot("domain-observe-second")

    async def first_plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: observed.append("first"))

    async def second_plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: observed.append("second"))

    await first.mount(first_plugin, name="first-observer")
    await second.mount(second_plugin, name="second-observer")
    event = _memory_written_event()
    async with _bound_root(first):
        await first.context.observe(MEMORY_WRITTEN_EVENT, event)
    async with _bound_root(second):
        await second.context.observe(MEMORY_WRITTEN_EVENT, event)

    assert observed == ["first", "second"]


@pytest.mark.asyncio
async def test_event_bus_does_not_bridge_into_plugin_composition() -> None:
    observed: list[MemoryWritten] = []
    root = CompositionRoot("event-bus-is-core-only")

    async def plugin(ctx) -> None:
        await ctx.on(MEMORY_WRITTEN_EVENT, observed.append)

    await root.mount(plugin, name="domain-observer")
    async with _bound_root(root):
        await EventBus().fanout(_memory_written_event())

    assert observed == []


@pytest.mark.asyncio
async def test_domain_observe_event_rejects_inherited_wrong_task_binding() -> None:
    from agent.plugins.snapshot import get_lifecycle_runtime_snapshot

    root = CompositionRoot("domain-observe-wrong-task")
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
        await ctx.on(MEMORY_WRITTEN_EVENT, lambda _: None)

    await root.mount(plugin, name="domain-observer")
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


@pytest.mark.asyncio
async def test_retrieval_completed_event_payload() -> None:
    observed: list[RetrievalCompleted] = []
    root = CompositionRoot("retrieval-completed")

    async def plugin(ctx) -> None:
        await ctx.on(RETRIEVAL_COMPLETED_EVENT, lambda event: observed.append(event))

    await root.mount(plugin, name="retrieval-observer")

    request = RetrievalRequest(
        message="original",
        session_key="session",
        channel="test",
        chat_id="chat",
        history=[],
        session_metadata={},
    )
    result = MemoryQueryResult(
        text_block="memory block",
        records=[
            MemoryRecord(
                id="memory-1",
                kind="event",
                summary="a long enough memory summary",
                score=0.91,
                engine_kind="fake",
                signals={"confidence_label": "certain", "forced": True},
                injected=True,
            )
        ],
        trace={"route_decision": "RETRIEVE", "hyde_hypotheses": ["aux query"]},
        raw={"rewritten_query": "rewritten"},
    )

    async with _bound_root(root):
        await root.context.observe(
            RETRIEVAL_COMPLETED_EVENT,
            build_retrieval_completed(request, result),
        )

    assert len(observed) == 1
    event = observed[0]
    assert event.query == "rewritten"
    assert event.orig_query == "original"
    assert event.route_decision == "RETRIEVE"
    assert event.aux_queries == ["aux query"]
    assert event.injected_count == 1
    assert event.hits[0].item_id == "memory-1"
    assert event.hits[0].confidence_label == "certain"
    assert event.hits[0].forced is True
    assert event.hits[0].metadata["forced"] is True


def test_domain_event_contract_imports_without_phase_runtime() -> None:
    code = (
        "from agent.turn_events.observe import ("
        "RETRIEVAL_COMPLETED_EVENT, MEMORY_WRITTEN_EVENT); "
        "import sys; "
        "assert 'agent.lifecycle.phases.after_turn' not in sys.modules; "
        "assert RETRIEVAL_COMPLETED_EVENT.name == 'memory.retrieval.completed'; "
        "assert MEMORY_WRITTEN_EVENT.name == 'memory.written'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _committed_event() -> TurnCommitted:
    return TurnCommitted(
        session_key="session",
        channel="test",
        chat_id="chat",
        input_message="hello",
        persisted_user_message="hello",
        assistant_response="world",
        tools_used=[],
    )


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
