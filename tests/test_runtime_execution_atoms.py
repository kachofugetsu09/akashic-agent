from __future__ import annotations

# pyright: reportPrivateUsage=false

import ast
import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugins.errors import RootRetired
from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    ObserveEventKey,
    ServiceKey,
)
from agent.plugin_composition.tasks import TaskControl
from agent.plugins.manager import PluginManager
from agent.plugins.service_call import _bind_service
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    get_current_runtime_snapshot,
    reset_runtime_snapshot,
)
from bus.event_bus import EventBus


_CORE_FILES = (
    "agent/plugin_composition/context.py",
    "agent/plugin_composition/tasks.py",
    "agent/plugins/artifact_pins.py",
    "agent/plugins/service_call.py",
    "agent/plugins/service_hold.py",
    "agent/plugins/snapshot.py",
)


def test_core_runtime_files_do_not_import_control_domain() -> None:
    """Keep the runtime lease atoms independent from the old Turn owner."""

    root = Path(__file__).parents[1]
    violations: list[str] = []
    for relative_path in _CORE_FILES:
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            else:
                continue
            for name in names:
                if name == "agent.control" or name.startswith("agent.control."):
                    violations.append(f"{relative_path}:{node.lineno}:{name}")
    assert violations == []


class _Lease:
    def __init__(self) -> None:
        self.active = True
        self.release_count = 0

    async def release(self) -> None:
        if not self.active:
            return
        self.active = False
        self.release_count += 1

    async def run(self, call):
        try:
            return await call()
        finally:
            await self.release()


async def _service_root(
    name: str,
    key: ServiceKey[str],
    value: str,
) -> CompositionRoot:
    root = CompositionRoot(name)

    async def plugin(ctx) -> None:
        _ = await ctx.provide(key, value)

    _ = await root.mount(plugin, name=f"{name}-provider")
    return root


def _bind_root(root: CompositionRoot, store: RuntimeSnapshotStore) -> None:
    cast(Any, root)._bind_lease(lambda: store.acquire_composition_root(root))


@pytest.mark.asyncio
async def test_service_call_is_single_key_stable_only_and_releases() -> None:
    key = ServiceKey[str]("fixture-service")
    stable_root = await _service_root("stable-root", key, "stable")
    latest_root = await _service_root("latest-root", key, "latest")
    compiler = RuntimeSnapshotCompiler()
    stable = compiler.compile({}, composition_root=stable_root)
    latest = compiler.compile({}, composition_root=latest_root)
    store = RuntimeSnapshotStore()
    store.install(stable)
    transaction = store.begin_publish(latest)
    await store.commit_latest(transaction)
    service_call = _bind_service(store, key)

    async def read(service: str) -> str:
        assert get_current_runtime_snapshot() is stable
        return service

    assert await service_call.call(read) == "stable"
    missing = _bind_service(store, ServiceKey[str]("missing-service"))
    with pytest.raises(CompositionError) as caught:
        await missing.call(read)
    assert caught.value.code == "INACTIVE_SERVICE"
    assert stable.lease_count == 0
    assert latest.lease_count == 0

    await store.close()
    await stable_root.dispose()
    await latest_root.dispose()


@pytest.mark.asyncio
async def test_service_call_rejects_inherited_binding_and_cleans_cancel() -> None:
    key = ServiceKey[str]("fixture-service")
    root = await _service_root("service-root", key, "value")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    store.install(snapshot)
    service_call = _bind_service(store, key)
    parent_lease = store.lease()
    token = bind_runtime_snapshot(parent_lease)

    async def read(service: str) -> str:
        return service

    try:
        child = asyncio.create_task(service_call.call(read))
        with pytest.raises(CompositionError) as caught:
            await child
        assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"
        assert snapshot.lease_count == 1
    finally:
        reset_runtime_snapshot(token)
        await parent_lease.release()

    started = asyncio.Event()

    async def block(_service: str) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    call = asyncio.create_task(service_call.call(block))
    await started.wait()
    assert snapshot.lease_count == 1
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert snapshot.lease_count == 0

    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_root_scope_keeps_owning_root_and_rejects_retired_root() -> None:
    old_root = CompositionRoot("old-root")
    new_root = CompositionRoot("new-root")
    compiler = RuntimeSnapshotCompiler()
    old_snapshot = compiler.compile(
        {}, composition_root=old_root, snapshot_revision="old"
    )
    new_snapshot = compiler.compile(
        {}, composition_root=new_root, snapshot_revision="new"
    )
    store = RuntimeSnapshotStore()
    _bind_root(old_root, store)
    _bind_root(new_root, store)
    store.install(old_snapshot)

    async def background() -> object:
        async with old_root.context.root_scope():
            current = get_current_runtime_snapshot()
            assert current is old_snapshot
            return current.composition_root

    assert await asyncio.create_task(background()) is old_root
    assert old_snapshot.lease_count == 0

    parent_lease = store.lease()
    token = bind_runtime_snapshot(parent_lease)
    try:
        inherited = asyncio.create_task(background())
        with pytest.raises(CompositionError) as caught:
            await inherited
        assert caught.value.code == "RUNTIME_SNAPSHOT_BINDING_MISMATCH"
    finally:
        reset_runtime_snapshot(token)
        await parent_lease.release()

    transaction = store.begin_publish(new_snapshot)
    await store.commit(transaction)
    with pytest.raises(RootRetired):
        async with old_root.context.root_scope():
            pytest.fail("retired Root must not enter a new scope")

    new_lease = store.lease()
    token = bind_runtime_snapshot(new_lease)
    try:
        with pytest.raises(CompositionError) as caught:
            async with old_root.context.root_scope():
                pytest.fail("a Root binding must not be overwritten")
        assert caught.value.code == "ROOT_MISMATCH"
    finally:
        reset_runtime_snapshot(token)
        await new_lease.release()

    await store.close()
    await old_root.dispose()
    await new_root.dispose()


@pytest.mark.asyncio
async def test_root_scope_rejects_release_from_another_task() -> None:
    root = CompositionRoot("owner-root")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    _bind_root(root, store)
    store.install(snapshot)
    control = TaskControl()
    start = control.bind_start(ServiceKey[object]("agents"))
    entered = asyncio.Event()
    finish = asyncio.Event()

    async with root.context.root_scope():
        scope = root.context.capture_scope()

        async def run() -> str:
            entered.set()
            await finish.wait()
            return "terminal"

        task = start.claim("session", "task", scope, run, lambda: finish.set())
        await entered.wait()
        assert snapshot.lease_count == 2
        with pytest.raises(CompositionError) as caught:
            await scope.release()
        assert caught.value.code == "TASK_MISMATCH"
        assert snapshot.lease_count == 2

    assert snapshot.lease_count == 1
    finish.set()
    assert await task.wait() == "terminal"
    assert snapshot.lease_count == 0

    await control.aclose()
    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_observer_task_keeps_the_source_root() -> None:
    event = ObserveEventKey[None]("fixture.observe")
    root = CompositionRoot("observer-root")
    seen: list[tuple[object | None, asyncio.Task[object] | None]] = []

    async def plugin(ctx) -> None:
        async def observe(_event: None) -> None:
            async with ctx.root_scope():
                seen.append(
                    (
                        get_current_runtime_snapshot(),
                        cast(asyncio.Task[object] | None, asyncio.current_task()),
                    )
                )

        _ = await ctx.on(event, observe)

    _ = await root.mount(plugin, name="observer")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store = RuntimeSnapshotStore()
    _bind_root(root, store)
    store.install(snapshot)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    owner = cast(asyncio.Task[object] | None, asyncio.current_task())
    try:
        await root.context.observe(event, None)
        assert len(seen) == 1
        seen_snapshot, seen_task = seen[0]
        assert seen_snapshot is snapshot
        assert seen_task is not owner
        assert snapshot.lease_count == 1
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert snapshot.lease_count == 0
    await store.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_observe_rejects_another_root_before_callbacks() -> None:
    event = ObserveEventKey[None]("fixture.observe")
    source_root = CompositionRoot("source-root")
    other_root = CompositionRoot("other-root")
    callback_count = 0

    async def plugin(ctx) -> None:
        def observe(_event: None):
            nonlocal callback_count
            callback_count += 1

            async def run() -> None:
                return None

            return run()

        _ = await ctx.on(event, observe)

    _ = await source_root.mount(plugin, name="observer")
    other_snapshot = RuntimeSnapshotCompiler().compile(
        {}, composition_root=other_root
    )
    store = RuntimeSnapshotStore()
    store.install(other_snapshot)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        with pytest.raises(CompositionError) as caught:
            await source_root.context.observe(event, None)
        assert caught.value.code == "ROOT_MISMATCH"
        assert callback_count == 0
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    await store.close()
    await source_root.dispose()
    await other_root.dispose()


@pytest.mark.asyncio
async def test_core_task_control_cancels_old_root_after_reload(tmp_path) -> None:
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path)
    store = manager.snapshot_store
    old_root = CompositionRoot("old-root")
    new_root = CompositionRoot("new-root")
    _bind_root(old_root, store)
    _bind_root(new_root, store)
    compiler = RuntimeSnapshotCompiler()
    old_snapshot = compiler.compile(
        {}, composition_root=old_root, snapshot_revision="old"
    )
    new_snapshot = compiler.compile(
        {}, composition_root=new_root, snapshot_revision="new"
    )
    store.install(old_snapshot)
    service_key = ServiceKey[object]("agents")
    started = asyncio.Event()
    finish = asyncio.Event()
    cancel_calls: list[str] = []

    async with old_root.context.root_scope():
        old_scope = old_root.context.capture_scope()

        async def run_old() -> str:
            started.set()
            await finish.wait()
            return "old-terminal"

        def cancel_old() -> None:
            cancel_calls.append("old")
            finish.set()

        old_task = manager.task_start(service_key).claim(
            "same-session",
            "old-task",
            old_scope,
            run_old,
            cancel_old,
        )
        await started.wait()

    assert old_snapshot.lease_count == 1
    transaction = store.begin_publish(new_snapshot)
    await store.commit(transaction)
    assert old_snapshot.lease_count == 1

    async with new_root.context.root_scope():
        next_scope = new_root.context.capture_scope()
    try:
        with pytest.raises(RuntimeError, match="scope"):
            manager.task_start(ServiceKey[object]("agents")).claim(
                "same-session",
                "new-task",
                next_scope,
                lambda: asyncio.sleep(0, result="unreachable"),
                lambda: None,
            )
    finally:
        await next_scope.release()

    assert await manager.task_cancel(ServiceKey[object]("agents")).cancel(
        "old-task"
    )
    assert await old_task.wait() == "old-terminal"
    assert cancel_calls == ["old"]
    assert old_snapshot.lease_count == 0
    await store.wait_for_snapshot_drained(old_snapshot)

    async with new_root.context.root_scope():
        final_scope = new_root.context.capture_scope()
    new_task = manager.task_start(service_key).claim(
        "same-session",
        "new-task",
        final_scope,
        lambda: asyncio.sleep(0, result="new-terminal"),
        lambda: None,
    )
    assert await new_task.wait() == "new-terminal"
    assert new_snapshot.lease_count == 0

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_task_control_close_finishes_after_caller_cancel() -> None:
    control = TaskControl()
    lease = _Lease()
    run_started = asyncio.Event()
    cancel_started = asyncio.Event()
    allow_cancel = asyncio.Event()
    finish = asyncio.Event()

    async def run() -> str:
        run_started.set()
        await finish.wait()
        return "terminal"

    async def cancel() -> None:
        cancel_started.set()
        await allow_cancel.wait()
        finish.set()

    task = control.bind_start(ServiceKey[object]("agents")).claim(
        "session",
        "task",
        lease,
        run,
        cancel,
    )
    await run_started.wait()
    close_call = asyncio.create_task(control.aclose())
    await cancel_started.wait()
    close_call.cancel()
    await asyncio.sleep(0)
    assert not close_call.done()
    assert lease.active

    allow_cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await close_call
    assert await task.wait() == "terminal"
    assert not lease.active
    assert lease.release_count == 1


@pytest.mark.asyncio
async def test_task_cancel_finishes_after_caller_cancel() -> None:
    control = TaskControl()
    lease = _Lease()
    run_started = asyncio.Event()
    cancel_started = asyncio.Event()
    allow_cancel = asyncio.Event()
    finish = asyncio.Event()

    async def run() -> str:
        run_started.set()
        await finish.wait()
        return "terminal"

    async def cancel() -> None:
        cancel_started.set()
        await allow_cancel.wait()
        finish.set()

    service_key = ServiceKey[object]("agents")
    task = control.bind_start(service_key).claim(
        "session",
        "task",
        lease,
        run,
        cancel,
    )
    await run_started.wait()
    cancel_call = asyncio.create_task(
        control.bind_cancel(service_key).cancel("task")
    )
    await cancel_started.wait()
    cancel_call.cancel()
    await asyncio.sleep(0)
    assert not cancel_call.done()
    assert lease.active

    allow_cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await cancel_call
    assert await task.wait() == "terminal"
    assert lease.release_count == 1
    assert not await control.bind_cancel(service_key).cancel("task")
    await control.aclose()


@pytest.mark.asyncio
async def test_task_wait_cancel_does_not_release_owner_early() -> None:
    control = TaskControl()
    lease = _Lease()
    started = asyncio.Event()
    finish = asyncio.Event()

    async def run() -> str:
        started.set()
        await finish.wait()
        return "terminal"

    task = control.bind_start(ServiceKey[object]("agents")).claim(
        "session",
        "task",
        lease,
        run,
        lambda: finish.set(),
    )
    await started.wait()
    waiter = asyncio.create_task(task.wait())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert lease.active

    finish.set()
    assert await task.wait() == "terminal"
    assert not lease.active
    assert lease.release_count == 1
    await control.aclose()
