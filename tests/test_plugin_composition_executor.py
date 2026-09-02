from __future__ import annotations

import asyncio
import threading

import pytest

from agent.plugin_composition import (
    EXECUTOR_SERVICE,
    CompositionError,
    CompositionRoot,
    ExecutorService,
    Fiber,
    HealthHandle,
    SyncTask,
)


async def _mount_executor(root: CompositionRoot, max_workers: int) -> Fiber:
    service = ExecutorService(max_workers=max_workers)
    return await root.mount(service.apply, name=service.name)


@pytest.mark.asyncio
async def test_parallel_sync_runs_concurrently_and_preserves_result_order() -> None:
    barrier = threading.Barrier(2)
    root = CompositionRoot("executor-order")
    await _mount_executor(root, 2)
    executor = root.context.require(EXECUTOR_SERVICE)

    def run(value: str) -> str:
        _ = barrier.wait(timeout=1)
        return value

    results = await executor.parallel_sync(
        (
            SyncTask("first", lambda: run("first")),
            SyncTask("second", lambda: run("second")),
        )
    )

    assert results == ("first", "second")


@pytest.mark.asyncio
async def test_parallel_sync_waits_all_and_aggregates_errors() -> None:
    completed: list[str] = []
    root = CompositionRoot("executor-errors")
    await _mount_executor(root, 2)
    executor = root.context.require(EXECUTOR_SERVICE)

    def fail(name: str, error: Exception) -> None:
        completed.append(name)
        raise error

    with pytest.raises(BaseExceptionGroup) as caught:
        await executor.parallel_sync(
            (
                SyncTask("first", lambda: fail("first", ValueError("first"))),
                SyncTask("second", lambda: fail("second", RuntimeError("second"))),
            )
        )

    assert sorted(completed) == ["first", "second"]
    assert {type(error) for error in caught.value.exceptions} == {
        ValueError,
        RuntimeError,
    }


@pytest.mark.asyncio
async def test_parallel_sync_worker_cannot_access_context() -> None:
    root = CompositionRoot("executor-context-boundary")
    await _mount_executor(root, 1)
    executor = root.context.require(EXECUTOR_SERVICE)

    with pytest.raises(BaseExceptionGroup) as caught:
        await executor.parallel_sync(
            (SyncTask("escape", lambda: root.context.generation_id),)
        )

    error = caught.value.exceptions[0]
    assert isinstance(error, CompositionError)
    assert error.code == "CONTEXT_IN_SYNC_WORKER"


@pytest.mark.asyncio
async def test_parallel_sync_worker_cannot_mutate_saved_health_handle() -> None:
    root = CompositionRoot("executor-health-boundary")
    await _mount_executor(root, 1)
    handles: list[HealthHandle] = []

    async def plugin(ctx) -> None:
        handles.append(await ctx.health("worker", required=True))

    await root.mount(plugin, name="plugin")
    executor = root.context.require(EXECUTOR_SERVICE)

    with pytest.raises(BaseExceptionGroup) as caught:
        await executor.parallel_sync(
            (SyncTask("escape", lambda: handles[0].degrade("thread write")),)
        )

    error = caught.value.exceptions[0]
    assert isinstance(error, CompositionError)
    assert error.code == "CONTEXT_IN_SYNC_WORKER"
    assert root.receipt().required_degraded == ()


@pytest.mark.asyncio
async def test_parallel_sync_worker_cannot_read_saved_fiber_handle() -> None:
    root = CompositionRoot("executor-fiber-boundary")
    await _mount_executor(root, 1)
    handle = await root.context.mount(lambda _: None, name="plugin")
    executor = root.context.require(EXECUTOR_SERVICE)

    with pytest.raises(BaseExceptionGroup) as caught:
        await executor.parallel_sync(
            (SyncTask("escape", lambda: handle.state),)
        )

    error = caught.value.exceptions[0]
    assert isinstance(error, CompositionError)
    assert error.code == "CONTEXT_IN_SYNC_WORKER"


@pytest.mark.asyncio
async def test_parallel_sync_cancellation_joins_running_thread() -> None:
    started = threading.Event()
    release = threading.Event()
    root = CompositionRoot("executor-cancel")
    await _mount_executor(root, 1)
    executor = root.context.require(EXECUTOR_SERVICE)

    def run() -> str:
        started.set()
        _ = release.wait(timeout=2)
        return "done"

    call = asyncio.create_task(
        executor.parallel_sync((SyncTask("running", run),))
    )
    assert await asyncio.to_thread(started.wait, 1)
    _ = call.cancel()
    await asyncio.sleep(0)
    assert call.done() is False
    _ = call.cancel()
    await asyncio.sleep(0)
    assert call.done() is False
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.mark.asyncio
async def test_parallel_sync_cancellation_drops_queued_task() -> None:
    started = threading.Event()
    release = threading.Event()
    queued_ran = threading.Event()
    root = CompositionRoot("executor-cancel-queued")
    await _mount_executor(root, 1)
    executor = root.context.require(EXECUTOR_SERVICE)

    def running() -> str:
        started.set()
        _ = release.wait(timeout=2)
        return "running"

    def queued() -> str:
        queued_ran.set()
        return "queued"

    call = asyncio.create_task(
        executor.parallel_sync(
            (
                SyncTask("running", running),
                SyncTask("queued", queued),
            )
        )
    )
    assert await asyncio.to_thread(started.wait, 1)
    _ = call.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await call
    assert queued_ran.is_set() is False


@pytest.mark.asyncio
async def test_executor_provider_dispose_closes_pool_and_removes_service() -> None:
    root = CompositionRoot("executor-dispose")
    provider = await _mount_executor(root, 1)
    executor = root.context.require(EXECUTOR_SERVICE)

    await provider.dispose()

    assert root.context.get(EXECUTOR_SERVICE) is None
    with pytest.raises(CompositionError) as caught:
        await executor.parallel_sync((SyncTask("late", lambda: "late"),))
    assert caught.value.code == "EXECUTOR_CLOSED"


@pytest.mark.asyncio
async def test_parallel_sync_empty_batch_is_valid() -> None:
    root = CompositionRoot("executor-empty")
    await _mount_executor(root, 1)
    executor = root.context.require(EXECUTOR_SERVICE)

    assert await executor.parallel_sync(()) == ()
