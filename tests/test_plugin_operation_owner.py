"""操作截止与实际资源退出分开观察；迟到任务不能重新取得提交权。"""
import asyncio
import threading

import pytest

from agent.plugin_composition import CompositionRoot, RuntimeScope
from agent.plugins._operation import OperationBusyError, OperationTimeoutError, complete_critical
from agent.plugins.manager import PluginManager, _copy_in_thread
from agent.plugins.snapshot import RuntimeSnapshotCompiler
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def manager(tmp_path, kind=PluginManager):
    initialize_plugin_workspace(tmp_path)
    return kind([], event_bus=EventBus(), workspace=tmp_path)


async def settle(task):
    """测试只观察已持有的任务，不重复发送取消。"""
    await asyncio.wait((task,))
    if not task.cancelled():
        return task.exception()
    return None


@pytest.mark.asyncio
async def test_timeout_retains_cancel_suppressing_build_and_terminate_joins(tmp_path):
    entered, swallowed, release, stopped = (asyncio.Event() for _ in range(4))
    commits, cancellations = [], []

    class Host(PluginManager):
        async def _load_all(self):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellations.append("cancel")
                swallowed.set()
                await release.wait()
            self._check_operation_commit()
            commits.append("late commit")

        async def _terminate_all(self):
            stopped.set()

    host = manager(tmp_path, Host)
    host.POST_PUBLISH_TIMEOUT_SECONDS = 0.05
    load = asyncio.create_task(host.load_all())
    await entered.wait()
    with pytest.raises(OperationTimeoutError):
        await load
    build = host._operation
    await swallowed.wait()
    assert build.revoked and not build.task.done()
    with pytest.raises(OperationBusyError):
        await host.start_runtime()
    first = asyncio.create_task(host.terminate_all())
    second = asyncio.create_task(host.terminate_all())
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(item, OperationTimeoutError) for item in results)
    shutdown = host._operation
    assert shutdown is not build and not shutdown.task.done()
    assert not stopped.is_set()
    assert cancellations == ["cancel"]
    release.set()
    await settle(shutdown.task)
    assert stopped.is_set() and commits == []
    assert host._operation is shutdown
    with pytest.raises(RuntimeError, match="停止接纳"):
        await host.load_all()


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_cancel_started_effect_twice(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    closes, cancellations = [], []
    host = manager(tmp_path)
    root = CompositionRoot("pending-close")

    async def close():
        closes.append("close")
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellations.append("cancel")
            raise

    await root.context.effect(lambda: close)
    host._building_roots[root] = ()
    first = asyncio.create_task(host.terminate_all())
    await entered.wait()
    shutdown = host._operation
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = asyncio.create_task(host.terminate_all())
    await asyncio.wait((shutdown.task,), timeout=0)
    assert host._operation is shutdown and root in host._building_roots
    assert closes == ["close"] and cancellations == []
    release.set()
    await second
    assert host._building_roots == {}
    assert closes == ["close"] and cancellations == []
    await host.terminate_all()
    assert host._operation is shutdown


@pytest.mark.asyncio
async def test_failed_effect_keeps_owner_until_explicit_terminate_retry(tmp_path):
    host = manager(tmp_path)
    root = CompositionRoot("failed-close")
    attempts = []

    async def close():
        attempts.append("close")
        if len(attempts) == 1:
            raise OSError("connection still open")

    await root.context.effect(lambda: close)
    host._building_roots[root] = ()
    with pytest.raises(BaseExceptionGroup):
        await host.terminate_all()
    failed = host._operation
    assert failed.task.done() and root in host._building_roots
    assert attempts == ["close"]
    await host.terminate_all()
    assert host._operation is not failed
    assert attempts == ["close", "close"] and host._building_roots == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("release_started", [False, True])
async def test_deadline_restores_only_the_unreleased_old_root(tmp_path, release_started):
    entered, finish_close = asyncio.Event(), asyncio.Event()

    class Host(PluginManager):
        async def _load_all(self):
            entered.set()
            await self._replace_formal_root({}, expected_ref=self._selection.read())

    host = manager(tmp_path, Host)
    root = CompositionRoot("old")

    async def close():
        entered.set()
        await finish_close.wait()

    if release_started:
        await root.context.effect(lambda: close)
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    host._snapshot_store.install(snapshot)
    lease = None if release_started else host._snapshot_store.lease(snapshot.snapshot_id)
    host.POST_PUBLISH_TIMEOUT_SECONDS = 0.05
    update = asyncio.create_task(host.load_all())
    await entered.wait()
    with pytest.raises(OperationTimeoutError):
        await update
    operation = host._operation
    if release_started:
        assert not operation.task.done()
        assert not snapshot.accepting_leases
        finish_close.set()
    await settle(operation.task)
    assert host.current_snapshot is snapshot
    assert snapshot.accepting_leases is not release_started
    if lease is not None:
        await lease.release()
    await host.terminate_all()


@pytest.mark.asyncio
async def test_prepare_thread_keeps_directory_until_actual_completion(tmp_path):
    entered, finish = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    target = tmp_path / "prepared"

    def write():
        target.mkdir()
        loop.call_soon_threadsafe(entered.set)
        finish.wait()
        (target / "receipt").write_text("thread completed")

    class Host(PluginManager):
        async def _load_all(self):
            try:
                await _copy_in_thread(write)
                self._check_operation_commit()
            finally:
                # 线程仍在写时到达这里会让测试的真实文件操作失败。
                (target / "receipt").unlink()
                target.rmdir()

    host = manager(tmp_path, Host)
    host.POST_PUBLISH_TIMEOUT_SECONDS = 0.05
    load = asyncio.create_task(host.load_all())
    await entered.wait()
    try:
        with pytest.raises(OperationTimeoutError):
            await load
        operation = host._operation
        assert target.is_dir() and not operation.task.done()
        with pytest.raises(OperationBusyError):
            await host.prepare_candidate("next")
    finally:
        finish.set()
        await settle(host._operation.task)
        await host.terminate_all()
    assert not target.exists()


@pytest.mark.asyncio
async def test_same_manager_lease_rejects_update_and_terminate_before_side_effects(tmp_path):
    host = manager(tmp_path)
    root = CompositionRoot("leased")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    host._snapshot_store.install(snapshot)
    async with RuntimeScope(await host._snapshot_store.acquire()):
        with pytest.raises(RuntimeError, match="lease"):
            await host.load_all()
        with pytest.raises(RuntimeError, match="lease"):
            await host.terminate_all()
    assert host._operation is None and not host._stopping
    assert snapshot.accepting_leases
    await host.terminate_all()


@pytest.mark.asyncio
async def test_cancel_and_real_cleanup_failure_remain_distinguishable(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def close():
        entered.set()
        await release.wait()
        raise OSError("actual cleanup failure")

    class Host(PluginManager):
        async def _load_all(self):
            await complete_critical(close())

    host = manager(tmp_path, Host)
    load = asyncio.create_task(host.load_all())
    await entered.wait()
    operation = host._operation
    load.cancel()
    with pytest.raises(asyncio.CancelledError):
        await load
    release.set()
    error = await settle(operation.task)
    assert isinstance(error, BaseExceptionGroup)
    assert any(isinstance(item, asyncio.CancelledError) for item in error.exceptions)
    assert any(isinstance(item, OSError) for item in error.exceptions)
    await host.terminate_all()


@pytest.mark.asyncio
async def test_committed_fact_survives_cancel_and_blocks_late_admission(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    root = CompositionRoot("committed")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)

    class Host(PluginManager):
        async def _load_all(self):
            operation = self._check_operation_commit()
            # 模拟主协调器在同步 durable commit 后登记实际提交回执。
            self._snapshot_store.install(snapshot)
            operation.committed = snapshot
            entered.set()
            await complete_critical(release.wait())
            self._check_operation_commit()
            snapshot.accepting_leases = True

    host = manager(tmp_path, Host)
    load = asyncio.create_task(host.load_all())
    await entered.wait()
    operation = host._operation
    load.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await load
    assert any("已经提交" in note for note in cancelled.value.__notes__)
    assert operation.committed is snapshot and not snapshot.accepting_leases
    release.set()
    await settle(operation.task)
    assert host.current_snapshot is snapshot and not snapshot.accepting_leases
    await host.terminate_all()


@pytest.mark.asyncio
async def test_background_publication_is_accepted_without_inheriting_caller_lease(tmp_path, monkeypatch):
    from agent.plugins.snapshot import get_current_runtime_lease
    from tests.test_plugin_fresh_root import installed_pair, candidate

    host = installed_pair(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    await host.load_all()
    stable = host.current_snapshot
    checked = await candidate(host, tmp_path)
    tx_id = checked.generations["changed@lab"].reload_tx_id
    update = host._reload_journal.update_for_reload(tx_id)
    assert update is not None
    switch = host._switch_ready

    async def blocked(plugin_id, *, update_id=None):
        assert get_current_runtime_lease() is None
        assert host._check_operation_commit() is host._operation
        entered.set()
        await release.wait()
        return await switch(plugin_id, update_id=update_id)

    monkeypatch.setattr(host, "_switch_ready", blocked)
    try:
        async with RuntimeScope(await host._snapshot_store.acquire()):
            host.start_update_publication(update.update_id)
            operation = host._operation
            host.start_update_publication(update.update_id)
            assert host._operation is operation
            await entered.wait()
            assert host.current_snapshot is stable
            assert host.update_is_publishing(update.update_id)
        release.set()
        await settle(operation.task)
        assert host.current_snapshot is not stable
        assert operation.committed is host.current_snapshot
    finally:
        release.set()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_plugin_raised_cancel_does_not_authorize_automatic_rebuild(tmp_path):
    attempts = []

    class Host(PluginManager):
        async def _load_all(self):
            await self._replace_formal_root({}, expected_ref=self._selection.read())

        async def _compile_topology_snapshot(self, generations):
            attempts.append("compile")
            raise asyncio.CancelledError("plugin cancelled its initialization")

    host = manager(tmp_path, Host)
    root = CompositionRoot("previous")
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    host._snapshot_store.install(snapshot)
    with pytest.raises(asyncio.CancelledError):
        await host.load_all()
    assert attempts == ["compile"]
    assert host._operation.revoked
    assert host.current_snapshot is snapshot and not snapshot.accepting_leases
    await host.terminate_all()


@pytest.mark.asyncio
async def test_cancel_revokes_before_an_already_ready_worker_can_commit(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    commits = []

    class Host(PluginManager):
        async def _load_all(self):
            entered.set()
            await release.wait()
            self._check_operation_commit()
            commits.append("committed")

    host = manager(tmp_path, Host)
    caller = asyncio.create_task(host.load_all())
    await entered.wait()
    operation = host._operation
    # 先把 worker 排到 ready 队列，再取消调用者；不能依赖调用者先被调度。
    release.set()
    caller.cancel()
    assert operation.revoked
    with pytest.raises(asyncio.CancelledError):
        await caller
    await settle(operation.task)
    assert commits == []
    await host.terminate_all()
