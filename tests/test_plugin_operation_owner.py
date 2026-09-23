"""操作截止与实际资源退出分开观察；迟到任务不能重新取得提交权。"""
import asyncio
import threading

import pytest

from agent.plugin_composition import CompositionRoot, FiberState
from agent.plugins._operation import OperationBusyError, OperationTimeoutError, complete_critical
from agent.plugins.manager import PluginManager, _copy_in_thread
from agent.plugins.snapshot import RuntimeSnapshotCompiler
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


@pytest.mark.asyncio
async def test_operation_owner_observes_one_finite_local_update(tmp_path) -> None:
    """Observe revoke, physical settlement, and explicit recovery on a local update."""
    plugins = tmp_path / "plugins"
    local_plugin = plugins / "local_owner"
    unrelated_plugin = plugins / "unrelated"
    local_plugin.mkdir(parents=True)
    unrelated_plugin.mkdir(parents=True)
    local_file = local_plugin / "plugin.py"
    local_file.write_text(
        "api_version = 3\nname = 'local_owner'\nversion = '1.0.0'\n"
        "async def apply(ctx):\n    return None\n",
        encoding="utf-8",
    )
    (unrelated_plugin / "plugin.py").write_text(
        "api_version = 3\nname = 'unrelated'\nversion = '1.0.0'\n"
        "async def apply(ctx):\n    return None\n",
        encoding="utf-8",
    )
    host = manager(tmp_path, plugin_dirs=(plugins,))
    try:
        LOCAL_APPLY_ENTERED.clear()
        LOCAL_APPLY_RELEASE.clear()
        global LOCAL_RETRY_ALLOWED
        LOCAL_RETRY_ALLOWED = False
        await host.load_all()
        unrelated = host.generation("unrelated")
        assert unrelated is not None
        local_file.write_text(
            "api_version = 3\nname = 'local_owner'\nversion = '2.0.0'\n"
            "import asyncio\n"
            "import tests.test_plugin_operation_owner as harness\n"
            "async def apply(ctx):\n"
            "    if not harness.LOCAL_RETRY_ALLOWED:\n"
            "        harness.LOCAL_APPLY_ENTERED.set()\n"
            "        try:\n"
            "            await harness.LOCAL_APPLY_RELEASE.wait()\n"
            "        except asyncio.CancelledError:\n"
            "            await harness.LOCAL_APPLY_RELEASE.wait()\n",
            encoding="utf-8",
        )
        selected_before = host._selection.read()
        caller = asyncio.create_task(host.reconcile_changed())
        await LOCAL_APPLY_ENTERED.wait()
        operation = host._operation
        assert operation is not None and not operation.task.done()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert operation.revoked
        LOCAL_APPLY_RELEASE.set()
        await settle(operation.task)
        assert host._selection.read() != selected_before
        assert host.generation("local_owner") is None
        assert host.generation("unrelated") is unrelated
        async with unrelated.fiber.context.runtime_scope():
            assert unrelated.fiber.context.runtime.plugin_id == "unrelated"
        assert unrelated.fiber.state is FiberState.ACTIVE

        LOCAL_RETRY_ALLOWED = True
        recovered_result = await host.retry_runtime_recovery("local_owner")
        assert recovered_result["publication_state"] == "recovered"
        assert recovered_result["plugin_id"] == "local_owner"
        recovered = host.generation("local_owner")
        assert recovered is not None and recovered.fiber.state is FiberState.ACTIVE
        assert recovered.instance.version == "2.0.0"
        assert host.generation("unrelated") is unrelated
    finally:
        LOCAL_APPLY_RELEASE.set()
        LOCAL_RETRY_ALLOWED = True
        await host.terminate_all()


LOCAL_APPLY_ENTERED = asyncio.Event()
LOCAL_APPLY_RELEASE = asyncio.Event()
LOCAL_RETRY_ALLOWED = False


def manager(tmp_path, kind=PluginManager, *, plugin_dirs=()):
    initialize_plugin_workspace(tmp_path)
    return kind(list(plugin_dirs), event_bus=EventBus(), workspace=tmp_path)


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
    host.BOOT_COMMIT_TIMEOUT_SECONDS = 0.05
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
    with pytest.raises(OSError, match="connection still open"):
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
    host.BOOT_COMMIT_TIMEOUT_SECONDS = 0.05
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
    host.BOOT_COMMIT_TIMEOUT_SECONDS = 0.05
    load = asyncio.create_task(host.load_all())
    await entered.wait()
    try:
        with pytest.raises(OperationTimeoutError):
            await load
        operation = host._operation
        assert target.is_dir() and not operation.task.done()
        with pytest.raises(OperationBusyError):
            await host.reconcile_changed()
    finally:
        finish.set()
        await settle(host._operation.task)
        await host.terminate_all()
    assert not target.exists()


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
