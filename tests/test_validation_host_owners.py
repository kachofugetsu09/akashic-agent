"""验证宿主构建取消、lease 和模块关闭使用真实 owner，不创建子 Manager。"""
import asyncio
import contextvars
import sys

import pytest
from agent.plugin_composition import ServiceKey

from tests.test_plugin_business_validation import MODULE, prepare
from tests.test_plugin_install import _commit


async def candidate(tmp_path):
    source, _, _, log, manager = prepare(tmp_path)
    await manager.load_all()
    module = MODULE.replace(
        "inject = (MESSAGE_WRITERS, SESSION_ADMISSION, TASKS)",
        'inject = (MESSAGE_WRITERS, SESSION_ADMISSION, TASKS, ServiceKey("core.mobile_ui.v1"), ServiceKey("core.interaction_undo"))',
    )
    (source / "plugin.py").write_text(module + "\nmarker = 'owner-tests'\n")
    _commit(source)
    result, _ = await manager.install_candidate(
        source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
    )
    return log, manager, result.update_id


@pytest.mark.asyncio
async def test_validation_lease_retains_root_and_module_until_released(tmp_path):
    log, manager, update_id = await candidate(tmp_path)
    lease = None
    try:
        undo_key = ServiceKey("core.interaction_undo")
        # 第一候选与真实隔离宿主都只有拒绝端口，不能获得正式撤销 owner。
        candidate_undo = manager.latest_snapshot.composition_root.context.require(undo_key)
        with pytest.raises(RuntimeError, match="禁止撤销正式 interaction"):
            await candidate_undo.undo_latest("formal")
        with pytest.raises(RuntimeError, match="仍有 lease"):
            async with manager.open_validation(update_id) as scope:
                host = next(iter(manager._validation_hosts.values()))
                with pytest.raises(RuntimeError, match="禁止撤销正式 interaction"):
                    await scope.require(undo_key).undo_latest("formal")
                ui = scope.require(ServiceKey("core.mobile_ui.v1"))
                # 没有继承 runtime context 的请求也必须只租用验证 Store。
                query_lease = await asyncio.create_task(
                    ui._capture_query_lease(), context=contextvars.Context(),
                )
                assert query_lease.snapshot is host.snapshot_store.current
                assert query_lease.snapshot is not manager.current_snapshot
                await query_lease.release()
                lease = await host.snapshot_store.acquire()
                snapshot = lease.snapshot
                modules = tuple(item.module_path for item in snapshot.generations.values())
        assert manager._validation_hosts[host.identity] is host
        assert not host.closed
        assert host.root is snapshot.composition_root
        assert all(name in sys.modules for name in modules)
        assert host.messages.reader("formal").snapshot() == ()
        assert host.parent_lease.active
        await lease.release()
        lease = None
        await manager.retry_validation_cleanup(host.identity)
        assert host.closed and not host.parent_lease.active
        assert all(name not in sys.modules for name in modules)
        assert host.snapshot_store.current is None
    finally:
        if lease is not None:
            await lease.release()
        await manager.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_cancelled_build_retains_failed_unpublished_root(tmp_path, monkeypatch):
    log, manager, update_id = await candidate(tmp_path)
    entered = asyncio.Event()
    blocked = asyncio.Event()
    fail_cleanup = True
    calls = 0
    original = manager._mount_generation_composition

    async def mount(root, generation):
        nonlocal calls
        await original(root, generation)
        async def cleanup():
            nonlocal calls
            calls += 1
            if fail_cleanup:
                raise OSError("unpublished root still owns module")
        root._defer_internal_cleanup("test-build-owner", cleanup)
        entered.set()
        await blocked.wait()

    monkeypatch.setattr(manager, "_mount_generation_composition", mount)
    async def run():
        async with manager.open_validation(update_id):
            pytest.fail("cancelled build became ready")

    task = asyncio.create_task(run())
    try:
        await entered.wait()
        host = next(iter(manager._validation_hosts.values()))
        modules = tuple(item.module_path for item in host.generations)
        assert host.snapshot_store.current is None
        assert host.root not in manager._building_roots
        task.cancel()
        outcome, = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(outcome, BaseException)
        assert manager._validation_hosts[host.identity] is host
        assert not host.closed and host.parent_lease.active
        assert calls == 1
        assert all(name in sys.modules for name in modules)
        assert host.messages.reader("formal").snapshot() == ()
        fail_cleanup = False
        await manager.retry_validation_cleanup(host.identity)
        assert calls == 2 and host.closed
        assert all(name not in sys.modules for name in modules)
    finally:
        fail_cleanup = False
        blocked.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await manager.terminate_all()
        log.close()
