"""实际卸载等待整个旧 Root 关闭，关闭失败时保留 cache 与资源 owner。"""

import asyncio
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bootstrap.app import AppRuntime
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_fresh_root import module
from tests.test_plugin_install import _commit, _write_v3_plugin


def installed_app(tmp_path, *, fail_close=False):
    """安装两个实际插件，沿 App 的卸载入口观察资源关闭与 cache 删除。"""
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    for name in ("target", "peer"):
        source = tmp_path / name
        _write_v3_plugin(source, name=name, module_source=module(
            name, fail_close=fail_close and name == "target",
        ))
        _commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    initialize_plugin_workspace(workspace)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    app = SimpleNamespace(core=SimpleNamespace(plugin_manager=host), workspace=workspace)
    return host, app, home / "cache/lab/target"


@pytest.mark.asyncio
async def test_uninstall_waits_for_snapshot_and_fork_then_closes_whole_root(tmp_path, monkeypatch):
    host, app, cache = installed_app(tmp_path)
    lease = fork = task = None
    finish_drain = asyncio.Event()
    try:
        await host.load_all()
        old = host.current_snapshot
        target = old.generations["target@lab"]
        target_state = old.composition_root.context.require(ServiceKey("target.state"))
        peer_state = old.composition_root.context.require(ServiceKey("peer.state"))
        marker = target.data_dir / "keep.txt"
        marker.write_text("user data")
        lease = host.snapshot_store.lease()
        fork = lease.fork()
        waiting = asyncio.Event()
        drain_started = asyncio.Event()
        original_wait = host.snapshot_store.wait_for_no_leases
        original_drain = host.snapshot_store._on_drained

        async def wait(snapshot):
            if snapshot is old:
                waiting.set()
            await original_wait(snapshot)

        monkeypatch.setattr(host.snapshot_store, "wait_for_no_leases", wait)

        async def hold_drain(snapshot):
            if snapshot is old:
                drain_started.set()
                await finish_drain.wait()
            await original_drain(snapshot)

        monkeypatch.setattr(host.snapshot_store, "_on_drained", hold_drain)
        task = asyncio.create_task(AppRuntime._uninstall_plugin(cast(Any, app), "target@lab"))
        await waiting.wait()
        assert not task.done() and cache.is_dir()
        assert old.lease_count == 2 and target_state["closes"] == 0
        await lease.release()
        assert old.lease_count == 1 and not task.done() and cache.is_dir()
        await fork.release()
        await drain_started.wait()
        assert not task.done() and cache.is_dir()
        assert target_state["closes"] == peer_state["closes"] == 1
        assert target.module_path in sys.modules
        finish_drain.set()
        result = await task
        assert result["pluginId"] == "target@lab"
        assert old.lease_count == 0
        assert target_state["closes"] == peer_state["closes"] == 1
        assert target.scope.closed and target.module_path not in sys.modules
        assert "target@lab" not in host.current_snapshot.generations
        assert host.current_snapshot.composition_root.context.require(ServiceKey("peer.state")) is not peer_state
        assert not cache.exists() and marker.read_text() == "user data"
        assert "target@lab" not in host._draining_generations
    finally:
        for owned in (fork, lease):
            if owned is not None and owned.active:
                await owned.release()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finish_drain.set()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_cancelled_uninstall_rejoins_the_same_running_drain(tmp_path, monkeypatch):
    """调用者取消不删除 cache；重试只等待已经登记的原 snapshot drain。"""
    host, app, cache = installed_app(tmp_path)
    first = None
    finish_drain = asyncio.Event()
    drain_started = asyncio.Event()
    try:
        await host.load_all()
        old = host.current_snapshot
        target = old.generations["target@lab"]
        original_drain = host.snapshot_store._on_drained

        async def hold_drain(snapshot):
            if snapshot is old:
                drain_started.set()
                await finish_drain.wait()
            await original_drain(snapshot)

        monkeypatch.setattr(host.snapshot_store, "_on_drained", hold_drain)
        first = asyncio.create_task(AppRuntime._uninstall_plugin(cast(Any, app), "target@lab"))
        await drain_started.wait()
        operation = host._operation
        assert operation is not None
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.wait((operation.task,))
        assert cache.is_dir() and target.module_path in sys.modules
        retry_waiting = asyncio.Event()
        original_wait = host.snapshot_store.wait_for_snapshot_drained

        async def wait_for_retired(snapshot):
            if snapshot is old:
                retry_waiting.set()
            await original_wait(snapshot)

        monkeypatch.setattr(host.snapshot_store, "wait_for_snapshot_drained", wait_for_retired)
        retry = asyncio.create_task(AppRuntime._uninstall_plugin(cast(Any, app), "target@lab"))
        await retry_waiting.wait()
        assert not retry.done() and cache.is_dir()
        finish_drain.set()
        result = await retry
        assert result["pluginId"] == "target@lab"
        assert not cache.exists() and target.scope.closed
        assert host.current_snapshot.accepting_leases
    finally:
        finish_drain.set()
        if first is not None and not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        await host.terminate_all()


@pytest.mark.asyncio
async def test_uninstall_resource_close_failure_preserves_cache_and_owner(tmp_path):
    host, app, cache = installed_app(tmp_path, fail_close=True)
    try:
        await host.load_all()
        old = host.current_snapshot
        target = old.generations["target@lab"]
        state = old.composition_root.context.require(ServiceKey("target.state"))
        marker = target.data_dir / "keep.txt"
        marker.write_text("user data")
        with pytest.raises(Exception):
            await AppRuntime._uninstall_plugin(cast(Any, app), "target@lab")
        assert state["closes"] == 1
        assert old.snapshot_id in host.snapshot_store.retained_snapshot_ids
        assert target.module_path in sys.modules
        assert cache.is_dir() and marker.read_text() == "user data"
    finally:
        await host.terminate_all()
