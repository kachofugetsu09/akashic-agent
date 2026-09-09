"""从真实已安装候选观察隔离 SQLite 数据，正式库始终保持原提交。"""
from contextlib import closing
import sqlite3

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.test_plugin_install import _commit, _write_v3_plugin


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_candidate_copies_committed_wal_and_writes_only_its_database(tmp_path, directory):
    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    relative = "shared/registry.sqlite3" if directory else "registry.sqlite3"
    declaration = 'workspace_roots = ("shared",)' if directory else 'workspace_files = ("registry.sqlite3",)'
    module = '''import sqlite3
from contextlib import closing
from agent.plugin_composition import ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = ()
DECLARATION
async def apply(ctx, config):
    path = ctx.runtime.workspace / "RELATIVE"
    def probe(value=None):
        with closing(sqlite3.connect(path)) as db:
            if value is not None:
                db.execute("UPDATE records SET value=?", (value,))
                db.commit()
            return "old", db.execute("SELECT value FROM records").fetchone()[0], path
    await ctx.provide(ServiceKey("data.probe"), probe)
'''.replace("DECLARATION", declaration).replace("RELATIVE", relative)
    _write_v3_plugin(source, name="probe", module_source=module)
    _commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        for suffix in ("-wal", "-shm"):
            (path.parent / ("report" + suffix)).write_text("ordinary product file")
    with closing(sqlite3.connect(path)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE records(value TEXT NOT NULL)")
        writer.execute("INSERT INTO records VALUES ('latest committed')")
        writer.commit()
        assert path.with_name(path.name + "-wal").stat().st_size > 0
        host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
        try:
            await host.load_all()
            (source / "plugin.py").write_text(module.replace('return "old"', 'return "new"'))
            _commit(source)
            _, status = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
            assert status["candidate_state"] == "latest_ready"
            probe = host.latest_snapshot.composition_root.context.require(ServiceKey("data.probe"))
            version, value, candidate_path = probe()
            assert (version, value) == ("new", "latest committed")
            assert candidate_path != path
            if directory:
                for suffix in ("-wal", "-shm"):
                    assert (candidate_path.parent / ("report" + suffix)).read_text() == "ordinary product file"
            assert probe("candidate only")[1] == "candidate only"
            assert writer.execute("SELECT value FROM records").fetchone()[0] == "latest committed"
            await host.drop_candidate("probe@lab")
            assert writer.execute("SELECT value FROM records").fetchone()[0] == "latest committed"
        finally:
            await host.terminate_all()


@pytest.mark.asyncio
async def test_locked_candidate_database_fails_without_changing_stable(tmp_path, monkeypatch):
    """真实独占锁必须拒绝候选并清理副本，不能挂住 Core 或改写原库。"""
    import agent.plugins.manager as manager

    source, workspace, home = (tmp_path / name for name in ("source", "workspace", "home"))
    from tests.test_plugin_business_validation import MODULE
    _write_v3_plugin(source, name="probe", module_source=MODULE)
    _commit(source)
    install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        stable = host.current_snapshot
        path = workspace / "plugin-data/probe-lab/locked.sqlite3"
        with closing(sqlite3.connect(path)) as writer:
            writer.execute("CREATE TABLE records(value TEXT)")
            writer.execute("INSERT INTO records VALUES ('original')")
            writer.commit()
            writer.execute("BEGIN EXCLUSIVE")
            (source / "plugin.py").write_text((source / "plugin.py").read_text() + "\nmarker = 'new'\n")
            _commit(source)
            monkeypatch.setattr(manager, "_SQLITE_BACKUP_LOCK_TIMEOUT_SECONDS", 0.0)
            with pytest.raises(RuntimeError, match="SQLite.*等待锁超时"):
                await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="locked-copy")
            assert host.current_snapshot is stable
            assert host.read_update("locked-copy").phase == "rolled_back"
            assert not list((workspace / "runtime/plugin-validation").rglob("locked.sqlite3"))
            assert writer.execute("SELECT value FROM records").fetchall() == [("original",)]
            writer.rollback()
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["install", "validation"])
async def test_candidate_copy_keeps_loop_live_and_finishes_before_cancel_cleanup(tmp_path, monkeypatch, phase):
    """复制线程未退出时不释放 scope 或删除目录，且事件循环仍可处理其他工作。"""
    import asyncio
    import threading
    import agent.plugins.manager as manager
    from tests.test_plugin_business_validation import prepare, MODULE

    source, workspace, _, log, host = prepare(tmp_path)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    copied = []
    real_copy = manager._copy_validation_tree

    def blocked_copy(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(10):
            raise TimeoutError("test copy was not released by the event loop")
        value = real_copy(*args, **kwargs)
        copied.append(args[1])
        return value

    try:
        await host.load_all()
        stable = host.current_snapshot
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'new'\n")
        _commit(source)
        if phase == "validation":
            result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])

            async def run():
                async with host.open_validation(result.update_id):
                    pytest.fail("cancelled validation entered its body")
        else:
            async def run():
                await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        monkeypatch.setattr(manager, "_copy_validation_tree", blocked_copy)
        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(entered.wait(), 3)
            # 必须在 worker 仍受阻时执行主循环回调，再取消外层调用。
            responsive = loop.create_future()
            loop.call_soon(responsive.set_result, True)
            assert await responsive
            task.cancel()
            cancelled = loop.create_future()
            loop.call_soon(cancelled.set_result, True)
            await cancelled
            assert not task.done()
            assert not copied
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert copied
        assert all(not path.exists() for path in copied)
        assert host.current_snapshot is stable
        assert host._validation_hosts == {}
        assert host.latest_snapshot.lease_count == 0
    finally:
        release.set()
        await host.terminate_all()
        log.close()
