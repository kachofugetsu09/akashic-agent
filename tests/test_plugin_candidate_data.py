"""本地更新保留 live peer 与插件数据 owner。"""
from contextlib import closing
import sqlite3

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import ServiceKey
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.test_plugin_install import _commit, _write_v3_plugin


async def install_update(host, source, update_id):
    """Wait for the live install owner after its accepted selection receipt."""

    accepted = await host.install(
        source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        update_id=update_id,
    )
    assert accepted.selection == "selected"
    operation = host._operation
    assert operation is not None
    await operation.task


@pytest.mark.asyncio
async def test_local_update_keeps_unrelated_plugin_on_same_root(tmp_path):
    """只换目标 Fiber，无关 peer 保留原实例和状态。"""
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    module = '''from agent.plugin_composition import ServiceKey
api_version = 3
name = "NAME"
version = "1.0.0"
inject = ()
async def apply(ctx):
    await ctx.provide(ServiceKey("NAME.state"), [])
'''
    for name in ("changed", "peer"):
        source = tmp_path / name
        _write_v3_plugin(source, name=name, module_source=module.replace("NAME", name))
        _commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    initialize_plugin_workspace(workspace)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        old_peer = root.context.require(ServiceKey("peer.state"))
        old_peer.append("stable")
        old_changed = host.generation("changed@lab")
        source = tmp_path / "changed"
        (source / "plugin.py").write_text(module.replace("NAME", "changed") + "\nrevision = 2\n")
        _commit(source)
        await install_update(host, source, "replace-changed")
        assert host.live_root is root
        assert host.generation("changed@lab") is not old_changed
        assert root.context.require(ServiceKey("peer.state")) is old_peer
        assert old_peer == ["stable"]
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_plugin_update_uses_same_formal_database(tmp_path, directory):
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
async def apply(ctx):
    path = ctx.runtime.workspace / "RELATIVE"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE records(value TEXT NOT NULL)")
            db.execute("INSERT INTO records VALUES ('prepared by plugin')")
            db.commit()
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
        initialize_plugin_workspace(workspace)
        host = PluginManager([], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache")
        try:
            await host.load_all()
            (source / "plugin.py").write_text(module.replace('return "old"', 'return "new"'))
            _commit(source)
            await install_update(host, source, "replace-data")
            root = host.live_root
            assert root is not None
            probe = root.context.require(ServiceKey("data.probe"))
            version, value, live_path = probe()
            assert (version, value) == ("new", "latest committed")
            assert live_path == path
            if directory:
                for suffix in ("-wal", "-shm"):
                    assert (live_path.parent / ("report" + suffix)).exists()
            assert probe("live update")[1] == "live update"
            assert writer.execute("SELECT value FROM records").fetchone()[0] == "live update"
        finally:
            await host.terminate_all()



@pytest.mark.asyncio
async def test_locked_plugin_data_does_not_block_unrelated_local_update(tmp_path):
    """未声明为工作集的插件业务库不参与代码更新。"""
    from tests.test_plugin_update_operation_lease import installed_host

    async with installed_host(tmp_path) as (host, source, workspace, _):
        path = workspace / "plugin-data/target-lab/locked.sqlite3"
        with closing(sqlite3.connect(path)) as database:
            database.execute("CREATE TABLE records(value TEXT)")
            database.execute("INSERT INTO records VALUES ('original')")
            database.commit()
            database.execute("BEGIN EXCLUSIVE")
            await install_update(host, source, "replace-locked")
            generation = host.generation("target@lab")
            assert generation is not None
            assert (generation.data_dir / "locked.sqlite3") == path
            assert database.execute("SELECT value FROM records").fetchall() == [("original",)]
            database.rollback()
