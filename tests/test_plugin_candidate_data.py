"""从真实已安装候选观察隔离 SQLite 数据，正式库始终保持原提交。"""
from contextlib import closing
import sqlite3

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import ServiceKey
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.test_plugin_install import _commit, _write_v3_plugin


@pytest.mark.asyncio
async def test_candidate_rebuilds_even_an_unrelated_plugin_in_its_own_root(tmp_path):
    """完整候选不借用正式插件实例，候选状态变化不会串到 stable。"""
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
        stable = host.current_snapshot
        old_peer = stable.composition_root.context.require(ServiceKey("peer.state"))
        old_peer.append("stable")
        source = tmp_path / "changed"
        (source / "plugin.py").write_text(module.replace("NAME", "changed") + "\nrevision = 2\n")
        _commit(source)
        _, status = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
        assert status["candidate_state"] == "latest_ready"
        candidate = host.latest_snapshot
        new_peer = candidate.composition_root.context.require(ServiceKey("peer.state"))
        assert new_peer is not old_peer
        assert new_peer == []
        new_peer.append("candidate")
        assert old_peer == ["stable"]
        await host.drop_candidate("changed@lab")
        assert host.current_snapshot is stable
        assert old_peer == ["stable"]
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_plugin_initializes_candidate_data_without_copying_formal_database(tmp_path, directory):
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
            _, status = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
            assert status["candidate_state"] == "latest_ready"
            probe = host.latest_snapshot.composition_root.context.require(ServiceKey("data.probe"))
            version, value, candidate_path = probe()
            assert (version, value) == ("new", "prepared by plugin")
            assert candidate_path != path
            if directory:
                for suffix in ("-wal", "-shm"):
                    assert not (candidate_path.parent / ("report" + suffix)).exists()
            assert probe("candidate only")[1] == "candidate only"
            assert writer.execute("SELECT value FROM records").fetchone()[0] == "latest committed"
            await host.drop_candidate("probe@lab")
            assert writer.execute("SELECT value FROM records").fetchone()[0] == "latest committed"
        finally:
            await host.terminate_all()



@pytest.mark.asyncio
async def test_locked_formal_data_does_not_block_candidate(tmp_path):
    """底座不打开插件业务库，因此它的独占锁不成为候选装配前提。"""
    from tests.test_plugin_update_operation_lease import installed_host

    async with installed_host(tmp_path) as (host, source, workspace, _):
        path = workspace / "plugin-data/target-lab/locked.sqlite3"
        with closing(sqlite3.connect(path)) as database:
            database.execute("CREATE TABLE records(value TEXT)")
            database.execute("INSERT INTO records VALUES ('original')")
            database.commit()
            database.execute("BEGIN EXCLUSIVE")
            result, _ = await host.install_candidate(source=str(source), marketplace="lab", ref_name="", sparse_paths=[])
            assert host.read_update(result.update_id).ready
            assert not (host.latest_snapshot.generations["target@lab"].data_dir / "locked.sqlite3").exists()
            await host.discard_update(result.update_id)
            assert database.execute("SELECT value FROM records").fetchall() == [("original",)]
            database.rollback()
