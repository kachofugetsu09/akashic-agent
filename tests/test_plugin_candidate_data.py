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
