"""局部更新保留同一 Root 和未受影响的 Fiber。"""
import pytest
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_install import _commit, _write_v3_plugin

MODULE = '''from agent.plugin_composition import ServiceKey, RUNTIME_STARTED
api_version = 3
name = "NAME"
version = "1.0.0"
STATE = {"mounts": 0, "starts": 0, "closes": 0}
async def apply(ctx):
    STATE["mounts"] += 1
    def started(_):
        STATE["starts"] += 1
    async def close():
        STATE["closes"] += 1
        if FAIL_CLOSE and STATE["closes"] == 1:
            raise OSError("connection still open")
    await ctx.effect(lambda: close)
    await ctx.provide(ServiceKey("NAME.state"), STATE)
    await ctx.on(RUNTIME_STARTED, started)
    if REJECT_FORMAL and (ctx.runtime.workspace / "reject-formal").exists():
        raise ValueError("formal rejected after acquisition")
'''

def module(name, *, reject=False, fail_close=False):
    return MODULE.replace("NAME", name).replace("REJECT_FORMAL", repr(reject)).replace("FAIL_CLOSE", repr(fail_close))

def installed_pair(tmp_path):
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    for name in ("changed", "peer"):
        source = tmp_path / name
        _write_v3_plugin(source, name=name, module_source=module(name))
        _commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    initialize_plugin_workspace(workspace)
    return PluginManager(
        [], event_bus=EventBus(), workspace=workspace, installed_cache_root=home / "cache",
    )

@pytest.mark.asyncio
async def test_live_update_replaces_only_target_fiber_in_one_root(tmp_path):
    """A local install replaces its target while the peer keeps its owner."""
    host = installed_pair(tmp_path)
    try:
        await host.load_all()
        root = host.live_root
        old = host.generation("changed@lab")
        peer = host.generation("peer@lab")
        assert root is not None and old is not None and old.fiber is not None
        assert peer is not None and peer.fiber is not None
        peer_fiber = peer.fiber
        source = tmp_path / "changed"
        (source / "plugin.py").write_text(module("changed") + "\nrevision = 2\n")
        _commit(source)

        accepted = await host.install(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
            update_id="fresh-target",
        )
        assert accepted.state == "accepted"
        operation = host._operation
        assert operation is not None
        await operation.task

        current = host.generation("changed@lab")
        assert current is not None and current is not old
        assert current.fiber is not None and current.fiber is not old.fiber
        assert host.live_root is root
        assert host.generation("peer@lab") is peer
        assert peer.fiber is peer_fiber
        assert old.instance.module.STATE["closes"] == 1
        assert current.instance.module.STATE["starts"] == 1
    finally:
        await host.terminate_all()
