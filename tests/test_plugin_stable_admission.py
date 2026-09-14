"""闭接纳初始化不能给后台工作传递提交前的运行许可。"""
import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.selection import PluginSelection, SelectionWriteError
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_commit", [False, True])
async def test_started_background_work_waits_for_durable_commit(tmp_path, monkeypatch, fail_commit):
    source = tmp_path / "plugins" / "startup"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text('''import asyncio
from agent.plugin_composition import RUNTIME_STARTED
name = "startup"
version = "1.0"
api_version = 3
effects = []
queued = asyncio.Event()
acted = asyncio.Event()
async def apply(ctx):
    async def work():
        queued.set()
        async with ctx.runtime_scope():
            effects.append("accepted work")
            acted.set()
    async def start(event):
        await ctx.spawn(work(), name="startup-work")
        await queued.wait()
    await ctx.on(RUNTIME_STARTED, start)
''')
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    host = PluginManager([source.parent], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=tmp_path / "home" / "cache")
    commit = host._selection.commit
    modules = []

    def inspect_commit(components, *, expected_ref):
        snapshot = host._publication.candidate
        module = snapshot.generations["startup"].instance.module
        modules.append(module)
        assert module.queued.is_set()
        assert module.effects == []
        assert not snapshot.accepting_leases
        if fail_commit:
            raise SelectionWriteError(operation="commit", target_ref=None, outcome="unchanged",
                                      observed_ref=expected_ref, observation_error=None)
        return commit(components, expected_ref=expected_ref)

    monkeypatch.setattr(host._selection, "commit", inspect_commit)
    try:
        if fail_commit:
            with pytest.raises(SelectionWriteError):
                await host.load_all()
            assert PluginSelection(workspace).read() is None
            assert modules[0].effects == [] and not modules[0].acted.is_set()
        else:
            await host.load_all()
            await modules[0].acted.wait()
            assert PluginSelection(workspace).read() is not None
            assert modules[0].effects == ["accepted work"]
    finally:
        await host.terminate_all()
