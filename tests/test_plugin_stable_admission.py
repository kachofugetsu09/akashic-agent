"""首次 selection 提交前不能运行插件代码。"""
import asyncio
import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.selection import PluginSelection, SelectionWriteError
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_commit", [False, True])
async def test_initial_start_waits_for_durable_selection_commit(tmp_path, monkeypatch, fail_commit):
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
    await ctx.on(RUNTIME_STARTED, start)
''')
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    host = PluginManager([source.parent], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=tmp_path / "home" / "cache")
    commit = host._selection.commit
    pending = []

    def inspect_commit(components, *, expected_ref):
        generation = host._active_generations["startup"]
        pending.append(generation)
        assert generation.instance is None
        assert generation.fiber is None
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
            assert pending[0].instance is None
        else:
            await host.load_all()
            generation = host.generation("startup")
            assert generation is not None and generation.instance is not None
            module = generation.instance.module
            await asyncio.wait_for(module.acted.wait(), 5)
            assert PluginSelection(workspace).read() is not None
            assert module.effects == ["accepted work"]
    finally:
        await host.terminate_all()
