from dataclasses import replace

import pytest

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.model import PluginRuntime
from agent.plugin_composition.overlay import CompositionOverlay
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    get_current_runtime_snapshot,
    lease_runtime_snapshot,
)


@pytest.mark.asyncio
async def test_selected_plugin_forks_overlay_but_replaced_plugin_cannot_capture_it(
    tmp_path,
):
    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    contexts = {}

    async def mounted(ctx):
        contexts[ctx.runtime.generation_id, ctx.runtime.plugin_id] = ctx

    runtime = PluginRuntime("kept", "stable", tmp_path, tmp_path, tmp_path, {})
    store = RuntimeSnapshotStore()
    try:
        await stable.mount(mounted, name="kept", runtime=runtime)
        await stable.mount(
            mounted, name="changed", runtime=replace(runtime, plugin_id="changed")
        )
        await candidate.mount(
            mounted,
            name="changed",
            runtime=replace(
                runtime,
                plugin_id="changed",
                generation_id="candidate",
            ),
        )
        overlay = CompositionOverlay(
            stable,
            candidate,
            plugin_ids=frozenset({"kept", "changed"}),
            replaced_plugin_ids=frozenset({"changed"}),
        )
        snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=overlay)
        store.install(snapshot)
        async with lease_runtime_snapshot(store):
            for selected in [
                contexts["stable", "kept"],
                contexts["candidate", "changed"],
            ]:
                async with selected.runtime_scope():
                    assert get_current_runtime_snapshot() is snapshot
                async with selected.capture_runtime_scope():
                    assert get_current_runtime_snapshot() is snapshot
            with pytest.raises(RuntimeError, match="未绑定"):
                contexts["stable", "changed"].capture_runtime_scope()
    finally:
        await store.close()
        await candidate.dispose()
        await stable.dispose()
