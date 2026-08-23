from pathlib import Path

import pytest

from agent.plugin_composition import (
    BACKGROUND_JOBS,
    BackgroundJobDefinition,
    CompositionRoot,
    IntervalTrigger,
    PluginBackgroundJobs,
    PluginRuntime,
)
from agent.plugins.generation import GateResult, PluginContributions, PluginGeneration
from agent.plugins.generation_activity_host import ActivityHost
from agent.plugins.generation_job_host import BackgroundJobActivityAdapter
from agent.plugins.manager import PluginManager
from agent.plugins.scope import PluginScope
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
from bus.event_bus import EventBus


def _generation(plugin_dir: Path) -> PluginGeneration:
    return PluginGeneration(
        plugin_id="emotion",
        generation_id="emotion:test",
        module_path="plugins.emotion",
        source_revision="source",
        config_revision="config",
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        config=None,
        instance=object(),
        scope=PluginScope("emotion"),
        contributions=PluginContributions(manifest={}),
        gate_result=GateResult(
            gate_id="gate",
            plugin_id="emotion",
            candidate_revision="source",
            status="passed",
            checks=(),
        ),
    )


@pytest.mark.asyncio
async def test_snapshot_freezes_exact_background_job_catalog(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "emotion"
    plugin_dir.mkdir()
    root = CompositionRoot("emotion:test")
    jobs = PluginBackgroundJobs(root.instance_token)
    _ = await root.context.provide(BACKGROUND_JOBS, jobs)

    async def apply(ctx) -> None:
        await ctx.require(BACKGROUND_JOBS).register(
            ctx,
            BackgroundJobDefinition(
                name="merge_pending",
                triggers=(IntervalTrigger(60),),
                handler_export="runtime.merge_pending",
            ),
        )

    _ = await root.mount(
        apply,
        name="emotion",
        inject=(BACKGROUND_JOBS,),
        runtime=PluginRuntime(
            plugin_id="emotion",
            plugin_dir=plugin_dir,
            data_dir=plugin_dir / "data",
            workspace=plugin_dir / "workspace",
            config=None,
        ),
    )
    generation = _generation(plugin_dir)
    snapshot = RuntimeSnapshotCompiler().compile(
        {generation.plugin_id: generation},
        composition_root=root,
    )

    catalog = snapshot.background_job_catalog
    assert catalog is not None
    binding = catalog.job("merge_pending")
    assert binding is not None
    assert binding.generation_id == generation.generation_id
    assert snapshot.background_job_catalog_identity == catalog.identity
    assert catalog.root_instance_token is root.instance_token
    store = RuntimeSnapshotStore()
    store.install(snapshot)
    await store.close()


@pytest.mark.asyncio
async def test_manager_provides_and_compiles_background_job_service(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "emotion"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import (\n"
        "    BACKGROUND_JOBS, BackgroundJobDefinition, IntervalTrigger,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'emotion'\n"
        "version = '1.0.0'\n"
        "inject = (BACKGROUND_JOBS,)\n"
        "async def merge_pending(ctx):\n"
        "    return None\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(BACKGROUND_JOBS).register(ctx, BackgroundJobDefinition(\n"
        "        name='merge_pending',\n"
        "        triggers=(IntervalTrigger(60),),\n"
        "        handler_export='merge_pending',\n"
        "    ))\n",
        encoding="utf-8",
    )
    event_bus = EventBus()
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    manager.bind_activity_host(
        ActivityHost(
            (
                BackgroundJobActivityAdapter(
                    manager.snapshot_store,
                    workspace=str(workspace),
                ),
            )
        )
    )

    await manager.load_all()

    snapshot = manager.current_snapshot
    assert snapshot is not None
    catalog = snapshot.background_job_catalog
    assert catalog is not None
    binding = catalog.job("merge_pending")
    generation = manager.generation("emotion")
    assert binding is not None and generation is not None
    assert binding.generation_id == generation.generation_id
    await manager.terminate_all()
