from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugins.composable import ComposablePlugin
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.registry import plugin_registry
from bus.event_bus import EventBus


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


@pytest.mark.asyncio
async def test_v3_namespace_loader_waits_for_service_not_scan_order(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "a_consumer",
        "from pydantic import BaseModel\n"
        "from agent.plugin_composition import PLUGIN_ASSETS, ServiceKey\n"
        "api_version = 3\n"
        "name = 'a_consumer'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "inject = (VALUE, PLUGIN_ASSETS)\n"
        "observed = None\n"
        "disposed = False\n"
        "class Config(BaseModel):\n"
        "    suffix: str = 'default'\n"
        "async def apply(ctx, config):\n"
        "    global observed, disposed\n"
        "    observed = (ctx.require(VALUE), ctx.runtime.plugin_id, "
        "ctx.runtime.workspace.name, config.suffix)\n"
        "    await ctx.require(PLUGIN_ASSETS).register_skill(ctx, 'skills')\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label='consumer')\n",
    )
    consumer_skill = tmp_path / "plugins" / "a_consumer" / "skills" / "consumer-probe"
    consumer_skill.mkdir(parents=True)
    (consumer_skill / "SKILL.md").write_text(
        "---\nname: consumer-probe\ndescription: probe\n---\nbody\n",
        encoding="utf-8",
    )
    _write_plugin(
        tmp_path / "plugins",
        "z_provider",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'z_provider'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "async def apply(ctx, config):\n"
        "    await ctx.provide(VALUE, 'ready')\n",
    )
    config_dir = tmp_path / "workspace" / "plugin-data" / "a_consumer-builtin"
    config_dir.mkdir(parents=True)
    (config_dir / "config.local.toml").write_text(
        "suffix = 'configured'\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    consumer = manager.generation("a_consumer")
    snapshot = manager.current_snapshot
    assert consumer is not None and snapshot is not None
    assert isinstance(consumer.instance, ComposablePlugin)
    assert consumer.instance.module.observed == (
        "ready",
        "a_consumer",
        "workspace",
        "configured",
    )
    assert snapshot.composition_root is not None
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.services == (
        "core.plugin_assets",
        "fixture.value",
    )
    assert consumer.contributions.skill_roots == (
        tmp_path / "plugins" / "a_consumer" / "skills",
    )
    assert (
        next(
            plugin
            for plugin in manager.active_plugins()
            if plugin.plugin_id == "a_consumer"
        ).skill_roots
        == consumer.contributions.skill_roots
    )
    assert tuple(item.name for item in snapshot.composition_topology.fibers) == (
        "a_consumer",
        "z_provider",
    )

    await manager.terminate_all()

    assert consumer.instance.module.disposed is True


@pytest.mark.asyncio
async def test_v3_loader_fails_loud_when_required_service_never_appears(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "waiting",
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'waiting'\n"
        "version = '1.0.0'\n"
        "inject = (ServiceKey('never.provided'),)\n"
        "def apply(ctx, config):\n"
        "    raise AssertionError('pending plugin must not apply')\n",
    )
    manager = _manager(tmp_path)

    with pytest.raises(RuntimeError, match="never.provided"):
        await manager.load_all()

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_reload_keeps_old_root_until_snapshot_lease_drains(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "reloadable",
        "from agent.plugin_composition import PLUGIN_ASSETS\n"
        "api_version = 3\n"
        "name = 'reloadable'\n"
        "version = '1.0.0'\n"
        "inject = (PLUGIN_ASSETS,)\n"
        "marker = 'old'\n"
        "disposed = False\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(PLUGIN_ASSETS).register_skill(ctx, 'skills')\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label=marker)\n",
    )
    old_skill = plugin_dir / "skills" / "reload-old"
    old_skill.mkdir(parents=True)
    (old_skill / "SKILL.md").write_text(
        "---\nname: reload-old\ndescription: old\n---\nold\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_generation = manager.generation("reloadable")
    old_snapshot = manager.current_snapshot
    assert old_generation is not None and old_snapshot is not None
    assert old_snapshot.plugin_skill_index is not None
    assert set(old_snapshot.plugin_skill_index.records) == {"reload-old"}
    lease = manager._snapshot_store.lease()

    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import PLUGIN_ASSETS\n"
        "api_version = 3\n"
        "name = 'reloadable'\n"
        "version = '1.0.0'\n"
        "inject = (PLUGIN_ASSETS,)\n"
        "marker = 'new'\n"
        "disposed = False\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(PLUGIN_ASSETS).register_skill(ctx, 'skills')\n"
        "    def cleanup():\n"
        "        global disposed\n"
        "        disposed = True\n"
        "    await ctx.effect(lambda: cleanup, label=marker)\n",
        encoding="utf-8",
    )
    new_skill = old_skill.with_name("reload-new")
    old_skill.rename(new_skill)
    (new_skill / "SKILL.md").write_text(
        "---\nname: reload-new\ndescription: new\n---\nnew\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("reloadable")
    assert candidate is not None

    result = await manager.publish_prepared("reloadable")

    assert result["publication_state"] == "committed"
    assert manager.current_snapshot is not old_snapshot
    assert manager.current_snapshot is not None
    assert manager.current_snapshot.plugin_skill_index is not None
    assert set(manager.current_snapshot.plugin_skill_index.records) == {"reload-new"}
    assert set(old_snapshot.plugin_skill_index.records) == {"reload-old"}
    assert old_generation.instance.module.disposed is False
    await lease.release()
    await manager._snapshot_store.retry_drains()
    assert old_generation.instance.module.disposed is True

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_installed_v3_candidate_rebuilds_runtime_then_promotes(
    tmp_path: Path,
) -> None:
    plugin_base = tmp_path / "home" / "cache" / "lab" / "installed_v3"
    stable_root = plugin_base / ".artifacts" / "1.0.0-aaaa"
    latest_root = plugin_base / ".artifacts" / "2.0.0-bbbb"
    stable_root.mkdir(parents=True)
    latest_root.mkdir(parents=True)
    source = (
        "from agent.plugin_composition import PLUGIN_ASSETS\n"
        "api_version = 3\n"
        "name = 'installed_v3'\n"
        "version = '1.0.0'\n"
        "inject = (PLUGIN_ASSETS,)\n"
        "applied = []\n"
        "disposed = []\n"
        "async def apply(ctx, config):\n"
        "    workspace = str(ctx.runtime.workspace)\n"
        "    applied.append(workspace)\n"
        "    assets = ctx.require(PLUGIN_ASSETS)\n"
        "    await assets.register_skill(ctx, 'skills')\n"
        "    await assets.register_dashboard(ctx, 'dashboard.py')\n"
        "    def cleanup():\n"
        "        disposed.append(workspace)\n"
        "    await ctx.effect(lambda: cleanup, label='runtime')\n"
    )
    (stable_root / "plugin.py").write_text(source, encoding="utf-8")
    (latest_root / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    for root in (stable_root, latest_root):
        skill_dir = root / "skills" / "installed-v3"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: installed-v3\ndescription: probe\n---\nbody\n",
            encoding="utf-8",
        )
        (root / "dashboard.py").write_text(
            "def register(app, plugin_dir, workspace):\n"
            "    @app.get('/api/installed-v3')\n"
            "    def probe(): return {'ok': True}\n",
            encoding="utf-8",
        )
    stable_pointer = ArtifactPointer(".artifacts/1.0.0-aaaa")
    latest_pointer = ArtifactPointer(".artifacts/2.0.0-bbbb")
    write_pointers(plugin_base, stable=stable_pointer, latest=stable_pointer)
    write_plugin_manifest(
        {"installed_v3@lab": True},
        plugins_home=tmp_path / "home",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.generation("installed_v3@lab")
    assert stable is not None
    stable_lease = manager.snapshot_store.lease()

    write_pointers(plugin_base, stable=stable_pointer, latest=latest_pointer)
    result = (await manager.reconcile_changed())[0]
    candidate = manager.ready_candidate

    assert result["publication_state"] == "latest_ready"
    assert candidate is not None
    assert "plugin-validation" in candidate.instance.module.applied[-1]
    promoted = await manager.switch_ready("installed_v3@lab")

    assert promoted["publication_state"] == "promoted"
    assert candidate.instance.module.applied[-1] == str(tmp_path / "workspace")
    assert candidate.contributions.skill_roots == (latest_root / "skills",)
    assert candidate.contributions.dashboard_module == latest_root / "dashboard.py"
    assert candidate.skill_catalog is not None
    assert "installed-v3" in candidate.skill_catalog.names
    assert any(
        "plugin-validation" in workspace
        for workspace in candidate.instance.module.disposed
    )
    assert stable.instance.module.disposed == []
    await stable_lease.release()
    await manager.snapshot_store.retry_drains()
    assert stable.instance.module.disposed == [str(tmp_path / "workspace")]

    await manager.terminate_all()
