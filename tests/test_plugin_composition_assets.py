from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugin_composition import (
    PLUGIN_ASSETS,
    CompositionRoot,
    PluginAssets,
    PluginRuntime,
)
from agent.plugins.dashboard_host import PluginDashboardHost
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id="assets",
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=object(),
    )


@pytest.mark.asyncio
async def test_v3_assets_compile_into_skill_and_dashboard_hosts(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "assets",
        "from agent.plugin_composition import PLUGIN_ASSETS\n"
        "api_version = 3\n"
        "name = 'assets'\n"
        "version = '1.0.0'\n"
        "inject = (PLUGIN_ASSETS,)\n"
        "async def apply(ctx, config):\n"
        "    assets = ctx.require(PLUGIN_ASSETS)\n"
        "    await assets.register_skill(ctx, 'skills')\n"
        "    await assets.register_dashboard(ctx, 'dashboard.py')\n",
    )
    skill_dir = plugin_dir / "skills" / "asset-probe"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: asset-probe\ndescription: probe\n---\nbody\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace):\n"
        "    @app.get('/api/assets')\n"
        "    def read_assets():\n"
        "        return {'plugin': plugin_dir.name, 'workspace': workspace.name}\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager.load_all()

    generation = manager.generation("assets")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert generation.contributions.skill_roots == (plugin_dir / "skills",)
    assert generation.contributions.dashboard_module == plugin_dir / "dashboard.py"
    assert snapshot.plugin_skill_index is not None
    assert "asset-probe" in snapshot.plugin_skill_index.records
    active = manager.active_plugins()
    assert active[0].skill_roots == (plugin_dir / "skills",)
    dashboard = PluginDashboardHost(
        workspace=workspace,
        memory_admin=object(),
        memory_store=object(),
        core_routes=(),
    )
    dashboard.prepare_snapshot(snapshot)
    assert len(snapshot.dashboard_bindings) == 1
    assert snapshot.dashboard_bindings[0].plugin_id == "assets"
    root = snapshot.composition_root
    assert root is not None
    assert "assets:asset:skill:skills" in root.receipt().effects
    assert "assets:asset:dashboard:dashboard.py" in root.receipt().effects

    await manager.terminate_all()

    assert root.receipt().services == ()
    assert root.receipt().effects == ()


@pytest.mark.asyncio
async def test_plugin_assets_reject_symlink_escape(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    outside = tmp_path / "outside"
    plugin_dir.mkdir()
    outside.mkdir()
    (plugin_dir / "skills").symlink_to(outside, target_is_directory=True)
    root = CompositionRoot("assets-escape")
    assets = PluginAssets()
    _ = await root.context.provide(PLUGIN_ASSETS, assets)

    async def plugin(ctx) -> None:
        service = ctx.require(PLUGIN_ASSETS)
        await service.register_skill(ctx, "skills")

    _ = await root.mount(
        plugin,
        name="assets",
        inject=(PLUGIN_ASSETS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("插件资产路径越界" in error for error in receipt.errors)
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_assets_roll_back_duplicate_dashboard(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "first.py").write_text("", encoding="utf-8")
    (plugin_dir / "second.py").write_text("", encoding="utf-8")
    root = CompositionRoot("assets-duplicate")
    assets = PluginAssets()
    _ = await root.context.provide(PLUGIN_ASSETS, assets)

    async def plugin(ctx) -> None:
        service = ctx.require(PLUGIN_ASSETS)
        await service.register_dashboard(ctx, "first.py")
        await service.register_dashboard(ctx, "second.py")

    _ = await root.mount(
        plugin,
        name="assets",
        inject=(PLUGIN_ASSETS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("只能声明一个 Dashboard" in error for error in receipt.errors)
    assert dict(assets.freeze()) == {}
    await root.dispose()
