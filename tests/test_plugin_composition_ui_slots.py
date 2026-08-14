from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path

import pytest

from agent.plugin_composition import (
    UI_SLOTS,
    CompositionError,
    CompositionRoot,
    Context,
    PluginRuntime,
    PluginUiSlots,
)
from agent.plugins.dashboard_host import PluginDashboardHost
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus


class _LeakedPluginUiSlots(PluginUiSlots):
    def _register(
        self,
        plugin_id: str,
        path: Path,
    ) -> Callable[[], None]:
        _ = super()._register(plugin_id, path)
        return lambda: None


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=object(),
    )


@pytest.mark.asyncio
async def test_v3_ui_slot_compiles_into_dashboard_host(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "ui_probe",
        "from agent.plugin_composition import UI_SLOTS\n"
        "api_version = 3\n"
        "name = 'ui_probe'\n"
        "version = '1.0.0'\n"
        "inject = (UI_SLOTS,)\n"
        "async def apply(ctx, config):\n"
        "    slots = ctx.require(UI_SLOTS)\n"
        "    await slots.register_dashboard(ctx, 'dashboard.py')\n",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace):\n"
        "    @app.get('/api/ui-probe')\n"
        "    def read_probe():\n"
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

    generation = manager.generation("ui_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert generation.contributions.dashboard_module == plugin_dir / "dashboard.py"
    dashboard = PluginDashboardHost(
        workspace=workspace,
        memory_admin=object(),
        memory_store=object(),
        core_routes=(),
    )
    dashboard.prepare_snapshot(snapshot)
    assert len(snapshot.dashboard_bindings) == 1
    assert snapshot.dashboard_bindings[0].plugin_id == "ui_probe"
    root = snapshot.composition_root
    assert root is not None
    assert "ui_probe:ui-slot:dashboard:dashboard.py" in root.receipt().effects

    await manager.terminate_all()

    assert root.receipt().services == ()
    assert root.receipt().effects == ()


@pytest.mark.asyncio
async def test_plugin_ui_slots_freezes_dashboard_declaration(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "ui_freeze"
    plugin_dir.mkdir()
    dashboard = plugin_dir / "dashboard.py"
    dashboard.write_text("", encoding="utf-8")
    root = CompositionRoot("ui-slots-freeze")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)
    plugin_ctx: Context | None = None

    async def plugin(ctx: Context) -> None:
        nonlocal plugin_ctx
        plugin_ctx = ctx
        await ctx.require(UI_SLOTS).register_dashboard(ctx, "dashboard.py")

    _ = await root.mount(
        plugin,
        name="ui-freeze",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    assert plugin_ctx is not None

    frozen = slots.freeze()

    assert frozen["ui_freeze"].dashboard_module == dashboard
    with pytest.raises(CompositionError) as caught:
        await slots.register_dashboard(plugin_ctx, "dashboard.py")
    assert caught.value.code == "PLUGIN_UI_SLOTS_FROZEN"
    await root.dispose()
    assert root.receipt().effects == ()


@pytest.mark.asyncio
async def test_plugin_ui_slots_rejects_symlink_escape(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "ui_escape"
    plugin_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    (plugin_dir / "dashboard.py").symlink_to(outside)
    root = CompositionRoot("ui-slots-escape")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def plugin(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_dashboard(ctx, "dashboard.py")

    _ = await root.mount(
        plugin,
        name="ui-escape",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("Dashboard 路径越界" in error for error in receipt.errors)
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_ui_slots_rolls_back_duplicate_dashboard(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "ui_duplicate"
    plugin_dir.mkdir()
    (plugin_dir / "first.py").write_text("", encoding="utf-8")
    (plugin_dir / "second.py").write_text("", encoding="utf-8")
    root = CompositionRoot("ui-slots-duplicate")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def plugin(ctx: Context) -> None:
        service = ctx.require(UI_SLOTS)
        await service.register_dashboard(ctx, "first.py")
        await service.register_dashboard(ctx, "second.py")

    _ = await root.mount(
        plugin,
        name="ui-duplicate",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("只能声明一个 Dashboard" in error for error in receipt.errors)
    assert dict(slots.freeze()) == {}
    await root.dispose()


@pytest.mark.asyncio
async def test_plugin_ui_slots_oracle_kills_leaked_registration_mutant(
    tmp_path: Path,
) -> None:
    correct = await _disposed_registration_fixture(tmp_path / "correct", PluginUiSlots)
    mutant = await _disposed_registration_fixture(
        tmp_path / "mutant",
        _LeakedPluginUiSlots,
    )

    assert correct is False
    assert mutant is True


async def _disposed_registration_fixture(
    plugin_dir: Path,
    service_type: type[PluginUiSlots],
) -> bool:
    """Dispose one owner Fiber before freezing its remaining declarations."""

    plugin_dir.mkdir(parents=True)
    (plugin_dir / "dashboard.py").write_text("", encoding="utf-8")
    root = CompositionRoot(f"ui-slots-dispose:{service_type.__name__}")
    slots = service_type()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def plugin(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_dashboard(ctx, "dashboard.py")

    fiber = await root.mount(
        plugin,
        name="ui-slot-owner",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    await fiber.dispose()
    leaked = bool(slots.freeze())
    await root.dispose()
    return leaked
