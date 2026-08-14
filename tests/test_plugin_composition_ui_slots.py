from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    UI_SLOTS,
    CompositionError,
    CompositionRoot,
    Context,
    MobileUiDefinition,
    MobileUiNavigation,
    MobileUiQueryHandler,
    PluginRuntime,
    PluginUiSlots,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.dashboard_host import PluginDashboardHost
from agent.plugins.generation import MobileUiAsset
from agent.plugins.manager import PluginManager
from agent.plugins.mobile_ui import PluginMobileUiProvider
from bus.event_bus import EventBus


class _LeakedPluginUiSlots(PluginUiSlots):
    def _register_dashboard(
        self,
        plugin_id: str,
        path: Path,
    ) -> Callable[[], None]:
        _ = super()._register_dashboard(plugin_id, path)
        return lambda: None

    def _register_mobile(
        self,
        plugin_id: str,
        asset: MobileUiAsset,
        query: MobileUiQueryHandler,
        available: Callable[[], bool],
    ) -> Callable[[], None]:
        _ = super()._register_mobile(
            plugin_id,
            asset,
            query,
            available,
        )
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


def _mobile_definition(module: str = "mobile.js") -> MobileUiDefinition:
    return MobileUiDefinition(
        module=module,
        stylesheet="mobile.css",
        slots=("drawer.panel",),
    )


def _mobile_query(
    method: str,
    payload: dict[str, object],
    *,
    session_id: str | None,
    turn_id: str | None,
) -> dict[str, object]:
    return {"method": method, "payload": payload}


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
async def test_v3_mobile_ui_slot_compiles_into_generation_provider(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "mobile_probe",
        "from agent.plugin_composition import (\n"
        "    UI_SLOTS, MobileUiDefinition, MobileUiNavigation,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'mobile_probe'\n"
        "version = '1.0.0'\n"
        "inject = (UI_SLOTS,)\n"
        "available = True\n"
        "def query(method, payload, *, session_id, turn_id):\n"
        "    return {\n"
        "        'method': method, 'payload': payload,\n"
        "        'session_id': session_id, 'turn_id': turn_id,\n"
        "    }\n"
        "async def apply(ctx, config):\n"
        "    slots = ctx.require(UI_SLOTS)\n"
        "    await slots.register_mobile(\n"
        "        ctx,\n"
        "        MobileUiDefinition(\n"
        "            module='mobile.js',\n"
        "            stylesheet='mobile.css',\n"
        "            navigation=MobileUiNavigation(\n"
        "                label='Probe', description='Probe panel',\n"
        "            ),\n"
        "            slots=('drawer.panel',),\n"
        "        ),\n"
        "        query=query,\n"
        "        available=lambda: available,\n"
        "    )\n",
    )
    (plugin_dir / "mobile.js").write_text(
        "export const probe = true;\n",
        encoding="utf-8",
    )
    (plugin_dir / "mobile.css").write_text(
        ":host { display: block; }\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )

    await manager.load_all()

    generation = manager.generation("mobile_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    assert generation.contributions.mobile_ui_query is not None
    assert generation.contributions.mobile_ui_available is not None
    provider = PluginMobileUiProvider(manager)
    catalog = provider.catalog()
    item = cast(list[dict[str, object]], catalog["items"])[0]
    assert item["id"] == "mobile_probe"
    assert item["navigation"] == {
        "label": "Probe",
        "description": "Probe panel",
    }
    asset = provider.asset(
        "mobile_probe",
        generation.source_revision,
        "module",
        cast(str, item["module_sha256"]),
    )
    assert asset["content"] == "export const probe = true;\n"
    result = await provider.query(
        "mobile_probe",
        generation.source_revision,
        "probe.current",
        {"limit": 3},
        session_id="mobile:test",
        turn_id="turn-1",
    )
    assert result == {
        "method": "probe.current",
        "payload": {"limit": 3},
        "session_id": "mobile:test",
        "turn_id": "turn-1",
    }

    module = cast(Any, generation.instance.module)
    module.available = False
    assert provider.catalog()["items"] == []
    root = snapshot.composition_root
    assert root is not None
    assert "mobile_probe:ui-slot:mobile:mobile.js" in root.receipt().effects
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
async def test_plugin_ui_slots_rejects_mobile_asset_symlink_escape(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "mobile_escape"
    plugin_dir.mkdir()
    outside = tmp_path / "outside.js"
    outside.write_text("export default 1;", encoding="utf-8")
    (plugin_dir / "mobile.js").symlink_to(outside)
    (plugin_dir / "mobile.css").write_text("", encoding="utf-8")
    root = CompositionRoot("mobile-slots-escape")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def plugin(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_mobile(
            ctx,
            _mobile_definition(),
            query=_mobile_query,
        )

    _ = await root.mount(
        plugin,
        name="mobile-escape",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("mobile UI module 无效" in error for error in receipt.errors)
    await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("async_field", ("query", "available"))
async def test_plugin_ui_slots_rejects_async_mobile_handlers(
    tmp_path: Path,
    async_field: str,
) -> None:
    plugin_dir = tmp_path / f"mobile_async_{async_field}"
    plugin_dir.mkdir()
    (plugin_dir / "mobile.js").write_text("export default 1;", encoding="utf-8")
    (plugin_dir / "mobile.css").write_text("", encoding="utf-8")
    root = CompositionRoot(f"mobile-slots-async:{async_field}")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def async_handler(*args: object, **kwargs: object) -> object:
        return {}

    async def plugin(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_mobile(
            ctx,
            _mobile_definition(),
            query=(
                cast(Any, async_handler)
                if async_field == "query"
                else _mobile_query
            ),
            available=(
                cast(Any, async_handler)
                if async_field == "available"
                else None
            ),
        )

    _ = await root.mount(
        plugin,
        name=f"mobile-async-{async_field}",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("必须是同步函数" in error for error in receipt.errors)
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
async def test_plugin_ui_slots_rolls_back_duplicate_mobile_ui(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "mobile_duplicate"
    plugin_dir.mkdir()
    (plugin_dir / "mobile.js").write_text("export default 1;", encoding="utf-8")
    (plugin_dir / "mobile.css").write_text("", encoding="utf-8")
    root = CompositionRoot("mobile-slots-duplicate")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def plugin(ctx: Context) -> None:
        service = ctx.require(UI_SLOTS)
        await service.register_mobile(
            ctx,
            _mobile_definition(),
            query=_mobile_query,
        )
        await service.register_mobile(
            ctx,
            _mobile_definition(),
            query=_mobile_query,
        )

    _ = await root.mount(
        plugin,
        name="mobile-duplicate",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )

    receipt = root.receipt()
    assert receipt.ready is False
    assert any("只能声明一个 Mobile UI" in error for error in receipt.errors)
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


@pytest.mark.asyncio
async def test_plugin_ui_slots_oracle_kills_leaked_mobile_registration_mutant(
    tmp_path: Path,
) -> None:
    correct = await _disposed_mobile_registration_fixture(
        tmp_path / "mobile-correct",
        PluginUiSlots,
    )
    mutant = await _disposed_mobile_registration_fixture(
        tmp_path / "mobile-mutant",
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


async def _disposed_mobile_registration_fixture(
    plugin_dir: Path,
    service_type: type[PluginUiSlots],
) -> bool:
    """在冻结声明前释放一个 Mobile UI owner。"""

    plugin_dir.mkdir(parents=True)
    (plugin_dir / "mobile.js").write_text("export default 1;", encoding="utf-8")
    (plugin_dir / "mobile.css").write_text("", encoding="utf-8")
    root = CompositionRoot(f"mobile-slots-dispose:{service_type.__name__}")
    slots = service_type()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def plugin(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_mobile(
            ctx,
            _mobile_definition(),
            query=_mobile_query,
        )

    fiber = await root.mount(
        plugin,
        name="mobile-slot-owner",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    await fiber.dispose()
    leaked = bool(slots.freeze())
    await root.dispose()
    return leaked
