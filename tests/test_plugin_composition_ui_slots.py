from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    UI_SLOTS,
    CompositionError,
    CompositionRoot,
    MobileUiDefinition,
    MobileUiNavigation,
    PluginRuntime,
    PluginUiSlots,
    resolve_mobile_ui_asset,
)
from agent.plugin_composition.ui_slots import MobileUiSlot
from agent.plugins.manager import PluginManager
from agent.plugins.mobile_ui import (
    MobileUiPluginUnavailable,
    PluginMobileUiProvider,
)
from bus.event_bus import EventBus


def _runtime(plugin_dir: Path) -> PluginRuntime:
    return PluginRuntime(
        plugin_id=plugin_dir.name,
        generation_id="test-generation",
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=None,
    )


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _definition() -> MobileUiDefinition:
    return MobileUiDefinition(
        module="mobile.js",
        stylesheet="mobile.css",
        navigation=MobileUiNavigation(label="Probe", description="Probe panel"),
        slots=("drawer.panel",),
    )


def _query(
    method: str,
    payload: dict[str, object],
    *,
    session_id: str | None,
    turn_id: str | None,
) -> dict[str, object]:
    return {
        "method": method,
        "payload": payload,
        "session_id": session_id,
        "turn_id": turn_id,
    }


@pytest.mark.asyncio
async def test_ui_slots_freeze_descriptor_and_effect_cleanup(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "probe"
    plugin_dir.mkdir()
    (plugin_dir / "mobile.js").write_text("export const probe = true;\n", encoding="utf-8")
    (plugin_dir / "mobile.css").write_text(":host {}\n", encoding="utf-8")
    root = CompositionRoot("ui-slots")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def apply(ctx) -> None:
        await ctx.require(UI_SLOTS).register_mobile(
            ctx,
            _definition(),
            query=_query,
        )

    fiber = await root.mount(
        apply,
        name="probe",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    registry = slots.freeze()
    binding = registry["probe"]
    assert registry.descriptor("probe") is binding.descriptor
    assert binding.descriptor.owner == "probe"
    assert binding.descriptor.module_bytes == len(binding.asset.module.encode())
    assert registry.identity

    await fiber.dispose()
    assert slots.freeze() is registry
    assert len(registry) == 1
    assert not registry["probe"].is_live()
    await root.dispose()


@pytest.mark.asyncio
async def test_ui_slots_rejects_duplicate_and_frozen_registration(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "probe"
    plugin_dir.mkdir()
    (plugin_dir / "mobile.js").write_text("export default 1;", encoding="utf-8")
    (plugin_dir / "mobile.css").write_text("", encoding="utf-8")
    root = CompositionRoot("ui-slots-duplicate")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)
    captured: dict[str, Any] = {}

    async def apply(ctx) -> None:
        captured["ctx"] = ctx
        service = ctx.require(UI_SLOTS)
        await service.register_mobile(ctx, _definition(), query=_query)
        await service.register_mobile(ctx, _definition(), query=_query)

    _ = await root.mount(
        apply,
        name="probe",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    assert not root.receipt().ready
    assert any(
        "只能声明一个 Mobile UI" in (fiber.error or "")
        for fiber in root.receipt().fibers
    )
    assert len(slots.freeze()) == 0
    await root.dispose()

    root = CompositionRoot("ui-slots-frozen")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)
    ctx: Any = None

    async def register_once(current) -> None:
        nonlocal ctx
        ctx = current
        await ctx.require(UI_SLOTS).register_mobile(ctx, _definition(), query=_query)

    _ = await root.mount(
        register_once,
        name="probe",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    _ = slots.freeze()
    with pytest.raises(CompositionError, match="已冻结"):
        await slots.register_mobile(ctx, _definition(), query=_query)
    await root.dispose()


@pytest.mark.asyncio
async def test_ui_slots_rejects_symlink_and_async_handler(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "probe"
    plugin_dir.mkdir()
    outside = tmp_path / "outside.js"
    outside.write_text("export default 1;", encoding="utf-8")
    (plugin_dir / "mobile.js").symlink_to(outside)
    root = CompositionRoot("ui-slots-path")
    slots = PluginUiSlots()
    _ = await root.context.provide(UI_SLOTS, slots)

    async def apply(ctx) -> None:
        async def bad(*args: object, **kwargs: object) -> object:
            return {}

        await ctx.require(UI_SLOTS).register_mobile(
            ctx,
            _definition(),
            query=cast(Any, bad),
        )

    _ = await root.mount(
        apply,
        name="probe",
        inject=(UI_SLOTS,),
        runtime=_runtime(plugin_dir),
    )
    receipt = root.receipt()
    assert not receipt.ready
    assert any(
        "必须是同步函数" in (fiber.error or "") for fiber in receipt.fibers
    )
    await root.dispose()


@pytest.mark.parametrize(
    ("definition", "message"),
    (
        (
            MobileUiDefinition(
                module="mobile.js", slots=(cast(MobileUiSlot, "unknown.slot"),)
            ),
            "slots 无效",
        ),
        (
            MobileUiDefinition(
                module="mobile.js",
                navigation=MobileUiNavigation(label="", description="Probe"),
            ),
            "navigation 无效",
        ),
    ),
)
def test_mobile_ui_asset_rejects_invalid_metadata(
    tmp_path: Path,
    definition: MobileUiDefinition,
    message: str,
) -> None:
    plugin_dir = tmp_path / "probe"
    plugin_dir.mkdir()
    (plugin_dir / "mobile.js").write_text("export default 1;", encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        resolve_mobile_ui_asset(
            plugin_dir,
            module=definition.module,
            stylesheet=definition.stylesheet,
            navigation_label=(
                None
                if definition.navigation is None
                else definition.navigation.label
            ),
            navigation_description=(
                None
                if definition.navigation is None
                else definition.navigation.description
            ),
            slots=tuple(definition.slots),
        )


def test_mobile_ui_asset_rejects_size_over_budget(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "probe"
    plugin_dir.mkdir()
    (plugin_dir / "mobile.js").write_text("x" * (240 * 1024 + 1), encoding="utf-8")

    with pytest.raises(RuntimeError, match="超过协议安全预算"):
        resolve_mobile_ui_asset(
            plugin_dir,
            module="mobile.js",
            stylesheet=None,
            navigation_label=None,
            navigation_description=None,
            slots=(),
        )


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _plugin_source(version: str) -> str:
    return (
        "from agent.plugin_composition import UI_SLOTS, MobileUiDefinition\n"
        "api_version = 3\n"
        "name = 'ui_probe'\n"
        f"version = '{version}'\n"
        "inject = (UI_SLOTS,)\n"
        "def query(method, payload, *, session_id, turn_id):\n"
        "    return {'version': version, 'method': method, 'payload': payload}\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(UI_SLOTS).register_mobile(\n"
        "        ctx, MobileUiDefinition(module='mobile.js', slots=('drawer.panel',)),\n"
        "        query=query,\n"
        "    )\n"
    )


def _nested_plugin_source() -> str:
    return (
        "from agent.plugin_composition import UI_SLOTS, MobileUiDefinition\n"
        "api_version = 3\n"
        "name = 'ui_probe'\n"
        "version = '1'\n"
        "inject = (UI_SLOTS,)\n"
        "child_handle = None\n"
        "def query(method, payload, *, session_id, turn_id):\n"
        "    return {'status': 'ready', 'method': method}\n"
        "async def register_mobile(ctx):\n"
        "    await ctx.require(UI_SLOTS).register_mobile(\n"
        "        ctx, MobileUiDefinition(module='mobile.js', slots=('drawer.panel',)),\n"
        "        query=query,\n"
        "    )\n"
        "async def apply(ctx, config):\n"
        "    global child_handle\n"
        "    child_handle = await ctx.mount(\n"
        "        register_mobile, name='mobile-nested', inject=(UI_SLOTS,),\n"
        "    )\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ("dispose", "restart"))
async def test_published_snapshot_hides_nested_fiber_after_transition(
    tmp_path: Path,
    transition: str,
) -> None:
    plugin_dir = _write_plugin(tmp_path / "plugins", "ui_probe", _nested_plugin_source())
    (plugin_dir / "mobile.js").write_text("export const nested = true;\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()

    generation = manager.generation("ui_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    root = snapshot.composition_root
    assert root is not None
    child = cast(Any, generation.instance.module).child_handle
    assert child is not None
    provider = PluginMobileUiProvider(manager)
    assert provider.catalog()["items"]

    if transition == "dispose":
        await child.dispose()
    else:
        await child.restart()
        assert any(
            "已冻结" in (fiber.error or "") for fiber in root.receipt().fibers
        )

    assert provider.catalog()["items"] == []
    with pytest.raises(MobileUiPluginUnavailable):
        await provider.query(
            "ui_probe",
            generation.source_revision,
            "probe.current",
            {},
            session_id=None,
            turn_id=None,
        )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_publishes_v3_registry_without_generation_contribution(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(tmp_path / "plugins", "ui_probe", _plugin_source("1"))
    (plugin_dir / "mobile.js").write_text("export const version = 1;\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()

    generation = manager.generation("ui_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert snapshot.mobile_ui_registry is not None
    provider = PluginMobileUiProvider(manager)
    item = cast(list[dict[str, object]], provider.catalog()["items"])[0]
    assert item["id"] == "ui_probe"
    result = await provider.query(
        "ui_probe",
        generation.source_revision,
        "probe.current",
        {"limit": 1},
        session_id="mobile:test",
        turn_id="turn-1",
    )
    assert result == {
        "version": "1",
        "method": "probe.current",
        "payload": {"limit": 1},
    }
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_candidate_registry_stays_private_until_publish(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(tmp_path / "plugins", "ui_probe", _plugin_source("1"))
    (plugin_dir / "mobile.js").write_text("export const version = 1;\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    stable_generation = manager.generation("ui_probe")
    assert stable_generation is not None

    (plugin_dir / "plugin.py").write_text(_plugin_source("2"), encoding="utf-8")
    (plugin_dir / "mobile.js").write_text("export const version = 2;\n", encoding="utf-8")
    candidate = await manager.prepare_candidate("ui_probe")
    assert candidate is not None and candidate.runtime_snapshot is not None
    assert manager.current_snapshot is stable
    assert manager.current_snapshot.mobile_ui_registry is not None
    assert (
        candidate.runtime_snapshot.mobile_ui_registry is not None
        and candidate.runtime_snapshot.mobile_ui_registry["ui_probe"].asset.module
        == "export const version = 2;\n"
    )
    provider = PluginMobileUiProvider(manager)
    stable_item = cast(list[dict[str, object]], provider.catalog()["items"])[0]
    assert stable_item["module_bytes"] == len("export const version = 1;\n".encode())
    stable_result = await provider.query(
        "ui_probe",
        stable_generation.source_revision,
        "probe.current",
        {},
        session_id=None,
        turn_id=None,
    )
    assert stable_result["version"] == "1"
    await manager.discard_prepared("ui_probe")
    assert manager.current_snapshot is stable
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_publish_rebuilds_formal_ui_handler(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(tmp_path / "plugins", "ui_probe", _plugin_source("1"))
    (plugin_dir / "mobile.js").write_text("export const version = 1;\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    (plugin_dir / "plugin.py").write_text(_plugin_source("2"), encoding="utf-8")
    (plugin_dir / "mobile.js").write_text("export const version = 2;\n", encoding="utf-8")

    candidate = await manager.prepare_candidate("ui_probe")
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_registry = candidate.runtime_snapshot.mobile_ui_registry
    assert candidate_registry is not None

    result = await manager.publish_prepared("ui_probe")
    assert result["publication_state"] == "committed"
    snapshot = manager.current_snapshot
    generation = manager.generation("ui_probe")
    assert snapshot is not None and generation is not None
    registry = snapshot.mobile_ui_registry
    assert registry is not None
    assert registry["ui_probe"].asset.module == "export const version = 2;\n"
    assert registry["ui_probe"].query is not candidate_registry["ui_probe"].query
    provider = PluginMobileUiProvider(manager)
    response = await provider.query(
        "ui_probe",
        generation.source_revision,
        "probe.current",
        {},
        session_id=None,
        turn_id=None,
    )
    assert response["version"] == "2"
    await manager.terminate_all()
