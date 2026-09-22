"""普通 UI provider 的 Mobile 归属和封存边界。"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    CompositionError, CompositionRoot, MobileUiDefinition, MobileUiNavigation,
    PluginRuntime, SNAPSHOT_SEALING, SnapshotSealing, UI_SLOTS,
)
from agent.plugins.mobile_ui import MobileUiPluginUnavailable, PluginMobileUiProvider
from agent.plugins.snapshot import RuntimeSnapshot
from plugins.ui import plugin as ui_plugin


async def mount_owner(root, code, *, name="mobile", register=True):
    contexts = []

    def query(method, payload, *, session_id, turn_id):
        return {"owner": name, "method": method, "session_id": session_id, "turn_id": turn_id}

    async def apply(ctx):
        contexts.append(ctx)
        if register:
            await ctx.require(UI_SLOTS).register_mobile(
                ctx,
                MobileUiDefinition(
                    module="mobile.js", stylesheet="mobile.css",
                    navigation=MobileUiNavigation("Panel", "Owner panel"),
                    slots=("drawer.panel",),
                ),
                query=query,
            )

    await root.mount(
        apply, name=name, inject=(UI_SLOTS,),
        runtime=PluginRuntime(
            plugin_id=name, generation_id=name + "-generation", plugin_dir=code,
            data_dir=code / "data", workspace=code, config={},
        ),
    )
    return contexts[0]


@pytest.mark.asyncio
async def test_mobile_provider_seals_actual_owner_and_fixed_asset_bytes(tmp_path):
    root = CompositionRoot("mobile")
    module = tmp_path / "mobile.js"
    module.write_text("export function activate() {}")
    (tmp_path / "mobile.css").write_text(".panel { color: blue; }")
    original = module.read_text()
    try:
        await root.mount(ui_plugin.apply, name="renamed-ui")
        ctx = await mount_owner(root, tmp_path)
        slots = root.context.require(UI_SLOTS)
        with pytest.raises(RuntimeError, match="尚未封存"):
            slots.catalog()
        await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        catalog = slots.catalog()
        assert catalog.root_instance_token is root.instance_token
        binding = catalog["mobile"]
        assert binding.descriptor.owner == "mobile"
        assert binding.descriptor.slots == ("drawer.panel",)
        assert binding.is_live()
        module.write_text("changed")
        assert binding.asset.module == original
        snapshot = RuntimeSnapshot("mobile", {}, composition_root=root)
        generation = SimpleNamespace(plugin_id="mobile", generation_id="mobile-generation")
        assert PluginMobileUiProvider._mobile_ui_binding(snapshot, cast(Any, generation)) is binding
        with pytest.raises(CompositionError, match="已冻结"):
            await slots.register_mobile(
                ctx, MobileUiDefinition(module="mobile.js"), query=binding.query,
            )
    finally:
        await root.dispose()
    assert not binding.is_live()


@pytest.mark.asyncio
async def test_mobile_registration_rejects_cross_root_and_escaped_asset(tmp_path):
    left, right = CompositionRoot("left"), CompositionRoot("right")
    code = tmp_path / "owner"
    code.mkdir()
    foreign = tmp_path / "foreign.js"
    foreign.write_text("export function activate() {}")
    (code / "linked.js").symlink_to(foreign)

    def query(method, payload, *, session_id, turn_id):
        return {}

    try:
        for root in (left, right):
            await root.mount(ui_plugin.apply, name="ui")
        ctx = await mount_owner(left, code, register=False)
        other = await mount_owner(right, code, register=False)
        slots = ctx.require(UI_SLOTS)
        with pytest.raises(ValueError, match="实际 Root"):
            await slots.register_mobile(other, MobileUiDefinition(module="linked.js"), query=query)
        for path in ("../foreign.js", "linked.js", str(foreign)):
            with pytest.raises(RuntimeError, match="mobile UI"):
                await slots.register_mobile(ctx, MobileUiDefinition(module=path), query=query)
    finally:
        await right.dispose()
        await left.dispose()


@pytest.mark.asyncio
async def test_mobile_consumer_rejects_borrowed_root_provider():
    old, new = CompositionRoot("old"), CompositionRoot("new")
    try:
        await old.mount(ui_plugin.apply, name="ui")
        await old.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
        await new.context.provide(UI_SLOTS, old.context.require(UI_SLOTS))
        snapshot = RuntimeSnapshot("new", {}, composition_root=new)
        with pytest.raises(MobileUiPluginUnavailable, match="provider 不属于"):
            PluginMobileUiProvider._mobile_ui_binding(
                snapshot, cast(Any, SimpleNamespace(plugin_id="mobile"))
            )
    finally:
        await new.dispose()
        await old.dispose()
