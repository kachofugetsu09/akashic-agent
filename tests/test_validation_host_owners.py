"""两棵真实 Root 的 Mobile owner 隔离。"""

import pytest
from agent.plugin_composition import (
    CompositionRoot,
    MobileUiDefinition,
    MobileUiStaleRevision,
    PluginRuntime,
    UI_SLOTS,
)
from agent.plugins.mobile_ui import PluginMobileUiProvider
from plugins.ui.mobile import MobileUiSlots

@pytest.mark.asyncio
async def test_mobile_owner_isolation_uses_two_real_composition_roots(tmp_path):
    """Prove Mobile lookup and revisions cannot cross two live Roots."""

    left = CompositionRoot("validation-mobile-left")
    right = CompositionRoot("validation-mobile-right")

    async def provide_slots(ctx):
        await ctx.provide(UI_SLOTS, MobileUiSlots(ctx))

    def query(method, payload, *, session_id, turn_id):
        return {"method": method, "payload": payload}

    def owner_plugin():
        async def mount_owner(ctx):
            await ctx.require(UI_SLOTS).register_mobile(
                ctx,
                MobileUiDefinition(module="mobile.js"),
                query=query,
            )
        return mount_owner

    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    for plugin_dir in (left_dir, right_dir):
        plugin_dir.mkdir()
        (plugin_dir / "data").mkdir()
        (plugin_dir / "mobile.js").write_text("export const mobile = true;\n")

    left_provider = right_provider = None
    try:
        await left.mount(provide_slots, name="ui")
        await right.mount(provide_slots, name="ui")
        left_owner = await left.mount(
            owner_plugin(),
            name="mobile-owner",
            inject=(UI_SLOTS,),
            runtime=PluginRuntime(
                plugin_id="mobile-owner", generation_id="left-owner",
                plugin_dir=left_dir, data_dir=left_dir / "data",
                workspace=left_dir, config={},
            ),
        )
        right_owner = await right.mount(
            owner_plugin(),
            name="mobile-owner",
            inject=(UI_SLOTS,),
            runtime=PluginRuntime(
                plugin_id="mobile-owner", generation_id="right-owner",
                plugin_dir=right_dir, data_dir=right_dir / "data",
                workspace=right_dir, config={},
            ),
        )
        assert left_owner.state.value == "active"
        assert right_owner.state.value == "active"
        left_slots = left.context.require(UI_SLOTS)
        with pytest.raises(ValueError, match="跨实际 Root"):
            await left_slots.register_mobile(
                right_owner.context,
                MobileUiDefinition(module="mobile.js"),
                query=query,
            )

        left_provider = PluginMobileUiProvider(left)
        right_provider = PluginMobileUiProvider(right)
        left_catalog = await left_provider.catalog()
        right_catalog = await right_provider.catalog()
        left_items = left_catalog["items"]
        right_items = right_catalog["items"]
        assert isinstance(left_items, list) and isinstance(right_items, list)
        assert len(left_items) == len(right_items) == 1
        assert isinstance(left_items[0], dict) and isinstance(right_items[0], dict)
        left_revision = left_items[0]["revision"]
        right_revision = right_items[0]["revision"]
        assert isinstance(left_revision, str) and isinstance(right_revision, str)
        assert left_revision != right_revision
        with pytest.raises(MobileUiStaleRevision):
            await left_provider.query(
                "mobile-owner", right_revision, "test", {},
                session_id=None, turn_id=None,
            )
    finally:
        if left_provider is not None:
            await left_provider.aclose()
        if right_provider is not None:
            await right_provider.aclose()
        await left.dispose()
        await right.dispose()
