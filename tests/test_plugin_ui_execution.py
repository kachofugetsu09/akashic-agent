"""O/C1: bounded UI work is isolated by owner and drains its real Root scope."""
import asyncio
import threading
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from plugins.ui.contract import PluginUiDefinition, UI_SLOTS
from plugins.ui.contract import PluginUiQueryTimeout
from plugins.ui.contract import PLUGIN_UI
from plugins.ui.queries import LivePluginUiProvider
from plugins.ui.plugin_ui import PluginUiSlots


@pytest.mark.asyncio
async def test_slow_ui_owner_leaves_capacity_and_queued_timeout_never_runs(tmp_path, monkeypatch):
    root = CompositionRoot("ui-isolation")
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    lock = threading.Lock()
    calls = []

    async def slots(ctx):
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    await root.mount(slots, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))
    (tmp_path / "panel.js").write_text("export default {};\n")

    async def slow(ctx):
        def query(method, payload, *, session_id, turn_id):
            with lock:
                calls.append(payload["id"])
                if len(calls) >= 4:
                    loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "test did not release the physical query"
            return {"owner": "slow"}
        await ctx.require(UI_SLOTS).register_plugin_ui(ctx, PluginUiDefinition("panel.js"), query=query)

    async def fast(ctx):
        def query(method, payload, *, session_id, turn_id):
            return {"owner": "fast"}
        await ctx.require(UI_SLOTS).register_plugin_ui(ctx, PluginUiDefinition("panel.js"), query=query)

    for name, apply in (("slow", slow), ("fast", fast)):
        await root.mount(apply, name=name, inject=(UI_SLOTS,),
                         runtime=PluginRuntime(name, "ui", tmp_path, tmp_path, tmp_path, {}))
    catalog = await provider.catalog()
    items = cast(list[dict[str, object]], catalog["items"])
    revisions = {str(item["id"]): str(item["revision"]) for item in items}

    async def query(owner, identity):
        return await provider.query(owner, revisions[owner], "read", {"id": identity},
                                    session_id=None, turn_id=None)

    jobs = [asyncio.create_task(query("slow", i)) for i in range(8)]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert await asyncio.wait_for(query("fast", "fast"), 1) == {"owner": "fast"}
        jobs[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await jobs[0]
        monkeypatch.setattr("plugins.ui.queries.PLUGIN_UI_QUERY_TIMEOUT_SECONDS", 0.05)
        with pytest.raises(PluginUiQueryTimeout):
            await query("slow", "withdrawn")
        closing = asyncio.create_task(provider.aclose())
        tick = asyncio.Event()
        loop.call_soon(tick.set)
        await tick.wait()
        assert not closing.done()
        release.set()
        await asyncio.gather(*jobs[1:])
        await closing
        assert "withdrawn" not in calls
        assert sorted(calls) == list(range(8))
    finally:
        release.set()
        await asyncio.gather(*jobs, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_plugin_ui_slot_invalidation_follows_and_validates_owner(tmp_path):
    root = CompositionRoot("ui-slot-invalidation")
    emitted: list[dict[str, object]] = []
    got_first = asyncio.Event()
    got_second = asyncio.Event()

    async def slots(ctx):
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    await root.mount(slots, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))
    ui_slots = root.context.require(UI_SLOTS)
    (tmp_path / "panel.js").write_text("export default {};\n")

    captured: dict[str, object] = {}

    async def groups_owner(ctx):
        captured["groups"] = ctx
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx,
            PluginUiDefinition("panel.js", slots=("navigation.groups",)),
            query=lambda method, payload, *, session_id, turn_id: {},
        )

    async def other_owner(ctx):
        captured["other"] = ctx
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx,
            PluginUiDefinition("panel.js", slots=("header.end",)),
            query=lambda method, payload, *, session_id, turn_id: {},
        )

    for plugin_name, apply in (("projects", groups_owner), ("other", other_owner)):
        await root.mount(
            apply,
            name=plugin_name,
            inject=(UI_SLOTS,),
            runtime=PluginRuntime(plugin_name, "ui", tmp_path, tmp_path, tmp_path, {}),
        )

    async def emit(frame: dict[str, object]) -> None:
        emitted.append(frame)
        if len(emitted) == 1:
            got_first.set()
        elif len(emitted) == 2:
            got_second.set()

    follower = asyncio.create_task(provider.follow_slots(emit))
    try:
        # 让 follow_slots 先建立基线并挂起等待。
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="未登记 slot"):
            ui_slots.invalidate_slot(captured["other"], "navigation.groups")  # type: ignore[arg-type]

        ui_slots.invalidate_slot(captured["groups"], "navigation.groups")  # type: ignore[arg-type]
        await asyncio.wait_for(got_first.wait(), 2)
        assert emitted[0] == {"slot": "navigation.groups", "epoch": 1}

        ui_slots.invalidate_slot(captured["groups"], "navigation.groups")  # type: ignore[arg-type]
        await asyncio.wait_for(got_second.wait(), 2)
        assert emitted[1] == {"slot": "navigation.groups", "epoch": 2}
    finally:
        await provider.aclose()
        await asyncio.wait_for(follower, 2)
        await root.dispose()
