from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import (
    CompositionRoot,
    CompositionError,
    SerialEventKey,
    ServiceKey,
)
from agent.plugin_composition.model import PluginRuntime
from agent.plugin_composition.overlay import CompositionOverlay
from agent.plugins.generation import PluginGeneration
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    get_current_runtime_snapshot,
    lease_runtime_snapshot,
)
from bus.event_bus import EventBus


def _runtime(tmp_path: Path, plugin_id: str, generation_id: str) -> PluginRuntime:
    return PluginRuntime(
        plugin_id,
        generation_id,
        tmp_path,
        tmp_path,
        tmp_path,
        {},
    )


@pytest.mark.asyncio
async def test_selected_plugin_forks_overlay_but_replaced_plugin_cannot_capture_it(
    tmp_path,
):
    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    contexts = {}
    key = ServiceKey("overlay.dynamic")

    async def mounted(ctx):
        contexts[ctx.runtime.generation_id, ctx.runtime.plugin_id] = ctx
        if ctx.runtime.plugin_id == "changed":
            await ctx.provide(key, ctx.runtime.generation_id, binding_contributors=lambda: (ctx,))

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
        assert overlay.binding_contributors(key) == (contexts["candidate", "changed"],)
        assert stable.binding_contributors(key) == (contexts["stable", "changed"],)
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


@pytest.mark.asyncio
async def test_overlay_keeps_candidate_event_order_and_duplicate_owner_listeners(
    tmp_path: Path,
) -> None:
    event = SerialEventKey[object, object]("overlay.order")
    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    trace: list[str] = []

    def listener(owner: str):
        def record(_payload: object) -> None:
            trace.append(owner)

        return record

    async def stable_b(ctx) -> None:
        await ctx.on(event, listener("b"))

    async def stable_z(ctx) -> None:
        await ctx.on(event, listener("z"))

    async def stable_a(ctx) -> None:
        await ctx.on(event, listener("a"))
        await ctx.on(event, listener("a"))

    async def candidate_a(ctx) -> None:
        await ctx.on(event, listener("a"))
        await ctx.on(event, listener("a"))

    async def candidate_b(ctx) -> None:
        await ctx.on(event, listener("b"))

    async def candidate_z(ctx) -> None:
        await ctx.on(event, listener("z"))

    try:
        # 候选 Root 的注册顺序是 B -> Z -> A -> A。
        await stable.mount(
            stable_b,
            name="b",
            runtime=_runtime(tmp_path, "b", "stable-b"),
        )
        await stable.mount(
            stable_z,
            name="z",
            runtime=_runtime(tmp_path, "z", "stable-z"),
        )
        await stable.mount(
            stable_a,
            name="a",
            runtime=_runtime(tmp_path, "a", "stable-a"),
        )
        await candidate.mount(
            candidate_b,
            name="b",
            runtime=_runtime(tmp_path, "b", "candidate-b"),
        )
        await candidate.mount(
            candidate_z,
            name="z",
            runtime=_runtime(tmp_path, "z", "candidate-z"),
        )
        await candidate.mount(
            candidate_a,
            name="a",
            runtime=_runtime(tmp_path, "a", "candidate-a"),
        )

        overlay = CompositionOverlay(
            stable,
            candidate,
            plugin_ids=frozenset({"a", "b", "z"}),
            replaced_plugin_ids=frozenset({"a", "b", "z"}),
        )
        assert overlay.topology_view().listeners == (
            "serial:overlay.order:b",
            "serial:overlay.order:z",
            "serial:overlay.order:a",
            "serial:overlay.order:a",
        )
        await overlay.context.serial(event, None)
        assert trace == ["b", "z", "a", "a"]
    finally:
        await candidate.dispose()
        await stable.dispose()


@pytest.mark.asyncio
async def test_overlay_rejects_split_event_groups(tmp_path: Path) -> None:
    event = SerialEventKey[object, object]("overlay.split")
    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")

    async def stable_b(ctx) -> None:
        await ctx.on(event, lambda _: None)

    async def candidate_a(ctx) -> None:
        await ctx.on(event, lambda _: None)

    try:
        await stable.mount(
            stable_b,
            name="b",
            runtime=_runtime(tmp_path, "b", "stable-b"),
        )
        await candidate.mount(
            candidate_a,
            name="a",
            runtime=_runtime(tmp_path, "a", "candidate-a"),
        )
        with pytest.raises(CompositionError, match="同时属于 stable 与 candidate"):
            _ = CompositionOverlay(
                stable,
                candidate,
                plugin_ids=frozenset({"a", "b"}),
                replaced_plugin_ids=frozenset({"a"}),
            )
    finally:
        await candidate.dispose()
        await stable.dispose()


@pytest.mark.asyncio
async def test_candidate_frontier_includes_stable_optional_service_peer(
    tmp_path: Path,
) -> None:
    shared = ServiceKey[str]("overlay.shared")
    stable = CompositionRoot("stable")
    candidate = CompositionRoot("candidate")
    manager = PluginManager(
        [],
        event_bus=EventBus(),
        workspace=tmp_path / "manager",
    )

    async def stable_peer(ctx) -> None:
        async def child(child_ctx) -> None:
            assert child_ctx.require(shared) == "candidate"

        await ctx.inject((shared,), child, name="optional-peer")

    async def candidate_provider(ctx) -> None:
        await ctx.provide(shared, "candidate")

    try:
        await stable.mount(
            stable_peer,
            name="peer",
            runtime=_runtime(tmp_path, "peer", "stable-peer"),
        )
        await candidate.mount(
            candidate_provider,
            name="provider",
            runtime=_runtime(tmp_path, "provider", "candidate-provider"),
        )

        generations = cast(
            dict[str, PluginGeneration],
            {"provider": object(), "peer": object()},
        )
        additional = manager._candidate_composition_frontier(
            candidate,
            stable,
            generations,
            frozenset({"provider"}),
        )
        assert additional == frozenset({"peer"})
    finally:
        await manager.snapshot_store.close()
        await candidate.dispose()
        await stable.dispose()


@pytest.mark.asyncio
async def test_binding_contributors_end_with_the_service_effect() -> None:
    """动态归档贡献与原服务同生共死，其他 consumer 不能覆盖它。"""
    root = CompositionRoot("binding-lifetime")
    key = ServiceKey("binding-lifetime.value")
    try:
        effect = await root.context.provide(key, "owned", binding_contributors=lambda: (root.context,))
        assert root.binding_contributors(key) == (root.context,)
        with pytest.raises(CompositionError, match="已由"):
            await root.context.provide(key, "other", binding_contributors=lambda: ())
        await effect.aclose()
        with pytest.raises(RuntimeError, match="已失效"):
            root.binding_contributors(key)
    finally:
        await root.dispose()
