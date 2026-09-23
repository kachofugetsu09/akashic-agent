"""Mobile UI provider tests use real Root, registration Effect and Context scopes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

import pytest

import agent.plugins.mobile_ui as mobile_ui_module
from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    Context,
    Effect,
    Fiber,
    FiberState,
    MobileUiDefinition,
    MobileUiNavigation,
    MobileUiPluginUnavailable,
    MobileUiStaleRevision,
    PluginRuntime,
    ServiceKey,
    UI_SLOTS,
)
from agent.plugins.mobile_ui import PluginMobileUiProvider
from plugins.ui import plugin as ui_plugin


DEPENDENCY = ServiceKey[str]("test.mobile.ui.dependency")


class _OwnerState(TypedDict, total=False):
    context: Context
    effect: Effect


def _owner_context(state: _OwnerState) -> Context:
    context = state.get("context")
    assert isinstance(context, Context)
    return context


def _owner_effect(state: _OwnerState) -> Effect:
    effect = state.get("effect")
    assert isinstance(effect, Effect)
    return effect


def _item_revision(item: dict[str, object]) -> str:
    revision = item["revision"]
    assert isinstance(revision, str)
    return revision


def _item_sha(item: dict[str, object]) -> str:
    sha = item["module_sha256"]
    assert isinstance(sha, str)
    return sha


def _catalog_items(catalog: dict[str, object]) -> list[dict[str, object]]:
    items = catalog["items"]
    assert isinstance(items, list)
    assert all(isinstance(item, dict) for item in items)
    return items


async def mount_ui_owner(
    root: CompositionRoot,
    code: Path,
    *,
    name: str = "mobile",
    register: bool = True,
    available=None,
    query=None,
    inject: tuple[ServiceKey[object], ...] = (UI_SLOTS,),
) -> tuple[Fiber, _OwnerState]:
    state: _OwnerState = {}

    def default_query(method, payload, *, session_id, turn_id):
        return {
            "owner": name,
            "method": method,
            "payload": payload,
            "session_id": session_id,
            "turn_id": turn_id,
        }

    async def apply(ctx: Context) -> None:
        state["context"] = ctx
        if register:
            query_handler = default_query if query is None else query
            state["effect"] = await ctx.require(UI_SLOTS).register_mobile(
                ctx,
                MobileUiDefinition(
                    module="mobile.js",
                    stylesheet="mobile.css",
                    navigation=MobileUiNavigation("Panel", "Owner panel"),
                    slots=("drawer.panel",),
                ),
                query=query_handler,
                available=available,
            )

    fiber = await root.mount(
        apply,
        name=name,
        inject=inject,
        runtime=PluginRuntime(
            plugin_id=name,
            generation_id=name + "-generation",
            plugin_dir=code,
            data_dir=code / "data",
            workspace=code,
            config={},
        ),
    )
    return fiber, state


async def build_mobile_root(root: CompositionRoot, code: Path) -> tuple[Fiber, _OwnerState]:
    await root.mount(ui_plugin.apply, name="ui")
    return await mount_ui_owner(root, code)


def write_assets(code: Path) -> str:
    module = code / "mobile.js"
    module.write_text("export function activate() {}", encoding="utf-8")
    (code / "mobile.css").write_text(".panel { color: blue; }", encoding="utf-8")
    return module.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_mobile_provider_reads_live_registration_and_fixed_asset_bytes(tmp_path: Path):
    root = CompositionRoot("mobile")
    original = write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    try:
        _fiber, state = await build_mobile_root(root, tmp_path)
        context = _owner_context(state)
        catalog = await provider.catalog()
        assert len(_catalog_items(catalog)) == 1
        item = _catalog_items(catalog)[0]
        assert isinstance(_item_revision(item), str)
        assert isinstance(_item_sha(item), str)
        assert item["id"] == "mobile"
        assert item["module_bytes"] == len(original.encode())
        asset = await provider.asset(
            "mobile", _item_revision(item), "module", _item_sha(item),
        )
        assert asset["content"] == original
        (tmp_path / "mobile.js").write_text("changed", encoding="utf-8")
        assert (await provider.asset(
            "mobile", _item_revision(item), "module", _item_sha(item),
        ))["content"] == original
        result = await provider.query(
            "mobile", _item_revision(item), "inspect", {"ok": True},
            session_id="session", turn_id="turn",
        )
        assert result["owner"] == "mobile"
        assert context.fiber.state is FiberState.ACTIVE
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_registration_effect_replaces_identity_in_same_context(tmp_path: Path):
    root = CompositionRoot("reregister")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    try:
        await root.mount(ui_plugin.apply, name="ui")
        _fiber, state = await mount_ui_owner(root, tmp_path)
        slots = root.context.require(UI_SLOTS)
        old_binding = slots.bindings()[0]
        old_effect = _owner_effect(state)
        old_catalog = await provider.catalog()

        await old_effect.aclose()
        new_effect = await slots.register_mobile(
            _owner_context(state),
            MobileUiDefinition(module="mobile.js", stylesheet="mobile.css"),
            query=lambda method, payload, *, session_id, turn_id: {"new": True},
        )
        new_binding = slots.bindings()[0]
        new_catalog = await provider.catalog()
        assert new_binding.context is old_binding.context
        assert new_binding.registration_uuid != old_binding.registration_uuid
        assert _catalog_items(new_catalog)[0]["revision"] != _catalog_items(old_catalog)[0]["revision"]
        await new_effect.aclose()
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_hard_dependency_reactivation_gets_new_context_and_fence(tmp_path: Path):
    root = CompositionRoot("dependency-reload")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    dependency_a = None
    dependency_b = None
    dependency_a_code = tmp_path / "dependency-a"
    dependency_b_code = tmp_path / "dependency-b"
    dependency_a_code.mkdir()
    dependency_b_code.mkdir()

    async def provide_a(ctx):
        await ctx.provide(DEPENDENCY, "A")

    async def provide_b(ctx):
        await ctx.provide(DEPENDENCY, "B")

    def dependency_runtime(plugin_id: str, code: Path) -> PluginRuntime:
        return PluginRuntime(
            plugin_id=plugin_id,
            generation_id=plugin_id + "-generation",
            plugin_dir=code,
            data_dir=code / "data",
            workspace=code,
            config={},
        )

    try:
        await root.mount(ui_plugin.apply, name="ui")
        dependency_a = await root.mount(
            provide_a,
            name="dependency-a",
            runtime=dependency_runtime("dependency-a", dependency_a_code),
        )
        fiber, state = await mount_ui_owner(
            root,
            tmp_path,
            inject=(UI_SLOTS, DEPENDENCY),
        )
        old_context = _owner_context(state)
        old_catalog = await provider.catalog()
        old_item = _catalog_items(old_catalog)[0]
        old_binding = root.context.require(UI_SLOTS).bindings()[0]

        await dependency_a.dispose()
        assert fiber.state is FiberState.PENDING
        dependency_b = await root.mount(
            provide_b,
            name="dependency-b",
            runtime=dependency_runtime("dependency-b", dependency_b_code),
        )
        assert fiber.state is FiberState.ACTIVE
        new_context = _owner_context(state)
        new_binding = root.context.require(UI_SLOTS).bindings()[0]
        new_catalog = await provider.catalog()
        new_item = _catalog_items(new_catalog)[0]
        assert new_context is not old_context
        assert new_binding.context is new_context
        assert new_binding.registration_uuid != old_binding.registration_uuid
        assert _item_revision(new_item) != _item_revision(old_item)
        assert _item_sha(new_item) == _item_sha(old_item)
        assert new_item["module_bytes"] == old_item["module_bytes"]
        with pytest.raises(MobileUiStaleRevision):
            await provider.asset(
                "mobile", _item_revision(old_item), "module", _item_sha(old_item),
            )
        assert (await provider.asset(
            "mobile", _item_revision(new_item), "module", _item_sha(new_item),
        ))["content"] == (tmp_path / "mobile.js").read_text()
    finally:
        if dependency_b is not None:
            await dependency_b.dispose()
        if dependency_a is not None:
            await dependency_a.dispose()
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_ui_owner_replacement_rejects_stale_slots_and_accepts_new_slots(
    tmp_path: Path,
):
    root = CompositionRoot("ui-owner-reload")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    dependency_a = None
    dependency_b = None
    dependency_a_code = tmp_path / "dependency-a"
    dependency_b_code = tmp_path / "dependency-b"
    dependency_a_code.mkdir()
    dependency_b_code.mkdir()

    async def provide(ctx, value: str):
        await ctx.provide(DEPENDENCY, value)

    def runtime(plugin_id: str, code: Path) -> PluginRuntime:
        return PluginRuntime(
            plugin_id=plugin_id,
            generation_id=plugin_id + "-generation",
            plugin_dir=code,
            data_dir=code / "data",
            workspace=code,
            config={},
        )

    try:
        dependency_a = await root.mount(
            lambda ctx: provide(ctx, "A"),
            name="dependency-a",
            runtime=runtime("dependency-a", dependency_a_code),
        )
        ui_fiber = await root.mount(
            ui_plugin.apply,
            name="ui",
            inject=(DEPENDENCY,),
        )
        _owner, state = await mount_ui_owner(root, tmp_path)
        old_context = _owner_context(state)
        old_slots = root.context.require(UI_SLOTS)
        old_binding = old_slots.bindings()[0]
        await dependency_a.dispose()
        assert ui_fiber.state is FiberState.PENDING

        dependency_b = await root.mount(
            lambda ctx: provide(ctx, "B"),
            name="dependency-b",
            runtime=runtime("dependency-b", dependency_b_code),
        )
        assert ui_fiber.state is FiberState.ACTIVE
        new_slots = root.context.require(UI_SLOTS)
        assert new_slots is not old_slots
        assert new_slots.bindings()[0].registration_uuid != old_binding.registration_uuid
        with pytest.raises(CompositionError) as stale_bindings:
            old_slots.bindings()
        assert stale_bindings.value.code == "STALE_ACTIVATION"
        with pytest.raises(CompositionError) as stale_contributors:
            old_slots.contributors()
        assert stale_contributors.value.code == "STALE_ACTIVATION"
        with pytest.raises(CompositionError) as stale_registration:
            await old_slots.register_mobile(
                old_context,
                MobileUiDefinition(module="mobile.js"),
                query=lambda method, payload, *, session_id, turn_id: {},
            )
        assert stale_registration.value.code == "STALE_ACTIVATION"
        assert await provider.catalog()
        assert new_slots.bindings()[0].context is _owner_context(state)
    finally:
        if dependency_b is not None:
            await dependency_b.dispose()
        if dependency_a is not None:
            await dependency_a.dispose()
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_available_runs_in_real_target_permit_and_peer_is_unchanged(tmp_path: Path):
    root = CompositionRoot("available")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    observed: list[tuple[FiberState, bool]] = []
    peer_calls: list[str] = []

    def available() -> bool:
        context = _owner_context(owner_state)
        captured = context.capture_runtime_scope()
        captured._close()  # pyright: ignore[reportPrivateUsage]
        observed.append((context.fiber.state, True))
        return True

    owner_state: _OwnerState = {}
    try:
        await root.mount(ui_plugin.apply, name="ui")
        _fiber, owner_state = await mount_ui_owner(
            root, tmp_path, available=available,
        )
        peer_code = tmp_path / "peer"
        peer_code.mkdir()

        async def peer_apply(ctx):
            peer_calls.append(ctx.runtime.plugin_id)

        peer = await root.mount(
            peer_apply,
            name="peer",
            runtime=PluginRuntime(
                plugin_id="peer", generation_id="peer-generation",
                plugin_dir=peer_code, data_dir=peer_code / "data",
                workspace=peer_code, config={},
            ),
        )
        peer_identity = (peer.context, peer.context.fiber.activation_token, peer_calls[:])
        catalog = await provider.catalog()
        assert catalog["items"]
        assert observed == [(FiberState.ACTIVE, True)]
        assert (peer.context, peer.context.fiber.activation_token, peer_calls[:]) == peer_identity
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", ["OWNER_UNAVAILABLE", "STALE_ACTIVATION"])
async def test_mobile_available_false_and_errors_share_one_entry_contract(
    tmp_path: Path, error_code: str, monkeypatch,
):
    root = CompositionRoot("available-errors")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    asset_reads: list[str] = []
    handler_calls: list[str] = []
    original_asset_content = mobile_ui_module._asset_content

    def observe_asset_content(asset, kind):
        asset_reads.append(kind)
        return original_asset_content(asset, kind)

    def hidden_query(method, payload, *, session_id, turn_id):
        _ = payload, session_id, turn_id
        handler_calls.append(method)
        return {"ok": True}

    monkeypatch.setattr(mobile_ui_module, "_asset_content", observe_asset_content)
    try:
        await root.mount(ui_plugin.apply, name="ui")
        await mount_ui_owner(
            root,
            tmp_path,
            name="hidden",
            available=lambda: False,
            query=hidden_query,
        )
        hidden_catalog = await provider.catalog()
        assert hidden_catalog["items"] == []
        hidden_binding = root.context.require(UI_SLOTS).bindings()[0]
        hidden_revision = provider._plugin_revision(hidden_binding)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(MobileUiPluginUnavailable):
            await provider.asset(
                "hidden", hidden_revision, "module", hidden_binding.asset.module_sha256,
            )
        with pytest.raises(MobileUiPluginUnavailable):
            await provider.query(
                "hidden", hidden_revision, "hidden", {},
                session_id=None, turn_id=None,
            )
        assert asset_reads == []
        assert handler_calls == []

        error_code_dir = tmp_path / "error"
        error_code_dir.mkdir()
        (error_code_dir / "mobile.js").write_text("export const x = 1;")
        (error_code_dir / "mobile.css").write_text(".x {}")

        def fail_available() -> bool:
            raise CompositionError(error_code, "available failed")

        await mount_ui_owner(
            root, error_code_dir, name="broken", available=fail_available,
        )
        broken_binding = next(
            binding for binding in root.context.require(UI_SLOTS).bindings()
            if binding.descriptor.owner == "broken"
        )
        broken_revision = provider._plugin_revision(broken_binding)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(CompositionError, match="available failed"):
            await provider.catalog()
        with pytest.raises(CompositionError, match="available failed"):
            await provider.asset(
                "broken", broken_revision, "module", broken_binding.asset.module_sha256,
            )
        with pytest.raises(CompositionError, match="available failed"):
            await provider.query(
                "broken", broken_revision, "broken", {},
                session_id=None, turn_id=None,
            )
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_asset_and_query_retain_same_task_permit_during_unloading(
    tmp_path: Path,
):
    root = CompositionRoot("unloading-retain")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    dispose_task = None
    peer_task = None
    peer_release = asyncio.Event()
    peer_cleanup_count = 0
    try:
        await root.mount(ui_plugin.apply, name="ui")
        owner, state = await mount_ui_owner(root, tmp_path)
        peer_code = tmp_path / "peer"
        peer_code.mkdir()
        write_assets(peer_code)
        peer, peer_state = await mount_ui_owner(
            root, peer_code, name="peer", register=False,
        )
        context = _owner_context(state)
        peer_context = _owner_context(peer_state)

        async def setup_peer_cleanup():
            async def cleanup_peer():
                nonlocal peer_cleanup_count
                peer_cleanup_count += 1

            return cleanup_peer

        peer_effect = await peer_context.effect(
            setup_peer_cleanup, label="test:mobile-ui-peer",
        )
        peer_identity = (
            peer_context,
            peer.context.fiber.activation_token,
            peer.state,
            peer_context.fiber.state,
            peer_effect,
            peer_cleanup_count,
        )
        item = _catalog_items(await provider.catalog())[0]

        async with context.runtime_scope():
            dispose_task = asyncio.create_task(owner.dispose())
            await context._fiber._admission_closed.wait()  # pyright: ignore[reportPrivateUsage]
            peer_entered = asyncio.Event()

            async def peer_call():
                async with peer_context.runtime_scope():
                    peer_entered.set()
                    await peer_release.wait()

            peer_task = asyncio.create_task(peer_call())
            await peer_entered.wait()
            assert (
                peer_context,
                peer.context.fiber.activation_token,
                peer.state,
                peer_context.fiber.state,
                peer_effect,
                peer_cleanup_count,
            ) == peer_identity
            assert peer_cleanup_count == 0
            asset = await provider.asset(
                "mobile", _item_revision(item), "module", _item_sha(item),
            )
            assert asset["content"] == (tmp_path / "mobile.js").read_text()
            result = await provider.query(
                "mobile", _item_revision(item), "retained", {},
                session_id=None, turn_id=None,
            )
            assert result["method"] == "retained"
            with pytest.raises(MobileUiPluginUnavailable):
                await asyncio.create_task(
                    provider.asset(
                        "mobile", _item_revision(item), "module", _item_sha(item),
                    )
                )
            peer_release.set()
            await peer_task
            assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
            assert not peer_context._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]

        await dispose_task
        assert root.context.require(UI_SLOTS).bindings() == ()
        assert not context._fiber._in_flight_calls  # pyright: ignore[reportPrivateUsage]
    finally:
        peer_release.set()
        if peer_task is not None:
            await asyncio.gather(peer_task, return_exceptions=True)
        if dispose_task is not None:
            await asyncio.gather(dispose_task, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_ui_provider_rejects_new_tasks_during_ui_owner_unloading(
    tmp_path: Path,
):
    root = CompositionRoot("ui-unloading")
    write_assets(tmp_path)
    provider = PluginMobileUiProvider(root)
    ui_dispose_task = None
    try:
        ui_fiber = await root.mount(ui_plugin.apply, name="ui")
        await mount_ui_owner(root, tmp_path)
        item = _catalog_items(await provider.catalog())[0]
        ui_context, _slots = provider._ui_slots()  # pyright: ignore[reportPrivateUsage]
        async with ui_context.runtime_scope():
            ui_dispose_task = asyncio.create_task(ui_fiber.dispose())
            await ui_context._fiber._admission_closed.wait()  # pyright: ignore[reportPrivateUsage]
            retained_catalog = await provider.catalog()
            assert isinstance(retained_catalog["items"], list)
            with pytest.raises(MobileUiPluginUnavailable):
                await asyncio.create_task(provider.catalog())
            with pytest.raises(MobileUiPluginUnavailable):
                await asyncio.create_task(
                    provider.asset(
                        "mobile", _item_revision(item), "module", _item_sha(item),
                    )
                )
            with pytest.raises(MobileUiPluginUnavailable):
                await asyncio.create_task(
                    provider.query(
                        "mobile", _item_revision(item), "ui-unloading", {},
                        session_id=None, turn_id=None,
                    )
                )
            assert provider._admitted_queries == 0  # pyright: ignore[reportPrivateUsage]
        await ui_dispose_task
        assert ui_fiber.state.value == "disposed"
    finally:
        if ui_dispose_task is not None:
            await asyncio.gather(ui_dispose_task, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_mobile_registration_rejects_cross_root_escaped_assets_and_stale_provider(
    tmp_path: Path,
):
    left, right = CompositionRoot("left"), CompositionRoot("right")
    code = tmp_path / "owner"
    code.mkdir()
    foreign = tmp_path / "foreign.js"
    foreign.write_text("export function activate() {}")
    (code / "mobile.js").write_text("export function activate() {}")
    (code / "mobile.css").write_text(".panel {}")
    (code / "linked.js").symlink_to(foreign)
    provider = PluginMobileUiProvider(left)
    borrowed_root = CompositionRoot("borrowed")
    borrowed_provider = PluginMobileUiProvider(borrowed_root)
    borrowed_effect = None

    def query(method, payload, *, session_id, turn_id):
        return {}

    try:
        await left.mount(ui_plugin.apply, name="ui")
        await right.mount(ui_plugin.apply, name="ui")
        _left_fiber, left_state = await mount_ui_owner(left, code, register=False)
        _right_fiber, right_state = await mount_ui_owner(right, code, register=False)
        slots = left.context.require(UI_SLOTS)
        with pytest.raises(ValueError, match="实际 Root"):
            await slots.register_mobile(
                _owner_context(right_state), MobileUiDefinition(module="mobile.js"), query=query,
            )
        for path in ("../foreign.js", "linked.js", str(foreign)):
            with pytest.raises(RuntimeError, match="mobile UI"):
                await slots.register_mobile(
                    _owner_context(left_state), MobileUiDefinition(module=path), query=query,
                )

        borrowed_effect = await borrowed_root.context.provide(
            UI_SLOTS, left.context.require(UI_SLOTS),
        )
        with pytest.raises(MobileUiPluginUnavailable, match="不属于当前 Root"):
            await borrowed_provider.catalog()
    finally:
        if borrowed_effect is not None:
            await borrowed_effect.aclose()
        await borrowed_provider.aclose()
        await borrowed_root.dispose()
        await provider.aclose()
        await right.dispose()
        await left.dispose()


@pytest.mark.asyncio
async def test_mobile_provider_requires_ui_service(tmp_path: Path):
    root = CompositionRoot("missing")
    provider = PluginMobileUiProvider(root)
    try:
        with pytest.raises(MobileUiPluginUnavailable, match="没有 Mobile UI"):
            await provider.catalog()
    finally:
        await provider.aclose()
        await root.dispose()
