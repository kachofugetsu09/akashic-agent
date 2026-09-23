"""生产 Core 默认只解析安装制品，开发目录必须显式提供。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config_models import Config
from agent.plugin_composition import CompositionError, FiberState, ServiceKey
from bootstrap import tools as bootstrap
from core.net.http import SharedHttpResources
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def test_resolve_plugin_dirs_does_not_add_checkout_plugins_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)

    roots = bootstrap._resolve_plugin_dirs(tmp_path)

    assert roots == []
    checkout_plugins = Path(bootstrap.__file__).resolve().parents[1] / "plugins"
    assert checkout_plugins not in roots


def test_resolve_plugin_dirs_accepts_only_explicit_development_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = tmp_path / "development-plugins"
    extra = tmp_path / "extra-plugins"
    monkeypatch.setenv("AKASHIC_EXTRA_PLUGIN_DIRS", str(extra))

    roots = bootstrap._resolve_plugin_dirs(tmp_path, plugin_dirs=[development])

    assert roots == [development, extra]


@pytest.mark.asyncio
async def test_core_starts_with_no_checkout_plugins_and_keeps_manager_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    try:
        assert core.plugin_manager.discover() == []
        assert core.plugin_manager._dirs == []
        assert core.restart_gate.boot_id != "unmanaged"
        assert core.plugin_manager._host_boot_id == core.restart_gate.boot_id
        await core.start()
        root = core.plugin_manager.live_root
        assert root is not None
        assert core.plugin_manager.discover() == []
        assert core.plugin_manager.current_snapshot is None
        first = await core.inspect_modules()
        assert core.plugin_manager.live_root is root
        assert "identity:" in first
        assert "revision:" in first
        assert await core.inspect_modules() == first
    finally:
        await core.stop()
        await core.bus.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_core_inspection_cold_starts_one_live_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """冷查询只初始化一次，并继续读取同一正式 Root。"""

    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    try:
        assert core.plugin_manager.live_root is None
        first = await core.inspect_modules()
        root = core.plugin_manager.live_root
        assert root is not None
        assert core.plugin_manager.current_snapshot is None
        assert await core.inspect_modules() == first
        assert core.plugin_manager.live_root is root
    finally:
        await core.stop()
        await core.bus.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_warm_core_inspection_tracks_live_fibers_and_local_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """热查询读取同一 Root，并隔离真实局部失败分支。"""

    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    peer_fiber = None
    bad_fiber = None
    try:
        await core.start()
        manager = core.plugin_manager
        root = manager.live_root
        assert root is not None
        root_identity = root.instance_token
        initial_topology = root.topology_view()
        initial_text = await core.inspect_modules()
        assert f"identity: {initial_topology.identity}" in initial_text
        assert f"revision: {initial_topology.composition_revision}" in initial_text
        assert "fiber:" not in initial_text

        service = ServiceKey[list[str]]("test.inspect.peer")
        peer_value = ["peer-value"]
        peer_contexts = []
        peer_closes = []

        async def peer(ctx) -> None:
            peer_contexts.append(ctx)
            await ctx.provide(service, peer_value)

            async def close() -> None:
                peer_closes.append("closed")

            await ctx.effect(lambda: close)

        peer_fiber = await root.mount(peer, name="inspect-peer")
        assert peer_fiber.state is FiberState.ACTIVE
        assert len(peer_contexts) == 1
        peer_context = peer_contexts[0]
        peer_handle = peer_context.fiber
        peer_activation = peer_handle.activation_token
        peer_effects = tuple(peer_fiber.effects)
        assert peer_handle.state is FiberState.ACTIVE

        missing = ServiceKey[object]("test.inspect.missing")

        async def pending(ctx) -> None:
            ctx.require(missing)

        bad_fiber = await root.mount(
            pending, name="inspect-bad", inject=(missing,),
        )
        assert bad_fiber.state is FiberState.PENDING
        failed_receipt = root.receipt()
        assert not failed_receipt.ready
        assert "inspect-bad" in failed_receipt.required_pending
        assert bad_fiber.missing_services == (missing.name,)

        bad_scope_body: list[bool] = []
        with pytest.raises(CompositionError) as caught:
            async with bad_fiber.context.runtime_scope():
                bad_scope_body.append(True)
        assert caught.value.code == "OWNER_UNAVAILABLE"
        assert bad_scope_body == []
        assert not bad_fiber._in_flight_calls

        changed_topology = root.topology_view()
        assert changed_topology.identity != initial_topology.identity
        assert changed_topology.composition_revision > initial_topology.composition_revision
        mounted_text = await core.inspect_modules()
        assert f"identity: {changed_topology.identity}" in mounted_text
        assert f"revision: {changed_topology.composition_revision}" in mounted_text
        assert "fiber: <root> -> inspect-peer" in mounted_text
        assert "fiber: <root> -> inspect-bad" in mounted_text
        assert manager.live_root is root
        assert root.instance_token is root_identity
        assert not root.frozen

        async with peer_context.runtime_scope():
            assert peer_context.require(service) is peer_value
        assert peer_handle.activation_token is peer_activation
        assert peer_handle.state is FiberState.ACTIVE
        assert peer_fiber.context is peer_context
        assert tuple(peer_fiber.effects) == peer_effects
        assert not peer_fiber._in_flight_calls
        assert peer_closes == []

        async def reject_load_all() -> None:
            raise AssertionError("warm inspection must not reload the formal Root")

        async def reject_snapshot_acquire(snapshot_id=None, *, selector="stable"):
            raise AssertionError("warm inspection must not acquire a snapshot lease")

        def reject_snapshot_compile(
            generations, *, snapshot_revision="", composition_root=None,
        ):
            raise AssertionError("warm inspection must not compile a snapshot")

        def reject_ready_gate():
            raise AssertionError("warm inspection must not gate on the Root receipt")

        monkeypatch.setattr(manager, "load_all", reject_load_all)
        monkeypatch.setattr(manager.snapshot_store, "acquire", reject_snapshot_acquire)
        monkeypatch.setattr(manager._snapshot_compiler, "compile", reject_snapshot_compile)
        monkeypatch.setattr(root, "receipt", reject_ready_gate)

        inspected = await core.inspect_modules()
        assert manager.live_root is root
        assert f"identity: {changed_topology.identity}" in inspected
        assert "fiber: <root> -> inspect-bad" in inspected
        assert "identity:" in inspected
        assert "revision:" in inspected
        assert manager.live_root is root
        assert root.instance_token is root_identity
        assert peer_fiber.context is peer_context
        assert tuple(peer_fiber.effects) == peer_effects

        await bad_fiber.dispose()
        after_dispose = await core.inspect_modules()
        assert "fiber: <root> -> inspect-bad" not in after_dispose
        assert manager.live_root is root
        assert root.instance_token is root_identity
        assert peer_handle.activation_token is peer_activation
        assert peer_handle.state is FiberState.ACTIVE
        assert peer_fiber.context is peer_context
        assert tuple(peer_fiber.effects) == peer_effects
        await peer_fiber.dispose()
        assert peer_closes == ["closed"]
        assert not peer_fiber._in_flight_calls
    finally:
        await core.stop()
        await core.bus.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_unmanaged_core_runtime_gets_a_new_boot_id_per_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个无 supervisor 的真实 Core host 不能共享 transport boot identity。"""

    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first_http = SharedHttpResources()
    second_http = SharedHttpResources()
    first = bootstrap.build_core_runtime(Config(), first_workspace, first_http)
    second = bootstrap.build_core_runtime(Config(), second_workspace, second_http)
    try:
        assert first.restart_gate.boot_id != second.restart_gate.boot_id
        assert first.plugin_manager._host_boot_id == first.restart_gate.boot_id
        assert second.plugin_manager._host_boot_id == second.restart_gate.boot_id
    finally:
        await first.stop()
        await first.bus.aclose()
        await first_http.aclose()
        await second.stop()
        await second.bus.aclose()
        await second_http.aclose()


@pytest.mark.asyncio
async def test_build_core_runtime_keeps_explicit_plugin_dirs_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = tmp_path / "development-plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)

    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(
        Config(), workspace, http, plugin_dirs=[development]
    )
    try:
        assert core.plugin_manager._dirs == [development]
    finally:
        await core.stop()
        await core.bus.aclose()
        await http.aclose()
