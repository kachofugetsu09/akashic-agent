from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.scope import CleanupFailure, PluginScope


@pytest.mark.asyncio
async def test_disabled_cleanup_retries_a_generation_that_never_reached_a_snapshot(tmp_path):
    """显式禁用清理也能收敛加载回滚留下的 scope，不要求重启整个宿主。"""
    from agent.plugins.generation import PluginContributions, PluginGeneration
    from bus.event_bus import EventBus

    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                            installed_cache_root=tmp_path / "home/cache")
    scope = PluginScope("owner")
    attempts = 0

    def cleanup():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("still open")

    scope.defer("resource", cleanup)
    generation = PluginGeneration(
        plugin_id="owner", generation_id="failed-load", module_path="_failed_load",
        source_revision="source", config_revision="config", plugin_dir=tmp_path,
        data_dir=tmp_path / "data", instance=object(), scope=scope,
        contributions=PluginContributions({}),
    )
    try:
        with pytest.raises(RuntimeError, match="scope cleanup 未完成"):
            await manager._dispose_generation(generation, state="discarded")
        assert attempts == 1
        assert generation.runtime_snapshot is None
        await manager.reconcile_disabled_and_drain("owner")
        assert attempts == 2
        assert scope.closed
        assert "owner" not in manager._draining_generations
    finally:
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_failure_keeps_resource_and_earlier_dependencies_until_explicit_retry():
    scope = PluginScope("owner")
    events = []
    failing = True

    def close_consumer():
        events.append("consumer")
        if failing:
            raise OSError("still open")

    scope.defer("provider", lambda: events.append("provider"))
    scope.defer("consumer", close_consumer)
    scope.defer("last", lambda: events.append("last"))
    assert await scope.aclose() == [CleanupFailure("consumer", "still open")]
    assert events == ["last", "consumer"]
    assert not scope.closed
    with pytest.raises(RuntimeError):
        scope.defer("late", lambda: None)

    failing = False
    assert await scope.aclose() == []
    assert scope.closed
    assert events == ["last", "consumer", "consumer", "provider"]
    assert await scope.aclose() == []
    assert events == ["last", "consumer", "consumer", "provider"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_concurrent_close_and_repeated_cancel_join_one_cleanup(fails):
    scope = PluginScope("owner")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def cleanup():
        calls.append("close")
        entered.set()
        await release.wait()
        if fails:
            raise OSError("still open")

    scope.defer("resource", cleanup)
    first = asyncio.create_task(scope.aclose())
    await entered.wait()
    with pytest.raises(RuntimeError):
        scope.defer("late", lambda: None)
    second = asyncio.create_task(scope.aclose())
    first.cancel()
    asyncio.get_running_loop().call_soon(first.cancel)
    asyncio.get_running_loop().call_soon(release.set)
    with pytest.raises(asyncio.CancelledError):
        await first
    failures = await second
    assert failures == ([CleanupFailure("resource", "still open")] if fails else [])
    assert calls == ["close"]
    assert scope.closed is not fails


@pytest.mark.asyncio
async def test_cleanup_cancel_is_failure_and_keeps_its_dependency():
    scope = PluginScope("owner")
    provider = Mock()

    async def cleanup():
        raise asyncio.CancelledError

    scope.defer("provider", provider)
    scope.defer("consumer", cleanup)
    assert await scope.aclose() == [CleanupFailure("consumer", "CancelledError")]
    provider.assert_not_called()
    assert not scope.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["root", "scope"])
async def test_generation_disposal_keeps_failed_owner_and_module(monkeypatch, stage):
    """真实关闭入口不能在任一资源阶段失败后继续卸载其依赖。"""
    manager = object.__new__(PluginManager)
    scope = PluginScope("owner")
    cleanup = Mock(side_effect=OSError("still open") if stage == "scope" else None)
    scope.defer("asset", cleanup)
    generation = SimpleNamespace(
        plugin_id="owner", generation_id="generation", module_path="module",
        scope=scope, runtime_snapshot=object(), state="prepared",
    )
    manager._building_roots = {}
    manager._draining_generations = {}
    manager._scopes = {}
    manager._cleanup_failures = []
    manager._loaded = {"module"}
    manager._active_plugins = {"module": object()}
    manager._stable_aliases = {"module": "alias"}
    manager._snapshot_store = SimpleNamespace(pause_admission=Mock())
    dispose_root = AsyncMock(side_effect=OSError("still open") if stage == "root" else None)
    remove = Mock()
    monkeypatch.setattr(manager, "_record_root_failure", Mock())
    monkeypatch.setattr(manager, "_dispose_unreferenced_composition_root", dispose_root)
    monkeypatch.setattr(manager, "_remove_module_tree", remove)

    with pytest.raises((OSError, RuntimeError), match="still open"):
        await manager._dispose_generation(generation, state="discarded")
    assert manager._draining_generations["owner"] == [generation]
    assert manager._scopes["module"] is scope
    assert "module" in manager._loaded
    assert generation.state == "prepared"
    assert not scope.closed
    remove.assert_not_called()
    manager._snapshot_store.pause_admission.assert_called_once()
    if stage != "scope":
        cleanup.assert_not_called()

    dispose_root.side_effect = cleanup.side_effect = None
    await manager._dispose_generation(generation, state="discarded")
    assert manager._draining_generations == {}
    assert manager._scopes == {}
    assert scope.closed
    assert generation.state == "discarded"
    assert [call.args[0] for call in remove.call_args_list] == ["module", "alias"]


@pytest.mark.asyncio
async def test_discard_prepared_failure_keeps_candidate_and_does_not_abort(monkeypatch):
    manager = object.__new__(PluginManager)
    generation = SimpleNamespace(generation_id="generation")
    manager._prepared_generations = {"owner": generation}
    dispose = AsyncMock(side_effect=RuntimeError("cleanup pending"))
    abort = Mock()
    monkeypatch.setattr(manager, "_dispose_generation", dispose)
    monkeypatch.setattr(manager, "_abort_reload", abort)
    with pytest.raises(RuntimeError, match="cleanup pending"):
        await manager.discard_prepared("owner")
    assert manager._prepared_generations["owner"] is generation
    abort.assert_not_called()


@pytest.mark.asyncio
async def test_terminate_failure_keeps_scope_module_and_control_owner(monkeypatch):
    """全量关闭不能把尚未构造 generation 的失败 scope 清空。"""
    manager = object.__new__(PluginManager)
    manager._snapshot_store = SimpleNamespace(
        pause_admission=Mock(return_value=None), close=AsyncMock(),
    )
    manager._update_publication = None
    manager._validation_hosts = {}
    manager._plugin_tasks = SimpleNamespace(close=AsyncMock())
    manager._plugin_processes = SimpleNamespace(close=AsyncMock())
    manager._active_channel_generation = None
    manager._active_generations = {}
    manager._prepared_generations = {}
    manager._building_roots = {}
    manager._draining_generations = {}
    manager._cleanup_failures = []
    scope = PluginScope("owner")
    cleanup = Mock(side_effect=OSError("still open"))
    scope.defer("asset", cleanup)
    manager._scopes = {"module": scope}
    manager._loaded = {"module"}
    manager._active_plugins = {"module": object()}
    manager._stable_aliases = {}
    manager._owns_control_frames = True
    manager._control_frames = SimpleNamespace(close=Mock())
    remove = Mock()
    monkeypatch.setattr(manager, "_remove_module_tree", remove)

    with pytest.raises(RuntimeError, match="still open"):
        await manager.terminate_all()
    assert manager._scopes["module"] is scope
    assert "module" in manager._loaded
    assert manager._cleanup_failures == [CleanupFailure("asset", "still open")]
    manager._control_frames.close.assert_not_called()
    remove.assert_not_called()

    cleanup.side_effect = None
    await manager.terminate_all()
    assert manager._scopes == {}
    assert manager._loaded == set()
    manager._control_frames.close.assert_called_once()
