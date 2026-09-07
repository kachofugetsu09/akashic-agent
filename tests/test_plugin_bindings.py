import asyncio
import importlib
import shutil
import sys
from pathlib import Path

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.model import ServiceKey
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import get_current_runtime_snapshot, lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog

VALUE = ServiceKey("archive.test.value")
RESULT = ServiceKey("archive.test.result")


def write_plugins(path: Path):
    provider = path / "provider"
    provider.mkdir(parents=True)
    (provider / "helper.py").write_text("VALUE = 'A'\n")
    (provider / "asset.txt").write_text("asset A")
    (provider / "plugin.py").write_text("""
import os
from pydantic import BaseModel
from agent.plugin_composition import ServiceKey, RUNTIME_STARTED
from .helper import VALUE
api_version = 3
name = "provider"
version = "1.0.0"
class Config(BaseModel):
    prefix: str = "old:"
def is_active(services):
    return os.environ["ARCHIVE_PROVIDER_ACTIVE"] == "yes"
async def apply(ctx, config):
    state = {"text": config.prefix + VALUE, "started": False, "closed": False,
             "asset": (ctx.runtime.plugin_dir / "asset.txt").read_text()}
    async def start(event):
        state["started"] = True
    def setup():
        def cleanup():
            state["closed"] = True
        return cleanup
    await ctx.effect(setup)
    await ctx.on(RUNTIME_STARTED, start)
    await ctx.provide(ServiceKey("archive.test.value"), state)
""")
    consumer = path / "consumer"
    consumer.mkdir()
    (consumer / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "consumer"
version = "1.0.0"
inject = (ServiceKey("archive.test.value"),)
async def apply(ctx, config):
    await ctx.provide(ServiceKey("archive.test.result"), ctx.require(inject[0]))
""")


def manager(tmp_path, plugins):
    return PluginManager(
        plugin_dirs=plugins,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


@pytest.mark.asyncio
async def test_binding_restarts_without_source_and_keeps_exact_config_and_lifecycle(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    first = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await first.load_all()
        binding = Bindings(log, first._archive, first.open_binding)
        async with lease_runtime_snapshot(first.snapshot_store):
            identity = binding.bind(RESULT, {"choice": "fixed"})
            assert binding.bind(RESULT, {"choice": "fixed"}) == identity
        original_snapshot = first.current_snapshot
        assert original_snapshot is not None
        (plugins / "provider" / "helper.py").write_text("VALUE = 'B'\n")
        (plugins / "provider" / "asset.txt").write_text("asset B")
        monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "no")
        await first.terminate_all()
        shutil.rmtree(plugins)
        log.close()
        log = MessageLog(tmp_path / "messages.db")
        second = manager(tmp_path, [])
        recovered = Bindings(log, second._archive, second.open_binding)
        modules_before = set(sys.modules)
        async with recovered.open(identity, RESULT) as (state, metadata):
            assert state == {
                "text": "old:A",
                "asset": "asset A",
                "started": False,
                "closed": False,
            }
            assert metadata == {"choice": "fixed"}
            with pytest.raises(TypeError):
                metadata["choice"] = "changed"
            current = get_current_runtime_snapshot()
            assert current is not None
            assert current is not original_snapshot
            assert second.current_snapshot is None
        assert state["closed"] is True
        assert get_current_runtime_snapshot() is None
        assert not [
            name
            for name in set(sys.modules) - modules_before
            if name.startswith("_akashic_archive_")
        ]
        await second.terminate_all()
    finally:
        log.close()
        await first.terminate_all()


@pytest.mark.asyncio
async def test_missing_provider_and_runtime_mismatch_do_not_use_current_root(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    host = manager(tmp_path, [plugins])
    try:
        await host.load_all()
        current = host.current_snapshot
        provider = host.generation("provider")
        consumer = host.generation("consumer")
        with pytest.raises(RuntimeError, match="闭包不完整"):
            async with host.open_binding((consumer.archive_ref,)):
                pytest.fail("current provider must not fill the archive")
        descriptor = dict(host._archive.read_descriptor(provider.archive_ref))
        descriptor["runtime"] = {"python_tag": "other", "binding_api": 1}
        incompatible = host._archive.save_descriptor(descriptor)
        with pytest.raises(RuntimeError, match="不兼容"):
            async with host.open_binding((incompatible, consumer.archive_ref)):
                pytest.fail("incompatible archive must not load")
        assert host.current_snapshot is current
        async with host.open_binding(
            (provider.archive_ref, consumer.archive_ref)
        ) as scope:
            assert scope.require(RESULT)["text"] == "old:A"
        with pytest.raises(RuntimeError, match="关闭"):
            scope.require(RESULT)
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_loaded_generation_keeps_assets_and_late_imports_after_source_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    (plugins / "provider" / "late.py").write_text("VALUE = 'late A'\n")
    host = manager(tmp_path, [plugins])
    try:
        await host.load_all()
        generation = host.generation("provider")
        (plugins / "provider" / "asset.txt").write_text("asset B")
        (plugins / "provider" / "late.py").write_text("VALUE = 'late B'\n")
        alias = host._stable_aliases[generation.module_path]
        assert importlib.import_module(alias + ".late").VALUE == "late A"
        root, ready = await host._resolve_composition_root(
            host.current_snapshot.generations, force_fresh=True
        )
        assert ready
        try:
            assert root.service_value(RESULT)["asset"] == "asset A"
        finally:
            await root.dispose()
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_archive_close_drains_retained_scope_even_when_cancelled(
    tmp_path, monkeypatch, cancel
):
    from agent.plugins.snapshot import get_current_runtime_lease, RuntimeSnapshotStore

    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    host = manager(tmp_path, [plugins])
    await host.load_all()
    refs = tuple(g.archive_ref for g in host.current_snapshot.generations.values())
    retained = None
    state = None
    waiting = asyncio.Event()
    original_wait = RuntimeSnapshotStore.wait_for_no_leases

    async def observe_wait(store, snapshot):
        waiting.set()
        await original_wait(store, snapshot)

    monkeypatch.setattr(RuntimeSnapshotStore, "wait_for_no_leases", observe_wait)

    async def use_binding():
        nonlocal retained, state
        async with host.open_binding(refs) as scope:
            state = scope.require(RESULT)
            retained = get_current_runtime_lease().fork()

    task = asyncio.create_task(use_binding())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        assert not task.done()
        assert not state["closed"]
        if cancel:
            task.cancel()
        await retained.release()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert state["closed"]
    finally:
        if retained is not None:
            await retained.release()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_archive_apply_cannot_resolve_formal_delivery_port(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    path = plugins / "provider" / "plugin.py"
    path.write_text(
        path.read_text().replace(
            "async def apply(ctx, config):",
            "from agent.plugin_composition import DELIVERIES\n"
            "inject = (DELIVERIES,)\n"
            "async def apply(ctx, config):",
        )
    )
    host = manager(tmp_path, [plugins])
    try:
        await host.load_all()
        refs = tuple(g.archive_ref for g in host.current_snapshot.generations.values())
        with pytest.raises(RuntimeError, match="闭包不完整"):
            async with host.open_binding(refs):
                pytest.fail("archive must not borrow the live delivery owner")
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_capture_keeps_child_provider_dependency_and_excludes_unrelated_owner(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    (plugins / "consumer" / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "consumer"
version = "1.0.0"
async def apply(ctx, config):
    async def child(child_ctx):
        await child_ctx.provide(ServiceKey("archive.test.result"), child_ctx.require(ServiceKey("archive.test.value")))
    await ctx.mount(child, name="child-provider", inject=(ServiceKey("archive.test.value"),))
""")
    unrelated = plugins / "unrelated"
    unrelated.mkdir()
    (unrelated / "plugin.py").write_text("""
from agent.plugin_composition import DELIVERIES
api_version = 3
name = "unrelated"
version = "1.0.0"
inject = (DELIVERIES,)
async def apply(ctx, config):
    pass
""")
    host = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        assert len(host.current_snapshot.generations) == 3
        binding = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = binding.bind(RESULT, {})
        async with binding.open(identity, RESULT) as (state, _):
            assert state["text"] == "old:A"
            assert set(get_current_runtime_snapshot().generations) == {
                "consumer",
                "provider",
            }
    finally:
        log.close()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_capture_registry_contributor_uses_actual_live_context(
    tmp_path, monkeypatch
):
    from agent.plugin_composition import Context

    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    addon = plugins / "addon"
    addon.mkdir()
    (addon / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "addon"
version = "1.0.0"
inject = (ServiceKey("archive.test.value"),)
async def apply(ctx, config):
    value = ctx.require(inject[0])
    value["registration"] = ctx
    value["extra"] = "registered A"
""")
    host = manager(tmp_path, [plugins])
    other = manager(tmp_path / "other", [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        await other.load_all()
        context = host.current_snapshot.composition_root.context.require(RESULT)[
            "registration"
        ]
        foreign = other.current_snapshot.composition_root.context.require(RESULT)[
            "registration"
        ]
        binding = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store):
            for invalid in (foreign, Context(context._root, context._fiber)):
                with pytest.raises(ValueError, match="不属于"):
                    binding.bind(RESULT, {}, contributors=(invalid,))
            identity = binding.bind(
                RESULT, {"target": "extra"}, contributors=(context,)
            )
        async with binding.open(identity, RESULT) as (state, _):
            assert state["extra"] == "registered A"
            assert set(get_current_runtime_snapshot().generations) == {
                "addon",
                "consumer",
                "provider",
            }
    finally:
        log.close()
        await other.terminate_all()
        await host.terminate_all()
