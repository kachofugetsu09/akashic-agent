import asyncio
import importlib
from pathlib import Path
from collections.abc import Mapping
from typing import cast

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
async def apply(ctx):
    if os.environ["ARCHIVE_PROVIDER_ACTIVE"] != "yes":
        return
    config = Config.model_validate(ctx.config)
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
async def apply(ctx):
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
        current = host.current_snapshot
        assert current is not None
        root = await host._resolve_composition_root(dict(current.generations))
        assert root.receipt().ready
        try:
            value = cast(Mapping[str, object], root.service_value(RESULT))
            assert value["asset"] == "asset A"
        finally:
            await root.dispose()
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_binding_capture_keeps_declared_dependency_provenance(
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
async def apply(ctx):
    await ctx.provide(ServiceKey("archive.fixture.delivery"), {})
    async def child(child_ctx):
        await child_ctx.provide(ServiceKey("archive.test.result"), child_ctx.require(ServiceKey("archive.test.value")))
    await ctx.mount(child, name="child-provider", inject=(ServiceKey("archive.test.value"),))
""")
    unrelated = plugins / "unrelated"
    unrelated.mkdir()
    (unrelated / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "unrelated"
version = "1.0.0"
LEGACY_DELIVERY = ServiceKey("archive.fixture.delivery")
inject = (LEGACY_DELIVERY,)
async def apply(ctx):
    pass
""")
    host = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        assert len(host.current_snapshot.generations) == 3
        snapshot = host.current_snapshot
        assert snapshot.composition_root is not None
        binding = Bindings(log, host._archive, snapshot.composition_root)
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = binding.bind(RESULT, {})
        descriptor = log.read_binding(identity)
        root_descriptor = host._archive.read_descriptor(cast(str, descriptor["root_ref"]))
        component_ids = {
            cast(str, host._archive.read_descriptor(ref)["plugin_id"])
            for ref in cast(tuple[str, ...], root_descriptor["components"])
        }
        assert component_ids == {"consumer", "provider"}
        async with binding.open(identity, RESULT) as (state, _):
            assert state["text"] == "old:A"
            selected = get_current_runtime_snapshot()
            assert selected is snapshot
            assert set(selected.generations) == {
                "consumer",
                "provider",
                "unrelated",
            }
    finally:
        log.close()
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [False, True])
async def test_capture_registry_contributor_uses_actual_live_context(
    tmp_path, monkeypatch, dynamic
):
    from agent.plugin_composition import Context

    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    if dynamic:
        consumer = plugins / "consumer/plugin.py"
        consumer.write_text(consumer.read_text().replace(
            'ctx.provide(ServiceKey("archive.test.result"), ctx.require(inject[0]))',
            'ctx.provide(ServiceKey("archive.test.result"), ctx.require(inject[0]), '
            'binding_contributors=lambda: (ctx.require(inject[0])["registration"],))'))
    addon = plugins / "addon"
    addon.mkdir()
    (addon / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "addon"
version = "1.0.0"
inject = (ServiceKey("archive.test.value"),)
async def apply(ctx):
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
        snapshot = host.current_snapshot
        assert snapshot.composition_root is not None
        binding = Bindings(log, host._archive, snapshot.composition_root)
        async with lease_runtime_snapshot(host.snapshot_store):
            for invalid in (foreign, Context(context._root, context._fiber)):
                with pytest.raises(ValueError, match="不属于"):
                    if dynamic:
                        host.current_snapshot.composition_root.context.require(RESULT)["registration"] = invalid
                        binding.bind(RESULT, {})
                    else:
                        binding.bind(RESULT, {}, contributors=(invalid,))
            host.current_snapshot.composition_root.context.require(RESULT)["registration"] = context
            identity = binding.bind(
                RESULT, {"target": "extra"}, contributors=() if dynamic else (context,)
            )
        async with binding.open(identity, RESULT) as (state, _):
            assert state["extra"] == "registered A"
            selected = get_current_runtime_snapshot()
            assert selected is snapshot
            assert set(selected.generations) == {
                "addon",
                "consumer",
                "provider",
            }
    finally:
        log.close()
        await other.terminate_all()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_binding_open_uses_selected_scope_without_reopening_archive(
    tmp_path, monkeypatch,
):
    """打开旧 binding 只读取事实，并使用当前调用已经选定的 Root。"""
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    host = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        snapshot = host.current_snapshot
        assert snapshot is not None
        root = snapshot.composition_root
        assert root is not None
        bindings = Bindings(log, host._archive, root)  # type: ignore[arg-type]
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = bindings.bind(RESULT, {"choice": "stable"})
            expected = root.context.require(RESULT)

            def unexpected_archive_read(_identity):
                raise AssertionError("binding.open 不应重新读取历史 archive")

            monkeypatch.setattr(host._archive, "read_descriptor", unexpected_archive_read)
            async with bindings.open(identity, RESULT) as (value, metadata):
                assert value is expected
                assert metadata == {"choice": "stable"}
                assert get_current_runtime_snapshot() is snapshot
            assert get_current_runtime_snapshot() is snapshot
        assert get_current_runtime_snapshot() is None
    finally:
        log.close()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_binding_open_acquires_own_root_once_and_rejects_unrelated_scope(
    tmp_path, monkeypatch,
):
    """缺 scope 只从所属 Root 获取一次；已有其他 Root 时明确拒绝跨 Root。"""
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    host = manager(tmp_path, [plugins])
    other = manager(tmp_path / "other", [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        await other.load_all()
        snapshot = host.current_snapshot
        other_snapshot = other.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert other_snapshot is not None
        root = snapshot.composition_root
        bindings = Bindings(log, host._archive, root)  # type: ignore[arg-type]
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = bindings.bind(RESULT, {})

        acquire_calls = 0
        original_acquire = root._acquire_runtime_scope  # type: ignore[union-attr]

        async def acquire_once():
            nonlocal acquire_calls
            acquire_calls += 1
            return await original_acquire()

        monkeypatch.setattr(root, "_acquire_runtime_scope", acquire_once)
        async with bindings.open(identity, RESULT) as (value, _):
            assert value is root.context.require(RESULT)  # type: ignore[union-attr]
            assert get_current_runtime_snapshot() is snapshot
        assert acquire_calls == 1
        assert get_current_runtime_snapshot() is None

        async with lease_runtime_snapshot(other.snapshot_store):
            with pytest.raises(RuntimeError, match="所属 Root"):
                async with bindings.open(identity, RESULT):
                    pytest.fail("不相干 Root 不应静默改选 stable")
            assert get_current_runtime_snapshot() is other_snapshot
    finally:
        log.close()
        await other.terminate_all()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_binding_open_releases_fallback_scope_when_cancelled(tmp_path, monkeypatch):
    """fallback scope 被取消时仍释放 lease，不把取消变成残留运行时。"""
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    host = manager(tmp_path, [plugins])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        snapshot = host.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        bindings = Bindings(log, host._archive, snapshot.composition_root)  # type: ignore[arg-type]
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = bindings.bind(RESULT, {})

        entered = asyncio.Event()

        async def use_binding():
            async with bindings.open(identity, RESULT):
                entered.set()
                await asyncio.Future()

        task = asyncio.create_task(use_binding())
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert snapshot.lease_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert snapshot.lease_count == 0
        assert get_current_runtime_snapshot() is None
    finally:
        log.close()
        await host.terminate_all()
