import asyncio
import importlib
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import Mapping
from typing import cast

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.context import CompositionRoot
from agent.plugin_composition.model import FiberState, PluginRuntime, ServiceKey
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog

VALUE = ServiceKey("archive.test.value")
RESULT = ServiceKey("archive.test.result")
DELIVERY = ServiceKey("archive.fixture.delivery")


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
    await ctx.provide(ServiceKey("archive.test.secondary"), {"text": "secondary"})
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


def manager(tmp_path, plugins, *, message_log=None):
    return PluginManager(
        plugin_dirs=plugins,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
        message_log=message_log,
    )


@asynccontextmanager
async def live_binding_root(
    tmp_path, *, plugin_id="provider", with_drain_consumer=False
):
    """Build one live Root whose provider owns a cleanup Effect."""
    root = CompositionRoot(f"bindings-{plugin_id}")
    value = {"text": "old:A"}
    state = {
        "cleanup_calls": 0,
        "contributor_calls": 0,
        "unrelated_events": [],
    }
    consumer_released = asyncio.Event()
    state["consumer_released"] = consumer_released
    state["consumer_cleanup_calls"] = 0

    async def provider(ctx):
        state["senders_context"] = ctx

        def forbidden_contributors():
            state["contributor_calls"] += 1
            raise AssertionError("Bindings.open 不得读取 binding contributors")

        def setup():
            def cleanup():
                state["cleanup_calls"] += 1

            return cleanup

        await ctx.effect(setup, label="binding-provider-resource")
        await ctx.provide(
            RESULT,
            value,
            binding_contributors=forbidden_contributors,
        )

    async def drain_consumer(ctx):
        _ = ctx.require(RESULT)

        def setup():
            def cleanup():
                state["consumer_cleanup_calls"] += 1
                consumer_released.set()

            return cleanup

        await ctx.effect(setup, label="binding-drain-consumer")

    if with_drain_consumer:
        await root.mount(
            drain_consumer,
            name="binding-drain-consumer",
            inject=(RESULT,),
            runtime=PluginRuntime(
                "binding-drain-consumer",
                f"generation-{plugin_id}",
                tmp_path / "data",
                tmp_path / "workspace",
                tmp_path / "plugin",
                {},
            ),
        )
    fiber = await root.mount(
        provider,
        name=plugin_id,
        runtime=PluginRuntime(
            plugin_id,
            f"generation-{plugin_id}",
            tmp_path / "data",
            tmp_path / "workspace",
            tmp_path / "plugin",
            {},
        ),
    )
    try:
        yield root, fiber.context, value, state
    finally:
        await root.dispose()


def seed_binding(log, metadata):
    """Seed only the persisted descriptor fact used by open tests."""
    identity = "fixture-binding"
    log.save_binding(
        identity,
        {
            "version": 1,
            "root_ref": "fixture-root",
            "service": RESULT.name,
            "metadata": metadata,
        },
    )
    return identity


@pytest.mark.asyncio
async def test_loaded_generation_keeps_assets_and_late_imports_after_source_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    (plugins / "provider" / "late.py").write_text("VALUE = 'late A'\n")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [plugins])
    try:
        await host.load_all()
        generation = host.generation("provider")
        (plugins / "provider" / "asset.txt").write_text("asset B")
        (plugins / "provider" / "late.py").write_text("VALUE = 'late B'\n")
        assert importlib.import_module(generation.module_path + ".late").VALUE == "late A"
        root = host._live_root
        assert root is not None and root.receipt().ready
        value = cast(Mapping[str, object], root.service_value(RESULT))
        assert value["asset"] == "asset A"
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
from agent.plugin_composition.bindings import BINDINGS
api_version = 3
name = "consumer"
version = "1.0.0"
async def apply(ctx):
    delivery = {}
    await ctx.provide(ServiceKey("archive.fixture.delivery"), delivery)
    async def extra_child(extra_ctx):
        extra_ctx.require(ServiceKey("archive.test.secondary"))
        delivery["extra_context"] = extra_ctx
    await ctx.mount(
        extra_child, name="extra-child",
        inject=(ServiceKey("archive.test.secondary"),),
    )
    async def result_child(child_ctx):
        value = child_ctx.require(ServiceKey("archive.test.value"))
        await child_ctx.provide(ServiceKey("archive.test.result"), value)
        delivery["identity"] = child_ctx.require(BINDINGS).bind(
            ServiceKey("archive.test.result"), {"source": "real-child"},
            contributors=(delivery["extra_context"],),
        )
    await ctx.mount(
        result_child, name="result-child",
        inject=(ServiceKey("archive.test.value"), BINDINGS),
    )
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
    initialize_plugin_workspace(tmp_path / "workspace")
    log = MessageLog(tmp_path / "messages.db")
    host = manager(tmp_path, [plugins], message_log=log)
    try:
        await host.load_all()
        root = host._live_root
        assert root is not None
        binding = root.context.require(BINDINGS)
        delivery = root.context.require(DELIVERY)
        assert delivery["extra_context"].fiber.state is FiberState.ACTIVE
        identity = cast(str, delivery["identity"])
        descriptor = log.read_binding(identity)
        root_descriptor = host._archive.read_descriptor(cast(str, descriptor["root_ref"]))
        component_ids = {
            cast(str, host._archive.read_descriptor(ref)["plugin_id"])
            for ref in cast(tuple[str, ...], root_descriptor["components"])
        }
        assert component_ids == {"consumer", "provider"}
        expected = root.context.require(RESULT)
        async with binding.open(identity, RESULT) as (state, metadata):
            assert state["text"] == "old:A"
            assert state is expected
            assert metadata == descriptor["metadata"]
            assert log.read_binding(identity) == descriptor
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
            'await ctx.provide(ServiceKey("archive.test.result"), ctx.require(inject[0]))',
            'await ctx.provide(ServiceKey("archive.test.result"), ctx.require(inject[0]), '
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
    initialize_plugin_workspace(tmp_path / "workspace")
    initialize_plugin_workspace(tmp_path / "other/workspace")
    log = MessageLog(tmp_path / "messages.db")
    other_log = MessageLog(tmp_path / "other/messages.db")
    host = manager(tmp_path, [plugins], message_log=log)
    other = manager(tmp_path / "other", [plugins], message_log=other_log)
    try:
        await host.load_all()
        await other.load_all()
        host_root = host._live_root
        other_root = other._live_root
        assert host_root is not None and other_root is not None
        context = host_root.context.require(RESULT)[
            "registration"
        ]
        foreign = other_root.context.require(RESULT)[
            "registration"
        ]
        binding = host_root.context.require(BINDINGS)
        result_value = host_root.context.require(RESULT)
        provider_context, _ = host_root._service_provider(RESULT)
        for invalid in (foreign, Context(context._root, context._fiber)):
            with pytest.raises(ValueError, match="不属于"):
                if dynamic:
                    result_value["registration"] = invalid
                    async with provider_context.runtime_scope():
                        binding.bind(RESULT, {})
                else:
                    async with provider_context.runtime_scope():
                        binding.bind(RESULT, {}, contributors=(invalid,))
        result_value["registration"] = context
        async with provider_context.runtime_scope():
            identity = binding.bind(
                RESULT, {"target": "extra"}, contributors=() if dynamic else (context,)
            )
        descriptor = log.read_binding(identity)
        expected = host_root.context.require(RESULT)
        async with binding.open(identity, RESULT) as (state, metadata):
            assert state["extra"] == "registered A"
            assert state is expected
            assert metadata == descriptor["metadata"]
            assert log.read_binding(identity) == descriptor
    finally:
        log.close()
        other_log.close()
        await other.terminate_all()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_binding_open_uses_selected_scope_without_reopening_archive(
    tmp_path,
):
    """打开旧 binding 只读取日志事实，并使用自己的 live provider。"""
    log = MessageLog(tmp_path / "messages.db")
    try:
        async with live_binding_root(tmp_path) as (root, context, expected, state):
            bindings = Bindings(log, object(), root)  # type: ignore[arg-type]
            identity = seed_binding(log, {"choice": "stable"})
            async with bindings.open(identity, RESULT) as (value, metadata):
                assert value is expected
                assert metadata == {"choice": "stable"}
                assert context._fiber._in_flight_calls
            assert not context._fiber._in_flight_calls
            assert state["cleanup_calls"] == 0
    finally:
        log.close()


@pytest.mark.asyncio
async def test_binding_open_acquires_own_root_once_and_rejects_unrelated_scope(
    tmp_path,
):
    """缺 scope 从所属 Root 获取；已有其他 Root 时明确拒绝跨 Root。"""
    log = MessageLog(tmp_path / "messages.db")
    try:
        async with live_binding_root(tmp_path / "host", plugin_id="host") as (
            root, context, expected, _state,
        ):
            async with live_binding_root(tmp_path / "other", plugin_id="other") as (
                _other_root, other_context, _other_value, _other_state,
            ):
                bindings = Bindings(log, object(), root)  # type: ignore[arg-type]
                identity = seed_binding(log, {})
                async with bindings.open(identity, RESULT) as (value, _):
                    assert value is expected
                    assert context._fiber._in_flight_calls
                assert not context._fiber._in_flight_calls

                with pytest.raises(RuntimeError, match="所属 Root"):
                    async with other_context.runtime_scope():
                        async with bindings.open(identity, RESULT):
                            pytest.fail("不相干 Root 不应静默改选 stable")
                assert not context._fiber._in_flight_calls
    finally:
        log.close()


@pytest.mark.asyncio
async def test_binding_open_keeps_admitted_provider_scope_during_unload(tmp_path):
    """已有 provider permit 可嵌套；新 Task 不得借用继承的 ContextVar。"""
    log = MessageLog(tmp_path / "messages.db")
    try:
        async with live_binding_root(
            tmp_path, with_drain_consumer=True,
        ) as (root, context, expected, state):
            bindings = Bindings(log, object(), root)  # type: ignore[arg-type]
            identity = seed_binding(log, {"choice": "stable"})
            admitted = asyncio.Event()
            unloading = state["consumer_released"]
            continue_work = asyncio.Event()
            nested_opened = asyncio.Event()
            rejected = asyncio.Event()
            release = asyncio.Event()
            rejection = {"body_ran": False, "message": ""}
            unrelated_scope_work = []

            async def unrelated(ctx):
                await ctx.on(RUNTIME_STARTED, lambda _event: state["unrelated_events"].append("started"))
                await ctx.on(RUNTIME_STOPPING, lambda _event: state["unrelated_events"].append("stopping"))

            unrelated_fiber = await root.mount(
                unrelated,
                name="binding-unrelated",
                runtime=PluginRuntime(
                    "binding-unrelated",
                    "generation-unrelated",
                    tmp_path / "unrelated-data",
                    tmp_path / "unrelated-workspace",
                    tmp_path / "unrelated-plugin",
                    {},
                ),
            )
            unrelated_context = unrelated_fiber.context
            unrelated_state = unrelated_context.fiber.state
            unrelated_activation = unrelated_context.fiber.activation_token
            unrelated_events = tuple(state["unrelated_events"])
            root_identity = root.instance_token

            async def reject_inherited_scope():
                try:
                    async with bindings.open(identity, RESULT):
                        rejection["body_ran"] = True
                except RuntimeError as error:
                    rejection["message"] = str(error)
                finally:
                    rejected.set()

            async def admitted_work():
                async with context.runtime_scope():
                    admitted.set()
                    await unloading.wait()
                    await continue_work.wait()
                    async with unrelated_context.runtime_scope():
                        unrelated_scope_work.append(unrelated_context.fiber.state)
                    async with bindings.open(identity, RESULT) as (value, metadata):
                        assert value is expected
                        assert metadata == {"choice": "stable"}
                    nested_opened.set()
                    child = asyncio.create_task(reject_inherited_scope())
                    await rejected.wait()
                    await child
                    await release.wait()

            holder = None
            dispose_task = None
            try:
                holder = asyncio.create_task(admitted_work())
                await admitted.wait()
                dispose_task = asyncio.create_task(context.fiber.dispose())
                await unloading.wait()
                assert context.fiber.state is FiberState.UNLOADING
                assert state["consumer_cleanup_calls"] == 1
                assert state["contributor_calls"] == 0
                assert root.instance_token is root_identity
                assert unrelated_context.fiber.state is unrelated_state
                assert unrelated_context.fiber.activation_token is unrelated_activation
                assert tuple(state["unrelated_events"]) == unrelated_events
                continue_work.set()
                await nested_opened.wait()
                await rejected.wait()
                assert unrelated_scope_work == [FiberState.ACTIVE]
                assert not rejection["body_ran"]
                assert "不接纳新调用" in rejection["message"]
                release.set()
                await holder
                await dispose_task
                assert context.fiber.state is FiberState.DISPOSED
                assert state["cleanup_calls"] == 1
                assert not context._fiber._in_flight_calls
                assert not state["senders_context"]._fiber._in_flight_calls
            finally:
                continue_work.set()
                release.set()
                if holder is not None:
                    try:
                        await holder
                    finally:
                        if dispose_task is not None:
                            await dispose_task
                elif dispose_task is not None:
                    await dispose_task
    finally:
        log.close()


@pytest.mark.asyncio
async def test_binding_open_releases_fallback_scope_when_cancelled(tmp_path):
    """取消 live provider scope 时释放 permit，不留下运行时残留。"""
    log = MessageLog(tmp_path / "messages.db")
    try:
        async with live_binding_root(tmp_path) as (root, context, _value, state):
            bindings = Bindings(log, object(), root)  # type: ignore[arg-type]
            identity = seed_binding(log, {})
            entered = asyncio.Event()

            async def use_binding():
                async with bindings.open(identity, RESULT):
                    entered.set()
                    await asyncio.Future()

            task = asyncio.create_task(use_binding())
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert len(context._fiber._in_flight_calls) == 1
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not context._fiber._in_flight_calls
            assert state["cleanup_calls"] == 0

            await root.dispose()
            with pytest.raises(RuntimeError, match="当前 runtime scope 不提供服务"):
                async with bindings.open(identity, RESULT):
                    pytest.fail("已移除 provider 不应继续打开")
            assert state["cleanup_calls"] == 1
    finally:
        log.close()
