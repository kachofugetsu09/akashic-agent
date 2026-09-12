from session.message import ContentReferences
import asyncio
import importlib
import inspect
import shutil
import sys
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
LEGACY_PROGRAM = ServiceKey("archive.test.legacy-program")


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
                cast(dict[str, object], metadata)["choice"] = "changed"
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
async def test_archived_absolute_program_import_uses_current_run_reply_boundary(
    tmp_path,
):
    """旧归档程序的绝对 import 会进入当前兼容边界。"""
    from plugins.reply_program.program import run_reply

    sources = tmp_path / "plugins"
    plugin = sources / "legacy_program"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
from plugins.reply_program.program import run_reply
api_version = 3
name = "legacy_program"
version = "1.0.0"
async def apply(ctx, config):
    await ctx.provide(ServiceKey("archive.test.legacy-program"), run_reply)
""")
    log = MessageLog(tmp_path / "messages.db")
    first = manager(tmp_path, [sources])
    try:
        await first.load_all()
        bindings = Bindings(log, first._archive, first.open_binding)
        async with lease_runtime_snapshot(first.snapshot_store):
            identity = bindings.bind(LEGACY_PROGRAM, {})
        await first.terminate_all()
        shutil.rmtree(sources)

        restored = manager(tmp_path, [])
        recovered = Bindings(log, restored._archive, restored.open_binding)
        try:
            async with recovered.open(identity, LEGACY_PROGRAM) as (program, _):
                assert program is run_reply
                assert "tool_names" in inspect.signature(program).parameters
        finally:
            await restored.terminate_all()
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
        assert current is not None
        provider = host.generation("provider")
        consumer = host.generation("consumer")
        provider_ref = provider.archive_ref
        consumer_ref = consumer.archive_ref
        assert provider_ref is not None and consumer_ref is not None
        with pytest.raises(RuntimeError, match="闭包不完整"):
            async with host.open_binding((consumer_ref,)):
                pytest.fail("current provider must not fill the archive")
        # 即使旧接口位于闭包尾部，也必须在打开任何组件源码前拒绝。
        for runtime in (
            {"python_tag": "other", "binding_api": 2},
            {"python_tag": sys.implementation.cache_tag, "binding_api": 1},
        ):
            descriptor = dict(host._archive.read_descriptor(consumer_ref))
            descriptor["runtime"] = runtime
            incompatible = host._archive.save_descriptor(descriptor)
            def unexpected_open(_identity):
                pytest.fail("incompatible closure must not open any component code")
            with monkeypatch.context() as patch:
                patch.setattr(host._archive, "open", unexpected_open)
                with pytest.raises(RuntimeError, match="不兼容"):
                    async with host.open_binding((provider_ref, incompatible)):
                        pytest.fail("incompatible archive must not load")
        assert host.current_snapshot is current
        async with host.open_binding(
            (provider_ref, consumer_ref)
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
        current = host.current_snapshot
        assert current is not None
        root, ready = await host._resolve_composition_root(dict(current.generations), force_fresh=True)
        assert ready
        try:
            value = cast(Mapping[str, object], root.service_value(RESULT))
            assert value["asset"] == "asset A"
        finally:
            await root.dispose()
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_archive_close_drains_retained_scope_even_when_cancelled(
    tmp_path, monkeypatch, cancel
):
    from agent.plugins.snapshot import get_current_runtime_lease, RuntimeSnapshotLease, RuntimeSnapshotStore

    monkeypatch.setenv("ARCHIVE_PROVIDER_ACTIVE", "yes")
    plugins = tmp_path / "plugins"
    write_plugins(plugins)
    host = manager(tmp_path, [plugins])
    await host.load_all()
    current = host.current_snapshot
    assert current is not None
    raw_refs = tuple(g.archive_ref for g in current.generations.values())
    assert all(ref is not None for ref in raw_refs)
    refs = tuple(cast(str, ref) for ref in raw_refs)
    retained: RuntimeSnapshotLease | None = None
    state: Mapping[str, object] | None = None
    waiting = asyncio.Event()
    original_wait = RuntimeSnapshotStore.wait_for_no_leases

    async def observe_wait(store, snapshot):
        waiting.set()
        await original_wait(store, snapshot)

    monkeypatch.setattr(RuntimeSnapshotStore, "wait_for_no_leases", observe_wait)

    async def use_binding():
        nonlocal retained, state
        async with host.open_binding(refs) as scope:
            state = cast(Mapping[str, object], scope.require(RESULT))
            lease = get_current_runtime_lease()
            assert lease is not None
            retained = lease.fork()

    task: asyncio.Task[None] = asyncio.create_task(use_binding())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=5)
        assert not task.done()
        assert state is not None and retained is not None
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
            "from agent.plugin_composition import ServiceKey\n"
            "LEGACY_DELIVERY = ServiceKey('archive.fixture.delivery')\n"
            "inject = (LEGACY_DELIVERY,)\n"
            "async def apply(ctx, config):",
        )
    )
    consumer = plugins / "consumer" / "plugin.py"
    consumer.write_text(
        """
from agent.plugin_composition import ServiceKey
api_version = 3
name = "consumer"
version = "1.0.0"
async def apply(ctx, config):
    await ctx.provide(ServiceKey("archive.fixture.delivery"), {})
"""
    )
    host = manager(tmp_path, [plugins])
    try:
        await host.load_all()
        current = host.current_snapshot
        assert current is not None
        raw_refs = tuple(
            generation.archive_ref
            for plugin_id, generation in current.generations.items()
            if plugin_id == "provider"
        )
        assert all(ref is not None for ref in raw_refs)
        refs = tuple(cast(str, ref) for ref in raw_refs)
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
            assert set(get_current_runtime_snapshot().generations) == {
                "addon",
                "consumer",
                "provider",
            }
    finally:
        log.close()
        await other.terminate_all()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_content_binding_keeps_registered_protocol_owner_after_source_removal(
    tmp_path,
):
    from plugins.content.plugin import CONTENT, open_content

    sources = tmp_path / "plugins"
    sources.mkdir()
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "plugins" / "content",
        sources / "content",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    protocol = sources / "protocol"
    protocol.mkdir()
    (protocol / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
from session.message import ContentPart, ContentReferences
api_version = 3
name = "protocol"
version = "1.0.0"
inject = (ServiceKey("content.v2"),)
async def apply(ctx, config):
    async def decode(source, references):
        return ({"start": len(source.text), "end": len(source.text), "parts": (ContentPart("sample", "fixed A"),)},), {}
    await ctx.require(inject[0]).register(ctx, {
        "name": "sample", "content": {"sample": lambda part: ContentReferences()},
        "prompt": "Protocol A", "decode": decode,
    })
""")
    host = manager(tmp_path, [sources])
    log = MessageLog(tmp_path / "messages.db")
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            content = snapshot.composition_root.context.require(CONTENT)
            identity = content.save_binding(bindings)
            async with content.bind() as view:
                original = await view.decode("body")
        await host.terminate_all()
        shutil.rmtree(sources)
        restored = manager(tmp_path, [])
        bindings = Bindings(log, restored._archive, restored.open_binding)
        async with open_content(bindings, identity) as view:
            assert view.prompts == ("Protocol A",)
            assert await view.decode("body") == original
            assert set(view.checks) == {"text", "artifact_ref", "sample"}
        with pytest.raises(RuntimeError, match="释放"):
            await view.decode("late")
        await restored.terminate_all()
    finally:
        log.close()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_archived_context_accepts_public_summary_from_another_plugin(tmp_path):
    from datetime import UTC, datetime
    from agent.plugin_composition.models import ModelRequest
    from session.message import Input, Message, Output

    sources = tmp_path / "plugins"
    shutil.copytree(Path(__file__).resolve().parents[1] / "plugins" / "context", sources / "context",
                    ignore=shutil.ignore_patterns("__pycache__"))
    host = manager(tmp_path, [sources])
    log = MessageLog(tmp_path / "messages.db")
    key = ServiceKey("context.v2")
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store):
            identity = bindings.bind(key, {})
        await host.terminate_all()
        shutil.rmtree(sources)
        snapshot = (
            Message("u", "s", 0, datetime.now(UTC), "user", "conversation", Input(())),
            Message("a", "s", 1, datetime.now(UTC), "agent", "conversation", Output((), "complete")),
        )
        class Model:
            context_window = None
            max_tool_schemas = None
            def render(self, messages, *, after_seq, summary_reference=None):
                assert messages == snapshot and after_seq == 1
                return ModelRequest(())
            def estimate(self, request):
                return 1
        async with bindings.open(identity, key) as (context, metadata):
            request = context.build(snapshot,
                                    materials={"summary": {"reference": "saved", "source_message_ids": ("u", "a"), "content": "summary"}},
                                    model=Model(), max_output_tokens=1)
            assert "summary" in request.messages[0]["content"]
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [False, True])
async def test_archive_exposes_only_declared_message_reader_without_starting_runtime(tmp_path, declared):
    from session.message import Input
    from agent.plugin_composition.messages import MESSAGE_CATALOG

    sources = tmp_path / "plugins"
    provider = sources / "reader"
    provider.mkdir(parents=True)
    (provider / "plugin.py").write_text('''
from agent.plugin_composition import ServiceKey, RUNTIME_STARTED
from agent.plugin_composition.messages import MESSAGE_CATALOG
api_version = 3
name = "reader"
version = "1.0.0"
inject = DECLARED
async def apply(ctx, config):
    value = {"catalog": ctx.get(MESSAGE_CATALOG), "started": False}
    async def start(event):
        value["started"] = True
    await ctx.on(RUNTIME_STARTED, start)
    await ctx.provide(ServiceKey("reader.test"), value)
'''.replace("DECLARED", "(MESSAGE_CATALOG,)" if declared else "()"))
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    key = ServiceKey("reader.test")
    try:
        await host.load_all()
        await host.start_runtime()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root.context.require(key)["started"]
            identity = bindings.bind(key, {})
        await host.terminate_all()
        # 窄 reader 读取真实当前日志；归档代码与其启动生命周期仍保持独立。
        log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append("u", Input(()))
        async with bindings.open(identity, key) as (state, _):
            assert state["started"] is False
            if declared:
                assert state["catalog"].reader("s").get("u").message_id == "u"
            else:
                assert state["catalog"] is None
    finally:
        await host.terminate_all()
        log.close()
