import asyncio
from pathlib import Path

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.tasks import TASKS
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog, WriterExpired
from session.message import ContentPart, ContentReferences, Input
from tests.fixtures.formal_plugins import MINIMAL_MESSAGE_PLUGINS, install_formal_plugins


def write_plugins(root):
    for name in ("one", "two"):
        path = root / name
        path.mkdir(parents=True)
        (path / "plugin.py").write_text(f'''
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.tasks import TASKS
api_version = 3
name = "{name}"
version = "1.0.0"
inject = (MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, TASKS, BINDINGS)
async def apply(ctx):
    await ctx.provide(ServiceKey("probe.{name}"), ctx)
''')


@pytest.mark.asyncio
async def test_formal_capabilities_use_real_owner_and_task_holds_exact_runtime(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        one, two = (root.context.require(ServiceKey("probe." + name)) for name in ("one", "two"))
        writers = root.context.require(MESSAGE_WRITERS)
        state = root.context.require(OWNER_STATE)
        checks = {"text": lambda part: ContentReferences()}
        async with one.runtime_scope():
            factory = writers.bind(one, author="user", source="chat", body_types=(Input,), content=checks)
        checks["model.facts"] = lambda part: ContentReferences()
        async with one.runtime_scope():
            writer = factory("s")
        with pytest.raises(PermissionError):
            writer.append("forged", Input((ContentPart("model.facts", {}),)))
        with pytest.raises(PermissionError, match="命名空间"):
            writer.append("forged-metadata", Input(()), metadata={"two": {"tag": "fake"}})
        message = writer.append("u1", Input((ContentPart("text", "accepted"),)), metadata={"one": {"tag": "own"}})
        assert message.metadata == {"one": {"tag": "own"}}
        async with one.runtime_scope():
            first = state.open(one)
        first.transact(lambda tx: tx.save("same", {"value": 1}, expected_version=None))
        async with one.runtime_scope():
            assert state.open(one).read("same").value["value"] == 1
        async with two.runtime_scope():
            assert state.open(two).read("same") is None
        service = root.context.require(TASKS)
        async with one.runtime_scope():
            tasks = service.open(one)
            assert service.open(one) is tasks
        async with two.runtime_scope():
            assert service.open(two) is not tasks
        entered, released = asyncio.Event(), asyncio.Event()
        async def operation(task):
            assert one._fiber._call_owned_by_current_task() is not None
            output = factory("s")
            task.on_close(output.expire)
            entered.set()
            await released.wait()
            return output.append("u2", Input(()))
        async with one.runtime_scope():
            task = await tasks.admit("local-key", lambda slot: slot.start(operation))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            catalog = root.context.require(MESSAGE_CATALOG)
            assert catalog.snapshot_heads() == {"s": 0}
            assert one._fiber._in_flight_calls
            with pytest.raises(CompositionError, match="OwnerCall"):
                factory("s")
        finally:
            released.set()
        assert (await asyncio.wait_for(task.join(), 5)).message_id == "u2"
        assert not one._fiber._in_flight_calls
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", [False, True])
async def test_task_cancel_before_first_instruction_releases_admission_lease(tmp_path, reject):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        ctx = root.context.require(ServiceKey("probe.one"))
        captured = []
        async def operation(task):
            pytest.fail("cancelled admission must not run user code")
        def admit(slot):
            task = slot.start(operation)
            captured.append(task)
            if reject:
                raise ValueError("admission rejected")
            task.cancel()
            return task
        async with ctx.runtime_scope():
            tasks = root.context.require(TASKS).open(ctx)
            if reject:
                with pytest.raises(ValueError, match="admission rejected"):
                    await tasks.admit("work", admit)
            else:
                await tasks.admit("work", admit)
        assert len(ctx._fiber._in_flight_calls) == (0 if reject else 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(captured[0].join(), 5)
        assert not ctx._fiber._in_flight_calls
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_local_update_keeps_formal_message_owner_on_live_root(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        one = root.context.require(ServiceKey("probe.one"))
        two = root.context.require(ServiceKey("probe.two"))
        writers = root.context.require(MESSAGE_WRITERS)
        state = root.context.require(OWNER_STATE)
        async with one.runtime_scope():
            state.open(one).transact(lambda tx: tx.save("entry", {"value": 1}, expected_version=None))
            writer = writers.bind(one, author="user", source="chat", body_types=(Input,), content={})("s")
            writer.append("before", Input(()))
        (sources / "one/plugin.py").write_text((sources / "one/plugin.py").read_text() + "\nmarker = 'updated'\n")
        result = await host.reconcile_changed()
        assert result[0]["publication_state"] == "active"
        assert host.live_root is root
        assert root.receipt().ready
        updated = root.context.require(ServiceKey("probe.one"))
        assert updated is not one
        assert root.context.require(ServiceKey("probe.two")) is two
        async with updated.runtime_scope():
            assert state.open(updated).read("entry").value == {"value": 1}
            with pytest.raises(CompositionError, match="OwnerCall"):
                writers.bind(one, author="user", source="chat", body_types=(Input,), content={})
            writer = writers.bind(updated, author="user", source="chat", body_types=(Input,), content={})("s")
            writer.append("after", Input(()))
        async with two.runtime_scope():
            assert state.open(two).read("entry") is None
        assert root.context.require(MESSAGE_CATALOG).snapshot_heads() == {"s": 1}
        assert log.catalog().snapshot_heads() == {"s": 1}
        assert log.reader("s").get("before") is not None
        assert log.reader("s").get("after") is not None
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("formal_log", [False, True])
async def test_runtime_listener_requires_host_ports_without_archiving_them(tmp_path, formal_log):
    sources = tmp_path / "plugins"
    path = sources / "listener"
    path.mkdir(parents=True)
    (path / "plugin.py").write_text('''
from agent.plugin_composition import RUNTIME_STARTED, ServiceKey
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS
from session.message import Input
api_version = 3
name = "listener"
version = "1.0.0"
inject = (MESSAGE_CATALOG, MESSAGE_WRITERS)
async def apply(ctx):
    async def start(event):
        async with ctx.runtime_scope():
            writer = ctx.require(MESSAGE_WRITERS).bind(
                ctx, author="user", source="chat", body_types=(Input,), content={}
            )("s")
            writer.append("accepted", Input(()))
            await ctx.provide(ServiceKey("started"), ctx.require(MESSAGE_CATALOG).snapshot_heads())
    await ctx.on(RUNTIME_STARTED, start)
''')
    log = MessageLog(tmp_path / "sessions.db") if formal_log else None
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        if not formal_log:
            with pytest.raises(RuntimeError, match="消息能力需要 bootstrap"):
                await host.load_all()
        else:
            await host.load_all()
            await host.start_runtime()
            await host.start_runtime()
            root = host.live_root
            assert root is not None
            assert root.context.require(ServiceKey("started")) == {"s": 0}
            assert log.catalog().snapshot_heads() == {"s": 0}
    finally:
        await host.terminate_all()
        if log is not None:
            log.close()


@pytest.mark.asyncio
async def test_actual_conversation_plugin_accepts_without_model_or_reply_and_shares_source_task(tmp_path):
    from plugins.sources.plugin import SOURCES
    from session.message import Control

    plugin_home, _ = install_formal_plugins(tmp_path, MINIMAL_MESSAGE_PLUGINS)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=plugin_home / "cache", message_log=log)
    entered = asyncio.Event()
    async def program(task, reader, source):
        assert source == "conversation"
        entered.set()
        await asyncio.Event().wait()
    try:
        await host.load_all()
        generation = host.generation("sources@fixture")
        assert generation is not None and generation.fiber is not None
        sources_context = generation.fiber.context
        async with sources_context.runtime_scope():
            matches = tuple(item for item in sources_context.require(SOURCES).entries()
                            if item.name == "conversation")
        assert len(matches) == 1
        source = matches[0]
        async with source.context.runtime_scope():
            first = source.open("s")
            accepted = await first.accept("u1", Input((ContentPart("text", "saved without a model"),)))
            assert log.catalog().snapshot_heads() == {"s": 0}
            assert log.reader("s").get("u1") == accepted
            task = await first.start(program)
            await entered.wait()
            second = source.open("s")
            assert await second.start(program) is task
            await second.control("pause", Control("pause", 0), expected_head=0, handle=task.handle)
            with pytest.raises(asyncio.CancelledError):
                await task.join()
            assert await source.open("s").start(program) is None
    finally:
        await host.terminate_all()
        log.close()
