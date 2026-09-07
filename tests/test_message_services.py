import asyncio
from pathlib import Path

import pytest

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE
from agent.plugin_composition.tasks import TASKS
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import get_current_runtime_snapshot, lease_runtime_snapshot
from bus.event_bus import EventBus
from session.log import MessageLog, WriterExpired
from session.message import ContentPart, ContentReferences, Input


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
async def apply(ctx, config):
    await ctx.provide(ServiceKey("probe.{name}"), ctx)
''')


@pytest.mark.asyncio
async def test_formal_capabilities_use_real_owner_and_task_holds_exact_runtime(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root
            one, two = (root.context.require(ServiceKey("probe." + name)) for name in ("one", "two"))
            writers = root.context.require(MESSAGE_WRITERS)
            state = root.context.require(OWNER_STATE)
            checks = {"text": lambda part: ContentReferences()}
            factory = writers.bind(one, author="user", source="chat", body_types=(Input,), content=checks)
            checks["model.facts"] = lambda part: ContentReferences()
            writer = factory("s")
            with pytest.raises(PermissionError):
                writer.append("forged", Input((ContentPart("model.facts", {}),)))
            writer.append("u1", Input((ContentPart("text", "accepted"),)))
            first = state.open(one)
            first.transact(lambda tx: tx.save("same", {"value": 1}, expected_version=None))
            assert state.open(one).read("same").value["value"] == 1
            assert state.open(two).read("same") is None
            service = root.context.require(TASKS)
            tasks = service.open(one)
            assert service.open(one) is tasks
            assert service.open(two) is not tasks
            entered, released = asyncio.Event(), asyncio.Event()
            async def operation(task):
                assert get_current_runtime_snapshot() is snapshot
                output = factory("s")
                task.on_close(output.expire)
                entered.set()
                await released.wait()
                return output.append("u2", Input(()))
            task = await tasks.admit("local-key", lambda slot: slot.start(operation))
            await entered.wait()
            catalog = root.context.require(MESSAGE_CATALOG)
            assert catalog.snapshot_heads() == {"s": 0}
        assert snapshot.lease_count >= 1
        with pytest.raises(RuntimeError, match="runtime scope"):
            factory("s")
        released.set()
        assert (await task.join()).message_id == "u2"
        assert snapshot.lease_count == 0
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", [False, True])
async def test_task_cancel_before_first_instruction_releases_admission_lease(tmp_path, reject):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root
            ctx = root.context.require(ServiceKey("probe.one"))
            tasks = root.context.require(TASKS).open(ctx)
            baseline = snapshot.lease_count
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
            if reject:
                with pytest.raises(ValueError, match="admission rejected"):
                    await tasks.admit("work", admit)
            else:
                await tasks.admit("work", admit)
            assert snapshot.lease_count == baseline + (0 if reject else 1)
            with pytest.raises(asyncio.CancelledError):
                await captured[0].join()
            assert snapshot.lease_count == baseline
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_candidate_caps_reject_formal_log_state_tasks_and_bindings(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        snapshot = host.current_snapshot
        prepared = await host.prepare_candidate("one")
        assert prepared is not None
        generations = {**snapshot.generations, "one": prepared}
        candidate, ready = await host._resolve_composition_root(
            generations, candidate_owner=prepared, force_fresh=True,
        )
        assert ready
        try:
            ctx = candidate.context.require(ServiceKey("probe.one"))
            with pytest.raises(RuntimeError, match="candidate"):
                candidate.context.require(MESSAGE_CATALOG).snapshot_heads()
            with pytest.raises(RuntimeError, match="candidate"):
                candidate.context.require(MESSAGE_WRITERS).bind(ctx, author="user", source="s", body_types=(Input,), content={})
            with pytest.raises(RuntimeError, match="candidate"):
                candidate.context.require(OWNER_STATE).open(ctx)
            with pytest.raises(RuntimeError, match="正式 Task"):
                candidate.context.require(TASKS).open(ctx)
            with pytest.raises(RuntimeError, match="candidate"):
                candidate.context.require(BINDINGS).describe("missing", ServiceKey("missing"))
            assert log.catalog().snapshot_heads() == {}
            assert host.current_snapshot is snapshot
        finally:
            await candidate.dispose()
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
inject = ()
async def apply(ctx, config):
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
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    try:
        await host.load_all()
        if formal_log:
            await host.start_runtime()
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                assert snapshot.composition_root.context.require(ServiceKey("started")) == {"s": 0}
        else:
            with pytest.raises(CompositionError, match="core.message_writers"):
                await host.start_runtime()
    finally:
        await host.terminate_all()
        if log is not None:
            log.close()


@pytest.mark.asyncio
async def test_actual_conversation_plugin_accepts_without_model_or_reply_and_shares_source_task(tmp_path):
    import shutil
    from plugins.conversation.plugin import CONVERSATION
    from session.message import Control

    sources = tmp_path / "plugins"
    shutil.copytree(Path(__file__).resolve().parents[1] / "plugins" / "conversation", sources / "conversation",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    entered = asyncio.Event()
    async def program(task, reader, source):
        assert source == "conversation"
        entered.set()
        await asyncio.Event().wait()
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            open_source = snapshot.composition_root.context.require(CONVERSATION)
            first = open_source("s")
            accepted = await first.accept("u1", Input((ContentPart("text", "saved without a model"),)))
            assert log.catalog().snapshot_heads() == {"s": 0}
            assert log.reader("s").get("u1") == accepted
            task = await first.start(program)
            await entered.wait()
            second = open_source("s")
            assert await second.start(program) is task
            await second.control("pause", Control("pause", 0), expected_head=0, handle=task.handle)
            with pytest.raises(asyncio.CancelledError):
                await task.join()
            assert await open_source("s").start(program) is None
    finally:
        await host.terminate_all()
        log.close()
