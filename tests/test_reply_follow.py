import ast
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import inspect
import logging
from pathlib import Path
import shutil

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import PluginRuntime, ServiceKey
from agent.plugin_composition.channels import ChannelInboundMessage
from agent.plugin_composition.model import CompositionError
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG, MESSAGE_WRITERS, MessageCatalog, MessageReader,
)
from agent.plugin_composition.tasks import (
    RESTART_GATE, TASKS, Task, TaskServiceClosed,
)
from plugins.content.plugin import CONTENT
from plugins.reply.follow import follow
from plugins.sources.plugin import SOURCE_SESSION, SOURCES
from session.log import MessageLog
from session.message import ContentPart, Control, Input, Output


def _write_source(path, source):
    """在 fixture 写盘前静态解析并内存编译动态 Python 源码。"""
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    path.write_text(source)


@asynccontextmanager
async def running(
    tmp_path, program, *, lifecycle=False, fault_source=False,
    loading_source=False, before_watcher=None, before_start=None,
    expected_watcher_errors=None,
):
    sources = tmp_path / "plugins"
    for name in ("commands", "ui", "conversation", "sources", "content", "models"):
        shutil.copytree(Path(__file__).parents[1] / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    probe = sources / "probe"
    probe.mkdir()
    _write_source(probe / "plugin.py", '''
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.messages import MESSAGE_WRITERS
api_version = 3
name = "probe"
version = "1.0.0"
inject = (MESSAGE_WRITERS,)
async def apply(ctx):
    await ctx.provide(ServiceKey("probe"), ctx)
''')
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    event_bus = EventBus()
    host = PluginManager([sources], event_bus=event_bus, workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    watcher = None
    fault_fiber = None
    body_error: BaseException | None = None
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        if fault_source:
            mount_fault_source = _mount_fault_source
            if loading_source:
                async def mount_fault_source(ctx):
                    await _mount_fault_source(ctx, loading=True)
            fault_fiber = await root.mount(
                mount_fault_source,
                name="fault-source",
                inject=(
                    CONTENT, MESSAGE_CATALOG, MESSAGE_WRITERS, RESTART_GATE,
                    SOURCE_SESSION, SOURCES, TASKS,
                ),
                runtime=PluginRuntime(
                    "fault-source", "fault-source", tmp_path, tmp_path, tmp_path, {},
                ),
            )
        if before_watcher is not None:
            await before_watcher(log, host, fault_fiber)
        ctx = root.context.require(ServiceKey("probe"))
        registered = root.context.require(SOURCES)
        watcher = await ctx.spawn(
            follow(ctx, log.catalog(), registered,
                   lambda task, reader, source: program(ctx, task, reader, source)),
            name="follow",
        )
        if lifecycle:
            from agent.plugin_composition import RUNTIME_STOPPING
            async def stop(_event):
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
            await ctx.on(RUNTIME_STOPPING, stop)
        if before_start is not None:
            await before_start(log, host, watcher, fault_fiber)
        await asyncio.wait_for(host.start_runtime(), 10)
        if fault_source:
            yield log, host, watcher, fault_fiber
        else:
            yield log, host, watcher
    except BaseException as error:
        body_error = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if watcher is not None:
            if not watcher.done():
                watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                if expected_watcher_errors is None or not any(
                    _contains_exception(error, expected)
                    for expected in expected_watcher_errors
                ):
                    cleanup_errors.append(error)
        try:
            await host.terminate_all()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            log.close()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            await event_bus.aclose()
        except BaseException as error:
            cleanup_errors.append(error)
        cleanup_errors = [
            error for error in cleanup_errors
            if body_error is None or not _contains_exception(body_error, error)
        ]
        if cleanup_errors:
            if body_error is None:
                raise BaseExceptionGroup("reply follow fixture cleanup failed", cleanup_errors)
            raise BaseExceptionGroup(
                "reply follow fixture body and cleanup failed",
                [body_error, *cleanup_errors],
            )


async def accept(host, session, identity):
    generation = host.generation("sources")
    assert generation is not None and generation.fiber is not None
    context = generation.fiber.context
    async with context.runtime_scope():
        matches = tuple(item for item in context.require(SOURCES).entries()
                        if item.name == "conversation")
    assert len(matches) == 1
    source = matches[0]
    async with source.context.runtime_scope():
        return await source.open(session).accept(
            identity, Input((ContentPart("text", identity),)))


async def accept_fault(host, session, identity):
    """通过真实 Sources 注册表接纳测试专属来源输入。"""
    root = host.live_root
    assert root is not None
    return await root.context.require(SOURCES).accept(
        session,
        identity,
        ChannelInboundMessage(
            channel="fault",
            sender="user",
            chat_id=session,
            content=identity,
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )


async def _mount_fault_source(ctx, *, loading=False):
    """挂载一个独立 Source owner，测试来源结算而不关闭 conversation。"""
    catalog = ctx.require(MESSAGE_CATALOG)
    writers = ctx.require(MESSAGE_WRITERS)
    source_session = ctx.require(SOURCE_SESSION)
    tasks = ctx.require(TASKS)

    def open_source(session_id):
        reader = catalog.reader(session_id)
        inputs = writers.bind(
            ctx,
            author="user",
            source="fault",
            body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text},
        )(session_id)
        controls = writers.bind(
            ctx, author="app", source="fault", body_types=(Control,), content={},
        )(session_id)
        return source_session(
            reader=reader,
            inputs=inputs,
            controls=controls,
            tasks=tasks.open(ctx),
            restart_gate=ctx.require(RESTART_GATE),
        )

    async def accept_source(session_id, message_id, message):
        return await open_source(session_id).accept(
            message_id, Input((ContentPart("text", message.content),))
        )

    async def register_source():
        await ctx.require(SOURCES).register(
            ctx,
            name="fault",
            open=open_source,
            needs_reply=lambda reader: source_session.needs_reply(reader, "fault"),
            accept=accept_source,
            channels=("fault",),
        )

    if loading:
        from agent.plugin_composition import RUNTIME_STARTING
        await ctx.on(RUNTIME_STARTING, lambda _event: register_source())
    else:
        await register_source()


async def _mount_gated_source(
    ctx, *, source_name, input_session, input_id, text, channels=(),
    gate, registered, input_written=None, accepted=None,
):
    """在真实 child Fiber 的 LOADING apply 中登记并可选追加 Input。"""
    open_source, needs_reply = _source_definition(ctx, source_name)
    source_session = ctx.require(SOURCE_SESSION)

    async def accept_source(session_id, message_id, message):
        if accepted is not None:
            accepted.set()
        return await open_source(session_id).accept(
            message_id, Input((ContentPart("text", message.content),))
        )

    await ctx.require(SOURCES).register(
        ctx,
        name=source_name,
        open=open_source,
        needs_reply=needs_reply,
        accept=accept_source if channels else None,
        channels=channels,
    )
    registered.set()
    if input_written is not None:
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx,
            author="user",
            source=source_name,
            body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text},
        )(input_session)
        writer.append(input_id, Input((ContentPart("text", text),)))
        input_written.set()
    await gate.wait()


def _source_definition(ctx, source_name, opened=None):
    """为一个真实 owner 构造独立 SourceSession open 与判定函数。"""
    writers = ctx.require(MESSAGE_WRITERS)
    catalog = ctx.require(MESSAGE_CATALOG)
    source_session = ctx.require(SOURCE_SESSION)
    tasks = ctx.require(TASKS)

    def open_source(session_id):
        if opened is not None:
            opened.append((source_name, ctx, ctx.fiber.activation_token))
        inputs = writers.bind(
            ctx,
            author="user",
            source=source_name,
            body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text},
        )(session_id)
        controls = writers.bind(
            ctx, author="app", source=source_name,
            body_types=(Control,), content={},
        )(session_id)
        return source_session(
            reader=catalog.reader(session_id),
            inputs=inputs,
            controls=controls,
            tasks=tasks.open(ctx),
            restart_gate=ctx.require(RESTART_GATE),
        )

    return open_source, lambda reader: source_session.needs_reply(reader, source_name)


async def _register_source(ctx, source_name, opened=None):
    """在指定 active owner 下登记一个不绑定输入渠道的真实来源。"""
    async with ctx.runtime_scope():
        open_source, needs_reply = _source_definition(ctx, source_name, opened)
        return await ctx.require(SOURCES).register(
            ctx, name=source_name, open=open_source, needs_reply=needs_reply,
        )


async def _register_custom_source(ctx, source_name, open_source, needs_reply):
    """在真实 owner 下登记测试需要的窄来源变体。"""
    async with ctx.runtime_scope():
        return await ctx.require(SOURCES).register(
            ctx,
            name=source_name,
            open=open_source,
            needs_reply=needs_reply,
        )


async def _append_source_input(ctx, source_name, session_id, message_id, text):
    """使用真实 owner writer 向指定来源追加一条 Input。"""
    async with ctx.runtime_scope():
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx,
            author="user",
            source=source_name,
            body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text},
        )(session_id)
        return writer.append(
            message_id, Input((ContentPart("text", text),))
        )


def output(ctx, task, reader, source):
    from agent.plugin_composition.messages import MESSAGE_WRITERS
    from plugins.content.plugin import check_text

    writer = ctx.require(MESSAGE_WRITERS).bind(
        ctx, author="assistant", source=source, body_types=(Output,), content={"text": check_text}
    )(reader.session_id)
    task.on_close(writer.expire)
    return writer


def _exception_leaves(error):
    """展开清理错误组，便于核对原始 cleanup 与 caller 取消。"""
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            yield from _exception_leaves(child)
    else:
        yield error


def _contains_exception(error, target):
    """按对象身份判断错误是否已经由测试主体观察过。"""
    if error is target:
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_exception(child, target) for child in error.exceptions)
    return False


@pytest.mark.asyncio
async def test_follow_wakes_for_source_registered_after_history(tmp_path, monkeypatch):
    """历史 Input 在 Source 注册前存在时，注册激活必须唤醒一次真实回复。"""
    history_observed = asyncio.Event()
    input_written = asyncio.Event()
    input_processed = asyncio.Event()
    started = asyncio.Event()
    starts: list[tuple[str, tuple[str, ...]]] = []

    original_snapshot_heads = MessageCatalog.snapshot_heads

    def observe_history(catalog):
        heads = original_snapshot_heads(catalog)
        history_observed.set()
        return heads

    monkeypatch.setattr(MessageCatalog, "snapshot_heads", observe_history)
    original_catalog_follow = MessageCatalog.follow

    async def observe_catalog(catalog):
        async for heads in original_catalog_follow(catalog):
            yield heads
            if input_written.is_set():
                input_processed.set()

    monkeypatch.setattr(MessageCatalog, "follow", observe_catalog)
    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        writer.append(
            "late-output",
            Output((ContentPart("text", "late answer"),), "complete"),
        )
        started.set()

    conversation_fiber = None
    async with running(tmp_path, program) as (log, host, watcher):
        try:
            await asyncio.wait_for(history_observed.wait(), 2)
            generation = host.generation("conversation")
            assert generation is not None and generation.fiber is not None
            conversation_fiber = generation.fiber
            ctx = conversation_fiber.context
            async def register_late_source():
                async with ctx.runtime_scope():
                    writers = ctx.require(MESSAGE_WRITERS)
                    registered = ctx.require(SOURCES)
                    source_name = "late"
                    input_writer = writers.bind(
                        ctx,
                        author="user",
                        source=source_name,
                        body_types=(Input,),
                        content={"text": ctx.require(CONTENT).check_text},
                    )("history")
                    input_writer.append(
                        "late-input", Input((ContentPart("text", "late"),))
                    )
                    input_written.set()
                    await asyncio.wait_for(input_processed.wait(), 2)
                    catalog = ctx.require(MESSAGE_CATALOG)
                    source_session = ctx.require(SOURCE_SESSION)
                    tasks = ctx.require(TASKS)

                    def open_source(session_id):
                        reader = catalog.reader(session_id)
                        inputs = writers.bind(
                            ctx,
                            author="user",
                            source=source_name,
                            body_types=(Input,),
                            content={"text": ctx.require(CONTENT).check_text},
                        )(session_id)
                        controls = writers.bind(
                            ctx, author="app", source=source_name,
                            body_types=(Control,), content={},
                        )(session_id)
                        return source_session(
                            reader=reader,
                            inputs=inputs,
                            controls=controls,
                            tasks=tasks.open(ctx),
                            restart_gate=ctx.require(RESTART_GATE),
                        )

                    await registered.register(
                        ctx,
                        name=source_name,
                        open=open_source,
                        needs_reply=lambda reader: source_session.needs_reply(reader, source_name),
                    )

            await register_late_source()
            await asyncio.wait_for(started.wait(), 2)
            assert starts == [("history", "late")]
            messages = log.reader("history").snapshot()
            assert [message.message_id for message in messages] == [
                "late-input", "late-output",
            ]
            assert messages[-1].author == "assistant"
            assert messages[-1].source == "late"
            assert messages[-1].body == Output(
                (ContentPart("text", "late answer"),), "complete",
            )
            assert not watcher.done()
        finally:
            pass

    assert conversation_fiber is not None
    assert conversation_fiber.state.name == "DISPOSED"


@pytest.mark.asyncio
async def test_source_loading_registration_publishes_once_after_catalog_sweep(
    tmp_path, monkeypatch,
):
    """LOADING 来源在首轮历史 head 后只发布一次并完成一次真实回复。"""
    first_catalog_sweep = asyncio.Event()
    input_observed = asyncio.Event()
    published = asyncio.Event()
    program_started = asyncio.Event()
    snapshots: list[tuple[str, ...]] = []
    collector = None
    collector_joined = False
    mount_task = None
    mount_task_joined = False
    source_fiber = None
    reply_fiber = None
    source_task = None
    source_task_done = asyncio.Event()
    source_task_joined = asyncio.Event()
    join_calls = 0
    gate = asyncio.Event()
    registration_ready = asyncio.Event()
    input_written = asyncio.Event()
    original_snapshot_heads = MessageCatalog.snapshot_heads

    def observe_catalog(catalog):
        heads = original_snapshot_heads(catalog)
        if not first_catalog_sweep.is_set():
            first_catalog_sweep.set()
        if "loading-session" in heads:
            input_observed.set()
        return heads

    monkeypatch.setattr(MessageCatalog, "snapshot_heads", observe_catalog)

    original_join = Task.join

    async def observe_join(task):
        nonlocal join_calls
        result = None
        try:
            result = await original_join(task)
        except BaseException:
            if task is source_task:
                join_calls += 1
                source_task_joined.set()
            raise
        if task is source_task:
            join_calls += 1
            source_task_joined.set()
        return result

    monkeypatch.setattr(Task, "join", observe_join)

    async def before_start(log, host, watcher, fault_fiber):
        nonlocal collector, collector_joined, mount_task, mount_task_joined
        nonlocal source_fiber
        root = host.live_root
        assert root is not None
        registry = root.context.require(SOURCES)
        assert all(source.name != "fault" for source in registry.entries())

        async def collect_changes():
            async for entries in registry.changes():
                names = tuple(source.name for source in entries)
                snapshots.append(names)
                if "fault" in names:
                    published.set()
                    return

        collector = asyncio.create_task(collect_changes())
        try:
            await asyncio.wait_for(first_catalog_sweep.wait(), 2)
            mount_task = asyncio.create_task(root.mount(
                lambda ctx: _mount_gated_source(
                    ctx,
                    source_name="fault",
                    input_session="loading-session",
                    input_id="loading-input",
                    text="loading",
                    gate=gate,
                    registered=registration_ready,
                    input_written=input_written,
                ),
                name="loading-source",
                inject=(
                    CONTENT, MESSAGE_CATALOG, MESSAGE_WRITERS, RESTART_GATE,
                    SOURCE_SESSION, SOURCES, TASKS,
                ),
                runtime=PluginRuntime(
                    "loading-source", "loading-source", tmp_path, tmp_path, tmp_path, {},
                ),
            ))
            await asyncio.wait_for(registration_ready.wait(), 2)
            await asyncio.wait_for(input_written.wait(), 2)
            await asyncio.wait_for(input_observed.wait(), 2)
            assert all(source.name != "fault" for source in registry.entries())
            gate.set()
            done, _ = await asyncio.wait((mount_task,), timeout=2)
            assert mount_task in done, "loading source mount timed out"
            try:
                source_fiber = mount_task.result()
            finally:
                mount_task_joined = True
        except BaseException as error:
            gate.set()
            cleanup_errors = []
            if mount_task is not None and not mount_task_joined:
                try:
                    if not mount_task.done():
                        await asyncio.wait((mount_task,))
                    mount_task.result()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                finally:
                    mount_task_joined = True
            if collector is not None and not collector_joined:
                cancel_requested = False
                try:
                    if not collector.done():
                        cancel_requested = True
                        collector.cancel()
                    await collector
                except asyncio.CancelledError as cleanup_error:
                    if not (cancel_requested and cleanup_error.__cause__ is None):
                        cleanup_errors.append(cleanup_error)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                finally:
                    collector_joined = True
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "before_start cleanup failed", [error, *cleanup_errors]
                ) from error
            raise

    async def program(ctx, task, reader, source):
        nonlocal source_task, reply_fiber
        assert source == "fault"
        reply_fiber = ctx._fiber
        source_task = task
        task.on_done(source_task_done.set)
        writer = output(ctx, task, reader, source)
        writer.append(
            "loading-output",
            Output((ContentPart("text", "loading answer"),), "complete"),
        )
        program_started.set()

    body_error = None
    cleanup_errors = []
    try:
        async with running(
            tmp_path,
            program,
            before_start=before_start,
        ) as (log, host, watcher):
            await asyncio.wait_for(published.wait(), 2)
            assert sum("fault" in names for names in snapshots) == 1
            await asyncio.wait_for(program_started.wait(), 2)
            await asyncio.wait_for(source_task_done.wait(), 2)
            await asyncio.wait_for(source_task_joined.wait(), 2)
            messages = log.reader("loading-session").snapshot()
            assert [message.message_id for message in messages] == [
                "loading-input", "loading-output",
            ]
            assert messages[-1].source == "fault"
            assert messages[-1].body == Output(
                (ContentPart("text", "loading answer"),), "complete",
            )
            assert join_calls == 1
            assert source_fiber is not None
            assert reply_fiber is not None
            await asyncio.wait_for(source_fiber._calls_idle.wait(), 2)
            await asyncio.wait_for(reply_fiber._calls_idle.wait(), 2)
            assert not source_fiber._in_flight_calls
            assert not reply_fiber._in_flight_calls
            assert not watcher.done()
    except BaseException as error:
        body_error = error
    finally:
        gate.set()
        if mount_task is not None and not mount_task_joined:
            try:
                if not mount_task.done():
                    await asyncio.wait((mount_task,))
                mount_task.result()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            finally:
                mount_task_joined = True
        if collector is not None and not collector_joined:
            cancel_requested = False
            try:
                if not collector.done():
                    cancel_requested = True
                    collector.cancel()
                await collector
            except asyncio.CancelledError as cleanup_error:
                if not (cancel_requested and cleanup_error.__cause__ is None):
                    cleanup_errors.append(cleanup_error)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            finally:
                collector_joined = True
    if body_error is not None:
        if cleanup_errors:
            raise BaseExceptionGroup(
                "source loading cleanup failed", [body_error, *cleanup_errors]
            ) from body_error
        raise body_error
    if cleanup_errors:
        raise BaseExceptionGroup("source loading cleanup failed", cleanup_errors)


@pytest.mark.asyncio
async def test_sources_route_loading_dedicated_channel_without_default_fallback(tmp_path):
    """LOADING 专属渠道不能回退默认来源，ready 后原输入只接纳一次。"""
    gate = asyncio.Event()
    registration_ready = asyncio.Event()
    dedicated_accepted = asyncio.Event()
    healthy_done = asyncio.Event()
    mount_task = None
    mount_task_joined = False
    body_error = None
    cleanup_errors = []

    async def program(ctx, task, reader, source):
        if source == "conversation":
            writer = output(ctx, task, reader, source)
            writer.append(
                "healthy-output",
                Output((ContentPart("text", "healthy"),), "complete"),
            )
            healthy_done.set()

    async with running(tmp_path, program) as (log, host, watcher):
        try:
            root = host.live_root
            assert root is not None
            registry = root.context.require(SOURCES)
            mount_task = asyncio.create_task(root.mount(
                lambda ctx: _mount_gated_source(
                    ctx,
                    source_name="dedicated",
                    input_session="unused",
                    input_id="unused-input",
                    text="unused",
                    channels=("dedicated",),
                    gate=gate,
                    registered=registration_ready,
                    accepted=dedicated_accepted,
                ),
                name="dedicated-loading-source",
                inject=(
                    CONTENT, MESSAGE_CATALOG, MESSAGE_WRITERS, RESTART_GATE,
                    SOURCE_SESSION, SOURCES, TASKS,
                ),
                runtime=PluginRuntime(
                    "dedicated-loading-source", "dedicated-loading-source",
                    tmp_path, tmp_path, tmp_path, {},
                ),
            ))
            await asyncio.wait_for(registration_ready.wait(), 2)
            assert not mount_task.done()
            message = ChannelInboundMessage(
                channel="dedicated",
                sender="user",
                chat_id="route-session",
                content="route me",
                timestamp=datetime.now(timezone.utc),
                metadata={},
            )
            with pytest.raises(CompositionError) as error:
                await registry.accept("route-session", "route-input", message)
            assert error.value.code == "OWNER_UNAVAILABLE"
            assert not dedicated_accepted.is_set()
            assert log.catalog().snapshot_heads() == {}

            gate.set()
            done, _ = await asyncio.wait((mount_task,), timeout=2)
            assert mount_task in done, "dedicated loading source mount timed out"
            try:
                mount_task.result()
            finally:
                mount_task_joined = True
            await registry.accept("route-session", "route-input", message)
            await asyncio.wait_for(dedicated_accepted.wait(), 2)
            messages = log.reader("route-session").snapshot()
            assert [item.message_id for item in messages] == ["route-input"]
            await registry.accept(
                "healthy-session",
                "healthy-input",
                ChannelInboundMessage(
                    channel="default",
                    sender="user",
                    chat_id="healthy-session",
                    content="healthy",
                    timestamp=datetime.now(timezone.utc),
                    metadata={},
                ),
            )
            await asyncio.wait_for(healthy_done.wait(), 2)
            healthy_messages = log.reader("healthy-session").snapshot()
            assert [item.message_id for item in healthy_messages] == [
                "healthy-input", "healthy-output",
            ]
            assert healthy_messages[-1].source == "conversation"
            assert not watcher.done()
        except BaseException as error:
            body_error = error
        finally:
            gate.set()
            if mount_task is not None and not mount_task_joined:
                try:
                    if not mount_task.done():
                        await asyncio.wait((mount_task,))
                    mount_task.result()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                finally:
                    mount_task_joined = True
        if body_error is not None:
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "dedicated source cleanup failed", [body_error, *cleanup_errors]
                ) from body_error
            raise body_error
        if cleanup_errors:
            raise BaseExceptionGroup(
                "dedicated source cleanup failed", cleanup_errors
            )


@pytest.mark.asyncio
async def test_follow_ignores_revoked_source_snapshot_before_open(tmp_path, monkeypatch):
    """撤销通知未消费时，主循环仍按当前 registry 阻止旧 Source open。"""
    source_seen = asyncio.Event()
    removal_ready = asyncio.Event()
    removal_release = asyncio.Event()
    removal_delivered = asyncio.Event()
    catalog_seen = asyncio.Event()
    peer_done = asyncio.Event()
    stale_started = asyncio.Event()
    old_effect = None
    original_changes = None

    async def gated_changes(registry):
        seen_stale = False
        assert original_changes is not None
        async for entries in original_changes(registry):
            names = {source.name for source in entries}
            if "stale" in names:
                seen_stale = True
                source_seen.set()
            elif seen_stale and "stale" not in names:
                removal_ready.set()
                await removal_release.wait()
                yield entries
                removal_delivered.set()
                continue
            yield entries

    async def before_watcher(log, host, fault_fiber):
        nonlocal original_changes
        root = host.live_root
        assert root is not None
        registry = root.context.require(SOURCES)
        original_changes = type(registry).changes
        monkeypatch.setattr(type(registry), "changes", gated_changes)

    original_follow = MessageCatalog.follow

    async def observe_catalog(catalog):
        async for heads in original_follow(catalog):
            if "stale-session" in heads:
                catalog_seen.set()
            yield heads

    monkeypatch.setattr(MessageCatalog, "follow", observe_catalog)

    async def program(ctx, task, reader, source):
        writer = output(ctx, task, reader, source)
        if source == "stale":
            stale_started.set()
            writer.append(
                "stale-output",
                Output((ContentPart("text", "stale"),), "complete"),
            )
        elif source == "conversation":
            writer.append(
                "peer-output",
                Output((ContentPart("text", "peer"),), "complete"),
            )
            peer_done.set()

    async with running(
        tmp_path, program, fault_source=True, before_watcher=before_watcher,
    ) as (
        log, host, watcher, fault_fiber,
    ):
        try:
            old_ctx = fault_fiber.context
            old_effect = await _register_source(old_ctx, "stale")
            await asyncio.wait_for(source_seen.wait(), 2)
            await old_effect.aclose()
            assert fault_fiber.state.name == "ACTIVE"
            await asyncio.wait_for(removal_ready.wait(), 2)
            await _append_source_input(
                old_ctx, "stale", "stale-session", "stale-input", "stale",
            )
            await asyncio.wait_for(catalog_seen.wait(), 2)
            await asyncio.wait_for(accept(host, "peer", "peer-input"), 2)
            await asyncio.wait_for(peer_done.wait(), 2)
            assert not stale_started.is_set()
            assert [item.message_id for item in log.reader("stale-session").snapshot()] == [
                "stale-input",
            ]
            assert not watcher.done()
            removal_release.set()
            await asyncio.wait_for(removal_delivered.wait(), 2)
        finally:
            removal_release.set()
            if old_effect is not None:
                await old_effect.aclose()


@pytest.mark.asyncio
async def test_follow_queued_drive_rechecks_revoked_source_before_first_instruction(
    tmp_path, monkeypatch,
):
    """已排队的真实 drive 在首指令前发现撤销时不打开旧来源。"""
    source_visible = asyncio.Event()
    removal_ready = asyncio.Event()
    removal_release = asyncio.Event()
    removal_delivered = asyncio.Event()
    drive_created = asyncio.Event()
    drive_release = asyncio.Event()
    peer_done = asyncio.Event()
    source_opened = asyncio.Event()
    opened: list[tuple[str, object, object]] = []
    old_effect = None
    drive_task = None
    drive_task_joined = False
    original_changes = None
    loop = asyncio.get_running_loop()

    async def gated_changes(registry):
        seen_source = False
        assert original_changes is not None
        async for entries in original_changes(registry):
            names = {source.name for source in entries}
            if "queued" in names:
                seen_source = True
                source_visible.set()
            elif seen_source and "queued" not in names:
                removal_ready.set()
                await removal_release.wait()
                yield entries
                removal_delivered.set()
                continue
            yield entries

    async def before_watcher(log, host, fault_fiber):
        nonlocal original_changes
        root = host.live_root
        assert root is not None
        registry = root.context.require(SOURCES)
        original_changes = type(registry).changes
        monkeypatch.setattr(type(registry), "changes", gated_changes)

    original_factory = loop.get_task_factory()

    def gate_drive_factory(current_loop, coroutine, **kwargs):
        nonlocal drive_task
        frame = getattr(coroutine, "cr_frame", None)
        code = getattr(getattr(coroutine, "cr_code", None), "co_name", None)
        is_queued_drive = (
            code == "drive"
            and frame is not None
            and frame.f_locals.get("session_id") == "queued-session"
            and frame.f_locals.get("source_name") == "queued"
        )
        if not is_queued_drive:
            if original_factory is None:
                return asyncio.Task(coroutine, loop=current_loop, **kwargs)
            return original_factory(current_loop, coroutine, **kwargs)

        async def hold_drive():
            drive_created.set()
            await drive_release.wait()
            return await coroutine

        held = hold_drive()
        try:
            if original_factory is None:
                task = asyncio.Task(held, loop=current_loop, **kwargs)
            else:
                task = original_factory(current_loop, held, **kwargs)
        except BaseException:
            held.close()
            coroutine.close()
            raise
        drive_task = task
        return task

    async def program(ctx, task, reader, source):
        if source == "conversation":
            writer = output(ctx, task, reader, source)
            writer.append(
                "queued-peer-output",
                Output((ContentPart("text", "queued-peer"),), "complete"),
            )
            peer_done.set()

    async with running(
        tmp_path, program, fault_source=True, before_watcher=before_watcher,
    ) as (
        log, host, watcher, fault_fiber,
    ):
        try:
            old_ctx = fault_fiber.context
            open_source, needs_reply = _source_definition(
                old_ctx, "queued", opened
            )

            def queued_open(session_id):
                source_opened.set()
                return open_source(session_id)

            old_effect = await _register_custom_source(
                old_ctx, "queued", queued_open, needs_reply
            )
            await asyncio.wait_for(source_visible.wait(), 2)
            loop.set_task_factory(gate_drive_factory)
            await _append_source_input(
                old_ctx, "queued", "queued-session", "queued-input", "queued",
            )
            await asyncio.wait_for(drive_created.wait(), 2)
            await old_effect.aclose()
            await asyncio.wait_for(removal_ready.wait(), 2)
            assert fault_fiber.state.name == "ACTIVE"
            drive_release.set()
            assert drive_task is not None
            try:
                await drive_task
            finally:
                drive_task_joined = True
            assert not source_opened.is_set()
            await asyncio.wait_for(accept(host, "queued-peer", "p1"), 2)
            await asyncio.wait_for(peer_done.wait(), 2)
            assert log.reader("queued-session").snapshot()[0].message_id == (
                "queued-input"
            )
            assert not watcher.done()
        finally:
            loop.set_task_factory(original_factory)
            drive_release.set()
            removal_release.set()
            if drive_task is not None and not drive_task_joined:
                try:
                    await drive_task
                finally:
                    drive_task_joined = True
            if old_effect is not None:
                await old_effect.aclose()
            if removal_ready.is_set():
                await asyncio.wait_for(removal_delivered.wait(), 2)


@pytest.mark.asyncio
async def test_follow_eager_closed_source_allows_same_name_replacement(tmp_path):
    """eager drive 同步结束后，同名新来源仍能启动一次并保留健康 peer。"""
    closed_opened = asyncio.Event()
    new_started = asyncio.Event()
    peer_done = asyncio.Event()
    starts: list[tuple[str, str]] = []
    old_effect = None
    new_effect = None
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    body_error = None
    cleanup_errors = []

    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        if source == "eager":
            writer.append(
                "eager-output",
                Output((ContentPart("text", "eager answer"),), "complete"),
            )
            new_started.set()
        elif source == "conversation":
            writer.append(
                "eager-peer-output",
                Output((ContentPart("text", "peer"),), "complete"),
            )
            peer_done.set()

    async with running(tmp_path, program, fault_source=True) as (
        log, host, watcher, fault_fiber,
    ):
        try:
            old_ctx = fault_fiber.context
            old_open, needs_reply = _source_definition(old_ctx, "eager")

            def closed_open(session_id):
                closed_opened.set()
                raise TaskServiceClosed(
                    f"closed admission for {session_id}"
                )

            old_effect = await _register_custom_source(
                old_ctx, "eager", closed_open, needs_reply
            )
            loop.set_task_factory(asyncio.eager_task_factory)
            await _append_source_input(
                old_ctx, "eager", "eager-session", "old-input", "old",
            )
            await asyncio.wait_for(closed_opened.wait(), 2)
            assert old_effect is not None
            await old_effect.aclose()
            generation = host.generation("conversation")
            assert generation is not None and generation.fiber is not None
            new_ctx = generation.fiber.context
            new_open, new_needs_reply = _source_definition(
                new_ctx, "eager"
            )
            new_effect = await _register_custom_source(
                new_ctx, "eager", new_open, new_needs_reply
            )
            await _append_source_input(
                new_ctx, "eager", "eager-session", "new-input", "new",
            )
            await asyncio.wait_for(new_started.wait(), 2)
            assert starts.count(("eager-session", "eager")) == 1
            messages = log.reader("eager-session").snapshot()
            assert [item.message_id for item in messages] == [
                "old-input", "new-input", "eager-output",
            ]
            assert messages[-1].body == Output(
                (ContentPart("text", "eager answer"),), "complete",
            )
            await asyncio.wait_for(accept(host, "eager-peer", "p1"), 2)
            await asyncio.wait_for(peer_done.wait(), 2)
            assert starts.count(("eager-peer", "conversation")) == 1
            assert not watcher.done()
        except BaseException as error:
            body_error = error
        finally:
            loop.set_task_factory(original_factory)
            if new_effect is not None:
                try:
                    await new_effect.aclose()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if old_effect is not None:
                try:
                    await old_effect.aclose()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        if body_error is not None:
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "eager replacement cleanup failed",
                    [body_error, *cleanup_errors],
                ) from body_error
            raise body_error
        if cleanup_errors:
            raise BaseExceptionGroup(
                "eager replacement cleanup failed", cleanup_errors
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_child", ("watch_catalog", "watch_sources"))
async def test_follow_closes_watcher_coroutine_when_group_factory_fails(
    tmp_path, failed_child,
):
    """TaskGroup child factory failure closes only the unowned coroutine."""
    marker = RuntimeError("watcher factory failure: " + failed_child)
    rejected = []
    created = []
    observed_errors = []
    loop = asyncio.get_running_loop()
    original_factory = None

    async def program(ctx, task, reader, source):
        raise AssertionError("program must not start")

    async def before_start(log, host, watcher, fault_fiber):
        nonlocal original_factory
        original_factory = loop.get_task_factory()

        def factory(current_loop, coroutine, **kwargs):
            name = getattr(getattr(coroutine, "cr_code", None), "co_name", None)
            if name == failed_child:
                rejected.append(coroutine)
                raise marker
            if original_factory is None:
                task = asyncio.Task(coroutine, loop=current_loop, **kwargs)
            else:
                task = original_factory(current_loop, coroutine, **kwargs)
            created.append((name, task))
            return task

        loop.set_task_factory(factory)

    try:
        async with running(
            tmp_path,
            program,
            before_start=before_start,
            expected_watcher_errors=observed_errors,
        ) as (log, host, watcher):
            with pytest.raises(BaseExceptionGroup) as error:
                await watcher
            observed_errors.append(error.value)
            leaves = list(_exception_leaves(error.value))
            assert any(leaf is marker for leaf in leaves)
            assert len(rejected) == 1
            assert rejected[0].cr_frame is None
            if failed_child == "watch_sources":
                watcher_tasks = [
                    task for name, task in created if name == "watch_catalog"
                ]
                assert watcher_tasks and all(task.done() for task in watcher_tasks)
    finally:
        if loop.get_task_factory() is not original_factory:
            loop.set_task_factory(original_factory)


@pytest.mark.asyncio
async def test_follow_replacement_settles_old_source_before_new_source(tmp_path, monkeypatch):
    """同名换代等待旧 Task.join() 返回后才允许新 owner 处理。"""
    old_started, old_done, old_release = (asyncio.Event() for _ in range(3))
    old_join_entered, old_join_release, old_join_returned = (
        asyncio.Event() for _ in range(3)
    )
    new_started, peer_before_done, peer_during_done, peer_after_done = (
        asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    )
    input2_written, input2_processed = asyncio.Event(), asyncio.Event()
    starts: list[tuple[str, str]] = []
    opened: list[tuple[str, object, object]] = []
    old_task = None
    old_effect = None
    new_effect = None
    new_mount_task = None
    new_mount_joined = False
    peer_fiber = None
    peer_context = None
    peer_activation = None
    peer_effect = None
    peer_effect_cleanup = 0
    peer_lifecycle = None
    peer_effects = None
    new_owner_ready = asyncio.Event()
    new_owner_context: dict[str, object] = {}
    new_owner_effect: dict[str, object] = {}
    body_error = None
    cleanup_errors = []

    original_join = Task.join

    async def gated_join(task):
        if task is old_task:
            old_join_entered.set()
            await old_join_release.wait()
            result = await original_join(task)
            old_join_returned.set()
            return result
        return await original_join(task)

    monkeypatch.setattr(Task, "join", gated_join)

    original_catalog_follow = MessageCatalog.follow

    async def observe_catalog(catalog):
        async for heads in original_catalog_follow(catalog):
            yield heads
            if input2_written.is_set():
                input2_processed.set()

    monkeypatch.setattr(MessageCatalog, "follow", observe_catalog)

    async def program(ctx, task, reader, source):
        nonlocal old_task
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        if source == "conversation":
            peer_index = sum(
                item[1] == "conversation" for item in starts
            )
            peer_message = f"peer-{peer_index}"
            peer_body = f"peer-{peer_index}"
            writer.append(
                peer_message,
                Output((ContentPart("text", peer_body),), "complete"),
            )
            {
                1: peer_before_done,
                2: peer_during_done,
                3: peer_after_done,
            }[peer_index].set()
            return
        if len([item for item in starts if item[1] == "swap"]) == 1:
            old_task = task
            task.on_done(old_done.set)
            writer.append(
                "swap-output-1",
                Output((ContentPart("text", "first"),), "complete"),
            )
            old_started.set()
            await old_release.wait()
            return
        writer.append(
            "swap-output-2",
            Output((ContentPart("text", "second"),), "complete"),
        )
        new_started.set()

    async with running(tmp_path, program, fault_source=True) as (
        log, host, watcher, fault_fiber,
    ):
        try:
            old_ctx = fault_fiber.context
            conversation_generation = host.generation("conversation")
            assert (
                conversation_generation is not None
                and conversation_generation.fiber is not None
            )
            peer_fiber = conversation_generation.fiber
            peer_context = peer_fiber.context
            async with peer_context.runtime_scope():
                peer_activation = peer_context.fiber.activation_token

                def peer_setup():
                    def cleanup():
                        nonlocal peer_effect_cleanup
                        peer_effect_cleanup += 1

                    return cleanup

                peer_effect = await peer_context.effect(
                    peer_setup, label="reply-follow-peer-continuity"
                )
            peer_effects = tuple(peer_fiber.effects)
            peer_lifecycle = (
                peer_fiber._lifecycle_started,
                peer_fiber._stopping_completed,
                len(peer_effects),
            )

            def assert_peer_unchanged():
                assert peer_fiber is conversation_generation.fiber
                assert peer_context is peer_fiber.context
                assert peer_fiber._activation_token is peer_activation
                assert tuple(peer_fiber.effects) == peer_effects
                assert (
                    peer_fiber._lifecycle_started,
                    peer_fiber._stopping_completed,
                    len(peer_fiber.effects),
                ) == peer_lifecycle
                assert peer_effect_cleanup == 0
                assert peer_effect is not None and not peer_effect._closed

            await _append_source_input(
                old_ctx, "swap", "swap-session", "swap-input-1", "first",
            )
            old_effect = await _register_source(old_ctx, "swap", opened)
            await asyncio.wait_for(old_started.wait(), 2)
            await asyncio.wait_for(accept(host, "peer-before", "p1"), 2)
            await asyncio.wait_for(peer_before_done.wait(), 2)
            assert_peer_unchanged()

            await old_effect.aclose()
            root = host.live_root
            assert root is not None

            async def mount_new_owner(ctx):
                new_owner_context["value"] = ctx
                new_owner_effect["value"] = await _register_source(
                    ctx, "swap", opened
                )
                new_owner_ready.set()

            new_mount_task = asyncio.create_task(root.mount(
                mount_new_owner,
                name="swap-replacement-owner",
                inject=(
                    CONTENT, MESSAGE_CATALOG, MESSAGE_WRITERS, RESTART_GATE,
                    SOURCE_SESSION, SOURCES, TASKS,
                ),
                runtime=PluginRuntime(
                    "swap-replacement-owner", "swap-replacement-owner",
                    tmp_path, tmp_path, tmp_path, {},
                ),
            ))
            await asyncio.wait_for(new_owner_ready.wait(), 2)
            done, _ = await asyncio.wait((new_mount_task,), timeout=2)
            assert new_mount_task in done, "replacement owner mount timed out"
            try:
                new_fiber = new_mount_task.result()
            finally:
                new_mount_joined = True
            new_ctx = new_owner_context["value"]
            new_effect = new_owner_effect["value"]
            assert new_ctx is new_fiber.context
            assert new_ctx is not peer_context
            assert new_ctx is not old_ctx
            assert_peer_unchanged()
            assert not new_started.is_set()
            await _append_source_input(
                new_ctx, "swap", "swap-session", "swap-input-2", "second",
            )
            input2_written.set()
            await asyncio.wait_for(input2_processed.wait(), 2)
            assert not new_started.is_set()
            await asyncio.wait_for(accept(host, "peer-during", "p2"), 2)
            await asyncio.wait_for(peer_during_done.wait(), 2)
            assert_peer_unchanged()
            old_release.set()
            await asyncio.wait_for(old_done.wait(), 2)
            await asyncio.wait_for(old_join_entered.wait(), 2)
            assert not old_join_returned.is_set()
            assert not new_started.is_set()
            old_join_release.set()
            await asyncio.wait_for(old_join_returned.wait(), 2)
            await asyncio.wait_for(new_started.wait(), 2)
            await asyncio.wait_for(accept(host, "peer-after", "p3"), 2)
            await asyncio.wait_for(peer_after_done.wait(), 2)
            assert_peer_unchanged()
            assert opened[0][1] is old_ctx
            assert opened[1][1] is new_ctx
            assert opened[0][2] is not opened[1][2]
            swap_messages = log.reader("swap-session").snapshot()
            assert [message.message_id for message in swap_messages] == [
                "swap-input-1", "swap-output-1", "swap-input-2", "swap-output-2",
            ]
            assert starts.count(("swap-session", "swap")) == 2
            assert starts.count(("peer-before", "conversation")) == 1
            assert starts.count(("peer-during", "conversation")) == 1
            assert starts.count(("peer-after", "conversation")) == 1
            assert not watcher.done()
        except BaseException as error:
            body_error = error
        finally:
            old_release.set()
            old_join_release.set()
            if new_mount_task is not None and not new_mount_joined:
                try:
                    if not new_mount_task.done():
                        await asyncio.wait((new_mount_task,))
                    new_mount_task.result()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                finally:
                    new_mount_joined = True
            if new_effect is None:
                new_effect = new_owner_effect.get("value")
            if old_effect is not None:
                try:
                    await old_effect.aclose()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if new_effect is not None:
                try:
                    await new_effect.aclose()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if peer_effect is not None:
                try:
                    await peer_effect.aclose()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        if body_error is not None:
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "replacement cleanup failed", [body_error, *cleanup_errors]
                ) from body_error
            raise body_error
        if cleanup_errors:
            raise BaseExceptionGroup("replacement cleanup failed", cleanup_errors)


@pytest.mark.asyncio
async def test_follow_revoke_without_replacement_does_not_restart_old_source(
    tmp_path, monkeypatch,
):
    """撤销来源且无换代时只排空旧 Task，不以旧 registration 自恢复。"""
    entered, draining, release, source_done = (asyncio.Event() for _ in range(4))
    starts: list[tuple[str, str]] = []
    source_task: dict[str, object] = {}
    dispose_task = None
    dispose_task_joined = False
    source_joined = asyncio.Event()
    source_join_error: BaseException | None = None
    original_join = Task.join

    async def observe_join(task):
        nonlocal source_join_error
        if task is source_task.get("value"):
            try:
                result = await original_join(task)
            except BaseException as error:
                source_join_error = error
                source_joined.set()
                raise
            source_joined.set()
            return result
        return await original_join(task)

    monkeypatch.setattr(Task, "join", observe_join)

    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        if source != "fault":
            return
        source_task["value"] = task
        task.on_done(source_done.set)
        _ = output(ctx, task, reader, source)
        entered.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await release.wait()

    async with running(tmp_path, program, fault_source=True) as (
        log, host, watcher, fault_fiber,
    ):
        try:
            await asyncio.wait_for(accept_fault(host, "revoked", "r1"), 2)
            await asyncio.wait_for(entered.wait(), 2)
            dispose_task = asyncio.create_task(fault_fiber.dispose())
            await asyncio.wait_for(draining.wait(), 2)
            assert not dispose_task.done()
            release.set()
            try:
                await dispose_task
            finally:
                dispose_task_joined = True
            await asyncio.wait_for(source_done.wait(), 2)
            await asyncio.wait_for(source_joined.wait(), 2)
            assert isinstance(source_join_error, asyncio.CancelledError)
            assert starts == [("revoked", "fault")]
            assert not watcher.done()
        finally:
            release.set()
            if dispose_task is not None and not dispose_task_joined:
                try:
                    await dispose_task
                finally:
                    dispose_task_joined = True


@pytest.mark.asyncio
async def test_follow_unrelated_source_does_not_retry_source_settlement_failure(
    tmp_path, monkeypatch,
):
    """来源独有结算失败后，无关登记不重试失败来源，健康来源仍只产一条输出。"""
    bad_started, cleanup_started, cleanup_release = (asyncio.Event() for _ in range(3))
    healthy_done, source_done, settlement_stopped = (
        asyncio.Event(), asyncio.Event(), asyncio.Event()
    )
    main_action_completed = asyncio.Event()
    bad_joined = asyncio.Event()
    bad_task: dict[str, object] = {}
    child_error: dict[str, BaseException] = {}
    bad_error = RuntimeError("bad source settlement")
    starts: list[tuple[str, str]] = []
    unrelated_effect = None
    fault_source = None
    unrelated_delivered = False
    loop = asyncio.get_running_loop()

    follow_module = inspect.getmodule(follow)
    assert follow_module is not None
    original_warning = follow_module.logger.warning

    def observe_warning(message, *args, **kwargs):
        error = kwargs.get("exc_info")
        if (
            message == "回复来源任务结算失败，停止当前 source drive"
            and error is child_error.get("value")
            and error.__cause__ is bad_error
        ):
            settlement_stopped.set()
        return original_warning(message, *args, **kwargs)

    monkeypatch.setattr(follow_module.logger, "warning", observe_warning)

    original_changes = None
    original_entries = None

    async def before_watcher(log, host, fault_fiber):
        nonlocal original_changes, original_entries
        root = host.live_root
        assert root is not None
        registry = root.context.require(SOURCES)
        original_changes = type(registry).changes
        original_entries = type(registry).entries

        async def observe_changes(current_registry):
            nonlocal unrelated_delivered
            assert original_changes is not None
            async for entries in original_changes(current_registry):
                if any(source.name == "unrelated" for source in entries):
                    unrelated_delivered = True
                yield entries

        def observe_entries(current_registry):
            assert original_entries is not None
            entries = original_entries(current_registry)
            if unrelated_delivered and any(
                source.name == "unrelated" for source in entries
            ):
                loop.call_soon(main_action_completed.set)
            return entries

        monkeypatch.setattr(type(registry), "changes", observe_changes)
        monkeypatch.setattr(type(registry), "entries", observe_entries)

    original_join = Task.join

    async def observe_join(task):
        if task is bad_task.get("value"):
            try:
                result = await original_join(task)
            except BaseException:
                bad_joined.set()
                raise
            bad_joined.set()
            return result
        return await original_join(task)

    monkeypatch.setattr(Task, "join", observe_join)

    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        if source == "conversation":
            writer.append(
                "healthy-output",
                Output((ContentPart("text", "healthy"),), "complete"),
            )
            healthy_done.set()
            return
        bad_task["value"] = task
        task.on_done(source_done.set)
        bad_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            child_error["value"] = error
            cleanup_started.set()
            await cleanup_release.wait()
            raise error from bad_error

    async with running(
        tmp_path, program, fault_source=True, before_watcher=before_watcher,
    ) as (
        log, host, watcher, fault_fiber,
    ):
        try:
            root = host.live_root
            assert root is not None
            registry = root.context.require(SOURCES)
            fault_source = next(
                source for source in registry.entries() if source.name == "fault"
            )
            await asyncio.wait_for(accept_fault(host, "bad", "b1"), 2)
            await asyncio.wait_for(bad_started.wait(), 2)
            bad_task["value"].cancel()
            await asyncio.wait_for(cleanup_started.wait(), 2)
            cleanup_release.set()
            await asyncio.wait_for(source_done.wait(), 2)
            await asyncio.wait_for(bad_joined.wait(), 2)
            await asyncio.wait_for(settlement_stopped.wait(), 2)
            assert fault_fiber.state.name == "ACTIVE"
            assert fault_source is not None

            generation = host.generation("conversation")
            assert generation is not None and generation.fiber is not None
            unrelated_effect = await _register_source(
                generation.fiber.context, "unrelated"
            )
            await asyncio.wait_for(main_action_completed.wait(), 2)
            assert any(source is fault_source for source in registry.entries())
            await asyncio.wait_for(accept(host, "healthy", "h1"), 2)
            await asyncio.wait_for(healthy_done.wait(), 2)
            assert starts.count(("bad", "fault")) == 1
            healthy_outputs = [
                message for message in log.reader("healthy").snapshot()
                if message.source == "conversation" and isinstance(message.body, Output)
            ]
            assert len(healthy_outputs) == 1
            assert healthy_outputs[0].body == Output(
                (ContentPart("text", "healthy"),), "complete",
            )
            assert [item.message_id for item in log.reader("bad").snapshot()] == ["b1"]
            assert not watcher.done()
        finally:
            cleanup_release.set()
            if unrelated_effect is not None:
                await unrelated_effect.aclose()


@pytest.mark.asyncio
async def test_log_follower_coalesces_interrupts_without_blocking_other_sessions(tmp_path):
    entered, draining, release, other_done, finished = (asyncio.Event() for _ in range(5))
    starts = []
    async def program(ctx, task, reader, source):
        writer = output(ctx, task, reader, source)
        starts.append((reader.session_id, tuple(m.message_id for m in reader.snapshot())))
        if reader.session_id == "one" and not entered.is_set():
            entered.set()
            try:
                await asyncio.Future()
            finally:
                draining.set()
                await release.wait()
        else:
            writer.append("answer:" + reader.session_id, Output((ContentPart("text", "answer"),), "complete"))
            (other_done if reader.session_id == "other" else finished).set()

    async with running(tmp_path, program) as (log, host, watcher):
        await asyncio.wait_for(accept(host, "one", "u1"), 2)
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(accept(host, "one", "u2"), 2)
        await asyncio.wait_for(draining.wait(), 2)
        await asyncio.wait_for(accept(host, "one", "u3"), 2)
        await asyncio.wait_for(accept(host, "other", "v1"), 2)
        await asyncio.wait_for(other_done.wait(), 2)
        assert not finished.is_set()
        assert [m.message_id for m in log.reader("one").snapshot()] == ["u1", "u2", "u3"]
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        assert starts == [("one", ("u1",)), ("other", ("v1",)), ("one", ("u1", "u2", "u3"))]
        assert not watcher.done()


@pytest.mark.asyncio
async def test_stopping_follower_cancels_decisions_and_waits_for_real_cleanup(tmp_path):
    entered, draining, release = (asyncio.Event() for _ in range(3))
    held = []
    async def program(ctx, task, reader, source):
        held.append(output(ctx, task, reader, source))
        entered.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await release.wait()

    async with running(tmp_path, program) as (log, host, watcher):
        await asyncio.wait_for(accept(host, "one", "u1"), 2)
        await asyncio.wait_for(entered.wait(), 2)
        watcher.cancel()
        await asyncio.wait_for(draining.wait(), 2)
        assert not watcher.done()
        from session.log import WriterExpired
        with pytest.raises(WriterExpired):
            held[0].append("stale", Output((), "quiet"))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(watcher, 2)
        assert [m.message_id for m in log.reader("one").snapshot()] == ["u1"]


@pytest.mark.asyncio
async def test_manager_shutdown_stops_reply_before_closing_task_service(tmp_path):
    entered, draining, release = (asyncio.Event() for _ in range(3))
    held = []
    async def program(ctx, task, reader, source):
        held.append(output(ctx, task, reader, source))
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            draining.set()
            await release.wait()
    async with running(tmp_path, program, lifecycle=True) as (log, host, watcher):
        await asyncio.wait_for(accept(host, "one", "u1"), 2)
        await asyncio.wait_for(entered.wait(), 2)
        shutdown = asyncio.create_task(host.terminate_all())
        try:
            await asyncio.wait_for(draining.wait(), 2)
            assert not shutdown.done()
            from session.log import WriterExpired
            with pytest.raises(WriterExpired):
                held[0].append("stale", Output((), "quiet"))
        finally:
            release.set()
            await asyncio.wait_for(shutdown, 2)
        assert len(held) == 1
        assert watcher.cancelled()
        assert [m.message_id for m in log.reader("one").snapshot()] == ["u1"]


@pytest.mark.asyncio
async def test_follow_waits_for_monitor_and_source_physical_cleanup_after_caller_cancel(
    tmp_path, monkeypatch,
):
    """monitor 未退出时 caller 取消也必须等 Source Task 的真实 finally。"""
    entered, draining, body_release = (asyncio.Event() for _ in range(3))
    monitor_blocked, monitor_cancelled, monitor_release = (asyncio.Event() for _ in range(3))
    from agent.plugin_composition.context import RuntimeScope

    original_wait = RuntimeScope.wait_admission_closed

    async def delayed_wait(scope):
        await original_wait(scope)
        if not monitor_blocked.is_set():
            monitor_blocked.set()
            while True:
                try:
                    await monitor_release.wait()
                    return
                except asyncio.CancelledError:
                    monitor_cancelled.set()

    monkeypatch.setattr(RuntimeScope, "wait_admission_closed", delayed_wait)

    async def program(ctx, task, reader, source):
        _ = output(ctx, task, reader, source)
        entered.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await body_release.wait()

    source_dispose = None
    async with running(tmp_path, program) as (log, host, watcher):
        try:
            await asyncio.wait_for(accept(host, "one", "u1"), 2)
            await asyncio.wait_for(entered.wait(), 2)
            generation = host.generation("sources")
            assert generation is not None and generation.fiber is not None
            source_dispose = asyncio.create_task(generation.fiber.dispose())
            await asyncio.wait_for(monitor_blocked.wait(), 2)
            watcher.cancel()
            await asyncio.wait_for(monitor_cancelled.wait(), 2)
            assert not source_dispose.done()
            assert not watcher.done()
            monitor_release.set()
            await asyncio.wait_for(draining.wait(), 2)
            assert not source_dispose.done()
            body_release.set()
            with pytest.raises(asyncio.CancelledError):
                await watcher
            await source_dispose
            assert generation.fiber.state.name == "DISPOSED"
        finally:
            body_release.set()
            monitor_release.set()
            if not watcher.done():
                watcher.cancel()
            if source_dispose is not None and not source_dispose.done():
                await source_dispose


@pytest.mark.asyncio
async def test_follow_preserves_caller_cancel_and_source_cleanup_error(tmp_path):
    """caller 取消与真实 Source Task cleanup 错误必须同时可观察。"""
    entered, draining, cleanup_called, release = (asyncio.Event() for _ in range(4))
    cleanup_error = RuntimeError("source cleanup failed")

    async def program(ctx, task, reader, source):
        _ = output(ctx, task, reader, source)

        def cleanup():
            cleanup_called.set()
            raise cleanup_error

        task.on_close(cleanup)
        entered.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await release.wait()

    with pytest.raises(BaseExceptionGroup) as raised:
        async with running(tmp_path, program) as (log, host, watcher):
            try:
                await asyncio.wait_for(accept(host, "one", "u1"), 2)
                await asyncio.wait_for(entered.wait(), 2)
                watcher.cancel()
                await asyncio.wait_for(cleanup_called.wait(), 2)
                await asyncio.wait_for(draining.wait(), 2)
                release.set()
                await watcher
            finally:
                release.set()
    leaves = tuple(_exception_leaves(raised.value))
    assert any(leaf is cleanup_error for leaf in leaves)
    assert any(isinstance(leaf, asyncio.CancelledError) for leaf in leaves)


@pytest.mark.asyncio
async def test_follow_preserves_extra_caller_cancel_after_admission_monitor_stop(tmp_path, monkeypatch):
    """admission monitor 的内部取消与额外 caller 取消不能互相吞掉。"""
    entered, draining, body_release = (asyncio.Event() for _ in range(3))
    monitor_returned = asyncio.Event()
    from agent.plugin_composition.context import RuntimeScope

    original_wait = RuntimeScope.wait_admission_closed

    async def record_wait(scope):
        await original_wait(scope)
        monitor_returned.set()

    monkeypatch.setattr(RuntimeScope, "wait_admission_closed", record_wait)

    async def program(ctx, task, reader, source):
        _ = output(ctx, task, reader, source)
        entered.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await body_release.wait()

    source_dispose = None
    async with running(tmp_path, program) as (log, host, watcher):
        try:
            await asyncio.wait_for(accept(host, "one", "u1"), 2)
            await asyncio.wait_for(entered.wait(), 2)
            generation = host.generation("sources")
            assert generation is not None and generation.fiber is not None
            source_dispose = asyncio.create_task(generation.fiber.dispose())
            await asyncio.wait_for(monitor_returned.wait(), 2)
            watcher.cancel()
            await asyncio.wait_for(draining.wait(), 2)
            assert not watcher.done()
            body_release.set()
            with pytest.raises(asyncio.CancelledError):
                await watcher
            await source_dispose
        finally:
            body_release.set()
            if not watcher.done():
                watcher.cancel()
            if source_dispose is not None and not source_dispose.done():
                await source_dispose


@pytest.mark.asyncio
async def test_follow_uses_public_join_when_task_factory_rejects_new_tasks(tmp_path):
    """已有 Source Task 的物理结算不依赖另一个 join waiter Task。"""
    entered, draining, release = (asyncio.Event() for _ in range(3))
    factory_set = asyncio.Event()
    created = []
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()

    async def program(ctx, task, reader, source):
        _ = output(ctx, task, reader, source)
        entered.set()

        def rejecting_factory(_loop, coroutine, **_kwargs):
            created.append(coroutine)
            raise RuntimeError("continuous task factory rejection")

        loop.set_task_factory(rejecting_factory)
        factory_set.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await release.wait()

    async with running(tmp_path, program) as (log, host, watcher):
        try:
            await asyncio.wait_for(accept(host, "one", "u1"), 2)
            await asyncio.wait_for(entered.wait(), 2)
            async with asyncio.timeout(2):
                await factory_set.wait()
            before = len(created)
            watcher.cancel()
            async with asyncio.timeout(2):
                await draining.wait()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await watcher
            assert len(created) == before
        finally:
            release.set()
            if not watcher.done():
                watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
            finally:
                loop.set_task_factory(original_factory)
            for coroutine in created:
                if inspect.getcoroutinestate(coroutine) != inspect.CORO_CLOSED:
                    coroutine.close()


@pytest.mark.asyncio
async def test_follow_preserves_same_turn_external_cancel_on_exact_drive(
    tmp_path, monkeypatch,
):
    """同一 exact drive 的 marker 与外部取消同拍时，外部事实不能丢。"""
    entered, draining, body_release = (asyncio.Event() for _ in range(3))
    monitor_returned = asyncio.Event()
    external_delivered = asyncio.Event()
    source_done = asyncio.Event()
    events: list[tuple[str, int]] = []
    drive_tasks: list[asyncio.Task[object]] = []
    source_task: dict[str, object] = {}
    source_result: dict[str, object] = {}
    from agent.plugin_composition.context import RuntimeScope

    original_create_task = asyncio.TaskGroup.create_task
    original_wait = RuntimeScope.wait_admission_closed

    def record_drive(group, coroutine, **kwargs):
        task = original_create_task(group, coroutine, **kwargs)
        if coroutine.cr_code is not None and coroutine.cr_code.co_name == "drive":
            drive_tasks.append(task)
        return task

    async def arrange_same_turn(scope):
        result = await original_wait(scope)
        if not monitor_returned.is_set():
            monitor_returned.set()

            def external_cancel():
                drive = drive_tasks[0]
                events.append(("before_external", drive.cancelling()))
                drive.cancel("caller cancellation")
                events.append(("after_external", drive.cancelling()))
                external_delivered.set()

            asyncio.get_running_loop().call_soon(external_cancel)
        return result

    monkeypatch.setattr(asyncio.TaskGroup, "create_task", record_drive)
    monkeypatch.setattr(RuntimeScope, "wait_admission_closed", arrange_same_turn)

    async def program(ctx, task, reader, source):
        _ = output(ctx, task, reader, source)
        source_task["value"] = task
        task.on_done(source_done.set)
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            draining.set()
            await body_release.wait()
        source_result["value"] = "source stopped normally"
        return source_result["value"]

    source_dispose = None
    source_dispose_retrieved = False
    async with running(tmp_path, program) as (log, host, watcher):
        try:
            await asyncio.wait_for(accept(host, "one", "u1"), 2)
            await asyncio.wait_for(entered.wait(), 2)
            generation = host.generation("sources")
            assert generation is not None and generation.fiber is not None
            source_dispose = asyncio.create_task(generation.fiber.dispose())
            await asyncio.wait_for(monitor_returned.wait(), 2)
            await asyncio.wait_for(external_delivered.wait(), 2)
            assert events[0][0] == "before_external"
            assert events[0][1] == 1
            assert events[1][0] == "after_external"
            assert events[1][1] >= 2
            assert not drive_tasks[0].done()
            await asyncio.wait_for(draining.wait(), 2)
            body_release.set()
            with pytest.raises(asyncio.CancelledError):
                await drive_tasks[0]
            await asyncio.wait_for(source_done.wait(), 2)
            assert await source_task["value"].join() == "source stopped normally"
            assert source_result["value"] == "source stopped normally"
        finally:
            body_release.set()
            try:
                if source_dispose is not None:
                    if not source_dispose.done():
                        await source_dispose
                    if not source_dispose_retrieved:
                        try:
                            source_dispose.result()
                        finally:
                            source_dispose_retrieved = True
            finally:
                if not watcher.done():
                    watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
        assert generation.fiber.state.name == "DISPOSED"


@pytest.mark.asyncio
async def test_follow_preserves_child_cancelled_error_and_cleanup_cause(
    tmp_path,
):
    """Source Task 的原始 CancelledError 与 cleanup cause 必须保留。"""
    entered, cleanup_started, release = (asyncio.Event() for _ in range(3))
    cleanup_error = RuntimeError("async source cleanup failed")
    child_error: dict[str, asyncio.CancelledError] = {}

    async def cleanup():
        cleanup_started.set()
        await release.wait()
        raise cleanup_error

    async def program(ctx, task, reader, source):
        _ = output(ctx, task, reader, source)
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            child_error["value"] = error
            try:
                await cleanup()
            except RuntimeError as cleanup_failure:
                raise error from cleanup_failure
            raise

    with pytest.raises(BaseException) as raised:
        async with running(tmp_path, program) as (log, host, watcher):
            try:
                await asyncio.wait_for(accept(host, "one", "u1"), 2)
                await asyncio.wait_for(entered.wait(), 2)
                watcher.cancel()
                await asyncio.wait_for(cleanup_started.wait(), 2)
                release.set()
                await watcher
            finally:
                release.set()

    child = child_error["value"]
    leaves = tuple(_exception_leaves(raised.value))
    assert any(leaf is child for leaf in leaves)
    assert child.__cause__ is cleanup_error
    assert any(
        isinstance(leaf, asyncio.CancelledError) and leaf is not child
        for leaf in leaves
    )


@pytest.mark.asyncio
async def test_follow_logs_lone_source_child_settlement_failure_and_keeps_peer(
    tmp_path, caplog,
):
    """无 caller 取消时，带 cleanup cause 的 child CE 只停止当前来源。"""
    healthy_started, healthy_release, healthy_ready, healthy_ready_or_cancel = (
        asyncio.Event() for _ in range(4)
    )
    healthy_output_release, healthy_done, healthy_outcome = (asyncio.Event() for _ in range(3))
    bad_started, cleanup_started, cleanup_release, source_done = (
        asyncio.Event() for _ in range(4)
    )
    cleanup_error = RuntimeError("isolated source cleanup failed")
    child_error: dict[str, asyncio.CancelledError] = {}
    source_task: dict[str, object] = {}
    starts: list[tuple[str, str]] = []
    warning_seen = asyncio.Event()
    settlement_observed = asyncio.Event()
    warning_records = []

    class WarningSignal(logging.Handler):
        def emit(self, record):
            warning_records.append(record)
            child = child_error.get("value")
            if (
                child is not None
                and record.exc_info is not None
                and _contains_exception(record.exc_info[1], child)
            ):
                warning_seen.set()

    warning_handler = WarningSignal()
    reply_logger = logging.getLogger("plugins.reply.follow")
    reply_logger.addHandler(warning_handler)

    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        if source == "conversation":
            healthy_started.set()
            try:
                await healthy_release.wait()
                healthy_ready.set()
                healthy_ready_or_cancel.set()
                await healthy_output_release.wait()
            except asyncio.CancelledError:
                healthy_outcome.set()
                healthy_ready_or_cancel.set()
                raise
            writer.append("healthy-output", Output((ContentPart("text", "healthy"),), "complete"))
            healthy_done.set()
            healthy_outcome.set()
            return

        source_task["value"] = task
        def mark_source_settled():
            source_done.set()
            settlement_observed.set()

        task.on_done(mark_source_settled)
        bad_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError as error:
            child_error["value"] = error
            cleanup_started.set()
            await cleanup_release.wait()
            raise error from cleanup_error

    source_dispose = None
    source_dispose_retrieved = False
    try:
        with caplog.at_level("WARNING", logger="plugins.reply.follow"):
            async with running(tmp_path, program, fault_source=True) as (
                log, host, watcher, fault_fiber,
            ):
                try:
                    await asyncio.wait_for(accept(host, "healthy", "h1"), 2)
                    await asyncio.wait_for(healthy_started.wait(), 2)
                    await asyncio.wait_for(accept_fault(host, "bad", "b1"), 2)
                    await asyncio.wait_for(bad_started.wait(), 2)

                    source_dispose = asyncio.create_task(fault_fiber.dispose())
                    await asyncio.wait_for(cleanup_started.wait(), 2)
                    cleanup_release.set()
                    done, pending = await asyncio.wait((source_dispose,), timeout=2)
                    assert source_dispose in done
                    assert not pending
                    try:
                        source_dispose.result()
                    finally:
                        source_dispose_retrieved = True
                    await asyncio.wait_for(settlement_observed.wait(), 2)
                    await asyncio.wait_for(source_done.wait(), 2)
                    with pytest.raises(asyncio.CancelledError) as joined:
                        await source_task["value"].join()

                    child = child_error["value"]
                    assert isinstance(joined.value, asyncio.CancelledError)
                    assert child.__cause__ is cleanup_error
                    await asyncio.wait_for(warning_seen.wait(), 2)
                    assert any(
                        record.exc_info is not None
                        and _contains_exception(record.exc_info[1], child)
                        and child.__cause__ is cleanup_error
                        for record in warning_records
                    )
                    healthy_release.set()
                    await asyncio.wait_for(healthy_ready_or_cancel.wait(), 2)
                    assert healthy_ready.is_set()
                    healthy_output_release.set()
                    await asyncio.wait_for(healthy_outcome.wait(), 2)
                    assert healthy_done.is_set()
                    assert not watcher.done()
                    assert starts.count(("bad", "fault")) == 1
                    healthy_outputs = [
                        message
                        for message in log.reader("healthy").snapshot()
                        if message.source == "conversation"
                        and isinstance(message.body, Output)
                    ]
                    assert len(healthy_outputs) == 1
                    healthy_message = healthy_outputs[0]
                    assert (healthy_message.author, healthy_message.source) == (
                        "assistant",
                        "conversation",
                    )
                    assert healthy_message.body == Output(
                        (ContentPart("text", "healthy"),), "complete"
                    )
                finally:
                    cleanup_release.set()
                    healthy_release.set()
                    healthy_output_release.set()
                    try:
                        if healthy_started.is_set():
                            await asyncio.wait_for(healthy_outcome.wait(), 2)
                    finally:
                        if source_dispose is not None:
                            if not source_dispose.done():
                                try:
                                    await source_dispose
                                finally:
                                    source_dispose_retrieved = True
                            if not source_dispose_retrieved:
                                try:
                                    source_dispose.result()
                                finally:
                                    source_dispose_retrieved = True
    finally:
        reply_logger.removeHandler(warning_handler)

    assert any(
        record.exc_info is not None
        and record.exc_info[1] is child_error["value"]
        and child_error["value"].__cause__ is cleanup_error
        for record in warning_records
    )


@pytest.mark.asyncio
async def test_follow_same_session_peer_does_not_replay_failed_source_history(
    tmp_path, monkeypatch,
):
    """同 Session peer 新消息不重放带 cleanup cause 的旧 Source Input。"""
    session_id = "late-failure-peer"
    fault_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    first_source_done = asyncio.Event()
    second_source_done = asyncio.Event()
    first_join_finished = asyncio.Event()
    second_join_finished = asyncio.Event()
    warning_seen = asyncio.Event()
    peer_processed = asyncio.Event()
    peer_output_done = asyncio.Event()
    new_fault_output_done = asyncio.Event()
    cleanup_error = RuntimeError("late source cleanup failed")
    child_error: dict[str, asyncio.CancelledError] = {}
    source_tasks: dict[str, Task] = {}
    join_observations: dict[str, dict[str, object]] = {
        "first": {},
        "second": {},
    }
    join_finished = {
        "first": first_join_finished,
        "second": second_join_finished,
    }
    starts: list[tuple[str, str]] = []
    warning_records = []

    class WarningSignal(logging.Handler):
        def emit(self, record):
            warning_records.append(record)
            child = child_error.get("value")
            if (
                child is not None
                and record.exc_info is not None
                and _contains_exception(record.exc_info[1], child)
            ):
                warning_seen.set()

    warning_handler = WarningSignal()
    reply_logger = logging.getLogger("plugins.reply.follow")
    reply_logger.addHandler(warning_handler)

    original_task_join = Task.join

    async def observe_task_join(task):
        label = next(
            (name for name, source_task in source_tasks.items() if task is source_task),
            None,
        )
        if label is None:
            return await original_task_join(task)

        observation = join_observations[label]
        try:
            try:
                result = await original_task_join(task)
            except BaseException as error:
                observation["error"] = error
                raise
            observation["result"] = result
            return result
        finally:
            observation["task"] = task
            join_finished[label].set()

    monkeypatch.setattr(Task, "join", observe_task_join)

    original_source_names = MessageReader.source_names

    def observe_source_names(reader):
        names = original_source_names(reader)
        if reader.session_id == session_id and reader.get("peer-input") is not None:
            peer_processed.set()
        return names

    monkeypatch.setattr(MessageReader, "source_names", observe_source_names)

    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        if source == "conversation":
            writer.append(
                "peer-output",
                Output((ContentPart("text", "peer answer"),), "complete"),
            )
            peer_output_done.set()
            return

        fault_runs = sum(item == (session_id, "fault") for item in starts)
        if fault_runs == 1:
            source_tasks["first"] = task
            task.on_done(first_source_done.set)
            fault_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError as error:
                child_error["value"] = error
                cleanup_started.set()
                await cleanup_release.wait()
                raise error from cleanup_error

        source_tasks["second"] = task
        task.on_done(second_source_done.set)
        writer.append(
            "new-fault-output",
            Output((ContentPart("text", "new fault answer"),), "complete"),
        )
        new_fault_output_done.set()
        return "new fault completed"

    try:
        async with running(tmp_path, program) as (log, host, watcher):
            try:
                root = host.live_root
                assert root is not None
                conversation = host.generation("conversation")
                assert conversation is not None and conversation.fiber is not None
                conversation_context = conversation.fiber.context

                # 1. Persist the old Source Input before its owner is registered.
                async with conversation_context.runtime_scope():
                    historical_writer = conversation_context.require(
                        MESSAGE_WRITERS
                    ).bind(
                        conversation_context,
                        author="user",
                        source="fault",
                        body_types=(Input,),
                        content={
                            "text": conversation_context.require(CONTENT).check_text,
                        },
                    )(session_id)
                    historical = historical_writer.append(
                        "old-fault-input",
                        Input((ContentPart("text", "old fault question"),)),
                    )

                fault_fiber = await root.mount(
                    _mount_fault_source,
                    name="late-fault-source",
                    inject=(
                        CONTENT, MESSAGE_CATALOG, MESSAGE_WRITERS, RESTART_GATE,
                        SOURCE_SESSION, SOURCES, TASKS,
                    ),
                    runtime=PluginRuntime(
                        "late-fault-source", "late-fault-source",
                        tmp_path, tmp_path, tmp_path, {},
                    ),
                )
                registry = root.context.require(SOURCES)
                await asyncio.wait_for(fault_started.wait(), 2)
                fault_source = next(
                    source for source in registry.entries() if source.name == "fault"
                )
                first_task = source_tasks["first"]
                first_task.cancel()
                await asyncio.wait_for(cleanup_started.wait(), 2)
                cleanup_release.set()
                await asyncio.wait_for(first_source_done.wait(), 2)
                await asyncio.wait_for(first_join_finished.wait(), 2)
                await asyncio.wait_for(warning_seen.wait(), 2)

                child = child_error["value"]
                first_join = join_observations["first"]
                assert first_join["task"] is first_task
                assert first_join["error"] is child
                assert isinstance(first_join["error"], asyncio.CancelledError)
                assert child.__cause__ is cleanup_error
                assert any(
                    record.exc_info is not None
                    and record.exc_info[1] is child
                    and child.__cause__ is cleanup_error
                    for record in warning_records
                )
                assert first_task.done and not first_task.active
                await asyncio.wait_for(fault_fiber._calls_idle.wait(), 2)
                assert not fault_fiber._in_flight_calls
                assert fault_fiber.state.name == "ACTIVE"
                assert any(source is fault_source for source in registry.entries())
                reader = log.reader(session_id)
                failed_source_head = reader.head(source="fault")
                assert failed_source_head == historical.seq

                # 2. A same-Session peer Input reaches the follower's real source scan.
                await accept(host, session_id, "peer-input")
                await asyncio.wait_for(peer_processed.wait(), 2)
                await asyncio.wait_for(peer_output_done.wait(), 2)
                assert fault_fiber.state.name == "ACTIVE"
                assert any(source is fault_source for source in registry.entries())
                assert starts.count((session_id, "fault")) == 1
                assert reader.head(source="fault") == failed_source_head
                peer_outputs = [
                    message for message in reader.snapshot()
                    if message.source == "conversation" and isinstance(message.body, Output)
                ]
                assert len(peer_outputs) == 1
                assert peer_outputs[0].body == Output(
                    (ContentPart("text", "peer answer"),), "complete",
                )
                fault_messages = reader.read(source="fault")
                assert [(message.message_id, message.body) for message in fault_messages] == [
                    (
                        "old-fault-input",
                        Input((ContentPart("text", "old fault question"),)),
                    ),
                ]
                assert not any(
                    message.source == "fault"
                    and isinstance(message.body, Control)
                    and message.body.action == "failure"
                    for message in reader.snapshot()
                )

                # 3. A new Input still runs once through the registered SourceSession.
                await accept_fault(host, session_id, "new-fault-input")
                await asyncio.wait_for(new_fault_output_done.wait(), 2)
                second_task = source_tasks["second"]
                await asyncio.wait_for(second_source_done.wait(), 2)
                await asyncio.wait_for(second_join_finished.wait(), 2)
                second_join = join_observations["second"]
                assert second_join["task"] is second_task
                assert "error" not in second_join
                assert second_join["result"] == "new fault completed"
                assert second_task.done and not second_task.active
                assert starts.count((session_id, "fault")) == 2
                assert reader.get("old-fault-input") == historical
                assert reader.get("new-fault-input") is not None
                new_output = reader.get("new-fault-output")
                assert new_output is not None
                assert new_output.body == Output(
                    (ContentPart("text", "new fault answer"),), "complete",
                )
                assert not watcher.done()
                assert not any(
                    message.source == "fault"
                    and isinstance(message.body, Control)
                    and message.body.action == "failure"
                    for message in reader.snapshot()
                )
            finally:
                cleanup_release.set()
    finally:
        reply_logger.removeHandler(warning_handler)


@pytest.mark.asyncio
async def test_follow_logs_source_on_close_failure_and_keeps_peer(tmp_path, caplog):
    """无 caller 取消时，Source Task 同步 on_close 失败不击穿健康来源。"""
    healthy_started, healthy_release, healthy_ready, healthy_ready_or_cancel = (
        asyncio.Event() for _ in range(4)
    )
    healthy_output_release, healthy_done, healthy_outcome = (asyncio.Event() for _ in range(3))
    bad_started, draining, body_release = (asyncio.Event() for _ in range(3))
    sync_error = RuntimeError("source writer close failed")
    source_task: dict[str, object] = {}
    starts: list[tuple[str, str]] = []
    warning_seen = asyncio.Event()
    settlement_observed = asyncio.Event()
    warning_records = []

    class WarningSignal(logging.Handler):
        def emit(self, record):
            warning_records.append(record)
            if (
                record.exc_info is not None
                and _contains_exception(record.exc_info[1], sync_error)
            ):
                warning_seen.set()

    warning_handler = WarningSignal()
    reply_logger = logging.getLogger("plugins.reply.follow")
    reply_logger.addHandler(warning_handler)

    def fail_on_close():
        raise sync_error

    async def program(ctx, task, reader, source):
        starts.append((reader.session_id, source))
        writer = output(ctx, task, reader, source)
        if source == "conversation":
            healthy_started.set()
            try:
                await healthy_release.wait()
                healthy_ready.set()
                healthy_ready_or_cancel.set()
                await healthy_output_release.wait()
            except asyncio.CancelledError:
                healthy_outcome.set()
                healthy_ready_or_cancel.set()
                raise
            writer.append("healthy-output", Output((ContentPart("text", "healthy"),), "complete"))
            healthy_done.set()
            healthy_outcome.set()
            return

        source_task["value"] = task
        task.on_done(settlement_observed.set)
        task.on_close(fail_on_close)
        bad_started.set()
        try:
            await asyncio.Future()
        finally:
            draining.set()
            await body_release.wait()

    source_dispose = None
    source_dispose_retrieved = False
    try:
        with caplog.at_level("WARNING", logger="plugins.reply.follow"):
            async with running(tmp_path, program, fault_source=True) as (
                log, host, watcher, fault_fiber,
            ):
                try:
                    await asyncio.wait_for(accept(host, "healthy", "h1"), 2)
                    await asyncio.wait_for(healthy_started.wait(), 2)
                    await asyncio.wait_for(accept_fault(host, "bad", "b1"), 2)
                    await asyncio.wait_for(bad_started.wait(), 2)

                    source_dispose = asyncio.create_task(fault_fiber.dispose())
                    await asyncio.wait_for(draining.wait(), 2)
                    body_release.set()
                    done, pending = await asyncio.wait((source_dispose,), timeout=2)
                    assert source_dispose in done
                    assert not pending
                    try:
                        source_dispose.result()
                    finally:
                        source_dispose_retrieved = True
                    await asyncio.wait_for(settlement_observed.wait(), 2)
                    healthy_release.set()
                    await asyncio.wait_for(healthy_ready_or_cancel.wait(), 2)
                    assert healthy_ready.is_set()
                    healthy_output_release.set()
                    await asyncio.wait_for(healthy_outcome.wait(), 2)
                    assert healthy_done.is_set()
                    await asyncio.wait_for(warning_seen.wait(), 2)
                    assert not watcher.done()
                    assert starts.count(("bad", "fault")) == 1
                    healthy_outputs = [
                        message
                        for message in log.reader("healthy").snapshot()
                        if message.source == "conversation"
                        and isinstance(message.body, Output)
                    ]
                    assert len(healthy_outputs) == 1
                    healthy_message = healthy_outputs[0]
                    assert (healthy_message.author, healthy_message.source) == (
                        "assistant",
                        "conversation",
                    )
                    assert healthy_message.body == Output(
                        (ContentPart("text", "healthy"),), "complete"
                    )
                finally:
                    body_release.set()
                    healthy_release.set()
                    healthy_output_release.set()
                    try:
                        if healthy_started.is_set():
                            await asyncio.wait_for(healthy_outcome.wait(), 2)
                    finally:
                        if source_dispose is not None:
                            if not source_dispose.done():
                                try:
                                    await source_dispose
                                finally:
                                    source_dispose_retrieved = True
                            if not source_dispose_retrieved:
                                try:
                                    source_dispose.result()
                                finally:
                                    source_dispose_retrieved = True
    finally:
        reply_logger.removeHandler(warning_handler)

    assert any(
        record.exc_info is not None
        and _contains_exception(record.exc_info[1], sync_error)
        for record in warning_records
    )
