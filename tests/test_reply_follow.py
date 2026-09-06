import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.conversation.plugin import CONVERSATION
from plugins.reply.follow import follow
from plugins.sources.plugin import SOURCES
from session.log import MessageLog
from session.message import ContentPart, Input, Output


@asynccontextmanager
async def running(tmp_path, program, *, lifecycle=False):
    sources = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins/conversation", sources / "conversation",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(Path(__file__).parents[1] / "plugins/sources", sources / "sources",
                    ignore=shutil.ignore_patterns("__pycache__"))
    probe = sources / "probe"
    probe.mkdir()
    (probe / "plugin.py").write_text('''
from agent.plugin_composition import ServiceKey
api_version = 3
name = "probe"
version = "1.0.0"
inject = ()
async def apply(ctx, config):
    await ctx.provide(ServiceKey("probe"), ctx)
''')
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log)
    watcher = None
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context.require(ServiceKey("probe"))
            registered = snapshot.composition_root.context.require(SOURCES)
            watcher = await ctx.spawn(follow(ctx, log.catalog(), registered,
                                             lambda task, reader, source: program(ctx, task, reader, source)), name="follow")
        if lifecycle:
            from agent.plugin_composition import RUNTIME_STOPPING
            async def stop(_event):
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
            await ctx.on(RUNTIME_STOPPING, stop)
            await host.start_runtime()
        yield log, host, watcher
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        await host.terminate_all()
        log.close()


async def accept(host, session, identity):
    async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
        return await snapshot.composition_root.context.require(CONVERSATION)(session).accept(
            identity, Input((ContentPart("text", identity),)))


def output(ctx, task, reader, source):
    from agent.plugin_composition.messages import MESSAGE_WRITERS
    from plugins.content.plugin import check_text

    writer = ctx.require(MESSAGE_WRITERS).bind(
        ctx, author="assistant", source=source, body_types=(Output,), content={"text": check_text}
    )(reader.session_id)
    task.on_close(writer.expire)
    return writer


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
        await accept(host, "one", "u1")
        await asyncio.wait_for(entered.wait(), 2)
        await accept(host, "one", "u2")
        await asyncio.wait_for(draining.wait(), 2)
        await accept(host, "one", "u3")
        await accept(host, "other", "v1")
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
        await accept(host, "one", "u1")
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
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.lease_count == 1


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
        await accept(host, "one", "u1")
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
