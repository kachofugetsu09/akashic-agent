"""验证普通渠道请求使用声明者权限，并参与真实 generation 排空。"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.plugin_composition import CHANNELS, CompositionError, TIMERS
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog

MODULE = '''import asyncio
from agent.plugin_composition import (
    CHANNELS, TIMERS, ChannelCapability, ChannelDefinition, ChannelReady, StopReceipt, InboundIdentity)
api_version = 3
name = "request_channel"
version = "1.0.0"
inject = (CHANNELS, TIMERS)
factory_context = None
declaration_context = None
closed = asyncio.Event()
stopped = asyncio.Event()
def build_channel(context):
    global factory_context
    factory_context = context
    class Adapter:
        def attach_runtime(self, ports):
            pass
        async def start(self):
            self.finish = asyncio.Event()
            ready = asyncio.Event()
            async def serve():
                ready.set()
                await self.finish.wait()
            self.server = await context.spawn_owned(serve(), name="test-listener")
            await ready.wait()
            return ChannelReady(context.binding_token)
        def open_admission(self):
            pass
        def close_admission(self):
            closed.set()
        async def stop(self):
            self.finish.set()
            await self.server
            stopped.set()
            return StopReceipt(context.binding_token, resources_closed=True)
        async def deliver(self, request):
            raise AssertionError("request scope test must not send")
    return Adapter()
async def apply(ctx, config):
    async def child(child_ctx):
        global declaration_context
        declaration_context = child_ctx
        await child_ctx.require(CHANNELS).register(child_ctx, ChannelDefinition(
            name="request-test", capabilities=frozenset({ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}),
            factory_export="build_channel", inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID, credential_paths=()))
    await ctx.mount(child, name="listener", inject=(CHANNELS,))
'''


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_request", [False, True])
async def test_request_scope_keeps_child_grants_and_blocks_shutdown_until_released(
    tmp_path: Path, cancel_request: bool,
) -> None:
    source = tmp_path / "plugins/request_channel"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text(MODULE)
    (source / "akashic.plugin.toml").write_text(
        'schema_version=1\nname="request_channel"\nversion="1.0.0"\n'
        'api_version=3\nentrypoint="plugin.py"\n'
    )
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager(
        [source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache", message_log=log,
    )
    shutdown = None
    request_task = None
    try:
        await host.load_all()
        await host.start_runtime()
        snapshot = host.current_snapshot
        assert snapshot is not None
        generation = snapshot.generations["request_channel"]
        module = generation.instance.module
        factory = module.factory_context
        assert factory.data_root == generation.data_dir
        assert factory.open_scope is not None
        async def forbidden_listener():
            raise AssertionError("listener was created outside start")
        with pytest.raises(RuntimeError, match="adapter.start"):
            await factory.spawn_owned(forbidden_listener(), name="late-listener")
        entered = asyncio.Event()
        release = asyncio.Event()
        retained = []

        async def request():
            async with factory.open_scope() as ctx:
                retained.append(ctx)
                assert ctx.plugin_id == "request_channel"
                assert ctx.require(CHANNELS) is not None
                with pytest.raises(CompositionError, match="未声明"):
                    ctx.require(TIMERS)

                async def inherited_task():
                    with pytest.raises(CompositionError, match="作用域"):
                        ctx.require(CHANNELS)
                    async with module.declaration_context.runtime_scope():
                        with pytest.raises(CompositionError, match="作用域"):
                            ctx.require(CHANNELS)

                await asyncio.create_task(inherited_task())
                entered.set()
                await release.wait()

        request_task = asyncio.create_task(request())
        await asyncio.wait_for(entered.wait(), 5)
        shutdown = asyncio.create_task(host.terminate_all())
        await asyncio.wait_for(module.closed.wait(), 5)
        assert not shutdown.done()
        assert not module.stopped.is_set()
        with pytest.raises(RuntimeError, match="admission"):
            async with factory.open_scope():
                raise AssertionError("closed channel accepted request")
        if cancel_request:
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        else:
            release.set()
            await request_task
        await asyncio.wait_for(shutdown, 5)
        assert module.stopped.is_set()
        with pytest.raises(CompositionError, match="作用域"):
            retained[0].require(CHANNELS)
    finally:
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        if shutdown is not None:
            await shutdown
        else:
            await host.terminate_all()
        log.close()
