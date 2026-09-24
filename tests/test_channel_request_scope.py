"""验证普通渠道请求使用声明者权限，并参与真实 generation 排空。"""
from __future__ import annotations

import asyncio
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import CHANNELS, CompositionError, FiberState, ServiceKey, TIMERS
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace

MODULE = '''import asyncio
from agent.plugin_composition import (
    CHANNELS, TIMERS, ChannelCapability, ChannelDefinition, ChannelReady, StopReceipt,
    InboundIdentity, ServiceKey)
from agent.plugin_composition.channels import CHANNEL_INPUT
api_version = 3
name = "request_channel"
version = "1.0.0"
inject = (CHANNELS, TIMERS)
factory_context = None
declaration_context = None
target_fiber = None
hard_consumer_exit = asyncio.Event()
consumer_release = asyncio.Event()
probe_after_unload = asyncio.Event()
scope_checked = asyncio.Event()
closed = asyncio.Event()
stopped = asyncio.Event()
HARD_SERVICE = ServiceKey("request.test.hard-service")
def build_channel(context):
    global factory_context
    factory_context = context
    class Adapter:
        def attach_runtime(self, ports):
            pass
        async def start(self):
            if context.open_scope is None:
                raise AssertionError("startup request scope missing")
            async with context.open_scope() as startup_scope:
                assert startup_scope.require(CHANNELS) is not None
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
            pass
        async def stop(self):
            self.finish.set()
            await self.server
            closed.set()
            stopped.set()
            return StopReceipt(context.binding_token, resources_closed=True)
        async def deliver(self, request):
            raise AssertionError("request scope test must not send")
    return Adapter()
async def apply(ctx):
    async def unused_input(*args):
        raise AssertionError("request test cannot accept input")
    await ctx.provide(CHANNEL_INPUT, unused_input)
    async def child(child_ctx):
        global declaration_context
        declaration_context = child_ctx
        await child_ctx.provide(HARD_SERVICE, object())
        await child_ctx.require(CHANNELS).register(child_ctx, ChannelDefinition(
            name="request-test", capabilities=frozenset({ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}),
            factory=build_channel, inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID))
    global target_fiber
    target_fiber = await ctx.mount(child, name="listener", inject=(CHANNELS, CHANNEL_INPUT))

    async def hard_consumer(consumer_ctx):
        service = consumer_ctx.require(HARD_SERVICE)
        async def cleanup():
            assert service is not None
            hard_consumer_exit.set()
            await consumer_release.wait()
        await consumer_ctx.effect(lambda: cleanup, label="request-hard-consumer")
    await ctx.mount(hard_consumer, name="hard-consumer", inject=(HARD_SERVICE,))
'''


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_request", [False, True])
async def test_request_scope_keeps_child_grants_and_blocks_shutdown_until_released(
    tmp_path: Path, cancel_request: bool,
) -> None:
    source = tmp_path / "plugins/request_channel"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text(MODULE)
    shutil.copytree(Path(__file__).parents[1] / "plugins/channels", source.parent / "channels", ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager(
        [source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache", message_log=log,
    )
    shutdown = None
    request_task = None
    module = None
    release = None
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        generation = host.generation("request_channel")
        assert generation is not None
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
                await module.probe_after_unload.wait()
                assert ctx.require(CHANNELS) is not None
                async with module.declaration_context.runtime_scope():
                    assert ctx.require(CHANNELS) is not None
                module.scope_checked.set()
                await release.wait()

        request_task = asyncio.create_task(request())
        await asyncio.wait_for(entered.wait(), 5)
        # The request remains in the original contributor scope while the
        # provider's local dispose drives its real hard consumer cleanup.
        shutdown = asyncio.create_task(module.target_fiber.dispose())
        await asyncio.wait_for(module.hard_consumer_exit.wait(), 5)
        assert module.target_fiber.state is FiberState.UNLOADING
        assert not shutdown.done()
        assert not module.stopped.is_set()
        module.probe_after_unload.set()
        await asyncio.wait_for(module.scope_checked.wait(), 5)
        with pytest.raises(CompositionError) as error:
            async with factory.open_scope():
                raise AssertionError("closed channel accepted request")
        assert error.value.code == "OWNER_UNAVAILABLE"
        if cancel_request:
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        else:
            release.set()
            await request_task
        module.consumer_release.set()
        await asyncio.wait_for(shutdown, 5)
        assert module.closed.is_set()
        assert module.stopped.is_set()
        with pytest.raises(CompositionError, match="作用域"):
            retained[0].require(CHANNELS)
    finally:
        if module is not None:
            module.consumer_release.set()
            module.probe_after_unload.set()
            module.scope_checked.set()
        if release is not None:
            release.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
        if shutdown is not None:
            await asyncio.gather(shutdown, return_exceptions=True)
        await host.terminate_all()
        log.close()
