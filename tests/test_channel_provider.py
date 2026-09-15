"""用真实 Root/Context 检查普通 Channel provider 的资源与回执归属。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from agent.plugin_composition import (
    CompositionRoot, PluginRuntime, RUNTIME_STARTING, RuntimeStarting,
    SNAPSHOT_SEALING, SnapshotSealing,
)
from agent.plugin_composition.admission import SOURCE_ADMISSION, SourceAdmission
from agent.plugin_composition.channel_io import (
    INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
    ChannelIdentity, ChannelAttachmentImport, ChannelAttachmentRead,
    unavailable, unavailable_input_custody,
)
from agent.plugin_composition.channels import (
    CHANNELS, ChannelCapability, ChannelDefinition, ChannelInboundMessage,
    ChannelReady, ControlResponseBodies, DeliveryStatus, RawInbound, StopReceipt,
)
from agent.plugin_composition.context import RuntimeScope
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
from plugins.channels import plugin


@asynccontextmanager
async def provider_root(tmp_path, adapter_factory, *, interrupt=None):
    root = CompositionRoot("provider-test")
    store = RuntimeSnapshotStore()
    admission = SourceAdmission(root.context, store, boot_id="host-boot", candidate=False)
    await root.context.provide(SOURCE_ADMISSION, admission)
    await root.context.provide(INPUT_CUSTODY, unavailable_input_custody())
    await root.context.provide(CHANNEL_IDENTITY, ChannelIdentity(unavailable, unavailable, unavailable))
    await root.context.provide(CHANNEL_ATTACHMENT_IMPORT, ChannelAttachmentImport(unavailable))
    await root.context.provide(CHANNEL_ATTACHMENT_READ, ChannelAttachmentRead(unavailable, unavailable))

    def runtime(name):
        return PluginRuntime(name, name + "-generation", tmp_path, tmp_path, tmp_path, {})

    await root.mount(plugin.apply, name="channels", inject=plugin.inject, runtime=runtime("channels"))

    async def contribute(ctx):
        capabilities = {ChannelCapability.OUTBOUND}
        if interrupt is not None:
            capabilities.add(ChannelCapability.CONTROL)
        await ctx.require(CHANNELS).register(ctx, ChannelDefinition(
            "probe", frozenset(capabilities), adapter_factory, None, interrupt=interrupt,
        ))

    await root.mount(contribute, name="probe", inject=(CHANNELS,), runtime=runtime("probe"))
    await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    transaction = store.begin_publish(snapshot)
    lease = store.retain_publication_target(transaction)
    async with RuntimeScope(lease):
        await root.context.serial(RUNTIME_STARTING, RuntimeStarting())
    await store.commit(transaction, after_open=admission.open)
    try:
        yield root, store, root.context.require(CHANNELS)
    finally:
        store.pause_admission()
        admission.close()
        await root.dispose()
        await store.close()


@pytest.mark.asyncio
async def test_scope_stops_adapter_before_listener_and_retains_failed_owner(tmp_path):
    events = []
    finished = asyncio.Event()
    stop_entered = asyncio.Event()
    stop_release = asyncio.Event()
    fail = True

    class Adapter:
        def __init__(self, context):
            self.context = context

        async def start(self):
            async def listen():
                await finished.wait()
                events.append("listener-finished")
            self.task = await self.context.spawn_owned(listen(), name="probe-listener")
            return ChannelReady(self.context.binding_token)

        async def deliver(self, request):
            raise AssertionError("此测试不得发送")

        async def stop(self):
            events.append("stop")
            stop_entered.set()
            await stop_release.wait()
            if fail:
                raise OSError("connection still open")
            finished.set()
            await self.task
            return StopReceipt(self.context.binding_token, True)

    async with provider_root(tmp_path, Adapter) as (root, _, channels):
        key = next(iter(channels._bindings))
        owner = channels._bindings[key]
        first = asyncio.create_task(root.dispose())
        await stop_entered.wait()
        joined = asyncio.Event()

        async def close_again():
            joined.set()
            await root.dispose()

        second = asyncio.create_task(close_again())
        await joined.wait()
        stop_release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        assert all(isinstance(result, BaseExceptionGroup) for result in results)
        assert events == ["stop"]
        assert channels._bindings[key] is owner
        assert not owner.adapter.task.done()
        fail = False
        await root.dispose()
        assert events == ["stop", "stop", "listener-finished"]
        assert not channels._bindings


@pytest.mark.asyncio
async def test_control_ack_unknown_effect_is_not_replayed(tmp_path):
    calls = []
    contexts = []

    async def interrupt(raw):
        calls.append("pause:" + raw.message_id)
        return True

    class Adapter:
        def __init__(self, context):
            self.context = context
            contexts.append(context)

        def attach_presentation(self, ports):
            self.ports = ports

        async def start(self):
            return ChannelReady(self.context.binding_token)

        async def deliver(self, request):
            calls.append("send:" + request.body)
            raise OSError("provider outcome unknown")

        async def stop(self):
            return StopReceipt(self.context.binding_token, True)

    async with provider_root(tmp_path, Adapter, interrupt=interrupt):
        raw = RawInbound("stop-1", ChannelInboundMessage(
            channel="probe", sender="user", chat_id="room", content="/stop",
            timestamp=datetime(2026, 9, 15, tzinfo=UTC),
        ))
        control = contexts[0].control
        bodies = ControlResponseBodies(interrupted="stopped", idle="idle")
        receipt = await control.interrupt(raw, response_bodies=bodies)
        assert receipt.accepted
        assert receipt.response.status is DeliveryStatus.FAILED
        duplicate = await control.interrupt(raw, response_bodies=bodies)
        assert not duplicate.accepted
        assert calls == ["pause:stop-1", "send:stopped"]
