"""用真实 Root/Context 检查普通 Channel provider 的资源与回执归属。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from agent.plugin_composition import CompositionError, CompositionRoot, FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.channel_io import (
    INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
    ChannelIdentity, ChannelAttachmentImport, ChannelAttachmentRead,
    unavailable, unavailable_input_custody,
)
from agent.plugin_composition.channels import (
    CHANNELS, CHANNEL_INPUT, ChannelCapability, ChannelDefinition, ChannelInboundMessage,
    ChannelReady, ControlResponseBodies, DeliveryStatus, RawInbound, StopReceipt,
    InboundIdentity,
)
from agent.plugin_composition.channels import AttachmentKind
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from session.artifact_store import ArtifactStore
from plugins.channels import plugin

LOCAL_CHANNEL_SERVICE = ServiceKey[object]("test.local.channel.service")


@asynccontextmanager
async def provider_root(tmp_path, adapter_factory, *, interrupt=None, opened=None, attachments=None):
    root = CompositionRoot("provider-test")
    await root.context.provide(HOST_INFO, HostInfo("host-boot", False))
    await root.context.provide(INPUT_CUSTODY, unavailable_input_custody())
    await root.context.provide(CHANNEL_IDENTITY, ChannelIdentity(unavailable, unavailable, unavailable))
    if attachments is None:
        attachment_import = ChannelAttachmentImport(unavailable)
        attachment_read = ChannelAttachmentRead(unavailable, unavailable)
    else:
        attachment_import = ChannelAttachmentImport(attachments.import_bytes)
        attachment_read = ChannelAttachmentRead(attachments.resolve_refs, attachments.acquire)
    await root.context.provide(CHANNEL_ATTACHMENT_IMPORT, attachment_import)
    await root.context.provide(CHANNEL_ATTACHMENT_READ, attachment_read)
    async def reject_input(*args, **kwargs):
        raise AssertionError("provider test must not accept input")
    await root.context.provide(CHANNEL_INPUT, reject_input)

    def runtime(name):
        return PluginRuntime(name, name + "-generation", tmp_path, tmp_path, tmp_path, {})

    await root.mount(plugin.apply, name="channels", inject=plugin.inject, runtime=runtime("channels"))

    async def contribute(ctx):
        capabilities = {ChannelCapability.OUTBOUND}
        if opened is not None:
            capabilities.add(ChannelCapability.INBOUND)
        if interrupt is not None:
            capabilities.add(ChannelCapability.CONTROL)
        def factory(context):
            adapter = adapter_factory(context)
            if opened is not None:
                adapter.opened = opened
            return adapter
        await ctx.require(CHANNELS).register(ctx, ChannelDefinition(
            "probe", frozenset(capabilities), factory,
            InboundIdentity.PROVIDER_MESSAGE_ID if opened is not None else None,
            interrupt=interrupt,
        ))

    inject = (CHANNELS, CHANNEL_INPUT) if opened is not None else (CHANNELS,)
    await root.mount(contribute, name="probe", inject=inject, runtime=runtime("probe"))
    try:
        if opened is not None:
            await asyncio.wait_for(opened.wait(), 5)
        yield root, root.context.require(CHANNELS)
    finally:
        await root.dispose()


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

    async with provider_root(tmp_path, Adapter) as (root, channels):
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

        def attach_runtime(self, ports):
            self.runtime_ports = ports

        def open_admission(self):
            self.opened.set()

        def close_admission(self):
            pass

        async def stop(self):
            return StopReceipt(self.context.binding_token, True)

    opened = asyncio.Event()
    async with provider_root(tmp_path, Adapter, interrupt=interrupt, opened=opened):
        await asyncio.wait_for(opened.wait(), 5)
        raw = RawInbound("stop-1", ChannelInboundMessage(
            channel="probe", sender="user", chat_id="room", content="/stop",
            timestamp=datetime(2026, 9, 15, tzinfo=UTC),
            metadata={},
        ))
        control = contexts[0].control
        bodies = ControlResponseBodies(interrupted="stopped", idle="idle")
        receipt = await control.interrupt(raw, response_bodies=bodies)
        assert receipt.accepted
        assert receipt.response.status is DeliveryStatus.FAILED
        duplicate = await control.interrupt(raw, response_bodies=bodies)
        assert not duplicate.accepted
        assert calls == ["pause:stop-1", "send:stopped"]


@pytest.mark.asyncio
async def test_local_channel_replacement_keeps_unrelated_binding_and_hard_consumer(tmp_path):
    """同一 Root 只替换目标 contribution，旧 Context 拒绝新请求且新 token 改变。"""

    old_opened, new_opened = asyncio.Event(), asyncio.Event()
    old_closed, new_closed = asyncio.Event(), asyncio.Event()
    hard_consumer_exit = asyncio.Event()
    release_hard_consumer = asyncio.Event()
    open_checks = []
    consumer_seen = []

    class Adapter:
        def __init__(self, context, contribution_context, opened, closed):
            self.context = context
            self.contribution_context = contribution_context
            self.opened = opened
            self.closed = closed

        async def start(self):
            return ChannelReady(self.context.binding_token)

        def attach_runtime(self, ports):
            self.ports = ports

        def open_admission(self):
            contribution_fiber = self.contribution_context._fiber
            assert contribution_fiber.state is FiberState.ACTIVE
            required = [
                entry for (fiber_id, _), entry in self.contribution_context._root._health_entries.items()
                if fiber_id == contribution_fiber.fiber_id and entry.required
            ]
            assert required and all(entry.active and entry.reason is None for entry in required)
            open_checks.append(self.context.binding_token)
            self.opened.set()

        def close_admission(self):
            pass

        async def deliver(self, request):
            raise AssertionError("local replacement must not send")

        async def stop(self):
            self.closed.set()
            return StopReceipt(self.context.binding_token, True)

    adapters = []

    class FixtureAdapter:
        def __init__(self, context):
            self.context = context
            self.starts = 0
            self.stops = 0

        async def start(self):
            self.starts += 1
            return ChannelReady(self.context.binding_token)

        async def deliver(self, request):
            raise AssertionError("unrelated binding must not send")

        async def stop(self):
            self.stops += 1
            return StopReceipt(self.context.binding_token, True)

    async with provider_root(tmp_path, FixtureAdapter) as (root, channels):
        # Replace the fixture's no-op probe with two real contribution Fibers.
        async def contribution(ctx, opened, closed, marker):
            await ctx.health("adapter")
            await ctx.provide(LOCAL_CHANNEL_SERVICE, marker)

            def factory(context):
                adapter = Adapter(context, ctx, opened, closed)
                adapters.append(adapter)
                return adapter

            await ctx.require(CHANNELS).register(
                ctx,
                ChannelDefinition(
                    "local-replace", frozenset({ChannelCapability.INBOUND}), factory,
                    InboundIdentity.PROVIDER_MESSAGE_ID,
                ),
            )

        old = await root.mount(
            lambda ctx: contribution(ctx, old_opened, old_closed, "old"),
            name="local-old",
            inject=(CHANNELS, CHANNEL_INPUT),
            runtime=PluginRuntime("local-old", "local-old", tmp_path, tmp_path, tmp_path, {}),
        )

        async def hard_consumer(ctx):
            consumer_seen.append(ctx.require(LOCAL_CHANNEL_SERVICE))

            async def cleanup():
                hard_consumer_exit.set()
                await release_hard_consumer.wait()

            await ctx.effect(lambda: cleanup, label="local-hard-consumer")

        hard = await root.mount(
            hard_consumer,
            name="local-hard-consumer",
            inject=(LOCAL_CHANNEL_SERVICE,),
            runtime=PluginRuntime("local-hard-consumer", "local-hard-consumer", tmp_path, tmp_path, tmp_path, {}),
        )
        await old_opened.wait()
        old_state = next(state for state in channels._bindings.values() if state.channel_name == "local-replace")
        old_context = old_state.plugin_context
        assert old_context is not None and old_state.factory_context is not None
        old_token = old_state.binding_token
        unrelated = next(state for state in channels._bindings.values() if state.channel_name == "probe")
        unrelated_token = unrelated.binding_token
        unrelated_context = unrelated.plugin_context
        unrelated_activation = None if unrelated_context is None else unrelated_context._fiber.activation_token
        unrelated_adapter = unrelated.adapter
        unrelated_starts = unrelated_adapter.starts
        unrelated_stops = unrelated_adapter.stops

        dispose = asyncio.create_task(old.dispose())
        await hard_consumer_exit.wait()
        assert old.state == FiberState.UNLOADING
        with pytest.raises(CompositionError) as error:
            async with old_state.factory_context.open_scope():
                raise AssertionError("old binding accepted a new request")
        assert error.value.code == "OWNER_UNAVAILABLE"
        assert unrelated.factory_context is not None
        async with unrelated.factory_context.open_scope() as unrelated_scope:
            assert unrelated_scope.require(CHANNELS) is not None
        assert unrelated_adapter.starts == unrelated_starts
        assert unrelated_adapter.stops == unrelated_stops
        release_hard_consumer.set()
        await dispose
        assert old_closed.is_set()

        fresh = await root.mount(
            lambda ctx: contribution(ctx, new_opened, new_closed, "new"),
            name="local-new",
            inject=(CHANNELS, CHANNEL_INPUT),
            runtime=PluginRuntime("local-new", "local-new", tmp_path, tmp_path, tmp_path, {}),
        )
        await new_opened.wait()
        new_state = next(state for state in channels._bindings.values() if state.channel_name == "local-replace")
        assert fresh.state == FiberState.ACTIVE
        assert len(adapters) == 2
        assert new_state.binding_token != old_token
        assert new_state.plugin_context is not old_context
        assert len(open_checks) == 2
        assert hard is not None
        assert hard.state == FiberState.ACTIVE
        assert consumer_seen == ["old", "new"]
        assert hard.context.require(LOCAL_CHANNEL_SERVICE) == "new"
        unchanged = next(state for state in channels._bindings.values() if state.channel_name == "probe")
        assert unchanged.binding_token == unrelated_token
        assert unchanged.plugin_context is unrelated_context
        assert unchanged.plugin_context is not None
        assert unchanged.plugin_context._fiber.activation_token is unrelated_activation
        assert unchanged.plugin_context._fiber.state == FiberState.ACTIVE
        assert root.context.require(HOST_INFO).boot_id == "host-boot"
    assert new_closed.is_set()


@pytest.mark.asyncio
async def test_channel_operation_task_creation_and_unentered_cancel_close_scopes(tmp_path):
    """真实 Task 工厂失败与首指令取消都释放 child scope，不影响无关 Fiber。"""

    from plugins.channels import provider as provider_module

    class Adapter:
        def __init__(self, context):
            self.context = context

        def attach_runtime(self, ports):
            self.ports = ports

        def open_admission(self):
            self.opened.set()

        def close_admission(self):
            pass

        async def start(self):
            return ChannelReady(self.context.binding_token)

        async def deliver(self, request):
            raise AssertionError("task factory test must not send")

        async def stop(self):
            return StopReceipt(self.context.binding_token, True)

    opened = asyncio.Event()
    calls = []

    async def interrupt(raw):
        calls.append(raw.message_id)
        return True

    async with provider_root(tmp_path, Adapter, interrupt=interrupt, opened=opened) as (root, channels):
        state = next(iter(channels._bindings.values()))
        assert state.factory_context is not None and state.factory_context.control is not None
        assert state.plugin_context is not None
        contributor_fiber = state.plugin_context._fiber
        control = state.factory_context.control
        bodies = ControlResponseBodies(interrupted="stopped", idle="idle")
        first = RawInbound("task-factory-failure", ChannelInboundMessage(
            channel="probe", sender="user", chat_id="room", content="/stop",
            timestamp=datetime(2026, 9, 15, tzinfo=UTC),
        ))
        second = RawInbound("task-cancelled-before-entry", ChannelInboundMessage(
            channel="probe", sender="user", chat_id="room", content="/stop",
            timestamp=datetime(2026, 9, 15, tzinfo=UTC),
        ))
        captured = []
        original_create_task = provider_module.asyncio.create_task
        try:
            def reject_control_task(coroutine, *, name=None, context=None):
                if name and name.startswith("channel-control:"):
                    captured.append(coroutine)
                    raise RuntimeError("synchronous create_task failure")
                kwargs = {} if name is None else {"name": name}
                if context is not None:
                    kwargs["context"] = context
                return original_create_task(coroutine, **kwargs)

            provider_module.asyncio.create_task = reject_control_task
            with pytest.raises(RuntimeError, match="synchronous create_task failure"):
                await control.interrupt(first, response_bodies=bodies)
            assert captured and captured[0].cr_frame is None
            assert calls == []
            assert not contributor_fiber._in_flight_calls

            def cancel_control_task(coroutine, *, name=None, context=None):
                if name and name.startswith("channel-control:"):
                    task = original_create_task(coroutine, name=name)
                    task.cancel()
                    return task
                kwargs = {} if name is None else {"name": name}
                if context is not None:
                    kwargs["context"] = context
                return original_create_task(coroutine, **kwargs)

            provider_module.asyncio.create_task = cancel_control_task
            with pytest.raises(asyncio.CancelledError):
                await control.interrupt(second, response_bodies=bodies)
            assert calls == []
            assert not contributor_fiber._in_flight_calls
        finally:
            provider_module.asyncio.create_task = original_create_task

        async def unrelated(ctx):
            await ctx.provide(LOCAL_CHANNEL_SERVICE, "unrelated")

        unrelated_fiber = await root.mount(unrelated, name="unrelated")
        async with unrelated_fiber.context.runtime_scope():
            assert unrelated_fiber.context.require(LOCAL_CHANNEL_SERVICE) == "unrelated"


@pytest.mark.asyncio
async def test_attachment_read_lease_drains_on_local_channel_stop_and_retries(tmp_path, monkeypatch):
    """目标 binding 卸载等待真实 read lease，停止失败保留同一 lease 供 retry。"""

    metadata = ArtifactStore(tmp_path / "artifact-metadata.db")
    artifact_root = tmp_path / "attachments"
    artifact_root.mkdir()
    attachments = ChannelAttachmentArtifactStore(
        workspace=artifact_root, metadata_store=metadata,
    )
    ref = await attachments.import_bytes(
        b"held-read", kind=AttachmentKind.FILE,
        filename="held.txt", media_type="text/plain",
    )
    target_opened = asyncio.Event()
    target_closed = asyncio.Event()
    hard_consumer_exit = asyncio.Event()
    release_hard_consumer = asyncio.Event()

    class Adapter:
        def __init__(self, context):
            self.context = context

        def attach_runtime(self, ports):
            self.ports = ports

        async def start(self):
            return ChannelReady(self.context.binding_token)

        def open_admission(self):
            pass

        def close_admission(self):
            pass

        async def deliver(self, request):
            raise AssertionError("attachment lease test must not send")

        async def stop(self):
            return StopReceipt(self.context.binding_token, True)

    class TargetAdapter(Adapter):
        def open_admission(self):
            target_opened.set()

        def close_admission(self):
            target_closed.set()

    async def hard_cleanup():
        hard_consumer_exit.set()
        await release_hard_consumer.wait()

    async with provider_root(
        tmp_path, Adapter, attachments=attachments,
    ) as (root, channels):
        target_fiber = None
        hard = None
        lease = None
        dispose = None
        real_lease = None
        real_aclose = None
        try:
            async def target(ctx):
                await ctx.provide(LOCAL_CHANNEL_SERVICE, "attachment-target")

                def factory(context):
                    return TargetAdapter(context)

                await ctx.require(CHANNELS).register(
                    ctx,
                    ChannelDefinition(
                        "attachment-target", frozenset({ChannelCapability.INBOUND}), factory,
                        InboundIdentity.PROVIDER_MESSAGE_ID,
                    ),
                )

            target_fiber = await root.mount(
                target,
                name="attachment-target",
                inject=(CHANNELS, CHANNEL_INPUT),
                runtime=PluginRuntime("attachment-target", "attachment-target", tmp_path, tmp_path, tmp_path, {}),
            )
            hard = await root.mount(
                lambda ctx: ctx.effect(
                    lambda: hard_cleanup, label="attachment-hard-consumer",
                ),
                name="attachment-hard-consumer",
                inject=(LOCAL_CHANNEL_SERVICE,),
                runtime=PluginRuntime("attachment-hard-consumer", "attachment-hard-consumer", tmp_path, tmp_path, tmp_path, {}),
            )
            await target_opened.wait()
            target_state = next(
                state for state in channels._bindings.values()
                if state.channel_name == "attachment-target"
            )
            assert target_state.factory_context is not None
            assert target_state.factory_context.attachment_read is not None
            lease = await target_state.factory_context.attachment_read.acquire(ref)
            assert await lease.read_bytes(max_bytes=64) == b"held-read"
            assert await lease.read_chunk(offset=0, max_bytes=4) == b"held"
            dispose = asyncio.create_task(target_fiber.dispose())
            await hard_consumer_exit.wait()
            assert target_fiber.state == FiberState.UNLOADING
            assert not dispose.done()
            assert await lease.read_bytes(max_bytes=64) == b"held-read"
            release_hard_consumer.set()
            await target_closed.wait()

            real_lease = lease._lease
            real_aclose = type(real_lease).aclose
            failed_once = True

            async def fail_underlying_close(instance):
                nonlocal failed_once
                if failed_once:
                    failed_once = False
                    raise OSError("read lease close failed")
                await real_aclose(instance)

            monkeypatch.setattr(type(real_lease), "aclose", fail_underlying_close)
            with pytest.raises(OSError, match="read lease close failed"):
                await lease.aclose()
            assert await lease.read_bytes(max_bytes=64) == b"held-read"
            monkeypatch.setattr(type(real_lease), "aclose", real_aclose)
            await lease.aclose()
            await dispose
            assert hard.state is not FiberState.FAILED
        finally:
            release_hard_consumer.set()
            if real_lease is not None and real_aclose is not None:
                monkeypatch.setattr(type(real_lease), "aclose", real_aclose)
            if lease is not None:
                await asyncio.gather(lease.aclose(), return_exceptions=True)
            if dispose is not None:
                await asyncio.gather(dispose, return_exceptions=True)
    metadata.close()
