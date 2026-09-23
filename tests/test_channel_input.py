import asyncio
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
import shutil

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.channels import (
    ChannelCapability, ChannelInboundMessage, ChannelReady,
    InboundIdentity, RawInbound, StopReceipt,
)
from agent.plugins.manager import PluginManager
from agent.plugin_composition import CompositionError, CompositionRoot, FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition.channels import CHANNELS
from agent.plugin_composition.channel_io import InputCustody
from bus.event_bus import EventBus
from bus.queue import MessageBus
from session.identities import ChannelIdentities
from session.log import MessageLog
from session.message import Input, Message
from plugins.sources import plugin as sources_plugin


class Adapter:
    def __init__(self, context):
        self.context = context
        self.opened = asyncio.Event()

    def attach_runtime(self, ports):
        self.ports = ports

    def open_admission(self):
        self.opened.set()

    def close_admission(self):
        pass

    async def start(self):
        return ChannelReady(self.context.binding_token)

    async def deliver(self, request):
        raise AssertionError("input acceptance must not send a reply")

    async def stop(self):
        return StopReceipt(self.context.binding_token, True)


class Custody(MessageBus):
    def __init__(self):
        super().__init__()
        self.prepared = asyncio.Event()
        self.prepare_gate = asyncio.Event()
        self.prepare_gate.set()
        self.reserved = asyncio.Event()
        self.reserve_gate = asyncio.Event()
        self.reserve_gate.set()
        self.committed = asyncio.Event()
        self.complete_gate = asyncio.Event()
        self.complete_gate.set()
        self.durable_reserved = asyncio.Event()
        self.durable_reserve_gate = asyncio.Event()
        self.durable_reserve_gate.set()
        self.settlement_started = asyncio.Event()
        self.settlement_gate = asyncio.Event()
        self.settlement_gate.set()
        self.fail_settlement_once = False
        self.completed = 0
        self.envelopes = []
        self.reject = False

    async def reserve_durable_inbound(self, raw):
        accepted = await super().reserve_durable_inbound(raw)
        self.durable_reserved.set()
        await self.durable_reserve_gate.wait()
        return accepted

    async def defer_durable_inbound(self, handoff_id):
        self.settlement_started.set()
        await self.settlement_gate.wait()
        if self.fail_settlement_once:
            self.fail_settlement_once = False
            raise OSError("injected durable settlement failure")
        return await super().defer_durable_inbound(handoff_id)

    async def settle_rejected_inbound(self, **kwargs):
        self.settlement_started.set()
        await self.settlement_gate.wait()
        if self.fail_settlement_once:
            self.fail_settlement_once = False
            raise OSError("injected durable settlement failure")
        await super().settle_rejected_inbound(**kwargs)

    async def prepare_channel_input(self, envelope):
        self.envelopes.append(envelope)
        self.prepared.set()
        await self.prepare_gate.wait()
        if self.reject:
            raise ValueError("rejected before commit")
        await super().prepare_channel_input(envelope)
        self.reserved.set()
        await self.reserve_gate.wait()

    async def complete_channel_input(self, envelope):
        self.committed.set()
        await self.complete_gate.wait()
        await super().complete_channel_input(envelope)
        self.completed += 1
        self.finished.set()


@asynccontextmanager
async def runtime(tmp_path, *, channel_name="probe", session_manager=None, recover=True, artifacts=None, inbound_store=None, admissions=None, durable_identities=False):
    sources = tmp_path / "plugins"
    for name in ("commands", "ui", "content", "models", "conversation", "sources", "channels"):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
            dirs_exist_ok=True,
        )
    log = MessageLog(tmp_path / "sessions.db")
    identity_store = ChannelIdentities(tmp_path / "sessions.db")
    custody = Custody()
    if session_manager is not None:
        inbound_store, admissions = session_manager.inbound_store, session_manager.admissions
    if inbound_store is not None:
        assert admissions is not None
        custody.bind_durable_inbound_store(inbound_store)
        custody.bind_session_admission_owner(admissions)
    identities, rollbacks, adapters = {}, [], []
    async def remember(channel, provider, recipient):
        identities[(channel, provider)] = recipient
        return channel, provider
    async def rollback(key):
        rollbacks.append(key)
        del identities[key]
        return True
    probe = sources / "probe_channel"
    probe.mkdir(exist_ok=True)
    probe_source = """from agent.plugin_composition import CHANNELS, ChannelDefinition, ChannelCapability, InboundIdentity
from agent.plugin_composition.channels import CHANNEL_INPUT
from tests.test_channel_input import Adapter
api_version = 3
name = 'probe_channel'
version = '1.0.0'
inject = (CHANNELS, CHANNEL_INPUT)
async def apply(ctx):
    await ctx.require(CHANNELS).register(ctx, ChannelDefinition(
        name=CHANNEL_NAME, capabilities=frozenset(CAPABILITIES),
        factory=Adapter, inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID))
"""
    capabilities = "[ChannelCapability.INBOUND]"
    if inbound_store is not None:
        capabilities = "[ChannelCapability.INBOUND, ChannelCapability.DURABLE_INBOUND]"
    (probe / "plugin.py").write_text(probe_source.replace("CHANNEL_NAME", repr(channel_name)).replace("CAPABILITIES", capabilities))
    # 插件在提交开放时自启一轮恢复；先挡住它，由本 fixture 的同步 recover 完成
    # 初次结算，避免断言与异步 claim 竞争。放行后 auto 恢复路径仍归插件。
    spawned_recovery_gate = asyncio.Event()

    async def gated_recover_durable_inbounds():
        await spawned_recovery_gate.wait()
        await custody.recover_durable_inbounds()

    host = PluginManager(
        [sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home", message_log=log,
        channel_attachment_store=artifacts, channel_identities=identity_store,
        input_custody=InputCustody(
            custody.prepare_channel_input, custody.complete_channel_input, custody.retain_channel_input,
            lambda raw: custody.reserve_durable_inbound(raw), custody.defer_durable_inbound,
            custody.settle_rejected_inbound, custody.has_pending_durable_inbound,
            custody.pending_durable_attachment_refs, gated_recover_durable_inbounds,
        ),
    )

    async def recover_input(raw):
        root = host.live_root
        if root is None:
            return False
        return await root.context.require(CHANNELS).recover_inbound(raw)

    custody.bind_durable_inbound_recoverer(recover_input)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        generation = host.generation("probe_channel")
        assert generation is not None
        channel = root.context.require(CHANNELS)
        if not durable_identities:
            channel._identity_rememberer = remember
            channel._identity_rollbacker = rollback
            channel._identity_resolver = lambda channel, provider: identities.get((channel, provider))
        key = next(
            key for key, state in channel._bindings.items()
            if state.generation_id == generation.generation_id and state.channel_name == channel_name
        )
        adapter = channel._bindings[key].adapter
        adapters.append(adapter)
        await asyncio.wait_for(adapter.opened.wait(), 5)
        if inbound_store is not None and recover:
            await custody.recover_durable_inbounds()
        spawned_recovery_gate.set()
        yield log, host, custody, identities, rollbacks, adapters[-1]
    finally:
        custody.prepare_gate.set()
        custody.reserve_gate.set()
        custody.complete_gate.set()
        await host.terminate_all()
        await custody.aclose()
        log.close()
        if identity_store is not None:
            identity_store.close()


def _source_context(host):
    """Return the selected sources Fiber that owns the live input service."""

    generation = host.generation("sources")
    if generation is None or generation.fiber is None:
        raise AssertionError("sources generation 未建立")
    return generation.fiber.context


def _generation_context(host, plugin_id):
    """Return one selected live generation Context for a declared service call."""

    generation = host.generation(plugin_id)
    if generation is None or generation.fiber is None:
        raise AssertionError(f"{plugin_id} generation 未建立")
    return generation.fiber.context


async def _registered_source(host, name):
    """Read one live source registration through its exact provider Context."""
    context = _source_context(host)
    async with context.runtime_scope():
        matches = tuple(item for item in context.require(sources_plugin.SOURCES).entries()
                        if item.name == name)
    assert len(matches) == 1
    return matches[0]


def channels(host):
    root = host.live_root
    if root is None:
        raise AssertionError("正式 live Root 尚未建立")
    return root.context.require(CHANNELS)


def binding_key(host, channel_name):
    """Find the live binding owned by the current generation."""
    generation = host.generation("probe_channel")
    if generation is None:
        raise AssertionError("当前 generation 不存在")
    channel = channels(host)
    return next(
        key for key, state in channel._bindings.items()
        if state.generation_id == generation.generation_id
        and state.channel_name == channel_name
    )


def raw():
    return RawInbound("u1", ChannelInboundMessage(
        channel="probe", chat_id="room", sender="user", content="hello",
        timestamp=datetime(2026, 9, 5, tzinfo=UTC), metadata={"session_key": "not-authority"},
    ), provider_identity="provider-user", recipient="room")


@pytest.mark.asyncio
async def test_durable_port_rejects_cross_channel_reservation(tmp_path):
    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("probe:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="probe",
            session_manager=manager,
        ) as (_, _, _, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            forged = RawInbound(
                "foreign-1",
                ChannelInboundMessage(
                    channel="other",
                    chat_id="room",
                    sender="foreign",
                    content="hello",
                    timestamp=datetime(2026, 9, 5, tzinfo=UTC),
                    metadata={
                        "session_key_override": "probe:room",
                        "provider_message_id": "foreign-1",
                        "durable_inbound": True,
                        "durable_handoff_id": "foreign-handoff",
                    },
                ),
            )
            with pytest.raises(RuntimeError, match="channel 与 exact binding"):
                await durable.reserve(forged)
            assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_durable_prepare_requires_prior_port_reservation(tmp_path):
    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
        ) as (_, _, _, _, _, adapter):
            with pytest.raises(RuntimeError, match="reservation 未绑定"):
                await adapter.context.ingress.admit(mobile_raw())
            assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_durable_recovery_waits_until_current_root_reopens(tmp_path):
    """暂停来源时保留 pending；不绕过关闭许可，重开后只恢复一次。"""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(mobile_raw())
            assert await durable.defer("handoff-1") is None

            channel = channels(host)
            key = binding_key(host, "akashic")
            await channel._stop_binding(key)
            await custody.recover_durable_inbounds()

            assert custody.completed == 0
            assert manager.inbound_store.list_inbound_handoffs()
            await host.retry_runtime_recovery("probe_channel")
            await custody.recover_durable_inbounds()
            assert custody.completed == 1
            assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_exact_root_input_commits_without_queue_model_or_delivery(tmp_path):
    initialize_plugin_workspace(tmp_path / "workspace")
    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        sources_context = _source_context(host)
        async with sources_context.runtime_scope():
            sources = sources_context.require(ServiceKey("sources.v2"))
            assert any(source.name == "conversation" for source in sources.entries())
        assert await adapter.context.ingress.admit(raw()) is True
        assert await adapter.context.ingress.admit(raw()) is False
        messages = log.reader("probe:room").snapshot()
        assert len(messages) == 1 and isinstance(messages[0].body, Input)
        assert messages[0].source == "conversation"
        assert messages[0].message_id == "u1"
        assert not log.reader("not-authority").snapshot()
        assert identities == {("probe", "provider-user"): "room"}
        assert rollbacks == [] and custody.completed == 1
        assert custody.inbound_size == 0
        assert not custody.envelopes[0].lease.active
        assert all(state.in_flight == 0 for state in channels(host)._bindings.values())


@pytest.mark.asyncio
async def test_sources_callback_scope_drains_without_fallback_or_unrelated_stop(tmp_path):
    from plugins.sources.plugin import CHANNEL_INPUT as SOURCE_INPUT, SOURCES

    source_marker = ServiceKey("test.selected-source")
    unrelated_marker = ServiceKey("test.unrelated-source")
    root = CompositionRoot("sources-scope-test")
    source_fiber = await root.mount(
        sources_plugin.apply,
        name="sources",
        inject=(),
        runtime=PluginRuntime("sources", "sources", tmp_path, tmp_path, tmp_path, {}),
    )
    entered = asyncio.Event()
    callback_scope_checked = asyncio.Event()
    release = asyncio.Event()
    hard_cleanup_started = asyncio.Event()
    release_hard_cleanup = asyncio.Event()
    cleanup_calls = 0

    async def contributor(ctx):
        await ctx.provide(source_marker, "selected")

        async def accept(session_id, message_id, message):
            entered.set()
            await release.wait()
            assert ctx.require(source_marker) == "selected"
            callback_scope_checked.set()
            return Message(
                message_id, session_id, 0, datetime.now(UTC), "user", "selected", Input(()),
            )

        await ctx.require(SOURCES).register(
            ctx,
            name="selected",
            open=lambda session_id: None,
            needs_reply=lambda reader: False,
            accept=accept,
            channels=("probe",),
        )

    selected = await root.mount(
        contributor,
        name="selected-source",
        inject=(SOURCES,),
        runtime=PluginRuntime("selected-source", "selected-source", tmp_path, tmp_path, tmp_path, {}),
    )

    async def hard_consumer(ctx):
        nonlocal cleanup_calls
        assert ctx.require(source_marker) == "selected"

        async def cleanup():
            nonlocal cleanup_calls
            cleanup_calls += 1
            hard_cleanup_started.set()
            await release_hard_cleanup.wait()

        await ctx.effect(lambda: cleanup, label="selected-hard-consumer")

    hard = await root.mount(
        hard_consumer,
        name="selected-hard-consumer",
        inject=(source_marker,),
        runtime=PluginRuntime("selected-hard-consumer", "selected-hard-consumer", tmp_path, tmp_path, tmp_path, {}),
    )

    async def unrelated(ctx):
        await ctx.provide(unrelated_marker, "unrelated")

    unrelated_fiber = await root.mount(
        unrelated,
        name="unrelated-source",
        runtime=PluginRuntime("unrelated-source", "unrelated-source", tmp_path, tmp_path, tmp_path, {}),
    )
    callback = None
    disposing = None
    try:
        accept = root.context.require(SOURCE_INPUT)
        callback = asyncio.create_task(accept(
            "probe:room", "selected-1", ChannelInboundMessage(
                "probe", "user", "room", "hello", datetime.now(UTC), {},
            ),
        ))
        await asyncio.wait_for(entered.wait(), 2)
        disposing = asyncio.create_task(selected.dispose())
        await hard_cleanup_started.wait()
        assert selected.state is FiberState.UNLOADING
        with pytest.raises(CompositionError) as error:
            await accept(
                "probe:room", "selected-2", ChannelInboundMessage(
                    "probe", "user", "room", "new", datetime.now(UTC), {},
                ),
            )
        assert error.value.code == "OWNER_UNAVAILABLE"
        async with unrelated_fiber.context.runtime_scope():
            assert unrelated_fiber.context.require(unrelated_marker) == "unrelated"
        assert not disposing.done()
        release.set()
        await callback_scope_checked.wait()
        result = await callback
        assert result.message_id == "selected-1"
        release_hard_cleanup.set()
        await disposing
        assert selected.state is FiberState.DISPOSED
        assert hard.state is FiberState.PENDING
        assert hard.dependency_store == {}
        assert cleanup_calls == 1
    finally:
        release.set()
        release_hard_cleanup.set()
        if callback is not None:
            await asyncio.gather(callback, return_exceptions=True)
        if disposing is not None:
            await asyncio.gather(disposing, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_conversation_source_needs_its_own_scope(tmp_path):
    initialize_plugin_workspace(tmp_path / "workspace")
    async with runtime(tmp_path) as (_, host, _, _, _, _):
        source = await _registered_source(host, "conversation")
        with pytest.raises(CompositionError) as error:
            source.open("probe:room")
        assert error.value.code == "OWNER_CALL_CONTEXT"
        async with source.context.runtime_scope():
            assert source.open("probe:room") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "committed"])
async def test_input_cancellation_respects_commit_boundary_and_closes_exact_lease(tmp_path, stage):
    initialize_plugin_workspace(tmp_path / "workspace")
    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        gate = custody.prepare_gate if stage == "prepare" else custody.complete_gate
        signal = custody.prepared if stage == "prepare" else custody.committed
        gate.clear()
        submit = asyncio.create_task(adapter.context.ingress.admit(raw()))
        try:
            await asyncio.wait_for(signal.wait(), 2)
            submit.cancel()
            if stage == "committed":
                assert len(log.reader("probe:room").snapshot()) == 1
                assert not submit.done()
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(submit, 2)
            assert not custody.envelopes[0].lease.active
            if stage == "prepare":
                assert not log.reader("probe:room").snapshot()
                assert identities == {} and rollbacks == [("probe", "provider-user")]
                assert await adapter.context.ingress.admit(raw()) is True
            else:
                assert identities == {("probe", "provider-user"): "room"}
                assert rollbacks == [] and custody.completed == 1
                assert await adapter.context.ingress.admit(raw()) is False
        finally:
            gate.set()
            await asyncio.gather(submit, return_exceptions=True)


@pytest.mark.asyncio
async def test_mobile_delete_retry_does_not_turn_committed_input_into_failed_acceptance(tmp_path, monkeypatch):
    from session.manager import SessionManager
    from session.store import SessionAdmissionConflictError
    import bus.queue as queue

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    store = manager.inbound_store
    complete = store.complete_inbound_handoff
    retry_started, retry_release = asyncio.Event(), asyncio.Event()
    calls = 0
    def fail_once(handoff_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected handoff delete failure")
        complete(handoff_id)
    monkeypatch.setattr(store, "complete_inbound_handoff", fail_once)
    original_retry = queue.MessageBus._retry_inbound_cleanup
    async def gated_retry(bus, key):
        retry_started.set()
        await retry_release.wait()
        await original_retry(bus, key)
    monkeypatch.setattr(queue.MessageBus, "_retry_inbound_cleanup", gated_retry)
    monkeypatch.setattr(queue, "_INBOUND_CLEANUP_RETRY_INITIAL_DELAY", 0)
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, identities, rollbacks, adapter):
            message = RawInbound("mobile-1", ChannelInboundMessage(
                channel="akashic", chat_id="room", sender="device:one", content="hello",
                timestamp=datetime(2026, 9, 5, tzinfo=UTC), metadata={
                    "session_key_override": "akashic:room", "provider_message_id": "mobile-1",
                    "durable_inbound": True, "durable_handoff_id": "handoff-1",
                },
            ), provider_identity="room", recipient="room")
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(message)
            assert await adapter.context.ingress.admit(message) is True
            await asyncio.wait_for(retry_started.wait(), 2)
            assert len(log.reader("akashic:room").snapshot()) == 1
            assert custody.durable_inbound_cleanup_pending(custody.envelopes[0])
            assert store.has_inbound_handoff(channel="akashic", session_key="akashic:room", provider_message_id="mobile-1")
            with pytest.raises(SessionAdmissionConflictError):
                manager.delete_session_with_audit("akashic:room")
            jobs = tuple(custody._inbound_cleanup_tasks.values())
            retry_release.set()
            await asyncio.wait_for(asyncio.gather(*jobs), 2)
            assert store.list_inbound_handoffs() == []
            assert not custody.envelopes[0].lease.active
            assert custody._durable_admissions == {}
            assert calls == 2 and rollbacks == []
            assert custody.inbound_size == 0
    finally:
        retry_release.set()
        manager.close()


@pytest.mark.asyncio
async def test_retry_uses_message_identity_without_copying_transport_clock_or_handoff(tmp_path):
    from agent.plugin_composition.channels import CHANNEL_INPUT
    initialize_plugin_workspace(tmp_path / "workspace")
    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        first = raw().message
        second = replace(first, timestamp=first.timestamp + timedelta(hours=1),
                         metadata={"durable_handoff_id": "another", "client_request_id": "retry"})
        context = _source_context(host)
        async with context.runtime_scope():
            accept = context.require(CHANNEL_INPUT)
            original = await accept("probe:room", "fixed", first)
            assert await accept("probe:room", "fixed", second) == original
        assert len(log.reader("probe:room").snapshot()) == 1
        assert original.body.parts[0].value == {"channel": "probe", "chat_id": "room", "sender": "user"}


@pytest.mark.asyncio
async def test_stop_commits_pause_before_draining_and_duplicate_does_not_pause_new_work(tmp_path):
    from agent.plugin_composition.channels import CHANNEL_INPUT
    from plugins.sources.session import needs_reply
    from session.message import Control

    initialize_plugin_workspace(tmp_path / "workspace")
    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        started, cancelled, drain = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def program(task, reader, source):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
                await drain.wait()
        source_context = _source_context(host)
        source = await _registered_source(host, "conversation")
        conversation_context = source.context
        async with source_context.runtime_scope(), conversation_context.runtime_scope():
            accept = source_context.require(CHANNEL_INPUT)
            await accept("probe:room", "first", raw().message)
            conversation = source.open("probe:room")
            task = await conversation.start(program)
            await asyncio.wait_for(started.wait(), 2)
            async def pause():
                async with source_context.runtime_scope():
                    return await source_context.require(CHANNEL_INPUT)(
                        "probe:room", "stop", replace(raw().message, content="/stop"))
            stop = asyncio.create_task(pause())
            try:
                signal = asyncio.create_task(cancelled.wait())
                done, _ = await asyncio.wait((stop, signal), timeout=2, return_when=asyncio.FIRST_COMPLETED)
                if stop in done:
                    await stop
                await asyncio.wait_for(signal, 2)
                messages = log.reader("probe:room").snapshot()
                assert isinstance(messages[-1].body, Control)
                assert messages[-1].body.action == "pause" and messages[-1].body.through_seq == messages[0].seq
                assert not needs_reply(messages, "conversation")
                assert not task.active and not stop.done()
                await accept("probe:room", "next", raw().message)
                assert needs_reply(log.reader("probe:room").snapshot(), "conversation")
                drain.set()
                paused = await asyncio.wait_for(stop, 2)
                assert await accept("probe:room", "stop", replace(raw().message, content="/stop")) == paused
                assert needs_reply(log.reader("probe:room").snapshot(), "conversation")
                assert len(log.reader("probe:room").snapshot()) == 3
            finally:
                drain.set()
                await asyncio.gather(stop, return_exceptions=True)


def mobile_raw(number=1, *, attachments=()):
    return RawInbound(f"mobile-{number}", ChannelInboundMessage(
        channel="akashic", chat_id="room", sender="device:one", content="hello",
        timestamp=datetime(2026, 9, 5, tzinfo=UTC), attachments=attachments, metadata={
            "session_key_override": "akashic:room", "provider_message_id": f"mobile-{number}",
            "durable_inbound": True, "durable_handoff_id": f"handoff-{number}",
        },
    ), provider_identity="room", recipient="room")


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_mobile_restart_replays_input_once_and_only_finishes_transport(
    tmp_path, committed,
):
    """重开当前 schema 只恢复输入交接，不构造旧执行器。"""
    from agent.plugin_composition.channels import CHANNEL_INPUT
    from session.admissions import SessionAdmissions
    from session.inbound_store import InboundHandoffStore
    from session.log import SessionAttributes

    # 1. 先落 durable 输入。
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    log.ensure_session("akashic:room", SessionAttributes())
    log.close()
    handoffs, admissions = InboundHandoffStore(path), SessionAdmissions(path)
    original = MessageBus()
    original.bind_durable_inbound_store(handoffs)
    original.bind_session_admission_owner(admissions)
    raw_message = mobile_raw()
    try:
        assert await original.reserve_durable_inbound(raw_message)
        reserved = handoffs.list_inbound_handoffs()
    finally:
        await original.aclose()
        handoffs.close()
        admissions.close()

    # 2. 分别模拟正文提交前与提交后进程结束，重开都不能重复正文。
    initialize_plugin_workspace(tmp_path / "workspace")
    if committed:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic") as (log, host, *rest):
            context = _source_context(host)
            async with context.runtime_scope():
                first = await context.require(CHANNEL_INPUT)(
                    "akashic:room", raw_message.message_id, raw_message.message)
    handoffs, admissions = InboundHandoffStore(path), SessionAdmissions(path)
    assert handoffs.list_inbound_handoffs() == reserved
    admissions.clear_stale()
    try:
        async with runtime(tmp_path, channel_name="akashic", inbound_store=handoffs,
                           admissions=admissions) as (log, host, custody, _, _, adapter):
            messages = log.reader("akashic:room").snapshot()
            assert len(messages) == 1 and messages[0].message_id == "mobile-1"
            if committed:
                assert messages[0] == first
            assert handoffs.list_inbound_handoffs() == []
            assert custody.inbound_size == 0 and custody.completed == 1
            assert custody._durable_admissions == {}
            assert custody.envelopes[0].snapshot_id == adapter.context.snapshot_id
            assert custody.envelopes[0].binding_token == adapter.context.binding_token
            assert not custody.envelopes[0].lease.active
    finally:
        handoffs.close()
        admissions.close()


@pytest.mark.asyncio
async def test_stop_transfers_reserve_only_handoff_and_revokes_old_port(tmp_path):
    """旧 binding 停止后，reserve-only row 由下一 Host 恢复且旧 port 失权。"""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    raw_message = mobile_raw()
    old_durable = None
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (log, _, _, _, _, adapter):
            old_durable = adapter.ports.durable_inbound
            assert old_durable is not None
            assert await old_durable.reserve(raw_message)
            assert log.reader("akashic:room").snapshot() == ()
        assert old_durable is not None
        with pytest.raises(KeyError):
            await old_durable.settle_rejected(
                session_key="akashic:room",
                provider_message_id="mobile-1",
            )
        assert manager.inbound_store.list_inbound_handoffs()

        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
        ) as (log, _, custody, _, _, adapter):
            assert [item.message_id for item in log.reader("akashic:room").snapshot()] == [
                "mobile-1"
            ]
            assert manager.inbound_store.list_inbound_handoffs() == []
            assert custody.completed == 1
            assert adapter.ports.durable_inbound is not None
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_adapter_stop_can_compensate_reservation_before_host_transfer(tmp_path):
    """Adapter-owned cancellation keeps its exact reservation until stop returns."""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, _, _, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(mobile_raw())
            order: list[str] = []

            async def stop_with_compensation():
                order.append("adapter.stop")
                await durable.defer("handoff-1")
                order.append("adapter.defer")
                return StopReceipt(adapter.context.binding_token, True)

            adapter.stop = stop_with_compensation
        assert order == ["adapter.stop", "adapter.defer"]
        assert manager.inbound_store.list_inbound_handoffs()
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_adapter_stop_failure_retains_reservation_owner_until_retry(tmp_path):
    """A failed adapter stop leaves its old reservation in the cleanup tombstone."""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(mobile_raw())

            async def fail_stop():
                raise RuntimeError("adapter stop injected failure")

            adapter.stop = fail_stop
            key = binding_key(host, "akashic")
            channel = channels(host)
            binding = channel._bindings[key]
            generation = host.generation("probe_channel")
            assert generation is not None and generation.fiber is not None
            probe_fiber = generation.fiber
            registration = next(
                effect for effect in probe_fiber.effects
                if effect.label == "channel:akashic"
            )
            adapter_owner = binding.adapter
            reservation = binding.durable_reservations["handoff-1"]
            with pytest.raises(RuntimeError) as error:
                await registration.aclose()
            assert str(error.value) == (
                "channel cleanup failed: adapter stop injected failure"
            )
            assert registration in probe_fiber.effects
            assert channel._bindings[key] is binding
            assert binding.adapter is adapter_owner
            assert channel._durable_reservation_owners["handoff-1"] == key
            assert binding.durable_reservations["handoff-1"] is reservation
            assert not custody._durable_admissions["handoff-1"].recoverable

            async def successful_stop():
                return StopReceipt(adapter.context.binding_token, True)

            adapter.stop = successful_stop
            await registration.aclose()
            assert registration not in probe_fiber.effects
            assert key not in channel._bindings
            assert custody._durable_admissions["handoff-1"].recoverable
            assert manager.inbound_store.has_inbound_handoff(
                channel="akashic", session_key="akashic:room",
                provider_message_id="mobile-1",
            )
            await probe_fiber.dispose()
            assert probe_fiber.state is FiberState.DISPOSED
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_reserve_cancellation_registers_owner_before_propagating(tmp_path):
    """Cancellation after Bus persistence cannot orphan the Host reservation."""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            custody.durable_reserve_gate.clear()
            reserving = asyncio.create_task(durable.reserve(mobile_raw()))
            await asyncio.wait_for(custody.durable_reserved.wait(), 2)
            reserving.cancel()
            custody.durable_reserve_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await reserving

            key = binding_key(host, "akashic")
            assert channels(host)._durable_reservation_owners[
                "handoff-1"
            ] == key
            assert await durable.defer("handoff-1") is None
            await custody.recover_durable_inbounds()
            assert custody.completed == 1
            assert not manager.inbound_store.list_inbound_handoffs()
    finally:
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["defer", "settle_rejected"])
async def test_durable_settlement_cancellation_waits_for_real_bus_operation(tmp_path, settlement):
    """取消只恢复给 caller；真实 durable settlement 完成后再报告取消。"""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            message = mobile_raw()
            assert await durable.reserve(message)
            custody.settlement_gate.clear()
            if settlement == "defer":
                operation = durable.defer("handoff-1")
            else:
                operation = durable.settle_rejected(
                    session_key="akashic:room", provider_message_id="mobile-1",
                )
            settling = asyncio.create_task(operation)
            try:
                await asyncio.wait_for(custody.settlement_started.wait(), 2)
                settling.cancel()
                first_cancel_delivered = asyncio.Event()
                asyncio.get_running_loop().call_soon(first_cancel_delivered.set)
                await first_cancel_delivered.wait()
                assert settling.cancelling() >= 1
                assert not settling.done()
                settling.cancel()
                second_cancel_delivered = asyncio.Event()
                asyncio.get_running_loop().call_soon(second_cancel_delivered.set)
                await second_cancel_delivered.wait()
                assert settling.cancelling() >= 2
                assert not settling.done()
            finally:
                custody.settlement_gate.set()
                await asyncio.gather(settling, return_exceptions=True)
            with pytest.raises(asyncio.CancelledError):
                await settling

            channel = channels(host)
            key = binding_key(host, "akashic")
            if settlement == "defer":
                assert manager.inbound_store.has_inbound_handoff(
                    channel="akashic", session_key="akashic:room",
                    provider_message_id="mobile-1",
                )
                assert custody._durable_admissions["handoff-1"].recoverable
                assert "handoff-1" not in channel._durable_reservation_owners
            else:
                assert not manager.inbound_store.has_inbound_handoff(
                    channel="akashic", session_key="akashic:room",
                    provider_message_id="mobile-1",
                )
                assert "handoff-1" not in custody._durable_admissions
            assert "handoff-1" not in channel._durable_reservation_owners
            assert "handoff-1" not in channel._bindings[key].durable_reservations
            assert custody.inbound_size == 0
    finally:
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["defer", "settle_rejected"])
async def test_durable_settlement_failure_retains_exact_reservation_until_retry(tmp_path, settlement):
    """Bus 结算失败保留真实 reservation owner，显式重试才完成同一回执。"""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(mobile_raw())
            custody.fail_settlement_once = True

            channel = channels(host)
            key = binding_key(host, "akashic")
            binding = channel._bindings[key]
            reservation = binding.durable_reservations["handoff-1"]
            with pytest.raises(OSError, match="injected durable settlement failure"):
                if settlement == "defer":
                    await durable.defer("handoff-1")
                else:
                    await durable.settle_rejected(
                        session_key="akashic:room", provider_message_id="mobile-1",
                    )

            assert channel._durable_reservation_owners["handoff-1"] == key
            assert binding.durable_reservations["handoff-1"] is reservation
            assert custody._durable_admissions["handoff-1"].recoverable is False
            assert manager.inbound_store.has_inbound_handoff(
                channel="akashic", session_key="akashic:room",
                provider_message_id="mobile-1",
            )

            if settlement == "defer":
                await durable.defer("handoff-1")
                assert custody._durable_admissions["handoff-1"].recoverable
                assert manager.inbound_store.has_inbound_handoff(
                    channel="akashic", session_key="akashic:room",
                    provider_message_id="mobile-1",
                )
            else:
                await durable.settle_rejected(
                    session_key="akashic:room", provider_message_id="mobile-1",
                )
                assert not custody._durable_admissions
                assert not manager.inbound_store.has_inbound_handoff(
                    channel="akashic", session_key="akashic:room",
                    provider_message_id="mobile-1",
                )
            assert "handoff-1" not in channel._durable_reservation_owners
            assert "handoff-1" not in binding.durable_reservations
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_same_host_local_retry_recovers_reserve_only_row(tmp_path):
    """同一 live Root 的 local retry 先重开 binding，再由当前 owner 恢复。"""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (log, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(mobile_raw())
            await host.retry_runtime_recovery("probe_channel")
            await custody.recover_durable_inbounds()
            await custody.committed.wait()
            messages = log.reader("akashic:room").snapshot()
            assert [item.message_id for item in messages] == ["mobile-1"]
            assert custody.completed == 1
            assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_recovery_failure_keeps_committed_root_and_pending_input(tmp_path):
    """恢复失败不回滚已经提交的选择，也不删 pending 输入。"""
    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager, recover=False) as (_, host, custody, _, _, adapter):
            durable = adapter.ports.durable_inbound
            assert await durable.reserve(mobile_raw())
            await durable.defer("handoff-1")
            original = host.live_root
            channel = channels(host)
            recover = channel._recover_inbound

            async def fail(*args, **kwargs):
                raise KeyError("temporary recovery failure")

            channel._recover_inbound = fail
            with pytest.raises(KeyError, match="temporary recovery failure"):
                await custody.recover_durable_inbounds()
            assert host.live_root is original
            assert manager.inbound_store.list_inbound_handoffs()
            channel._recover_inbound = recover
            await custody.recover_durable_inbounds()
            assert not manager.inbound_store.list_inbound_handoffs()
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_old_port_cannot_settle_handoff_reclaimed_by_next_generation(tmp_path):
    """同一 Host 的新 binding 接管后，旧 port 不能删除同一 handoff。"""

    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(
            tmp_path,
            channel_name="akashic",
            session_manager=manager,
            recover=False,
        ) as (_, host, _, _, _, adapter):
            old_port = adapter.ports.durable_inbound
            assert old_port is not None
            raw_message = mobile_raw()
            assert await old_port.reserve(raw_message)
            channel_host = channels(host)
            await host.retry_runtime_recovery("probe_channel")
            new_key = binding_key(host, "akashic")
            new_port = channel_host._bindings[new_key].adapter.ports.durable_inbound
            assert new_port is not None
            assert await new_port.reserve(raw_message)
            with pytest.raises(KeyError):
                await old_port.settle_rejected(
                    session_key="akashic:room",
                    provider_message_id="mobile-1",
                )
            assert manager.inbound_store.has_inbound_handoff(
                channel="akashic",
                session_key="akashic:room",
                provider_message_id="mobile-1",
            )
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_recovery_pages_past_live_owner_without_replaying_it(tmp_path, monkeypatch):
    from session.manager import SessionManager
    import bus.queue as queue

    monkeypatch.setattr(queue, "_DURABLE_INBOUND_RECOVERY_PAGE_SIZE", 1)
    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            custody.reserve_gate.clear()
            assert await custody.reserve_durable_inbound(mobile_raw(1))
            live = asyncio.create_task(adapter.context.ingress.admit(mobile_raw(1)))
            await asyncio.wait_for(custody.reserved.wait(), 2)
            for number in (2, 3):
                assert await custody.reserve_durable_inbound(mobile_raw(number))
                await custody.defer_durable_inbound(f"handoff-{number}")
            custody.reserve_gate.set()
            await custody.recover_durable_inbounds()
            assert await live is True
            assert len(log.reader("akashic:room").snapshot()) == 3
            assert [item.message_id for item in custody.envelopes].count("mobile-1") == 1
            assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_recovery_missing_session_retains_row_and_releases_batch_claims(tmp_path):
    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    original = MessageBus()
    original.bind_durable_inbound_store(manager.inbound_store)
    original.bind_session_admission_owner(manager.admissions)
    for number in (1, 2):
        assert await original.reserve_durable_inbound(mobile_raw(number))
    await original.aclose()
    assert manager.delete_session("akashic:room")
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager, recover=False) as (log, host, custody, _, _, adapter):
            with pytest.raises(KeyError, match="session 不存在"):
                await custody.recover_durable_inbounds()
            assert len(manager.inbound_store.list_inbound_handoffs()) == 2
            assert custody._recovery_claimed == set()
            assert log.reader("akashic:room").snapshot() == ()
            assert not custody.envelopes[0].lease.active
    finally:
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_bus", [False, True])
async def test_mobile_precommit_cancel_or_shutdown_keeps_exact_attachment_handoff(tmp_path, close_bus):
    from session.manager import SessionManager
    from agent.plugin_composition.channels import AttachmentKind, AttachmentRef

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    ref = AttachmentRef("image", AttachmentKind.IMAGE, "image.png", "image/png", 3, "a" * 64)
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            custody.reserve_gate.clear()
            message = mobile_raw(attachments=(ref,))
            durable = adapter.ports.durable_inbound
            assert durable is not None
            assert await durable.reserve(message)
            submit = asyncio.create_task(adapter.context.ingress.admit(message))
            await asyncio.wait_for(custody.reserved.wait(), 2)
            if close_bus:
                # Core 停机先终结 ingress，再释放进程接纳权。
                channel = channels(host)
                key = binding_key(host, "akashic")
                channel._close_admission(key)
            submit.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(submit, 2)
            if close_bus:
                await custody.aclose()
            assert not log.reader("akashic:room").snapshot()
            assert durable.pending_attachment_refs(session_key="akashic:room", provider_message_id="mobile-1") == (ref,)
            assert not custody.envelopes[0].lease.active
            if close_bus:
                assert custody._durable_admissions == {}
            else:
                assert custody._durable_admissions["handoff-1"].recoverable
                await durable.defer("handoff-1")
                assert manager.inbound_store.has_inbound_handoff(
                    channel="akashic",
                    session_key="akashic:room",
                    provider_message_id="mobile-1",
                )
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_committed_cancel_waits_for_durable_delete_and_exact_release(tmp_path):
    from session.manager import SessionManager
    from session.store import SessionAdmissionConflictError

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            custody.complete_gate.clear()
            assert await custody.reserve_durable_inbound(mobile_raw())
            submit = asyncio.create_task(adapter.context.ingress.admit(mobile_raw()))
            await asyncio.wait_for(custody.committed.wait(), 2)
            submit.cancel()
            assert len(log.reader("akashic:room").snapshot()) == 1
            assert custody.envelopes[0].lease.active
            assert manager.inbound_store.list_inbound_handoffs()
            with pytest.raises(SessionAdmissionConflictError):
                manager.delete_session_with_audit("akashic:room")
            custody.complete_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(submit, 2)
            assert manager.inbound_store.list_inbound_handoffs() == []
            assert not custody.envelopes[0].lease.active
            assert custody._durable_admissions == {}
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_postcommit_cleanup_shutdown_retains_recoverable_row(tmp_path, monkeypatch):
    from session.manager import SessionManager
    import bus.queue as queue

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    started = asyncio.Event()
    async def wait_for_shutdown(bus, key):
        started.set()
        await asyncio.Event().wait()
    def fail_delete(handoff_id):
        raise OSError("injected delete failure")
    monkeypatch.setattr(queue.MessageBus, "_retry_inbound_cleanup", wait_for_shutdown)
    monkeypatch.setattr(manager.inbound_store, "complete_inbound_handoff", fail_delete)
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            assert await custody.reserve_durable_inbound(mobile_raw())
            assert await adapter.context.ingress.admit(mobile_raw())
            await asyncio.wait_for(started.wait(), 2)
            await custody.aclose()
            assert len(log.reader("akashic:room").snapshot()) == 1
            assert len(manager.inbound_store.list_inbound_handoffs()) == 1
            assert not custody.envelopes[0].lease.active
            assert custody._durable_admissions == {}
            assert custody._inbound_cleanup_tasks == {}
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_prepare_waiting_on_handoff_lock_cannot_commit_after_close(tmp_path, monkeypatch):
    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            assert await custody.reserve_durable_inbound(mobile_raw())
            await custody._durable_handoff_lock.acquire()
            submit = asyncio.create_task(adapter.context.ingress.admit(mobile_raw()))
            await asyncio.wait_for(custody.prepared.wait(), 2)
            closing_started = asyncio.Event()
            stop = custody._stop_outbound_dispatcher
            async def mark_close():
                closing_started.set()
                await stop()
            monkeypatch.setattr(custody, "_stop_outbound_dispatcher", mark_close)
            closing = asyncio.create_task(custody.aclose())
            await asyncio.wait_for(closing_started.wait(), 2)
            custody._durable_handoff_lock.release()
            with pytest.raises(RuntimeError, match="message bus 已关闭"):
                await asyncio.wait_for(submit, 2)
            await asyncio.wait_for(closing, 2)
            assert not log.reader("akashic:room").snapshot()
            assert len(manager.inbound_store.list_inbound_handoffs()) == 1
            assert not custody.envelopes[0].lease.active
            assert custody._durable_admissions == {}
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_channel_input_imported_artifact_is_pinned_and_read_lease_closed(tmp_path, monkeypatch):
    from agent.plugin_composition.channels import AttachmentKind
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore

    MessageLog(tmp_path / "sessions.db").close()
    store = ArtifactStore(tmp_path / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=store)
    try:
        ref = await artifacts.import_bytes(b"evidence", kind=AttachmentKind.FILE,
                                           filename="evidence.txt", media_type="text/plain")
        opened = []
        acquire = artifacts.acquire
        async def track(ref):
            lease = await acquire(ref)
            opened.append(lease)
            return lease
        monkeypatch.setattr(artifacts, "acquire", track)
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, artifacts=artifacts) as (log, host, custody, _, _, adapter):
            message = replace(raw(), message=replace(raw().message, attachments=(ref,)))
            assert await adapter.context.ingress.admit(message)
            persisted = log.reader("probe:room").get("u1")
            assert [part.value for part in persisted.body.parts if part.kind == "artifact_ref"] == [ref.artifact_id]
            assert log.reader("probe:room").attachments("u1") == (ref,)
            assert len(opened) == 1
            with pytest.raises(RuntimeError, match="关闭"):
                await opened[0].read_bytes(max_bytes=100)
            assert not custody.envelopes[0].lease.active
            assert (tmp_path / "uploads/artifacts" / f"{ref.artifact_id}.bin").read_bytes() == b"evidence"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_mixed_legacy_recovery_keeps_one_page_and_still_accepts_exact_input(tmp_path, monkeypatch):
    import json
    from session.manager import SessionManager
    import bus.queue as queue

    monkeypatch.setattr(queue, "_DURABLE_INBOUND_RECOVERY_PAGE_SIZE", 2)
    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    original = MessageBus()
    original.bind_durable_inbound_store(manager.inbound_store)
    original.bind_session_admission_owner(manager.admissions)
    try:
        for number, created in [(1, "2020-01-01T00:00:00+00:00"), (2, "9999-01-01T00:00:00+00:00")]:
            manager.inbound_store.reserve_inbound_handoff(
                handoff_id=f"legacy-{number}", dedupe_key=f"akashic:room:legacy-{number}",
                channel="akashic", sender="device:one", chat_id="room", session_key="akashic:room",
                content="legacy input", timestamp="2026-09-05T00:00:00+00:00", media_json="[]",
                metadata_json=json.dumps({"client_message_id": f"legacy-{number}"}, separators=(",", ":")), created_at=created)
        assert await original.reserve_durable_inbound(mobile_raw())
        await original.aclose()
        initialize_plugin_workspace(tmp_path / "workspace")
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            assert custody.inbound_size == 1
            assert [message.message_id for message in log.reader("akashic:room").snapshot()] == ["mobile-1"]
            first = await custody.consume_inbound()
            assert first.handoff_id == "legacy-1"
            await custody.complete_inbound(first)
            assert custody.inbound_size == 1
            second = await custody.consume_inbound()
            assert second.handoff_id == "legacy-2"
            await custody.complete_inbound(second)
            assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        await original.aclose()
        manager.close()


@pytest.mark.asyncio
async def test_channel_identity_uses_real_owner_without_legacy_session_runtime(tmp_path):
    initialize_plugin_workspace(tmp_path / "workspace")
    async with runtime(tmp_path, durable_identities=True) as (log, host, custody, _, _, adapter):
        custody.reject = True
        with pytest.raises(ValueError, match="rejected before commit"):
            await adapter.context.ingress.admit(raw())
        with closing(ChannelIdentities(tmp_path / "sessions.db")) as identities:
            assert identities.load("probe") == {}
            assert identities.migration_completed("probe")
        assert log.catalog().snapshot_heads() == {}
        custody.reject = False
        assert await adapter.context.ingress.admit(raw())
        original = log.reader("probe:room").snapshot()
        assert len(original) == 1
    async with runtime(tmp_path, durable_identities=True) as (log, host, _, _, _, adapter):
        assert adapter.context.identity.resolve("provider-user") == "room"
        assert log.reader("probe:room").snapshot() == original
        assert await adapter.context.ingress.admit(raw()) is True
        assert log.reader("probe:room").snapshot() == original
