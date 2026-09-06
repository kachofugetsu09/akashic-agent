import asyncio
from contextlib import asynccontextmanager, closing
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.channels import (
    ChannelCapability, ChannelInboundMessage, ChannelReady, CoreChannelDefinition,
    InboundIdentity, RawInbound, StopReceipt,
)
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from bus.queue import MessageBus
from session.identities import ChannelIdentities
from session.log import MessageLog
from session.message import Input


class Adapter:
    def __init__(self, context):
        self.context = context

    def attach_runtime(self, ports):
        self.ports = ports

    def open_admission(self):
        pass

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
        self.completed = 0
        self.envelopes = []
        self.reject = False

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


@asynccontextmanager
async def runtime(tmp_path, *, channel_name="probe", session_manager=None, recover=True, artifacts=None, inbound_store=None, admissions=None, durable_identities=False):
    sources = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins/conversation", sources / "conversation",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    shutil.copytree(Path(__file__).parents[1] / "plugins/sources", sources / "sources",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    log = MessageLog(tmp_path / "sessions.db")
    identity_store = ChannelIdentities(tmp_path / "sessions.db") if durable_identities else None
    host = PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home", message_log=log,
                         channel_attachment_store=artifacts, channel_identities=identity_store)
    custody = Custody()
    if session_manager is not None:
        inbound_store, admissions = session_manager.inbound_store, session_manager.admissions
    if inbound_store is not None:
        custody.bind_durable_inbound_store(inbound_store)
        custody.bind_mobile_session_admission_owner(admissions)
    identities, rollbacks, adapters = {}, [], []
    async def remember(channel, provider, recipient):
        identities[(channel, provider)] = recipient
        return channel, provider
    async def rollback(key):
        rollbacks.append(key)
        del identities[key]
        return True
    def factory(context):
        adapter = Adapter(context)
        adapters.append(adapter)
        return adapter
    channel = host.channel_generation_host
    if not durable_identities:
        channel._identity_rememberer = remember
        channel._identity_rollbacker = rollback
    channel.bind_input_custody(custody)
    try:
        await host.load_all()
        await host.bind_core_channel_definitions((CoreChannelDefinition(
            name=channel_name, capabilities=frozenset({ChannelCapability.INBOUND}),
            factory=factory, inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            source_revision="test", config_revision="test", generation_id="test",
        ),))
        if inbound_store is not None:
            custody.bind_mobile_channel_inbound_recoverer(adapters[-1].ports.recovery_ingress.recover)
            if recover:
                await custody.recover_durable_inbounds()
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


def raw():
    return RawInbound("u1", ChannelInboundMessage(
        channel="probe", chat_id="room", sender="user", content="hello",
        timestamp=datetime(2026, 9, 5, tzinfo=UTC), metadata={"session_key": "not-authority"},
    ), provider_identity="provider-user", recipient="room")


@pytest.mark.asyncio
async def test_exact_root_input_commits_without_queue_model_or_delivery(tmp_path):
    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        assert await adapter.context.ingress.admit(raw()) is True
        assert await adapter.context.ingress.admit(raw()) is False
        messages = log.reader("probe:room").snapshot()
        assert len(messages) == 1 and isinstance(messages[0].body, Input)
        assert messages[0].message_id == "u1"
        assert not log.reader("not-authority").snapshot()
        assert identities == {("probe", "provider-user"): "room"}
        assert rollbacks == [] and custody.completed == 1
        assert custody.inbound_size == 0
        assert not custody.envelopes[0].lease.active
        assert host.current_snapshot.lease_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["prepare", "committed"])
async def test_input_cancellation_respects_commit_boundary_and_closes_exact_lease(tmp_path, stage):
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
                gate.set()
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
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, identities, rollbacks, adapter):
            message = RawInbound("mobile-1", ChannelInboundMessage(
                channel="akashic", chat_id="room", sender="device:one", content="hello",
                timestamp=datetime(2026, 9, 5, tzinfo=UTC), metadata={
                    "session_key_override": "akashic:room", "client_message_id": "mobile-1",
                    "mobile_v3_handoff": True, "mobile_handoff_id": "handoff-1",
                },
            ), provider_identity="room", recipient="room")
            assert await custody.reserve_mobile_channel_handoff(message)
            assert await adapter.context.ingress.admit(message) is True
            await asyncio.wait_for(retry_started.wait(), 2)
            assert len(log.reader("akashic:room").snapshot()) == 1
            assert custody.mobile_inbound_cleanup_pending(custody.envelopes[0])
            assert store.has_inbound_handoff(session_key="akashic:room", client_message_id="mobile-1")
            with pytest.raises(SessionAdmissionConflictError):
                manager.delete_session_with_audit("akashic:room")
            jobs = tuple(custody._inbound_cleanup_tasks.values())
            retry_release.set()
            await asyncio.wait_for(asyncio.gather(*jobs), 2)
            assert store.list_inbound_handoffs() == []
            assert not custody.envelopes[0].lease.active
            assert custody._mobile_v3_admissions == {}
            assert calls == 2 and rollbacks == []
            assert custody.inbound_size == 0
    finally:
        retry_release.set()
        manager.close()


@pytest.mark.asyncio
async def test_retry_uses_message_identity_without_copying_transport_clock_or_handoff(tmp_path):
    from agent.plugin_composition.channels import CHANNEL_INPUT
    from agent.plugins.snapshot import lease_runtime_snapshot
    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        first = raw().message
        second = replace(first, timestamp=first.timestamp + timedelta(hours=1),
                         metadata={"mobile_handoff_id": "another", "client_request_id": "retry"})
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            accept = snapshot.composition_root.context.require(CHANNEL_INPUT)
            original = await accept("probe:room", "fixed", first)
            assert await accept("probe:room", "fixed", second) == original
        assert len(log.reader("probe:room").snapshot()) == 1
        assert original.body.parts[0].value == {"channel": "probe", "chat_id": "room", "sender": "user"}


@pytest.mark.asyncio
async def test_stop_commits_pause_before_draining_and_duplicate_does_not_pause_new_work(tmp_path):
    from agent.plugin_composition.channels import CHANNEL_INPUT
    from agent.plugins.snapshot import lease_runtime_snapshot
    from plugins.conversation.plugin import CONVERSATION
    from plugins.conversation.source import needs_reply
    from session.message import Control

    async with runtime(tmp_path) as (log, host, custody, identities, rollbacks, adapter):
        started, cancelled, drain = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def program(task, reader, source):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
                await drain.wait()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            accept = root.require(CHANNEL_INPUT)
            await accept("probe:room", "first", raw().message)
            conversation = root.require(CONVERSATION)("probe:room")
            task = await conversation.start(program)
            await asyncio.wait_for(started.wait(), 2)
            async def pause():
                async with lease_runtime_snapshot(host.snapshot_store) as active:
                    return await active.composition_root.context.require(CHANNEL_INPUT)(
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
            "session_key_override": "akashic:room", "client_message_id": f"mobile-{number}",
            "mobile_v3_handoff": True, "mobile_handoff_id": f"handoff-{number}",
        },
    ), provider_identity="room", recipient="room")


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("legacy_database", [False, True])
async def test_mobile_restart_replays_input_once_and_only_finishes_transport(
    tmp_path, committed, legacy_database, monkeypatch,
):
    """同一新库或迁移库重开后只恢复输入交接，不构造旧执行器。"""
    import runpy
    import yoyo
    from agent.migrations.context import bind_migration_context
    from agent.plugin_composition.channels import CHANNEL_INPUT
    from agent.plugins.snapshot import lease_runtime_snapshot
    from session.admissions import SessionAdmissions
    from session.inbound_store import InboundHandoffStore
    from session.log import SessionAttributes
    from session.store import SessionStore

    # 1. 先落 durable 输入；旧库迁移必须保留这条尚未完成的交接。
    path = tmp_path / "sessions.db"
    if legacy_database:
        old = SessionStore(path)
        old.create_session(key="akashic:room")
        old.close()
    else:
        log = MessageLog(path)
        log.ensure_session("akashic:room", SessionAttributes())
        log.close()
    handoffs, admissions = InboundHandoffStore(path), SessionAdmissions(path)
    original = MessageBus()
    original.bind_durable_inbound_store(handoffs)
    original.bind_mobile_session_admission_owner(admissions)
    raw_message = mobile_raw()
    try:
        assert await original.reserve_mobile_channel_handoff(raw_message)
        reserved = handoffs.list_inbound_handoffs()
    finally:
        await original.aclose()
        handoffs.close()
        admissions.close()
    if legacy_database:
        monkeypatch.setattr(yoyo, "step", lambda callback: callback)
        with bind_migration_context(workspace=tmp_path, config_path=tmp_path / "config.toml"):
            for number, name in [(1, "message_log"), (2, "owner_records"), (3, "model_calls"),
                                 (5, "message_embeddings"), (6, "message_artifacts")]:
                module = runpy.run_path(str(Path(__file__).parents[1] / "migrations/yoyo" /
                                           f"20260905_{number:02d}_{name}.py"))
                module[f"migrate_{name}"](None)

    # 2. 分别模拟正文提交前与提交后进程结束，重开都不能重复正文。
    if committed:
        async with runtime(tmp_path, channel_name="akashic") as (log, host, *rest):
            async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                first = await snapshot.composition_root.context.require(CHANNEL_INPUT)(
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
            assert custody._mobile_v3_admissions == {}
            assert custody.envelopes[0].snapshot_id == host.current_snapshot.snapshot_id
            assert custody.envelopes[0].binding_token == adapter.context.binding_token
            assert not custody.envelopes[0].lease.active
    finally:
        handoffs.close()
        admissions.close()


@pytest.mark.asyncio
async def test_mobile_recovery_pages_past_live_owner_without_replaying_it(tmp_path, monkeypatch):
    from session.manager import SessionManager
    import bus.queue as queue

    monkeypatch.setattr(queue, "_DURABLE_INBOUND_RECOVERY_PAGE_SIZE", 1)
    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            custody.reserve_gate.clear()
            assert await custody.reserve_mobile_channel_handoff(mobile_raw(1))
            live = asyncio.create_task(adapter.context.ingress.admit(mobile_raw(1)))
            await asyncio.wait_for(custody.reserved.wait(), 2)
            for number in (2, 3):
                assert await custody.reserve_mobile_channel_handoff(mobile_raw(number))
                await custody.defer_mobile_channel_handoff(f"handoff-{number}")
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
    original.bind_mobile_session_admission_owner(manager.admissions)
    for number in (1, 2):
        assert await original.reserve_mobile_channel_handoff(mobile_raw(number))
    await original.aclose()
    assert manager.delete_session("akashic:room")
    try:
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
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            custody.reserve_gate.clear()
            message = mobile_raw(attachments=(ref,))
            assert await custody.reserve_mobile_channel_handoff(message)
            submit = asyncio.create_task(adapter.context.ingress.admit(message))
            await asyncio.wait_for(custody.reserved.wait(), 2)
            if close_bus:
                # Core 停机先终结 ingress，再释放进程接纳权。
                host.channel_generation_host.close_admission(host.current_snapshot.snapshot_id)
            submit.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(submit, 2)
            if close_bus:
                await custody.aclose()
            assert not log.reader("akashic:room").snapshot()
            assert custody.pending_mobile_attachment_refs(session_key="akashic:room", client_message_id="mobile-1") == (ref,)
            assert not custody.envelopes[0].lease.active
            if close_bus:
                assert custody._mobile_v3_admissions == {}
            else:
                assert custody._mobile_v3_admissions["handoff-1"].recoverable
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_committed_cancel_waits_for_durable_delete_and_exact_release(tmp_path):
    from session.manager import SessionManager
    from session.store import SessionAdmissionConflictError

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            custody.complete_gate.clear()
            assert await custody.reserve_mobile_channel_handoff(mobile_raw())
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
            assert custody._mobile_v3_admissions == {}
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
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            assert await custody.reserve_mobile_channel_handoff(mobile_raw())
            assert await adapter.context.ingress.admit(mobile_raw())
            await asyncio.wait_for(started.wait(), 2)
            await custody.aclose()
            assert len(log.reader("akashic:room").snapshot()) == 1
            assert len(manager.inbound_store.list_inbound_handoffs()) == 1
            assert not custody.envelopes[0].lease.active
            assert custody._mobile_v3_admissions == {}
            assert custody._inbound_cleanup_tasks == {}
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_mobile_prepare_waiting_on_handoff_lock_cannot_commit_after_close(tmp_path, monkeypatch):
    from session.manager import SessionManager

    manager = SessionManager(tmp_path / "transport")
    manager.save(manager.get_or_create("akashic:room"))
    try:
        async with runtime(tmp_path, channel_name="akashic", session_manager=manager) as (log, host, custody, _, _, adapter):
            assert await custody.reserve_mobile_channel_handoff(mobile_raw())
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
            assert custody._mobile_v3_admissions == {}
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_channel_input_imported_artifact_is_pinned_and_read_lease_closed(tmp_path, monkeypatch):
    import runpy
    import yoyo
    monkeypatch.setattr(yoyo, "step", lambda callback: callback)
    from agent.migrations.context import bind_migration_context
    from agent.plugin_composition.channels import AttachmentKind
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.store import SessionStore
    from session.artifact_store import ArtifactStore

    SessionStore(tmp_path / "sessions.db").close()
    store = ArtifactStore(tmp_path / "sessions.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=store)
    try:
        ref = await artifacts.import_bytes(b"evidence", kind=AttachmentKind.FILE,
                                           filename="evidence.txt", media_type="text/plain")
        with bind_migration_context(workspace=tmp_path, config_path=tmp_path / "config.toml"):
            for number, name in [(1, "message_log"), (2, "owner_records"), (3, "model_calls"),
                                 (5, "message_embeddings"), (6, "message_artifacts")]:
                module = runpy.run_path(str(Path(__file__).parents[1] / "migrations/yoyo" /
                                           f"20260905_{number:02d}_{name}.py"))
                module[f"migrate_{name}"](None)
        opened = []
        acquire = artifacts.acquire
        async def track(ref):
            lease = await acquire(ref)
            opened.append(lease)
            return lease
        monkeypatch.setattr(artifacts, "acquire", track)
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
    original.bind_mobile_session_admission_owner(manager.admissions)
    try:
        for number, created in [(1, "2020-01-01T00:00:00+00:00"), (2, "9999-01-01T00:00:00+00:00")]:
            manager.inbound_store.reserve_inbound_handoff(
                handoff_id=f"legacy-{number}", dedupe_key=f"akashic:room:legacy-{number}",
                channel="akashic", sender="device:one", chat_id="room", session_key="akashic:room",
                content="legacy input", timestamp="2026-09-05T00:00:00+00:00", media_json="[]",
                metadata_json=json.dumps({"client_message_id": f"legacy-{number}"}, separators=(",", ":")), created_at=created)
        assert await original.reserve_mobile_channel_handoff(mobile_raw())
        await original.aclose()
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
