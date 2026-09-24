from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.plugin_contracts import json_value
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelInboundMessage,
    DURABLE_ATTACHMENT_REFS,
    DURABLE_HANDOFF_ID,
    DURABLE_INBOUND_MARKER,
    DURABLE_PROVIDER_MESSAGE_ID,
    InboundEnvelope,
    InboundOwner,
    InboundState,
    JsonValue,
    RawInbound,
)
from bus.events import InboundMessage
from bus.queue import MessageBus
from session.inbound_store import HANDOFF_PROVIDER_IDENTITY_KEY, InboundHandoffStore
from session.manager import SessionManager
from session.store import SessionAdmissionConflictError


class _InboundLease:
    def __init__(
        self,
        close_gate: asyncio.Event | None = None,
        *,
        channel: str = "feishu",
    ) -> None:
        self.snapshot_id = "snapshot-1"
        self.generation_id = "generation-1"
        self.binding_token = "binding-1"
        self.channel_name = channel
        self.closed = 0
        self.closed_event = asyncio.Event()
        self.close_gate = close_gate
        self.close_started = asyncio.Event()

    @property
    def active(self) -> bool:
        return self.closed == 0

    async def aclose(self) -> None:
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.closed += 1
        self.closed_event.set()


def _v3_inbound(
    close_gate: asyncio.Event | None = None,
    *,
    message_id: str = "message-1",
    attachments: tuple[AttachmentRef, ...] = (),
    metadata: dict[str, JsonValue] | None = None,
    channel: str = "feishu",
) -> tuple[InboundEnvelope, _InboundLease]:
    lease = _InboundLease(close_gate, channel=channel)
    session_key = (metadata or {}).get("session_key_override", f"{channel}:chat-1")
    assert isinstance(session_key, str)
    envelope = InboundEnvelope(
        message_id=message_id,
        session_key=session_key,
        snapshot_id=lease.snapshot_id,
        generation_id=lease.generation_id,
        binding_token=lease.binding_token,
        message=ChannelInboundMessage(
            channel=channel,
            sender="user-1",
            chat_id="chat-1",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
            attachments=attachments,
        ),
        lease=lease,  # pyright: ignore[reportArgumentType] - focused inbound lease fake
    )
    return envelope, lease


def _durable_raw(
    *,
    session_key: str,
    handoff_id: str,
    message_id: str,
    provider_identity: str | None,
    recipient: str | None,
    sender: str = "display-sender",
    chat_id: str = "display-chat",
    content: str = "durable payload",
    metadata_extra: dict[str, JsonValue] | None = None,
    attachments: tuple[AttachmentRef, ...] = (),
) -> RawInbound:
    """Build one durable input whose provider and display identities differ."""

    metadata: dict[str, JsonValue] = {
        "session_key_override": session_key,
        DURABLE_PROVIDER_MESSAGE_ID: message_id,
        DURABLE_INBOUND_MARKER: True,
        DURABLE_HANDOFF_ID: handoff_id,
        "business": {"keep": True},
    }
    if metadata_extra is not None:
        metadata.update(metadata_extra)
    return RawInbound(
        message_id=message_id,
        provider_identity=provider_identity,
        recipient=recipient,
        message=ChannelInboundMessage(
            channel="identity-test",
            sender=sender,
            chat_id=chat_id,
            content=content,
            timestamp=datetime(2026, 9, 23, tzinfo=timezone.utc),
            metadata=metadata,
            attachments=attachments,
        ),
    )


def _insert_handoff(
    store: InboundHandoffStore,
    raw: RawInbound,
    metadata: dict[str, JsonValue],
    *,
    dedupe_key: str | None = None,
) -> tuple[str, bool]:
    """Insert a fixed test handoff without invoking recovery or rewriting it."""

    return store.reserve_inbound_handoff(
        handoff_id=cast(str, raw.message.metadata[DURABLE_HANDOFF_ID]),
        dedupe_key=dedupe_key,
        channel=raw.message.channel,
        sender=raw.message.sender,
        chat_id=raw.message.chat_id,
        session_key=cast(str, raw.message.metadata["session_key_override"]),
        content=raw.message.content,
        timestamp=raw.message.timestamp.isoformat(),
        media_json="[]",
        metadata_json=json.dumps(
            json_value(metadata), ensure_ascii=False, separators=(",", ":")
        ),
        created_at="2026-09-23T00:00:00+00:00",
    )


async def _durable_bus(manager: SessionManager) -> MessageBus:
    bus = MessageBus()
    try:
        bus.bind_durable_inbound_store(manager.inbound_store)
        bus.bind_session_admission_owner(manager.admissions)
    except BaseException:
        await bus.aclose()
        raise
    return bus


def _legacy_mobile_metadata(raw: RawInbound) -> dict[str, JsonValue]:
    return {
        "mobile_v3_handoff": True,
        "mobile_handoff_id": cast(str, raw.message.metadata[DURABLE_HANDOFF_ID]),
        "client_message_id": raw.message_id,
        "mobile_v3_attachment_refs": (),
        "session_key_override": cast(str, raw.message.metadata["session_key_override"]),
        "business": {"keep": True},
    }


def _raw_handoff_row(store: InboundHandoffStore, handoff_id: str) -> tuple[object, ...]:
    row = store._conn.execute(
        "SELECT * FROM inbound_handoffs WHERE handoff_id = ?",
        (handoff_id,),
    ).fetchone()
    assert row is not None
    return tuple(row)


@pytest.mark.asyncio
async def test_persistent_cleanup_failure_is_bounded_and_shutdown_cancels_retry(
    tmp_path: Path,
) -> None:
    store = InboundHandoffStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "akashic",
        "device:1",
        "persistent",
        "hello",
        metadata={
            "provider_message_id": "client-persistent",
            "durable_inbound": True,
        },
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    attempts = 0

    def always_fail(_handoff_id: str) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError("persistent delete failure")

    store.complete_inbound_handoff = always_fail  # type: ignore[method-assign]
    with pytest.raises(OSError, match="persistent delete failure"):
        await bus.complete_inbound(consumed)
    await asyncio.sleep(0.35)
    assert 2 <= attempts <= 4
    assert len(bus._inbound_accepted) == 1
    assert len(bus._inbound_cleanup_tasks) == 1
    await bus.aclose()
    assert bus._inbound_cleanup_tasks == {}
    assert (
        store._conn.execute(
            "SELECT 1 FROM inbound_handoffs WHERE handoff_id = ?",
            (consumed.handoff_id,),
        ).fetchone()
        is not None
    )
    store.close()


@pytest.mark.asyncio
async def test_cleanup_finalize_failure_is_fatal_and_observable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InboundHandoffStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    item = InboundMessage(
        "akashic",
        "device:1",
        "fatal",
        "hello",
        metadata={
            "provider_message_id": "client-fatal",
            "durable_inbound": True,
        },
    )
    await bus.publish_inbound(item)
    consumed = await bus.consume_inbound()
    original_complete = store.complete_inbound_handoff
    failed = True

    def fail_once(handoff_id: str) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise OSError("temporary delete failure")
        original_complete(handoff_id)

    store.complete_inbound_handoff = fail_once  # type: ignore[method-assign]

    async def fatal_finalize(_owner_key: int, _owner: object) -> None:
        raise RuntimeError("owner mismatch")

    bus._finalize_inbound_owner = fatal_finalize  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError, match="temporary delete failure"):
            await bus.complete_inbound(consumed)
        await asyncio.sleep(0.2)
    assert bus._inbound_cleanup_error is not None
    assert bus._inbound_cleanup_tasks == {}
    assert "event=runtime_fatal" in caplog.text
    assert "owner=message_bus.inbound_cleanup" in caplog.text
    with pytest.raises(RuntimeError, match="cleanup owner failed"):
        await bus.aclose()
    store.close()


@pytest.mark.asyncio
async def test_v3_mobile_reserve_loses_session_before_lock_without_orphan_row(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:delete-before-reserve"
    manager.save(manager.get_or_create(session_key))
    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.inbound_store)
    bus.bind_session_admission_owner(manager.admissions)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-delete-before-reserve",
        metadata={
            "session_key_override": session_key,
            "provider_message_id": "client-delete-before-reserve",
            "durable_inbound": True,
            "durable_handoff_id": "handoff-delete-before-reserve",
        },
    )
    raw = RawInbound(
        message_id=envelope.message_id,
        provider_identity=envelope.sender,
        recipient=envelope.chat_id,
        message=envelope.message,
    )
    await bus._durable_handoff_lock.acquire()
    reserving = asyncio.create_task(bus.reserve_durable_inbound(raw))
    await asyncio.sleep(0)
    assert not reserving.done()
    assert manager.delete_session(session_key) is True
    bus._durable_handoff_lock.release()

    with pytest.raises(KeyError, match="session 不存在"):
        await reserving
    assert manager.inbound_store.list_inbound_handoffs() == []
    assert (
        manager.admissions._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    await bus.aclose()
    manager.close()


@pytest.mark.asyncio
async def test_v3_mobile_reserve_waiting_on_lock_is_rejected_by_bus_close(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:close-before-reserve-lock"
    manager.save(manager.get_or_create(session_key))
    store = manager.inbound_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_session_admission_owner(manager.admissions)
    envelope, _ = _v3_inbound(
        channel="akashic",
        message_id="client-close-before-reserve-lock",
        metadata={
            "session_key_override": session_key,
            "provider_message_id": "client-close-before-reserve-lock",
            "durable_inbound": True,
            "durable_handoff_id": "handoff-close-before-reserve-lock",
        },
    )
    raw = RawInbound(
        message_id=envelope.message_id,
        provider_identity=envelope.sender,
        recipient=envelope.chat_id,
        message=envelope.message,
    )
    await bus._durable_handoff_lock.acquire()
    reserving = asyncio.create_task(bus.reserve_durable_inbound(raw))
    await asyncio.sleep(0)
    closing = asyncio.create_task(bus.aclose())
    await asyncio.sleep(0)
    assert bus._closed is True
    assert not reserving.done()
    assert not closing.done()
    bus._durable_handoff_lock.release()

    with pytest.raises(RuntimeError, match="message bus 已关闭"):
        await reserving
    await closing
    assert store.list_inbound_handoffs() == []
    assert bus._durable_admissions == {}
    assert (
        manager.admissions._conn.execute(
            "SELECT 1 FROM session_admissions WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        is None
    )
    manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('plugin,channel,metadata', [
    ('ordinary', 'feishu', {'session_key_override': 'other-session'}),
    ('core', 'akashic', {'session_key_override': 'other-session'}),
    ('ordinary', 'feishu', {'session_key_override': 'other-session', 'durable_inbound': True}),
    ('core', 'akashic', {'session_key_override': '', 'durable_inbound': True}),
])
async def test_channel_authority_rejects_untrusted_session_override_before_enqueue(plugin, channel, metadata):
    from types import SimpleNamespace
    from plugins.channels.provider import PluginChannels
    from agent.plugin_composition import ChannelCapability, InboundIdentity
    async def unused(*args):
        raise AssertionError('unauthorized input must not open runtime resources')
    host = object.__new__(PluginChannels)
    host._binding = lambda key: SimpleNamespace(
        plugin_id=plugin, channel_name=channel, capabilities=(ChannelCapability.INBOUND,),
        inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID, admission_open=True,
        stopping=False, stopped=False,
    )
    raw = RawInbound(message_id='raw', message=ChannelInboundMessage(
        channel=channel, sender='sender', chat_id='chat', content='input',
        timestamp=datetime.now(timezone.utc), metadata=metadata,
    ))
    with pytest.raises(RuntimeError, match='只属于'):
        await host._admit_inbound_scoped(('snapshot', channel), raw)


def test_host_info_boot_id_is_shared_across_roots():
    from agent.plugin_composition.host import HostInfo

    first = HostInfo("host-1", False)
    second = HostInfo("host-1", False)
    other = HostInfo("host-2", False)
    assert first.boot_id == second.boot_id
    assert first.boot_id != other.boot_id


def test_plugin_manager_passes_supervisor_boot_id_to_channel_host(tmp_path: Path) -> None:
    from agent.restart import RestartGate
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus

    gate = RestartGate(boot_id="supervisor-boot", supervised=False)
    manager = PluginManager([], event_bus=EventBus(), workspace=tmp_path, restart_gate=gate)
    assert manager._host_boot_id == gate.boot_id


def test_plugin_manager_without_gate_gets_one_fresh_host_boot_id(tmp_path: Path) -> None:
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus

    first = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "first")
    second = PluginManager([], event_bus=EventBus(), workspace=tmp_path / "second")

    assert first._host_boot_id
    assert first._host_boot_id != second._host_boot_id


@pytest.mark.asyncio
async def test_host_routes_recovery_by_persisted_channel_to_one_binding() -> None:
    from agent.plugin_composition.channels import ChannelCapability, InboundIdentity
    from agent.plugin_composition.context import CompositionRoot
    from plugins.channels.provider import PluginChannels, _ChannelBindingState

    async def unused(*args):
        return None

    root = CompositionRoot("channel-routing")
    async def apply(_ctx):
        return None

    fiber = await root.mount(apply, name="channel-owner")
    context = fiber.context

    def binding(channel, snapshot):
        return _ChannelBindingState(
            snapshot_id=snapshot, plugin_id="channel-owner", generation_id="test",
            channel_name=channel,
            capabilities=(ChannelCapability.INBOUND, ChannelCapability.DURABLE_INBOUND),
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            factory=lambda _context: pytest.fail("recovery must not start an adapter"),
            adapter=None, binding_token=channel, config={}, factory_context=None,
            plugin_context=context, activation_token=context.fiber.activation_token,
            admission_open=True,
        )

    host = object.__new__(PluginChannels)
    states = {("snapshot", channel): binding(channel, "snapshot") for channel in ("alpha", "beta")}
    host._bindings = states
    seen = []

    async def recover(key, raw):
        seen.append((key, raw.message.channel))
        return True

    host._recover_inbound = recover
    raw = RawInbound(
        message_id="beta-1",
        provider_identity="provider",
        recipient="room",
        message=ChannelInboundMessage(
            channel="beta",
            sender="provider",
            chat_id="room",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "durable_inbound": True,
                "durable_handoff_id": "handoff-beta-1",
                "provider_message_id": "beta-1",
                "session_key_override": "shared-session",
            },
        ),
    )
    try:
        assert await host.recover_inbound(raw) is True
        assert seen == [(("snapshot", "beta"), "beta")]

        states[("other", "beta")] = binding("beta", "other")
        with pytest.raises(RuntimeError, match="不唯一"):
            await host.recover_inbound(raw)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_recovery_skips_channel_without_current_owner_and_continues(
    tmp_path: Path,
) -> None:
    """One missing channel owner must not block another row in the page."""

    import json

    store = InboundHandoffStore(tmp_path / "sessions.db")
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    for channel in ("missing", "ready"):
        handoff_id = f"handoff-{channel}"
        provider_message_id = f"message-{channel}"
        store.reserve_inbound_handoff(
            handoff_id=handoff_id,
            dedupe_key=f"{channel}:shared:{provider_message_id}",
            channel=channel,
            sender="provider",
            chat_id="room",
            session_key="shared",
            content=channel,
            timestamp=datetime.now(timezone.utc).isoformat(),
            media_json="[]",
            metadata_json=json.dumps(
                {
                    "durable_inbound": True,
                    "durable_handoff_id": handoff_id,
                    "provider_message_id": provider_message_id,
                    "session_key_override": "shared",
                }
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    seen: list[str] = []

    async def recover(raw: RawInbound) -> bool:
        seen.append(raw.message.channel)
        return raw.message.channel == "ready"

    bus.bind_durable_inbound_recoverer(recover)
    try:
        await bus.recover_durable_inbounds()
        assert seen == ["missing", "ready"]
        assert bus._recovery_claimed == set()
        assert len(store.list_inbound_handoffs()) == 2
    finally:
        await bus.aclose()
        store.close()


@pytest.mark.asyncio
async def test_durable_identity_is_namespaced_by_channel(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "shared-session"
    manager.save(manager.get_or_create(session_key))
    store = manager.inbound_store
    bus = MessageBus()
    bus.bind_durable_inbound_store(store)
    bus.bind_session_admission_owner(manager.admissions)

    def raw(channel: str, handoff_id: str) -> RawInbound:
        return RawInbound(
            message_id="same-provider-id",
            provider_identity="provider",
            recipient="room",
            message=ChannelInboundMessage(
                channel=channel,
                sender="provider",
                chat_id="room",
                content=channel,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "durable_inbound": True,
                    "durable_handoff_id": handoff_id,
                    "provider_message_id": "same-provider-id",
                    "session_key_override": session_key,
                },
            ),
        )

    try:
        assert await bus.reserve_durable_inbound(raw("alpha", "handoff-alpha"))
        assert await bus.reserve_durable_inbound(raw("beta", "handoff-beta"))
        assert store.has_inbound_handoff(
            channel="alpha", session_key=session_key, provider_message_id="same-provider-id"
        )
        assert store.has_inbound_handoff(
            channel="beta", session_key=session_key, provider_message_id="same-provider-id"
        )
        assert not store.has_inbound_handoff(
            channel="gamma", session_key=session_key, provider_message_id="same-provider-id"
        )
        assert len(store.list_inbound_handoffs()) == 2
    finally:
        await bus.aclose()
        manager.close()


@pytest.mark.asyncio
async def test_live_durable_raw_cannot_use_legacy_provider_key(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "workspace")
    session_key = "legacy-key-session"
    manager.save(manager.get_or_create(session_key))
    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.inbound_store)
    bus.bind_session_admission_owner(manager.admissions)
    raw = RawInbound(
        message_id="legacy-id",
        message=ChannelInboundMessage(
            channel="alpha",
            sender="provider",
            chat_id="room",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "durable_inbound": True,
                "durable_handoff_id": "handoff-legacy",
                "client_message_id": "legacy-id",
                "session_key_override": session_key,
            },
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="durable inbound 缺少 durable handoff identity"):
            await bus.reserve_durable_inbound(raw)
        assert manager.inbound_store.list_inbound_handoffs() == []
    finally:
        await bus.aclose()
        manager.close()


@pytest.mark.asyncio
async def test_mobile_envelope_session_must_match_durable_handoff_before_reserve():
    envelope, lease = _v3_inbound(
        channel="akashic",
        metadata={
            "durable_inbound": True,
            "durable_handoff_id": "handoff-1",
            "provider_message_id": "message-1",
            "session_key_override": "akashic:authorized",
        },
    )
    envelope.session_key = "akashic:different"
    bus = MessageBus()
    with pytest.raises(RuntimeError, match="durable handoff identity"):
        await bus.prepare_channel_input(envelope)
    assert lease.closed == 1
    assert bus.inbound_size == 0


@pytest.mark.parametrize(
    ("provider_identity", "recipient"),
    [("exact-provider", "exact-recipient"), (None, None)],
    ids=("independent-identity", "explicit-none-pair"),
)
@pytest.mark.asyncio
async def test_durable_provider_identity_roundtrips_after_store_reopen(
    tmp_path: Path,
    provider_identity: str | None,
    recipient: str | None,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = "identity-test:session-1"
    manager = SessionManager(workspace)
    bus: MessageBus | None = None
    try:
        manager.save(manager.get_or_create(session_key))
        bus = await _durable_bus(manager)
        attachment = AttachmentRef(
            artifact_id="identity-artifact",
            kind=AttachmentKind.FILE,
            filename=None,
            media_type=None,
            size_bytes=0,
            sha256="a" * 64,
        )
        raw = _durable_raw(
            session_key=session_key,
            handoff_id="identity-handoff",
            message_id="provider-message-1",
            provider_identity=provider_identity,
            recipient=recipient,
            metadata_extra={"provider_option": "kept"},
            attachments=(attachment,),
        )
        assert await bus.reserve_durable_inbound(raw) is True
        assert await bus.reserve_durable_inbound(raw) is True
        rows = manager.inbound_store.list_inbound_handoffs()
        assert len(rows) == 1
        stored_metadata = json.loads(cast(str, rows[0]["metadata_json"]))
        assert stored_metadata[HANDOFF_PROVIDER_IDENTITY_KEY] == {
            "version": 1,
            "provider_identity": provider_identity,
            "recipient": recipient,
        }
        assert stored_metadata["provider_option"] == "kept"
        assert raw.message.metadata.get(HANDOFF_PROVIDER_IDENTITY_KEY) is None
        stored_row = _raw_handoff_row(manager.inbound_store, "identity-handoff")
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            manager.close()

    reopened = SessionManager(workspace)
    recovery_bus: MessageBus | None = None
    try:
        recovery_bus = await _durable_bus(reopened)
        recovered: list[RawInbound] = []

        async def observe_recovery(item: RawInbound) -> bool:
            recovered.append(item)
            return False

        recovery_bus.bind_durable_inbound_recoverer(observe_recovery)
        await recovery_bus.recover_durable_inbounds()
        assert len(recovered) == 1
        item = recovered[0]
        assert (item.provider_identity, item.recipient) == (
            provider_identity,
            recipient,
        )
        assert (item.message.sender, item.message.chat_id) == (
            "display-sender",
            "display-chat",
        )
        assert item.message.content == "durable payload"
        assert item.message.attachments == (attachment,)
        assert item.message.metadata["provider_option"] == "kept"
        assert item.message.metadata["business"] == {"keep": True}
        assert item.message.metadata[DURABLE_PROVIDER_MESSAGE_ID] == raw.message_id
        assert item.message.metadata[DURABLE_HANDOFF_ID] == "identity-handoff"
        assert HANDOFF_PROVIDER_IDENTITY_KEY not in item.message.metadata
        assert _raw_handoff_row(reopened.inbound_store, "identity-handoff") == stored_row
        assert reopened.inbound_store.list_inbound_handoffs()[0]["handoff_id"] == (
            "identity-handoff"
        )
        assert recovery_bus._recovery_claimed == set()
    finally:
        try:
            if recovery_bus is not None:
                await recovery_bus.aclose()
        finally:
            reopened.close()


@pytest.mark.asyncio
async def test_old_mobile_handoff_read_and_recovery_keep_legacy_projection_only(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = "identity-test:legacy"
    manager = SessionManager(workspace)
    bus: MessageBus | None = None
    try:
        manager.save(manager.get_or_create(session_key))
        raw = _durable_raw(
            session_key=session_key,
            handoff_id="legacy-handoff",
            message_id="legacy-provider-message",
            provider_identity=None,
            recipient=None,
        )
        legacy_metadata = _legacy_mobile_metadata(raw)
        store = manager.inbound_store
        assert _insert_handoff(
            store,
            raw,
            legacy_metadata,
            dedupe_key=f"mobile:{session_key}:{raw.message_id}",
        ) == ("legacy-handoff", True)
        row_before = _raw_handoff_row(store, "legacy-handoff")
        metadata_before = cast(str, row_before[-2]).encode("utf-8")
        bus = await _durable_bus(manager)
        recovered: list[RawInbound] = []

        async def observe_recovery(item: RawInbound) -> bool:
            recovered.append(item)
            return False

        bus.bind_durable_inbound_recoverer(observe_recovery)
        projected = store.read_inbound_handoff(
            channel=raw.message.channel,
            session_key=session_key,
            provider_message_id=raw.message_id,
        )
        assert projected is not None
        projected_metadata = json.loads(cast(str, projected["metadata_json"]))
        assert projected_metadata[DURABLE_INBOUND_MARKER] is True
        assert projected_metadata[DURABLE_HANDOFF_ID] == "legacy-handoff"
        assert projected_metadata[DURABLE_PROVIDER_MESSAGE_ID] == raw.message_id
        assert projected_metadata[DURABLE_ATTACHMENT_REFS] == []
        assert HANDOFF_PROVIDER_IDENTITY_KEY not in projected_metadata
        assert cast(str, _raw_handoff_row(store, "legacy-handoff")[-2]).encode(
            "utf-8"
        ) == metadata_before
        assert _raw_handoff_row(store, "legacy-handoff") == row_before

        await bus.recover_durable_inbounds()
        assert len(recovered) == 1
        assert (recovered[0].provider_identity, recovered[0].recipient) == (
            raw.message.sender,
            raw.message.chat_id,
        )
        assert HANDOFF_PROVIDER_IDENTITY_KEY not in recovered[0].message.metadata
        assert cast(str, _raw_handoff_row(store, "legacy-handoff")[-2]).encode(
            "utf-8"
        ) == metadata_before
        assert _raw_handoff_row(store, "legacy-handoff") == row_before
        assert bus._recovery_claimed == set()
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            manager.close()


@pytest.mark.asyncio
async def test_legacy_handoff_repush_is_idempotent_only_for_its_old_identity_pair(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = "identity-test:legacy-repush"
    manager = SessionManager(workspace)
    bus: MessageBus | None = None
    try:
        manager.save(manager.get_or_create(session_key))
        raw = _durable_raw(
            session_key=session_key,
            handoff_id="legacy-repush-handoff",
            message_id="legacy-repush-message",
            provider_identity="display-sender",
            recipient="display-chat",
        )
        _insert_handoff(
            manager.inbound_store,
            raw,
            _legacy_mobile_metadata(raw),
            dedupe_key=f"mobile:{session_key}:{raw.message_id}",
        )
        before = _raw_handoff_row(manager.inbound_store, "legacy-repush-handoff")
        bus = await _durable_bus(manager)
        requested = _durable_raw(
            session_key=session_key,
            handoff_id="legacy-repush-handoff",
            message_id="legacy-repush-message",
            provider_identity="display-sender",
            recipient="display-chat",
            metadata_extra=_legacy_mobile_metadata(raw),
        )
        assert await bus.reserve_durable_inbound(requested) is True
        assert _raw_handoff_row(manager.inbound_store, "legacy-repush-handoff") == before
        assert len(manager.inbound_store.list_inbound_handoffs()) == 1
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            manager.close()


@pytest.mark.parametrize(
    "change",
    ("different-pair", "explicit-none-pair", "business-metadata", "content"),
)
@pytest.mark.asyncio
async def test_legacy_handoff_repush_rejects_changed_identity(
    tmp_path: Path,
    change: str,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = f"identity-test:legacy-conflict:{change}"
    manager = SessionManager(workspace)
    bus: MessageBus | None = None
    try:
        manager.save(manager.get_or_create(session_key))
        original = _durable_raw(
            session_key=session_key,
            handoff_id="legacy-conflict-handoff",
            message_id="legacy-conflict-message",
            provider_identity=None,
            recipient=None,
        )
        _insert_handoff(
            manager.inbound_store,
            original,
            _legacy_mobile_metadata(original),
            dedupe_key=f"mobile:{session_key}:{original.message_id}",
        )
        before = _raw_handoff_row(manager.inbound_store, "legacy-conflict-handoff")
        pair = {
            "different-pair": ("other-provider", "other-recipient"),
            "explicit-none-pair": (None, None),
            "business-metadata": ("display-sender", "display-chat"),
            "content": ("display-sender", "display-chat"),
        }[change]
        extras = _legacy_mobile_metadata(original)
        if change == "business-metadata":
            extras["business"] = {"changed": True}
        content = "changed payload" if change == "content" else "durable payload"
        requested = _durable_raw(
            session_key=session_key,
            handoff_id="legacy-conflict-handoff",
            message_id="legacy-conflict-message",
            provider_identity=pair[0],
            recipient=pair[1],
            content=content,
            metadata_extra=extras,
        )
        bus = await _durable_bus(manager)
        with pytest.raises(RuntimeError, match="inbound handoff identity conflict"):
            await bus.reserve_durable_inbound(requested)
        assert _raw_handoff_row(manager.inbound_store, "legacy-conflict-handoff") == before
        assert bus._durable_admissions == {}
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            manager.close()


@pytest.mark.asyncio
async def test_new_handoff_repush_rejects_changed_provider_identity_pair(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = "identity-test:new-conflict"
    manager = SessionManager(workspace)
    bus: MessageBus | None = None
    try:
        manager.save(manager.get_or_create(session_key))
        original = _durable_raw(
            session_key=session_key,
            handoff_id="new-conflict-handoff",
            message_id="new-conflict-message",
            provider_identity="provider-original",
            recipient="recipient-original",
        )
        bus = await _durable_bus(manager)
        assert await bus.reserve_durable_inbound(original) is True
        before = _raw_handoff_row(manager.inbound_store, "new-conflict-handoff")
        changed = _durable_raw(
            session_key=session_key,
            handoff_id="new-conflict-handoff",
            message_id="new-conflict-message",
            provider_identity="provider-changed",
            recipient="recipient-original",
        )
        with pytest.raises(RuntimeError, match="inbound handoff identity conflict"):
            await bus.reserve_durable_inbound(changed)
        assert _raw_handoff_row(manager.inbound_store, "new-conflict-handoff") == before
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            manager.close()


@pytest.mark.asyncio
async def test_business_metadata_cannot_forge_reserved_handoff_identity(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    session_key = "identity-test:reserved-key"
    manager = SessionManager(workspace)
    bus: MessageBus | None = None
    try:
        manager.save(manager.get_or_create(session_key))
        raw = _durable_raw(
            session_key=session_key,
            handoff_id="reserved-key-handoff",
            message_id="reserved-key-message",
            provider_identity="provider",
            recipient="recipient",
            metadata_extra={
                HANDOFF_PROVIDER_IDENTITY_KEY: {
                    "version": 1,
                    "provider_identity": "forged",
                    "recipient": "forged-recipient",
                }
            },
        )
        original_metadata = dict(raw.message.metadata)
        bus = await _durable_bus(manager)
        with pytest.raises(ValueError, match="reserved identity key"):
            await bus.reserve_durable_inbound(raw)
        assert dict(raw.message.metadata) == original_metadata
        assert manager.inbound_store.list_inbound_handoffs() == []
        assert bus._durable_admissions == {}
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            manager.close()


@pytest.mark.parametrize(
    "bad_identity",
    (
        {"version": 2, "provider_identity": "provider", "recipient": "recipient"},
        {"version": 1, "provider_identity": "provider"},
        {"version": True, "provider_identity": "provider", "recipient": "recipient"},
        {"version": 1, "provider_identity": 5, "recipient": "recipient"},
        {"version": 1, "provider_identity": None, "recipient": "recipient"},
        {"version": 1, "provider_identity": "", "recipient": "recipient"},
    ),
    ids=("unknown-version", "missing-member", "bool-version", "wrong-type", "partial-pair", "invalid-text"),
)
@pytest.mark.asyncio
async def test_corrupt_new_identity_handoff_fails_without_consuming_row(
    tmp_path: Path,
    bad_identity: dict[str, JsonValue],
) -> None:
    store = InboundHandoffStore(tmp_path / "sessions.db")
    bus: MessageBus | None = None
    try:
        bus = MessageBus()
        bus.bind_durable_inbound_store(store)
        raw = _durable_raw(
            session_key="identity-test:corrupt",
            handoff_id="corrupt-identity-handoff",
            message_id="corrupt-identity-message",
            provider_identity="provider",
            recipient="recipient",
        )
        metadata: dict[str, JsonValue] = dict(raw.message.metadata)
        metadata[DURABLE_ATTACHMENT_REFS] = ()
        metadata[HANDOFF_PROVIDER_IDENTITY_KEY] = {
            "version": 1,
            "provider_identity": "provider",
            "recipient": "recipient",
        }
        _insert_handoff(store, raw, metadata)
        row = _raw_handoff_row(store, "corrupt-identity-handoff")
        corrupted = json.loads(cast(str, row[-2]))
        corrupted[HANDOFF_PROVIDER_IDENTITY_KEY] = bad_identity
        store._conn.execute(
            "UPDATE inbound_handoffs SET metadata_json = ? WHERE handoff_id = ?",
            (
                json.dumps(corrupted, separators=(",", ":")),
                "corrupt-identity-handoff",
            ),
        )
        store._conn.commit()
        before = _raw_handoff_row(store, "corrupt-identity-handoff")
        recover_calls: list[RawInbound] = []

        async def observe_recovery(item: RawInbound) -> bool:
            recover_calls.append(item)
            return True

        bus.bind_durable_inbound_recoverer(observe_recovery)
        with pytest.raises(ValueError):
            await bus.recover_durable_inbounds()
        assert recover_calls == []
        assert _raw_handoff_row(store, "corrupt-identity-handoff") == before
        assert bus._recovery_claimed == set()
    finally:
        try:
            if bus is not None:
                await bus.aclose()
        finally:
            store.close()


@pytest.mark.parametrize("dedupe_kind", ("null", "legacy"))
@pytest.mark.asyncio
async def test_corrupt_identity_cannot_disappear_from_legacy_lookup_or_repush(
    tmp_path: Path,
    dedupe_kind: str,
) -> None:
    store = InboundHandoffStore(tmp_path / "sessions.db")
    try:
        session_key = f"identity-test:corrupt-lookup:{dedupe_kind}"
        original = _durable_raw(
            session_key=session_key,
            handoff_id="corrupt-lookup-handoff",
            message_id="corrupt-lookup-message",
            provider_identity=None,
            recipient=None,
        )
        dedupe_key = (
            None
            if dedupe_kind == "null"
            else f"mobile:{session_key}:{original.message_id}"
        )
        _insert_handoff(
            store,
            original,
            _legacy_mobile_metadata(original),
            dedupe_key=dedupe_key,
        )
        before = _raw_handoff_row(store, "corrupt-lookup-handoff")
        corrupted = json.loads(cast(str, before[-2]))
        corrupted[HANDOFF_PROVIDER_IDENTITY_KEY] = {
            "version": 2,
            "provider_identity": "display-sender",
            "recipient": "display-chat",
        }
        store._conn.execute(
            "UPDATE inbound_handoffs SET metadata_json = ? WHERE handoff_id = ?",
            (
                json.dumps(corrupted, ensure_ascii=False, separators=(",", ":")),
                "corrupt-lookup-handoff",
            ),
        )
        store._conn.commit()
        corrupted_before = _raw_handoff_row(store, "corrupt-lookup-handoff")

        with pytest.raises(ValueError, match="version unsupported"):
            store.read_inbound_handoff(
                channel=original.message.channel,
                session_key=session_key,
                provider_message_id=original.message_id,
            )
        assert _raw_handoff_row(store, "corrupt-lookup-handoff") == corrupted_before

        retry = _durable_raw(
            session_key=session_key,
            handoff_id="corrupt-lookup-retry",
            message_id=original.message_id,
            provider_identity=original.message.sender,
            recipient=original.message.chat_id,
        )
        retry_metadata: dict[str, JsonValue] = dict(retry.message.metadata)
        retry_metadata[HANDOFF_PROVIDER_IDENTITY_KEY] = {
            "version": 1,
            "provider_identity": retry.provider_identity,
            "recipient": retry.recipient,
        }
        retry_metadata[DURABLE_ATTACHMENT_REFS] = ()
        with pytest.raises(ValueError, match="version unsupported"):
            _insert_handoff(
                store,
                retry,
                retry_metadata,
                dedupe_key=(
                    f"{retry.message.channel}:{session_key}:{retry.message_id}"
                ),
            )

        rows = store._conn.execute(
            "SELECT COUNT(*) FROM inbound_handoffs"
        ).fetchone()
        assert rows is not None and rows[0] == 1
        assert _raw_handoff_row(store, "corrupt-lookup-handoff") == corrupted_before
    finally:
        store.close()


@pytest.mark.asyncio
async def test_reserve_rejects_unknown_identity_version_without_inserting_row(
    tmp_path: Path,
) -> None:
    store = InboundHandoffStore(tmp_path / "sessions.db")
    try:
        raw = _durable_raw(
            session_key="identity-test:corrupt-request",
            handoff_id="corrupt-request-handoff",
            message_id="corrupt-request-message",
            provider_identity="provider",
            recipient="recipient",
        )
        metadata: dict[str, JsonValue] = dict(raw.message.metadata)
        metadata[HANDOFF_PROVIDER_IDENTITY_KEY] = {
            "version": 2,
            "provider_identity": "provider",
            "recipient": "recipient",
        }
        metadata[DURABLE_ATTACHMENT_REFS] = ()
        with pytest.raises(ValueError, match="version unsupported"):
            _insert_handoff(store, raw, metadata)
        assert store.list_inbound_handoffs() == []
    finally:
        store.close()
