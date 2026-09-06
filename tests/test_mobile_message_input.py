from types import SimpleNamespace
import asyncio
import json
import hashlib
import shutil
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent.plugins.manager import PluginManager
from bootstrap.core_channel_adapter import build_core_channel_definition
from bus.event_bus import EventBus
from bus.queue import MessageBus
from infra.channels.base import AttachmentStore
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from infra.mobile_realtime.attachments import AttachmentChunk
from session.artifact_store import ArtifactStore
from infra.mobile_realtime.channel import MobileRealtimeChannel, _command_hash
from infra.mobile_realtime.protocol import MessageSendCommand
from infra.mobile_realtime.storage import MobileRealtimeStorage
from session.admissions import SessionAdmissions
from session.identities import ChannelIdentities
from session.inbound_store import InboundHandoffStore
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Control, Input, Output
from tests.mobile_realtime.test_channel import _Runtime, _register_device


def command(session, number=0, **payload):
    identity = f'01ARZ3NDEKTSV4RRFFQ69G5{number:03d}'
    return MessageSendCommand.model_validate({
        'v': 1, 'kind': 'command', 'type': 'message.send', 'id': identity,
        'connection_epoch': 1, 'session_id': session,
        'payload': {'message_log_version': 2, 'client_message_id': identity,
                    'session_id': session, 'text': '原始文字', 'media_refs': [],
                    'client_created_at': '2026-09-06T00:00:00Z', **payload},
    })


@asynccontextmanager
async def runtime(tmp_path, *, device=None, store_type=InboundHandoffStore):
    """每次重开同库，使用真实 MessageBus、conversation 与 Channel binding。"""
    source = tmp_path / 'plugins'
    if not (source / "conversation").exists():
        shutil.copytree(Path(__file__).parents[1] / 'plugins/conversation', source / 'conversation',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(Path(__file__).parents[1] / 'plugins/sources', source / 'sources',
                        ignore=shutil.ignore_patterns('__pycache__'))
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    db = workspace / 'sessions.db'
    log = MessageLog(db)
    identities = ChannelIdentities(db)
    artifacts = ArtifactStore(db)
    physical = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=artifacts)
    admissions = SessionAdmissions(db)
    admissions.clear_stale()
    handoffs = store_type(db)
    storage = MobileRealtimeStorage(tmp_path / 'mobile.db')
    if device is None:
        device = uuid4().hex
        _register_device(storage, device)
    bus = MessageBus()
    bus.bind_durable_inbound_store(handoffs)
    bus.bind_mobile_session_admission_owner(admissions)
    channel = MobileRealtimeChannel(_Runtime(storage))
    channel.bind_messages(log.catalog())
    channel.bind_channel_attachment_store(physical)
    manager = PluginManager([source], event_bus=EventBus(), workspace=workspace,
        message_log=log, channel_identities=identities, channel_attachment_store=physical, installed_cache_root=tmp_path / 'cache')
    manager.channel_generation_host.bind_input_custody(bus)
    try:
        await channel.start(SimpleNamespace(bus=bus, attachment_store=AttachmentStore(tmp_path / 'uploads')))
        await manager.load_all()
        await manager.bind_core_channel_definitions((build_core_channel_definition(channel),))
        yield log, identities, manager, bus, channel, storage, device, handoffs
    finally:
        await manager.terminate_all()
        await channel.stop()
        await bus.aclose()
        for store in (log, identities, admissions, handoffs, storage, artifacts):
            store.close()


def append(log, session, identity, body, source='conversation'):
    return log.writer(session, author='程序作者', source=source, body_types=(type(body),),
                      content={'text': lambda _: ContentReferences()}).append(identity, body)


@pytest.mark.asyncio
async def test_mobile_input_and_reference_commit_original_facts_and_replay_once(tmp_path):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        target = append(log, session, 'target', Output((ContentPart('text', '引用原文'),), 'complete'), 'scheduler')
        frame = command(session, reply_to={'message_id': target.message_id})
        reply = await channel.handle_command(device_id=device, frame=frame)
        assert reply.type == 'message.send.ok'
        message = log.reader(session).get(frame.id)
        assert isinstance(message.body, Input)
        assert [(p.kind, p.value) for p in message.body.parts][1:] == [('text', '原始文字'), ('reply_ref', 'target')]
        assert not handoffs.list_inbound_handoffs()
        assert bus.inbound_size == 0
        assert (await channel.handle_command(device_id=device, frame=frame)).replayed
        assert len(log.reader(session).snapshot()) == 2
        assert manager.current_snapshot.lease_count == 0
        assert identities.load('akashic')


@pytest.mark.asyncio
async def test_mobile_rejects_invalid_targets_without_leaving_recovery_rows(tmp_path):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        append(log, session, 'first', Input(()))
        append(log, session, 'control', Control('pause', 0))
        append(log, f'akashic:{uuid4()}', 'elsewhere', Input(()))
        before = log.reader(session).snapshot()
        for i, target in enumerate(('missing', 'control', 'elsewhere')):
            frame = command(session, i, reply_to={'message_id': target})
            reply = await channel.handle_command(device_id=device, frame=frame)
            assert reply.type == 'message.send.error' and reply.payload['code'] == 'message_conflict'
            assert not handoffs.list_inbound_handoffs()
            again = await channel.handle_command(device_id=device, frame=frame)
            assert again.replayed and again.payload == reply.payload
        assert log.reader(session).snapshot() == before
        await bus.recover_durable_inbounds()
        assert log.reader(session).snapshot() == before


@pytest.mark.asyncio
async def test_mobile_pause_retry_and_recovery_use_controls_without_input_copy(tmp_path, monkeypatch):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        first = command(session)
        assert (await channel.handle_command(device_id=device, frame=first)).type == 'message.send.ok'
        stop = command(session, 1, text='/stop')
        assert (await channel.handle_command(device_id=device, frame=stop)).type == 'message.send.ok'
        retry = command(session, 2, retry_of_client_message_id=first.id)
        original = storage.complete_command
        def crash(**kwargs):
            if kwargs['command_id'] == retry.id:
                raise OSError('receipt write interrupted')
            return original(**kwargs)
        monkeypatch.setattr(storage, 'complete_command', crash)
        with pytest.raises(OSError, match='receipt write interrupted'):
            await channel.handle_command(device_id=device, frame=retry)
        facts = log.reader(session).snapshot()
        assert sum(isinstance(m.body, Input) for m in facts) == 1
        assert [m.body.action for m in facts if isinstance(m.body, Control)] == ['pause', 'resume']
        assert not handoffs.list_inbound_handoffs()
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        reply = await channel.handle_command(device_id=device, frame=retry)
        assert reply.type == 'message.send.ok' and reply.replayed
        assert log.reader(session).snapshot() == facts


class DeleteFailure(InboundHandoffStore):
    def complete_inbound_handoff(self, handoff_id):
        raise OSError('handoff delete interrupted')


@pytest.mark.asyncio
async def test_durable_rejection_survives_crash_before_handoff_cleanup(tmp_path):
    async with runtime(tmp_path, store_type=DeleteFailure) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        frame = command(session, reply_to={'message_id': 'later'})
        with pytest.raises(OSError, match='handoff delete interrupted'):
            await channel.handle_command(device_id=device, frame=frame)
        assert storage.read_command(device_id=device, command_id=frame.id).reply_type == 'message.send.error'
        assert len(handoffs.list_inbound_handoffs()) == 1
        # 即使目标随后变有效，已经拒绝的命令也不能恢复执行。
        append(log, session, 'later', Input(()))
        before = log.reader(session).snapshot()
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        await bus.recover_durable_inbounds()
        assert not handoffs.list_inbound_handoffs()
        assert log.reader(session).snapshot() == before
        reply = await channel.handle_command(device_id=device, frame=frame)
        assert reply.replayed and reply.payload['code'] == 'message_conflict'


def test_mobile_send_requires_new_protocol_and_one_real_message_identity():
    session = f'akashic:{uuid4()}'
    data = command(session).model_dump(mode='json')
    del data['payload']['message_log_version']
    with pytest.raises(ValidationError):
        MessageSendCommand.model_validate(data)
    for old in ({'client_message_id': command(session).id}, {'message_id': 'id', 'role': 'user'}):
        with pytest.raises(ValidationError):
            command(session, reply_to=old)


def upload(channel, device, session, identity):
    data = b'original upload survives rejection'
    transfers = channel._require_attachments()
    transfers.begin_upload(device_id=device, attachment_id=identity, session_id=session,
        filename='original.txt', content_type='text/plain', size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    transfers.append_chunk(device_id=device, chunk=AttachmentChunk(identity, 0, data))
    transfers.finish_upload(device_id=device, session_id=session, attachment_id=identity)
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize('attachment', [False, True])
async def test_rejected_first_input_has_no_session_or_claim_and_retains_attachment_evidence(tmp_path, attachment):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        upload_id = command(session, 100).id
        data = upload(channel, device, session, upload_id) if attachment else b''
        frame = command(session, reply_to={'message_id': 'missing'}, media_refs=[upload_id] if attachment else [])
        reply = await channel.handle_command(device_id=device, frame=frame)
        assert reply.payload['code'] == 'message_conflict'
        assert session not in log.catalog().snapshot_heads()
        assert not storage.has_session_claim(session)
        assert not log.reader(session).snapshot()
        assert not handoffs.list_inbound_handoffs()
        assert not storage.list_incomplete_attachment_imports()
        if attachment:
            mapping, = storage.list_attachment_imports(session_id=session, client_message_id=frame.id)
            assert mapping.phase == 'rejected' and 'message_conflict' in mapping.error
            assert Path(storage.read_attachment(upload_id).local_path).read_bytes() == data
            ref = channel._channel_attachment_store._metadata_store.get_attachment(mapping.artifact_id).ref
            lease = await channel._channel_attachment_store.acquire(ref)
            assert await lease.read_bytes(max_bytes=100) == data
            await lease.aclose()
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        await bus.recover_durable_inbounds()
        assert not log.reader(session).snapshot() and not storage.has_session_claim(session)
        assert not storage.list_incomplete_attachment_imports()


@pytest.mark.asyncio
async def test_mobile_attachment_is_bound_to_real_input_and_survives_reopen(tmp_path):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        upload_id = command(session, 100).id
        data = upload(channel, device, session, upload_id)
        frame = command(session, media_refs=[upload_id])
        reply = await channel.handle_command(device_id=device, frame=frame)
        assert reply.type == 'message.send.ok'
        mapping, = storage.list_attachment_imports(session_id=session, client_message_id=frame.id)
        message = log.reader(session).get(frame.id)
        assert message.body.parts[-1] == ContentPart('artifact_ref', mapping.artifact_id)
        assert mapping.phase == 'message_bound'
        assert not handoffs.list_inbound_handoffs()
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        assert not storage.list_incomplete_attachment_imports()
        assert Path(storage.read_attachment(upload_id).local_path).read_bytes() == data
        assert (await channel.handle_command(device_id=device, frame=frame)).replayed
        assert len(log.reader(session).snapshot()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('valid', [False, True])
async def test_cancelled_input_restarts_through_current_binding_and_uses_final_receipt(tmp_path, monkeypatch, valid):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        frame = command(session, **({} if valid else {'reply_to': {'message_id': 'missing'}}))
        entered, release = asyncio.Event(), asyncio.Event()
        original = bus.prepare_channel_input
        async def held(envelope):
            await original(envelope)
            entered.set()
            await release.wait()
        monkeypatch.setattr(bus, 'prepare_channel_input', held)
        send = asyncio.create_task(channel.handle_command(device_id=device, frame=frame))
        await asyncio.wait_for(entered.wait(), 3)
        assert len(handoffs.list_inbound_handoffs()) == 1
        observed = await channel.handle_command(device_id=device, frame=frame)
        assert observed.payload['code'] == 'command_in_progress'
        await bus.recover_durable_inbounds()
        assert not log.reader(session).snapshot()
        send.cancel()
        with pytest.raises(asyncio.CancelledError):
            await send
        assert len(handoffs.list_inbound_handoffs()) == 1
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        await bus.recover_durable_inbounds()
        assert not handoffs.list_inbound_handoffs()
        reply = await channel.handle_command(device_id=device, frame=frame)
        assert reply.replayed
        assert reply.type == ('message.send.ok' if valid else 'message.send.error')
        assert len(log.reader(session).snapshot()) == int(valid)
        assert storage.has_session_claim(session) is valid
        assert manager.current_snapshot.lease_count == 0


@pytest.mark.asyncio
async def test_rejection_cannot_expire_while_cross_database_cleanup_is_pending(tmp_path, monkeypatch):
    async with runtime(tmp_path, store_type=DeleteFailure) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        frame = command(session, reply_to={'message_id': 'later'})
        with pytest.raises(OSError):
            await channel.handle_command(device_id=device, frame=frame)
        future = datetime.now(UTC) + timedelta(days=20)
        assert storage.cleanup_command_receipts(device_id=device, now=future) == 0
        assert storage.read_command(device_id=device, command_id=frame.id).status == 'completed'
        monkeypatch.setattr('infra.mobile_realtime.channel._utc_now', lambda: future)
        append(log, session, 'later', Input(()))
        with pytest.raises(OSError):
            await channel.handle_command(device_id=device, frame=frame)
        assert len(log.reader(session).snapshot()) == 1
        assert len(storage.pending_message_rejections()) == 1
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        assert not handoffs.list_inbound_handoffs()
        assert not storage.pending_message_rejections()
        assert len(log.reader(session).snapshot()) == 1
        assert storage.cleanup_command_receipts(device_id=device, now=future) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["after_reserve", "after_rejection"])
async def test_existing_message_id_cannot_turn_a_different_failed_request_into_success(tmp_path, monkeypatch, stage):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        frame = command(session, text='另一个请求')
        append(log, session, frame.id, Input((ContentPart('text', '原始已提交事实'),)))
        before = log.reader(session).snapshot()
        if stage == "after_reserve":
            storage.reserve_command(device_id=device, command_id=frame.id, command_type=frame.type,
                                    request_hash=_command_hash(frame), created_at=datetime.now(UTC))
        else:
            original = storage.complete_command
            def crash(**kwargs):
                if kwargs['reply_type'] == 'message.send.error':
                    raise OSError('rejection receipt interrupted')
                return original(**kwargs)
            monkeypatch.setattr(storage, 'complete_command', crash)
            with pytest.raises(OSError, match='rejection receipt interrupted'):
                await channel.handle_command(device_id=device, frame=frame)
        assert storage.read_command(device_id=device, command_id=frame.id).status == 'processing'
        assert not storage.has_session_claim(session)
    async with runtime(tmp_path, device=device) as (log, identities, manager, bus, channel, storage, device, handoffs):
        if stage == "after_rejection":
            observed = await channel.handle_command(device_id=device, frame=frame)
            assert observed.payload['code'] == 'command_in_progress'
            await bus.recover_durable_inbounds()
        reply = await channel.handle_command(device_id=device, frame=frame)
        assert reply.replayed and reply.payload['code'] == 'message_conflict'
        assert log.reader(session).snapshot() == before
        assert not storage.has_session_claim(session)
        assert not handoffs.list_inbound_handoffs()


@pytest.mark.asyncio
async def test_claimed_missing_session_is_rejected_at_admission_without_recreation(tmp_path):
    async with runtime(tmp_path) as (log, identities, manager, bus, channel, storage, device, handoffs):
        session = f'akashic:{uuid4()}'
        storage.claim_session(device_id=device, session_id=session, created_at=datetime.now(UTC))
        reply = await channel.handle_command(device_id=device, frame=command(session))
        assert reply.payload['code'] == 'session_not_found'
        assert session not in log.catalog().snapshot_heads()
        assert not handoffs.list_inbound_handoffs()
        assert not storage.pending_message_rejections()
