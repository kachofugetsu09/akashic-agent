import asyncio
import base64
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from agent.config_models import MobileRealtimeConfig
from agent.plugin_composition.tasks import Tasks
from agent.plugins.snapshot import RuntimeSnapshotStore
from bootstrap.reply_status import RuntimeReplyStatus
from infra.channels.message_view import message_rows
from infra.mobile_realtime.auth import DeviceAuthenticator, device_proof_signing_bytes
from infra.mobile_realtime.channel import MobileRealtimeChannel
from infra.mobile_realtime.gateway import MobileGatewayRuntime, PairingApprovalRegistry, create_mobile_gateway_app
from infra.mobile_realtime.inbox import DurableInboxManager
from infra.mobile_realtime.key_protection import FileMasterKeyStore, KeysetManager
from infra.mobile_realtime.message_view import bounded_reply_status, message_json
from infra.mobile_realtime.pairing import PairingService
from infra.mobile_realtime.storage import DeviceRecord, MobileRealtimeStorage
from plugins.reply.status import ReplyState
from session.log import MessageLog
from session.message import ContentPart, Input, Output, Control
from tests.test_message_follow import status_root
from tests.test_mobile_message_log import append
from tests.test_message_log_migration import snapshot


@pytest.fixture
def gateway(tmp_path):
    loop = asyncio.new_event_loop()
    with closing(MessageLog(tmp_path / 'sessions.db')) as log, closing(MobileRealtimeStorage(tmp_path / 'mobile.db')) as storage:
        keyset = KeysetManager(tmp_path / 'keys', FileMasterKeyStore(tmp_path / 'master.json')).initialize(lan_hostname='localhost')
        private = ec.generate_private_key(ec.SECP256R1())
        public = base64.b64encode(private.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)).decode()
        device = uuid4().hex
        storage.register_device(DeviceRecord(device, public, 'fixture', datetime.now(timezone.utc), None, ()))
        runtime = MobileGatewayRuntime(config=MobileRealtimeConfig(), storage=storage,
            pairing=PairingService(storage, keyset, lan_endpoints=(), tunnel_endpoints=()),
            authenticator=DeviceAuthenticator(storage, keyset), inbox=DurableInboxManager(storage),
            approvals=PairingApprovalRegistry(loop), keyset=keyset)
        channel = MobileRealtimeChannel(runtime)
        channel.bind_messages(log.catalog())
        runtime.bind_channel(channel)
        with TestClient(create_mobile_gateway_app(runtime)) as client:
            yield log, runtime, client, device, private
        assert not runtime._message_followers and not log._listeners
    loop.close()


@contextmanager
def connected(gateway):
    log, runtime, client, device, private = gateway
    with client.websocket_connect('/ws') as ws:
        challenge = ws.receive_json()['payload']
        nonce = uuid4().hex
        signature = private.sign(device_proof_signing_bytes(server_id=challenge['server_id'],
            challenge_id=challenge['challenge_id'], challenge_nonce=challenge['nonce'],
            device_id=device, client_nonce=nonce), ec.ECDSA(hashes.SHA256()))
        ws.send_json({'v': 1, 'kind': 'control', 'type': 'device.proof', 'payload': {
            'challenge_id': challenge['challenge_id'], 'device_id': device,
            'client_nonce': nonce, 'signature': base64.b64encode(signature).decode()}})
        accepted = ws.receive_json()
        assert accepted['type'] == 'auth.accepted'
        epoch = accepted['connection_epoch']
        replay_through = runtime.storage.read_cursor(device).next_event_seq - 1
        ws.send_json({'v': 1, 'kind': 'control', 'type': 'resume', 'connection_epoch': epoch,
                      'payload': {'last_ack': 0, 'active_turns': []}})
        while True:
            frame = ws.receive_json()
            if frame['type'] == 'sync.completed' and frame['event_seq'] > replay_through:
                break
        yield ws, epoch


def follow(ws, epoch, session, after_seq, **fields):
    identity = '01ARZ3NDEKTSV4RRFFQ69G5FAV'
    ws.send_json({'v': 1, 'kind': 'command', 'type': 'session.follow', 'id': identity,
        'connection_epoch': epoch, 'session_id': session,
        'payload': {'message_log_version': 2, 'after_seq': after_seq, **fields}})
    reply = ws.receive_json()
    assert reply['type'] == 'session.follow.ok', reply
    return identity


def receive(ws, kind):
    while True:
        wire = ws.receive_json()
        assert wire['kind'] == 'control' and wire['type'] == 'session.message', wire
        if wire['payload']['type'] == kind:
            return wire['payload']
        assert wire['payload']['type'] == 'reply.status'


def test_authenticated_follow_splits_every_row_then_switches_reconnects_without_inbox_writes(gateway, tmp_path):
    log, runtime, client, device, private = gateway
    session, other = f'akashic:{uuid4()}', f'akashic:{uuid4()}'
    for i in range(15):
        append(log, session, str(i), Input((ContentPart('text', 'x' * 60000),)))
    append(log, session, 'large', Control('pause', 14, '🪷' * 100000))
    before = snapshot(tmp_path / 'sessions.db')
    with connected(gateway) as (ws, epoch):
        cursor = runtime.storage.read_cursor(device)
        identity = follow(ws, epoch, session, -1)
        pages = []
        while not pages or pages[-1]['has_more']:
            pages.append(receive(ws, 'messages.appended'))
        assert [page['after_seq'] for page in pages] == [-1] + [p['next_after_seq'] for p in pages[:-1]]
        assert len(pages) > 1 and all(p['through_seq'] == 15 for p in pages)
        rows = [row for page in pages for row in page['items']]
        expected = message_rows(log.reader(session).read_page(limit=200))
        assert rows[:-1] == expected[:-1]
        assert rows[-1]['message_ref']['byte_length'] == len(message_json(expected[-1]))
        assert runtime.storage.read_cursor(device) == cursor
        assert snapshot(tmp_path / 'sessions.db') == before
        with closing(sqlite3.connect(tmp_path / 'mobile.db')) as db:
            assert db.execute('select count(*) from mobile_command_receipts where command_id=?', (identity,)).fetchone() == (0,)
        append(log, session, 'late', Input(()))
        assert [row['id'] for row in receive(ws, 'messages.appended')['items']] == ['late']
        follow(ws, epoch, other, -1)
        assert not receive(ws, 'reply.status')['available']
        append(log, session, 'old-session', Input(()))
        append(log, other, 'other', Input(()))
        assert [row['id'] for row in receive(ws, 'messages.appended')['items']] == ['other']
    with connected(gateway) as (ws, epoch):
        follow(ws, epoch, session, 16)
        assert [row['id'] for row in receive(ws, 'messages.appended')['items']] == ['old-session']


def test_preview_large_unicode_and_generation_switch_remain_ephemeral(gateway, tmp_path):
    log, runtime, client, device, private = gateway
    session = f'akashic:{uuid4()}'
    state, tasks, store = ReplyState(), Tasks(), RuntimeSnapshotStore()
    root, first = client.portal.call(status_root, 'mobile-old', state)
    new_root, second = client.portal.call(status_root, 'mobile-new', ReplyState())
    store.install(first)
    runtime.channel.reply_status = RuntimeReplyStatus(store).follow
    entered, release = asyncio.Event(), asyncio.Event()
    full = '🪷完整草稿' * 100000
    async def operation(task):
        with state.open(task, session, 'conversation') as preview:
            with preview('answer') as delta:
                await delta({'content_delta': full, 'thinking_delta': full})
                entered.set()
                await release.wait()
                append(log, session, 'answer', Output((ContentPart('text', full),), 'complete'))
    async def start():
        task = await tasks.admit(session, lambda slot: slot.start(operation))
        await entered.wait()
        return task
    task = client.portal.call(start)
    try:
        with connected(gateway) as (ws, epoch):
            cursor = runtime.storage.read_cursor(device)
            follow(ws, epoch, session, -1)
            status = receive(ws, 'reply.status')
            preview = status['items'][0]['preview']
            assert preview['truncated'] and full.startswith(preview['text']) and full.startswith(preview['thinking'])
            assert len(message_json(status)) <= 240 * 1024
            assert log.reader(session).get('answer') is None and first.lease_count == 0
            client.portal.call(release.set)
            page = receive(ws, 'messages.appended')
            assert page['items'][0]['id'] == 'answer' and 'message_ref' in page['items'][0]
            assert log.reader(session).get('answer').body.parts[0].value == full
            async def promote():
                await store.commit(store.begin_publish(second))
                await store.wait_for_snapshot_drained(first)
            client.portal.call(promote)
            while receive(ws, 'reply.status')['snapshot_id'] != second.snapshot_id:
                pass
            assert runtime.storage.read_cursor(device) == cursor
    finally:
        client.portal.call(release.set)
        client.portal.call(task.join)
        client.portal.call(tasks.close)
        client.portal.call(store.close)
        client.portal.call(root.dispose)
        client.portal.call(new_root.dispose)


def test_runtime_stop_waits_for_status_reader_cleanup(gateway):
    log, runtime, client, device, private = gateway
    session = f'akashic:{uuid4()}'
    closing_started, release = asyncio.Event(), asyncio.Event()
    async def status(session_id):
        try:
            yield {'version': 2, 'session_id': session_id, 'snapshot_id': None, 'available': False, 'items': []}
            await asyncio.Event().wait()
        finally:
            closing_started.set()
            await release.wait()
    runtime.channel.reply_status = status
    with connected(gateway) as (ws, epoch):
        follow(ws, epoch, session, -1)
        receive(ws, 'reply.status')
        stopped = client.portal.start_task_soon(runtime.stop)
        client.portal.call(closing_started.wait)
        assert not stopped.done()
        client.portal.call(release.set)
        stopped.result(timeout=3)
        assert not log._listeners and not runtime._message_followers


@pytest.mark.parametrize('fields', [{'after_seq': True}, {'after_seq': -2}, {'after_seq': 1},
    {'message_log_version': 1}, {'message_log_version': True}, {'unknown': 1}])
def test_follow_rejects_invalid_cursor_or_version_without_subscription(gateway, fields):
    log, runtime, client, device, private = gateway
    session = f'akashic:{uuid4()}'
    with connected(gateway) as (ws, epoch):
        ws.send_json({'v': 1, 'kind': 'command', 'type': 'session.follow', 'id': '01ARZ3NDEKTSV4RRFFQ69G5FAV',
            'connection_epoch': epoch, 'session_id': session,
            'payload': {'message_log_version': 2, 'after_seq': -1, **fields}})
        assert ws.receive_json()['type'] == 'session.follow.error'
        assert not runtime._message_followers and not log._listeners


def test_preview_budget_keeps_all_activity_ids_and_original_values():
    items = [{'session_id': 's', 'source': 'source', 'handle': str(i), 'active': True,
              'preview': {'message_id': str(i), 'text': '🪷' * 100000, 'thinking': '思考' * 100000}} for i in range(8)]
    payload = {'type': 'reply.status', 'version': 2, 'session_id': 's', 'snapshot_id': 'root', 'available': True, 'items': items}
    before = message_json(payload)
    result = bounded_reply_status(payload)
    assert len(message_json(result)) <= 240 * 1024 and message_json(payload) == before
    assert [item['handle'] for item in result['items']] == [str(i) for i in range(8)]
    assert all(item['preview']['truncated'] for item in result['items'])


def test_replacement_closes_only_old_connection_and_stop_rejects_new_sessions(gateway):
    log, runtime, client, device, private = gateway
    session = f'akashic:{uuid4()}'
    append(log, session, 'input', Input(()))
    with connected(gateway) as (old, old_epoch):
        follow(old, old_epoch, session, 0)
        assert receive(old, 'reply.status')['snapshot_id'] is None
        with connected(gateway) as (current, epoch):
            # 同一设备第二条真实鉴权连接替换旧 epoch，旧 reader 已结束。
            assert not runtime._message_followers and not log._listeners
            follow(current, epoch, session, 0)
            receive(current, 'reply.status')
            append(log, session, 'control', Control('pause', 0, None))
            page = receive(current, 'messages.appended')
            assert page['items'][0]['body']['reason'] is None
            assert list(runtime._message_followers) == [runtime._connections[device].websocket]
            client.portal.call(runtime.stop)
            assert not runtime._connections and not log._listeners
    with client.websocket_connect('/ws') as ws:
        closed = ws.receive()
        assert closed['type'] == 'websocket.close' and closed['code'] == 1012


def test_failed_control_send_evicts_and_drains_reader(gateway, monkeypatch):
    log, runtime, client, device, private = gateway
    session = f'akashic:{uuid4()}'
    with connected(gateway) as (ws, epoch):
        follow(ws, epoch, session, -1)
        receive(ws, 'reply.status')
        connection = runtime._connections[device]
        closed = asyncio.Event()
        original_close = connection.websocket.close
        async def fail_send(data):
            raise OSError('controlled disconnected transport')
        async def close(**kwargs):
            await original_close(**kwargs)
            closed.set()
        monkeypatch.setattr(connection.websocket, 'send_text', fail_send)
        monkeypatch.setattr(connection.websocket, 'close', close)
        append(log, session, 'late', Input(()))
        client.portal.call(closed.wait)
        assert not runtime._message_followers and not log._listeners and not runtime._connections


def test_bad_frame_after_follow_keeps_protocol_error_visible_and_releases_reader(gateway):
    log, runtime, client, device, private = gateway
    session = f'akashic:{uuid4()}'
    with connected(gateway) as (ws, epoch):
        follow(ws, epoch, session, -1)
        receive(ws, 'reply.status')
        ws.send_text('{bad json')
        assert ws.receive_json()['type'] == 'protocol.error'
        assert not runtime._message_followers and not log._listeners
