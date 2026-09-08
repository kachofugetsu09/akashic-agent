import asyncio
import hashlib
import json
from typing import cast
from collections import deque
from contextlib import closing
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agent.config_models import MobileRealtimeConfig
from infra.channels.message_view import message_rows
from infra.mobile_realtime.auth import DeviceAuthenticator
from infra.mobile_realtime.channel import MobileCommandError, MobileRealtimeChannel
from fastapi import WebSocket
from starlette.types import Message

from infra.mobile_realtime.gateway import ActiveMobileConnection, MobileGatewayRuntime, PairingApprovalRegistry, create_mobile_gateway_app
from infra.mobile_realtime.inbox import DurableInboxManager
from infra.mobile_realtime.key_protection import FileMasterKeyStore, KeysetManager
from infra.mobile_realtime.pairing import PairingService
from infra.mobile_realtime.storage import MobileRealtimeStorage
from plugins.models.projection import check_facts
from session.log import MessageLog, SessionAttributes
from session.message import CallRef, ContentPart, ContentReferences, Control, Input, Output, ToolCall, ToolResult
from tests.mobile_realtime.test_channel import _Runtime, _generic_frame, _register_device
from tests.test_message_log_migration import snapshot


@pytest.fixture
def mobile(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log, closing(MobileRealtimeStorage(tmp_path / 'mobile.db')) as storage:
        device = uuid4().hex
        _register_device(storage, device)
        runtime = _Runtime(storage)
        channel = MobileRealtimeChannel(cast(MobileGatewayRuntime, runtime))
        channel.bind_messages(log.catalog())
        yield log, runtime, channel, device


def command(kind, session_id=None, **payload):
    return _generic_frame(frame_id='01ARZ3NDEKTSV4RRFFQ69G5FAV', command_type=kind,
                          session_id=session_id, payload={'message_log_version': 2, **payload})


def append(log, session, identity, body, call_ref=None):
    checks = {kind: lambda part: ContentReferences() for kind in ('text', 'history.transcript', 'history.record', 'history.provenance', 'history.turn_input', 'future.private')}
    checks['model.facts'] = check_facts
    return log.writer(session, author='真实作者', source='来源', body_types=(type(body),),
                      content=checks, call_ref=call_ref, check_call=lambda call: None).append(identity, body)


@pytest.mark.asyncio
async def test_mobile_history_reads_full_message_prefix_and_directory_without_old_context(mobile, tmp_path):
    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    empty = f'akashic:{uuid4()}'
    log.ensure_session(empty, SessionAttributes())
    log.ensure_session(f'akashic:{uuid4()}', SessionAttributes(visibility='internal'))
    log.ensure_session(f'web:{uuid4()}', SessionAttributes())
    append(log, session, 'i', Input(()))
    output = append(log, session, 'o', Output((ContentPart('text', '后台任务不能被隐藏'),), 'complete'))
    append(log, session, 'c', Control('abandon', output.seq, '放弃'))
    before = snapshot(tmp_path / 'sessions.db')
    await channel._list_sessions(device, command('session.list', page_size=1))
    first = runtime.events[-1]['payload']
    cursor = first['next_cursor']
    await channel._list_sessions(device, command('session.list', page_size=1, after_time=cursor['updated_at'], after_key=cursor['session_id']))
    second = runtime.events[-1]['payload']
    assert {first['items'][0]['session_id'], second['items'][0]['session_id']} == {session, empty}
    assert first['total'] == second['total'] == 2 and second['next_cursor'] is None
    assert all(item['title'] == '新对话' for page in (first, second) for item in page['items'])
    await channel._get_history(device, command('history.get', session, page_size=2))
    page = runtime.events[-1]['payload']
    assert page['items'] == message_rows(log.reader(session).read_page(limit=2))
    assert page['through_seq'] == 2 and page['next_after_seq'] == 1 and page['has_more']
    assert snapshot(tmp_path / 'sessions.db') == before
    append(log, session, 'later', Input((ContentPart('text', '新增'),)))
    await channel._get_history(device, command('history.get', session, after_seq=1, through_seq=2))
    page = runtime.events[-1]['payload']
    assert [item['id'] for item in page['items']] == ['c'] and not page['has_more']
    assert page['next_after_seq'] == page['through_seq'] == 2
    await channel._get_history(device, command('history.get', empty))
    assert runtime.events[-1]['payload']['items'] == []
    assert runtime.events[-1]['payload']['through_seq'] == -1
    with pytest.raises(MobileCommandError, match='会话不存在'):
        await channel._get_history(device, command('history.get', f'akashic:{uuid4()}'))


@pytest.mark.asyncio
async def test_mobile_tail_pages_keep_latest_manifest_and_load_older_on_request(mobile, tmp_path):
    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    for index in range(12):
        append(log, session, f'm{index}', Input((ContentPart('text', 'x' * 60000),)))
    before = snapshot(tmp_path / 'sessions.db')
    await channel._get_history(device, command('history.get', session, direction='backward', page_size=12))
    page = runtime.events[-1]['payload']
    assert page['direction'] == 'backward' and page['before_seq'] == 12
    assert page['items'][-1]['id'] == 'm11' and page['has_more']
    seen = []
    while True:
        assert len(json.dumps(page, ensure_ascii=False).encode()) < 240 * 1024
        assert page['next_after_seq'] == page['before_seq'] - 1
        assert page['next_before_seq'] == page['items'][0]['seq']
        seen = [row['id'] for row in page['items']] + seen
        if not page['has_more']:
            assert page['after_seq'] == -1
            break
        assert page['after_seq'] == page['next_before_seq'] - 1
        await channel._get_history(device, command('history.get', session, direction='backward',
            before_seq=page['next_before_seq'], through_seq=page['through_seq'], page_size=12))
        page = runtime.events[-1]['payload']
    assert seen == [f'm{index}' for index in range(12)]
    await channel._get_history(device, command('history.get', session, direction='backward', around_id='m5', page_size=2))
    page = runtime.events[-1]['payload']
    assert [row['id'] for row in page['items']] == ['m4', 'm5']
    assert page['around_id'] == 'm5' and page['through_seq'] == 11 and page['before_seq'] == 6
    with pytest.raises(MobileCommandError, match='目标消息不存在'):
        await channel._get_history(device, command('history.get', session, direction='backward', around_id='missing'))
    assert snapshot(tmp_path / 'sessions.db') == before


@pytest.mark.asyncio
async def test_display_pages_omit_hidden_archives_and_keep_legacy_downloads(mobile, tmp_path):
    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    append(log, session, 'archive', Output((
        ContentPart('history.record', {'private_archive': 'x' * 400000}),
        ContentPart('history.transcript', {'raw': '可见旧对话', 'completeness': 'unknown'}),
        ContentPart('text', '当前正文'),
    ), 'complete'))
    before = snapshot(tmp_path / 'sessions.db')
    await channel._get_history(device, command('history.get', session))
    old = runtime.events[-1]['payload']['items'][0]['message_ref']
    await channel._get_history(device, command('history.get', session, direction='backward', display_only=True))
    compact = runtime.events[-1]['payload']['items'][0]
    assert compact['body']['parts'] == [
        {'kind': 'history.record', 'display': 'unavailable'},
        {'kind': 'history.transcript', 'archive': {'raw': '可见旧对话', 'completeness': 'unknown'}},
        {'kind': 'text', 'value': '当前正文'},
    ]
    assert len(json.dumps(compact).encode()) < 1024
    legacy = channel.read_message_content(session_id=session, message_id='archive', byte_length=old['byte_length'], sha256=old['sha256'])
    assert len(legacy) > 400000 and json.loads(legacy)['body']['parts'][0]['archive']['private_archive']
    assert snapshot(tmp_path / 'sessions.db') == before
    append(log, session, 'large-visible', Input((ContentPart('text', 'x' * 80000),)))
    await channel._get_history(device, command('history.get', session, direction='backward', display_only=True, page_size=1))
    reference = runtime.events[-1]['payload']['items'][0]['message_ref']
    assert reference['display_only'] is True
    visible = channel.read_message_content(session_id=session, message_id='large-visible', byte_length=reference['byte_length'], sha256=reference['sha256'])
    assert json.loads(visible)['body']['parts'][0]['value'] == 'x' * 80000
    # 第一条权威记录保持原始归档，没有被展示适配器压缩。
    assert log.reader(session).get('archive').body.parts[0].value['private_archive'] == 'x' * 400000


@pytest.mark.asyncio
async def test_mobile_session_open_uses_message_log_and_preserves_raw_facts(mobile, tmp_path):
    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    log.ensure_session(session, SessionAttributes())
    before = snapshot(tmp_path / 'sessions.db')

    opened = await channel._open_session(
        device,
        _generic_frame(
            frame_id='01ARZ3NDEKTSV4RRFFQ69G5FAV',
            command_type='session.open',
            session_id=session,
            payload={},
        ),
    )

    assert opened.type == 'session.open.ok'
    assert opened.session_id == session
    assert runtime.events[-1] == {
        'event_type': 'session.updated',
        'session_id': session,
        'payload': {'session_id': session, 'state': 'opened'},
    }
    with pytest.raises(MobileCommandError, match='会话不存在'):
        await channel._open_session(
            device,
            _generic_frame(
                frame_id='01ARZ3NDEKTSV4RRFFQ69G5GAV',
                command_type='session.open',
                session_id=f'akashic:{uuid4()}',
                payload={},
            ),
        )
    assert snapshot(tmp_path / 'sessions.db') == before


@pytest.mark.asyncio
async def test_mobile_large_messages_download_whole_json_and_page_budget_never_truncates(mobile, tmp_path):
    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    log.save_binding('tool', {'root_ref': {'secret': 'root-secret'}, 'service': 'tools.v1',
                              'metadata': {'tool': {'name': 'original-tool'}, 'state': {'secret': 'binding-secret'}}})
    text = '完整内容🪷' * 30000
    facts = ContentPart('model.facts', {'call_record_id': 'call', 'tool_ids': {}, 'thinking': text,
                                       'continuation': {'binding_id': 'model', 'payload': {'private': 'model-secret'}}})
    output = append(log, session, 'o', Output((facts, ToolCall('tool', {'query': text}), ContentPart('future.private', {'private': 'part-secret'})), 'continue'))
    append(log, session, 'c', Control('abandon', output.seq, text))
    append(log, session, 'r', ToolResult(CallRef('o', 1), 'unknown', (ContentPart('text', text),)), CallRef('o', 1))
    append(log, session, 'a', Output((ContentPart('history.transcript', {'raw': text, 'completeness': 'unknown'}),), 'quiet'))
    for index in range(12):
        append(log, session, f'm{index}', Input((ContentPart('text', 'x' * 60000),)))
    before = snapshot(tmp_path / 'sessions.db')
    await channel._get_history(device, command('history.get', session, page_size=200))
    page = runtime.events[-1]['payload']
    assert page['has_more'] and len(page['items']) < 16
    assert len(json.dumps(page, ensure_ascii=False).encode()) < 240 * 1024
    expected = {row['id']: row for row in message_rows(log.reader(session).read_page(limit=200))}
    for row in page['items'][:4]:
        assert set(row) == {'id', 'session_id', 'seq', 'message_ref'}
        ref = row['message_ref']
        frame = command('message.content.prepare', session, message_id=row['id'], byte_length=ref['byte_length'], sha256=ref['sha256'])
        descriptor = channel.prepare_message_content(frame)
        assert descriptor['media_type'] == 'application/json' and descriptor['version'] == 2
        content = channel.read_message_content(session_id=session, message_id=row['id'], byte_length=ref['byte_length'], sha256=ref['sha256'])
        assert json.loads(content) == expected[row['id']]
        assert 'secret' not in content.decode()
        with pytest.raises(MobileCommandError, match='manifest'):
            channel.read_message_content(session_id=session, message_id=row['id'], byte_length=ref['byte_length'] + 1, sha256=ref['sha256'])
        with pytest.raises(MobileCommandError, match='不存在'):
            channel.read_message_content(session_id=f'akashic:{uuid4()}', message_id=row['id'], byte_length=ref['byte_length'], sha256=ref['sha256'])
    seen = page['items'][:]
    while page['has_more']:
        await channel._get_history(device, command('history.get', session, page_size=200, after_seq=page['next_after_seq'], through_seq=page['through_seq']))
        page = runtime.events[-1]['payload']
        seen.extend(page['items'])
    assert [row['id'] for row in seen] == list(expected)
    assert snapshot(tmp_path / 'sessions.db') == before


@pytest.mark.asyncio
@pytest.mark.parametrize('payload', [{}, {'message_log_version': 1}, {'message_log_version': True},
    {'message_log_version': 2, 'page': 1}, {'message_log_version': 2, 'after_seq': True},
    {'message_log_version': 2, 'through_seq': None}, {'message_log_version': 2, 'page_size': 0},
    {'message_log_version': 2, 'through_seq': 10}])
async def test_mobile_history_rejects_old_or_invalid_protocol(mobile, payload):
    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    log.ensure_session(session, SessionAttributes())
    with pytest.raises(MobileCommandError):
        await channel._get_history(device, _generic_frame(frame_id='01ARZ3NDEKTSV4RRFFQ69G5FAV', command_type='history.get', session_id=session, payload=payload))
    assert not runtime.events


@pytest.mark.asyncio
async def test_mobile_json_range_authentication_and_reopen(mobile, tmp_path):
    log, captured, channel, device = mobile
    session = f'akashic:{uuid4()}'
    append(log, session, 'large', Input((ContentPart('text', '🪷完整消息' * 30000),)))
    await channel._get_history(device, command('history.get', session))
    ref = captured.events[-1]['payload']['items'][0]['message_ref']
    keyset = KeysetManager(tmp_path / 'keys', FileMasterKeyStore(tmp_path / 'master-keys.json')).initialize(lan_hostname='localhost')
    runtime = MobileGatewayRuntime(config=MobileRealtimeConfig(), storage=captured.storage,
        pairing=PairingService(captured.storage, keyset, lan_endpoints=(), tunnel_endpoints=()),
        authenticator=DeviceAuthenticator(captured.storage, keyset), inbox=DurableInboxManager(captured.storage),
        approvals=PairingApprovalRegistry(asyncio.get_running_loop()), keyset=keyset)
    # 重开同一消息库后仍下载同一表示，不依赖内存正文缓存。
    with closing(MessageLog(tmp_path / 'sessions.db')) as reopened:
        channel = MobileRealtimeChannel(runtime)
        channel.bind_messages(reopened.catalog())
        runtime.bind_channel(channel)
        async def receive() -> Message:
            return {'type': 'websocket.disconnect'}

        async def send(_message: Message) -> None:
            return None

        websocket = WebSocket(
            {'type': 'websocket', 'path': '/', 'headers': [], 'query_string': b''},
            receive,
            send,
        )
        runtime._connections[device] = ActiveMobileConnection(
            websocket=websocket,
            connection_epoch=1,
            send_lock=asyncio.Lock(),
            pending_events=deque(),
            ready=True,
            delivery_task=None,
        )
        grant = runtime.message_content_tickets.issue(device_id=device, connection_epoch=1, session_id=session,
            message_id='large', byte_length=ref['byte_length'], sha256=ref['sha256'])
        client = TestClient(create_mobile_gateway_app(runtime))
        headers = {'Authorization': f'Bearer {grant.ticket}', 'Accept-Encoding': 'identity'}
        chunks = []
        for start in range(0, ref['byte_length'], 32768):
            response = client.get('/mobile/message-content/v2', headers={**headers, 'Range': f'bytes={start}-{min(start + 32767, ref["byte_length"] - 1)}', 'If-Range': f'"{ref["sha256"]}"'})
            assert response.status_code == 206, response.text
            assert response.headers['content-type'] == 'application/json'
            chunks.append(response.content)
        body = b''.join(chunks)
        assert len(body) == ref['byte_length'] and hashlib.sha256(body).hexdigest() == ref['sha256']
        assert json.loads(body)['body']['parts'][0]['value'] == '🪷完整消息' * 30000
        assert client.get('/mobile/message-content/v2', headers={**headers, 'Range': 'bytes=0-9', 'If-Range': '"bad"'}).status_code == 412
        assert client.get('/mobile/message-content/v2', headers={**headers, 'Range': 'bytes=0-9999999'}).status_code == 416
        runtime._connections[device].connection_epoch = 2
        assert client.get('/mobile/message-content/v2', headers={**headers, 'Range': 'bytes=0-9'}).status_code == 401
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('artifact_id,content,filename,media_type', [
    ('21f6c57ea37a477a98037f6ea74c5fb6', b'old attachment' * 15000, 'old.txt', 'text/plain'),
    ('opaque.File_' + 'x' * 100, b'', None, None),
])
async def test_artifact_download_uses_message_reference_and_core_bytes(mobile, tmp_path, artifact_id, content, filename, media_type):
    """真实历史引用经过 Mobile 命令与二进制编码，重放和跨会话不改变权威事实。"""
    from dataclasses import asdict
    import struct
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from infra.mobile_realtime.attachments import encode_attachment_chunk, MAX_ATTACHMENT_CHUNK_BYTES
    from infra.mobile_realtime.protocol import AttachmentDownloadCommand, parse_frame
    from session.artifact_store import ArtifactStore
    from session.artifacts import AttachmentKind, AttachmentRef

    log, runtime, channel, device = mobile
    session = f'akashic:{uuid4()}'
    other = f'akashic:{uuid4()}'
    source = tmp_path / 'original.bin'
    source.write_bytes(content)
    ref = AttachmentRef(artifact_id, AttachmentKind.FILE, filename, media_type, len(content), hashlib.sha256(content).hexdigest())
    with closing(ArtifactStore(tmp_path / 'sessions.db')) as metadata:
        artifacts = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=metadata)
        assert await artifacts.adopt_file_with_artifact_id(source, allowed_root=tmp_path, expected_ref=ref) == ref
        channel.bind_channel_attachment_store(artifacts)
        for sid, mid in ((session, 'message'), (other, 'other-message')):
            log.writer(sid, author='user', source='conversation', body_types=(Input,),
                       content={'artifact_ref': lambda part: ContentReferences(artifact_ids=(part.value,))}).append(
                           mid, Input((ContentPart('artifact_ref', artifact_id),)))
        before = snapshot(tmp_path / 'sessions.db')
        await channel._get_history(device, command('history.get', session))
        row = runtime.events[-1]['payload']['items'][0]
        assert row['attachments'] == [asdict(ref)]

        def request(sid, mid, offset, counter):
            frame = parse_frame(json.dumps({'v': 1, 'kind': 'command', 'type': 'attachment.download',
                'id': f'01ARZ3NDEKTSV4RRFFQ69G5{counter:03d}', 'connection_epoch': 1, 'session_id': sid,
                'payload': {'message_id': mid, 'artifact_id': row['attachments'][0]['artifact_id'], 'offset': offset}}))
            assert isinstance(frame, AttachmentDownloadCommand)
            return frame

        recovered = bytearray()
        offset = 0
        counter = 0
        while True:
            frame = request(session, row['id'], offset, counter)
            reply = await channel.handle_command(device_id=device, frame=frame)
            assert reply.type == 'attachment.download.ok'
            repeated = await channel.handle_command(device_id=device, frame=frame)
            assert repeated == reply
            binary = encode_attachment_chunk(reply.binary)
            size = struct.unpack('>I', binary[:4])[0]
            assert json.loads(binary[4:4 + size]) == {'artifact_id': artifact_id, 'offset': offset}
            chunk = binary[4 + size:]
            assert len(chunk) <= MAX_ATTACHMENT_CHUNK_BYTES
            recovered.extend(chunk)
            offset = reply.payload['next_offset']
            if reply.payload['complete']:
                break
            assert chunk
            counter += 1
        assert bytes(recovered) == content
        allowed = await channel.handle_command(device_id=device, frame=request(other, 'other-message', 0, 900))
        assert allowed.type == 'attachment.download.ok'
        rejected = await channel.handle_command(device_id=device, frame=request(other, 'message', 0, 901))
        assert rejected.type == 'attachment.download.error'
        assert rejected.payload['code'] == 'attachment_download_rejected'
        path = tmp_path / metadata.get_attachment(artifact_id).storage_key
        for counter, damage in ((902, 'missing'), (903, 'corrupt')):
            if damage == 'missing':
                path.unlink()
            else:
                path.write_bytes(b'changed bytes')
            failed_frame = request(session, 'message', 0, counter)
            failed = await channel.handle_command(device_id=device, frame=failed_frame)
            assert failed.type == 'attachment.download.error'
            assert failed.payload['code'] == 'attachment_download_failed'
            assert str(tmp_path) not in failed.payload['message']
            assert runtime.storage.read_command(device_id=device, command_id=failed_frame.id).status == 'completed'
            repeated_failure = await channel.handle_command(device_id=device, frame=failed_frame)
            assert repeated_failure.type == failed.type and repeated_failure.payload == failed.payload
            # 已完成成功回执的 bytes 也可能暂时无法读取，只拒绝本次下载重放。
            unavailable_replay = await channel.handle_command(device_id=device, frame=request(other, 'other-message', 0, 900))
            assert unavailable_replay.type == 'attachment.download.error'
            path.write_bytes(content)
        history = await channel.handle_command(device_id=device, frame=command('history.get', session))
        assert history.type == 'history.get.ok'
        assert snapshot(tmp_path / 'sessions.db') == before
