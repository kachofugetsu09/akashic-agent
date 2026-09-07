import asyncio
import hashlib
import json
from contextlib import closing
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from agent.config_models import MobileRealtimeConfig
from infra.channels.message_view import message_rows
from infra.mobile_realtime.auth import DeviceAuthenticator
from infra.mobile_realtime.channel import MobileCommandError, MobileRealtimeChannel
from infra.mobile_realtime.gateway import MobileGatewayRuntime, PairingApprovalRegistry, create_mobile_gateway_app
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
        channel = MobileRealtimeChannel(runtime)
        channel.bind_messages(log.catalog())
        yield log, runtime, channel, device


def command(kind, session_id=None, **payload):
    return _generic_frame(frame_id='01ARZ3NDEKTSV4RRFFQ69G5FAV', command_type=kind,
                          session_id=session_id, payload={'message_log_version': 2, **payload})


def append(log, session, identity, body, call_ref=None):
    checks = {kind: lambda part: ContentReferences() for kind in ('text', 'history.transcript', 'future.private')}
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
        runtime._connections[device] = SimpleNamespace(connection_epoch=1)
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
