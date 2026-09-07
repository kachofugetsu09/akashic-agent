import asyncio
from contextlib import aclosing, closing, suppress

import pytest
from fastapi.testclient import TestClient

from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.tasks import Tasks
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotStore
from bootstrap.reply_status import RuntimeReplyStatus
from bootstrap.chat_api import create_chat_app
from infra.channels.message_view import follow_messages, message_rows
from infra.channels.web_chat_channel import WebChatChannel
from plugins.reply.status import REPLY_STATUS, ReplyState
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input, Control, Output


@pytest.mark.asyncio
async def test_follow_pages_fixed_prefix_then_reconnect_from_last_seq(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        writer = log.writer('s', author='u', source='conversation', body_types=(Input, Control),
                            content={'text': lambda p: ContentReferences()})
        for i in range(106):
            writer.append(str(i), Input((ContentPart('text', str(i)),)))
        before = log.reader('s').snapshot()
        async with aclosing(follow_messages(log.reader('s'), after_seq=2)) as stream:
            first = await anext(stream)
            writer.append('late', Control('pause', 105, 'reason'))
            second, third, fourth = await anext(stream), await anext(stream), await anext(stream)
            assert [p['through_seq'] for p in (first, second, third, fourth)] == [105, 105, 105, 106]
            rows = [row for page in (first, second, third, fourth) for row in page['items']]
            assert rows == message_rows(log.reader('s').read_page(after_seq=2, limit=200))
            assert [p['next_after_seq'] for p in (first, second, third, fourth)] == [52, 102, 105, 106]
        assert log.reader('s').snapshot()[:-1] == before
        async with aclosing(follow_messages(log.reader('s'), after_seq=106)) as stream:
            pending = asyncio.create_task(anext(stream))
            writer.append('reconnected', Input(()))
            page = await asyncio.wait_for(pending, 3)
            assert [row['id'] for row in page['items']] == ['reconnected']
        assert not log._listeners


async def status_root(name, state):
    root = CompositionRoot(name)
    if state is not None:
        async def plugin(ctx):
            await ctx.provide(REPLY_STATUS, state.read)
        await root.mount(plugin, name='reply')
    return root, RuntimeSnapshotCompiler().compile({}, composition_root=root, snapshot_revision=name)


@pytest.mark.asyncio
async def test_status_switch_and_absence_never_pin_old_generation():
    old, new = ReplyState(), ReplyState()
    roots = [await status_root('old', old), await status_root('new', new), await status_root('absent', None)]
    store = RuntimeSnapshotStore()
    store.install(roots[0][1])
    tasks = Tasks()
    entered, release = asyncio.Event(), asyncio.Event()
    async def operation(task):
        with old.open(task, 's', 'conversation') as preview:
            with preview('draft') as delta:
                await delta({'content_delta': 'old preview'})
                entered.set()
                await release.wait()
    task = await tasks.admit('s', lambda slot: slot.start(operation))
    await asyncio.wait_for(entered.wait(), 3)
    try:
        async with aclosing(RuntimeReplyStatus(store).follow('s')) as stream:
            frame = await anext(stream)
            assert frame['items'][0]['preview']['text'] == 'old preview'
            assert roots[0][1].lease_count == 0
            await store.commit(store.begin_publish(roots[1][1]))
            frame = await asyncio.wait_for(anext(stream), 3)
            assert frame['snapshot_id'] == roots[1][1].snapshot_id and frame['available'] and frame['items'] == []
            await asyncio.wait_for(store.wait_for_snapshot_drained(roots[0][1]), 3)
            # 新 stable 即使没有 reply 也要清旧草稿，不能冒充可回复的空闲状态。
            await store.commit(store.begin_publish(roots[2][1]))
            frame = await asyncio.wait_for(anext(stream), 3)
            assert not frame['available'] and frame['items'] == []
        assert all(snapshot.lease_count == 0 for _, snapshot in roots)
    finally:
        release.set()
        await task.join()
        await tasks.close()
        await store.close()
        for root, _ in roots:
            await root.dispose()


@pytest.mark.asyncio
async def test_closed_status_and_cancelled_wait_release_subscriptions():
    state = ReplyState()
    root, snapshot = await status_root('closed', state)
    store = RuntimeSnapshotStore()
    store.install(snapshot)
    stream = RuntimeReplyStatus(store).follow('s')
    try:
        assert (await anext(stream))['available']
        state.close()
        while (await asyncio.wait_for(anext(stream), 3))['available']:
            pass
        pending = asyncio.create_task(anext(stream))
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await stream.aclose()
        assert snapshot.lease_count == 0
    finally:
        await stream.aclose()
        await store.close()
        await root.dispose()


def test_websocket_follow_reads_real_messages_switches_and_disconnects(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        def append(session, identity):
            return log.writer(session, author='真实作者', source='source', body_types=(Input,),
                              content={}).append(identity, Input(()))
        append('akashic:a', 'a0')
        append('akashic:a', 'a1')
        append('akashic:b', 'b0')
        channel = WebChatChannel()
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=log.catalog())
        def follow(ws, session, seq):
            ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': session,
                          'after_seq': seq, 'request_id': 'follow'})
            frame = ws.receive_json()
            assert frame['type'] == 'session.following' and frame['session_id'] == session
        def message(ws, session):
            while True:
                frame = ws.receive_json()
                assert frame['session_id'] == session
                if frame['type'] == 'messages.appended':
                    return frame
                assert frame['type'] == 'reply.status' and not frame['available']
        with TestClient(app) as client:
            with client.websocket_connect('/ws') as ws:
                follow(ws, 'akashic:a', 0)
                page = message(ws, 'akashic:a')
                assert [row['id'] for row in page['items']] == ['a1']
                append('akashic:a', 'a2')
                assert [row['id'] for row in message(ws, 'akashic:a')['items']] == ['a2']
                follow(ws, 'akashic:b', -1)
                append('akashic:a', 'old-session')
                assert [row['id'] for row in message(ws, 'akashic:b')['items']] == ['b0']
            with client.websocket_connect('/ws') as ws:
                follow(ws, 'akashic:a', 2)
                assert [row['id'] for row in message(ws, 'akashic:a')['items']] == ['old-session']
        assert not log._listeners and not channel._followers and not channel._connections


def test_websocket_preview_is_separate_until_same_id_commits(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        state = ReplyState()
        store, tasks = RuntimeSnapshotStore(), Tasks()
        channel = WebChatChannel()
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=log.catalog(),
                              reply_status=RuntimeReplyStatus(store).follow)
        with TestClient(app) as client:
            root, snapshot = client.portal.call(status_root, 'socket-reply', state)
            store.install(snapshot)
            entered, release = asyncio.Event(), asyncio.Event()
            async def operation(task):
                with state.open(task, 'akashic:s', 'conversation') as preview:
                    with preview('answer') as delta:
                        await delta({'call_record_id': 'model-call'})
                        await delta({'content_delta': '正在生成', 'thinking_delta': '思考'})
                        entered.set()
                        await release.wait()
                        return log.writer('akashic:s', author='assistant', source='conversation',
                            body_types=(Output,), content={'text': lambda p: ContentReferences()}
                        ).append('answer', Output((ContentPart('text', '完整回答'),), 'complete'))
            async def start():
                task = await tasks.admit('s', lambda slot: slot.start(operation))
                await entered.wait()
                return task
            try:
                with client.websocket_connect('/ws') as ws:
                    ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': 'akashic:s',
                                  'after_seq': -1, 'request_id': 'follow'})
                    assert ws.receive_json()['type'] == 'session.following'
                    assert ws.receive_json()['items'] == []
                    task = client.portal.call(start)
                    frame = ws.receive_json()
                    assert frame['type'] == 'reply.status'
                    assert frame['items'][0]['preview'] == {
                        'message_id': 'answer', 'text': '正在生成', 'thinking': '思考', 'call_record_id': 'model-call'}
                    assert log.reader('akashic:s').get('answer') is None
                    client.portal.call(release.set)
                    saved = client.portal.call(task.join)
                    committed, cleared = False, False
                    while not committed or not cleared:
                        frame = ws.receive_json()
                        if frame['type'] == 'messages.appended':
                            assert frame['items'][0]['id'] == saved.message_id == 'answer'
                            assert frame['items'][0]['body']['parts'][0]['value'] == '完整回答'
                            committed = True
                        elif frame['type'] == 'reply.status' and frame['items'] == []:
                            cleared = True
                assert not channel._followers and not log._listeners and snapshot.lease_count == 0
            finally:
                client.portal.call(release.set)
                client.portal.call(tasks.close)
                client.portal.call(store.close)
                client.portal.call(root.dispose)


def test_channel_stop_waits_for_active_subscriptions_to_close(tmp_path):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        store, state = RuntimeSnapshotStore(), ReplyState()
        finalizing, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def status(session_id):
            try:
                async with aclosing(RuntimeReplyStatus(store).follow(session_id)) as frames:
                    async for frame in frames:
                        yield frame
            finally:
                finalizing.set()
                await release.wait()
                finished.set()
        channel = WebChatChannel()
        app = create_chat_app(workspace=tmp_path, channel=channel, messages=log.catalog(), reply_status=status)
        with TestClient(app) as client:
            root, snapshot = client.portal.call(status_root, 'stop', state)
            store.install(snapshot)
            try:
                with client.websocket_connect('/ws') as ws:
                    ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': 'akashic:s',
                                  'after_seq': -1, 'request_id': 'follow'})
                    assert ws.receive_json()['type'] == 'session.following'
                    assert ws.receive_json()['type'] == 'reply.status'
                    stopping = client.portal.start_task_soon(channel.stop)
                    client.portal.call(finalizing.wait)
                    assert not stopping.done()
                    client.portal.call(release.set)
                    stopping.result(timeout=3)
                    assert finished.is_set() and not channel._followers and not log._listeners
                assert snapshot.lease_count == 0
            finally:
                client.portal.call(release.set)
                client.portal.call(store.close)
                client.portal.call(root.dispose)


@pytest.mark.parametrize('changes', [{'version': 1}, {'version': True}, {'after_seq': True},
                                      {'after_seq': -2}, {'after_seq': 2}, {'session_id': 'qq:a'}])
def test_websocket_follow_rejects_invalid_boundary(tmp_path, changes):
    with closing(MessageLog(tmp_path / 'sessions.db')) as log:
        channel = WebChatChannel()
        with TestClient(create_chat_app(workspace=tmp_path, channel=channel, messages=log.catalog())) as client:
            with client.websocket_connect('/ws') as ws:
                ws.send_json({'type': 'session.follow', 'version': 2, 'session_id': 'akashic:new',
                              'after_seq': -1, 'request_id': 'bad', **changes})
                assert ws.receive_json()['type'] == 'error'
                ws.send_json({'type': 'ping', 'request_id': 'alive'})
                assert ws.receive_json()['type'] == 'pong'
        assert not log._listeners and not channel._followers
