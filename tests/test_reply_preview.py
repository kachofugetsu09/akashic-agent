import asyncio
from datetime import UTC, datetime

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.models import ContextLengthError, LLMResponse
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.context.api import Summary
from plugins.reply.status import REPLY_STATUS, ReplyState
from session.message import ContentPart, ContentReferences, Input, Output
from session.log import WriterExpired
from tests.test_default_reply import application
from tests.test_message_react import runtime


@pytest.mark.asyncio
async def test_preview_commit_uses_allocated_id_and_slow_readers_get_current_snapshot(tmp_path):
    state = ReplyState()
    entered, release = asyncio.Event(), asyncio.Event()
    async def complete(request):
        await request.on_delta({'thinking_delta': '思考'})
        await request.on_delta({'content_delta': '第一段'})
        await request.on_delta({'content_delta': '第二段'})
        entered.set()
        await release.wait()
        return LLMResponse('第一段第二段', thinking='思考')
    async def invoke(key, arguments):
        pytest.fail('no tool call')
    async with runtime(tmp_path, complete, invoke, preview_state=state) as (conversation, log, store, run):
        follower = state.read.follow('s')
        assert await anext(follower) == ()
        await conversation.accept('u', Input((ContentPart('text', 'question'),)))
        task = await conversation.start(run)
        await asyncio.wait_for(entered.wait(), 3)
        items = await asyncio.wait_for(anext(follower), 3)
        assert len(items) == 1 and items[0].handle == task.handle and items[0].active
        draft = items[0].preview
        assert draft.text == '第一段第二段' and draft.thinking == '思考'
        assert log.reader('s').get(draft.message_id) is None
        assert state.read.snapshot('another-session') == ()
        release.set()
        saved = await task.join()
        assert saved.message_id == draft.message_id and isinstance(saved.body, Output)
        assert state.read.snapshot('s') == ()
        assert await asyncio.wait_for(anext(follower), 3) == ()
        await follower.aclose()
        assert state.read.snapshot('s') == ()


@pytest.mark.asyncio
async def test_cancel_withdraws_preview_before_uncooperative_provider_drains(tmp_path):
    state = ReplyState()
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def complete(request):
        await request.on_delta({'content_delta': '废弃草稿'})
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        with pytest.raises(asyncio.CancelledError):
            await request.on_delta({'content_delta': '不应复活'})
        return LLMResponse('废弃结果')
    async def invoke(key, arguments):
        pytest.fail('no tool call')
    async with runtime(tmp_path, complete, invoke, preview_state=state) as (conversation, log, store, run):
        await conversation.accept('u1', Input((ContentPart('text', 'one'),)))
        task = await conversation.start(run)
        await asyncio.wait_for(entered.wait(), 3)
        old_id = state.read.snapshot('s')[0].preview.message_id
        await conversation.accept('u2', Input((ContentPart('text', 'two'),)))
        item = state.read.snapshot('s')[0]
        assert not item.active and item.preview is None and not task.done
        await asyncio.wait_for(cancelled.wait(), 3)
        release.set()
        with pytest.raises(WriterExpired):
            await task.join()
        assert state.read.snapshot('s') == () and log.reader('s').get(old_id) is None
        assert [m.message_id for m in log.reader('s').snapshot()] == ['u1', 'u2']


@pytest.mark.asyncio
async def test_provider_capacity_retry_retires_old_draft_and_preserves_final_id(tmp_path):
    state = ReplyState()
    calls, ids = [], []
    async def complete(request):
        calls.append(request)
        ids.append(state.read.snapshot('s')[0].preview.message_id)
        await request.on_delta({'content_delta': '旧' if len(calls) == 1 else '新'})
        if len(calls) == 1:
            raise ContextLengthError('capacity')
        assert state.read.snapshot('s')[0].preview.text == '新'
        return LLMResponse('新')
    async def reduce(snapshot, prepared, request, model, projection, *, source, force):
        if not force:
            return prepared.summary
        assert state.read.snapshot('s')[0].preview is None
        return Summary('summary', ('old-user', 'old-reply'), 'preserved')
    async def invoke(key, arguments):
        pytest.fail('no tool call')
    async with runtime(tmp_path, complete, invoke, reducer=reduce, preview_state=state) as (conversation, log, store, run):
        log.save_binding('summary', {'target': 'summary'})
        writer = log.writer('s', author='test', source='conversation', body_types=(Input, Output),
            content={'text': lambda p: ContentReferences()})
        writer.append('old-user', Input((ContentPart('text', 'old'),)))
        writer.append('old-reply', Output((ContentPart('text', 'old'),), 'complete'))
        await conversation.accept('current', Input((ContentPart('text', 'new'),)))
        result = await (await conversation.start(run)).join()
        assert len(set(ids)) == 2 and result.message_id == ids[1]
        assert log.reader('s').get(ids[0]) is None and state.read.snapshot('s') == ()


@pytest.mark.asyncio
async def test_installed_reply_publishes_read_only_status_service(tmp_path):
    def streaming_driver(sources):
        path = sources / 'test_provider/plugin.py'
        text = path.read_text().replace('    calls = []', '''    import asyncio
    entered, release = asyncio.Event(), asyncio.Event()
    await ctx.provide(ServiceKey("fixture.preview-gates"), (entered, release))
    calls = []''').replace('            if len(calls) == 1:', '''            await request.on_delta({"content_delta": "真实预览"})
            if len(calls) == 1:
                entered.set()
                await release.wait()
            if len(calls) == 1:''')
        path.write_text(text)
    async with application(tmp_path, replying=True, extra_sources=streaming_driver) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            entered, release = ctx.require(ServiceKey('fixture.preview-gates'))
            read = ctx.require(REPLY_STATUS)
            await ctx.require(CHANNEL_INPUT)('test:room', 'u', ChannelInboundMessage('test', 'user', 'room', 'question', datetime.now(UTC), {}))
            await asyncio.wait_for(entered.wait(), 3)
            current = read.snapshot('test:room')[0]
            assert current.preview.text == '真实预览' and current.source == 'conversation'
            identity = current.preview.message_id
            release.set()
            async def finished():
                async for _ in log.catalog().follow():
                    messages = log.reader('test:room').snapshot()
                    if any(isinstance(m.body, Output) and m.body.finish == 'complete' for m in messages):
                        return
            await asyncio.wait_for(finished(), 3)
            assert log.reader('test:room').get(identity).body.finish == 'continue'
            async def drained():
                async for items in read.follow('test:room'):
                    if not items:
                        return
            await asyncio.wait_for(drained(), 3)
            assert read.snapshot('test:room') == ()


@pytest.mark.asyncio
async def test_reply_status_disposal_ends_old_generation_readers(tmp_path):
    async with application(tmp_path, replying=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            read = snapshot.composition_root.context.require(REPLY_STATUS)
        follower = read.follow('test:room')
        assert await anext(follower) == ()
        pending = asyncio.create_task(anext(follower))
    # 订阅只保留内存数据，没有 pin generation 或阻塞真正卸载。
    try:
        assert await asyncio.wait_for(pending, 3) == ()
    except StopAsyncIteration:
        pass
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(follower), 3)
