"""实际插件装配与 Message/Tools/Models 回执链；全部输入和数据库在临时目录。"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from agent.plugin_composition import CompositionRoot, ServiceKey
from agent.plugin_composition.model import PluginRuntime
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_contracts import ContentPart, Control, Input, Message, Output, ToolResult
from agent.plugin_contracts.models import CONTENT_VIEWS, MODEL_CALLS, RenderedContent
from agent.plugin_contracts.tools import CallSource
from plugins.content_view.plugin import ReadContent, prepare_view
from plugins.models.content import render_content
from plugins.models.projection import MessageProjection
from plugins.models.store import ModelsStore
from plugins.models.views import ContentViews
from plugins.react.plugin import _decode_request, _encode_request
from session.log import MessageLog
from tests.test_default_reply import application, live_root

LONG = '完整结果🙂汉字\n' * 3000 + 'END_OF_RESULT'
DRIVER = '''        async def complete(self, request):
            calls.append(request)
            step = len(calls)
            contents = [block["text"] for row in request.messages if row["role"] == "tool"
                        for block in row["content"] if block["type"] == "text"]
            schemas = {item["function"]["name"] for item in request.tools}
            assert "read_content" in schemas
            if step == 1:
                return LLMResponse(None, [ToolCall("long", "write_evidence", {})])
            if step == 2:
                assert LONG in contents and "short result" in contents
                return LLMResponse(None, [ToolCall("short-1", "write_evidence", {"short": True})])
            if step == 3:
                assert LONG not in contents
                folded = json.loads(contents[0])
                self.reference = folded["read_content"]
                self.placeholder = contents[0]
                return LLMResponse(None, [ToolCall("read-full", "read_content", self.reference)])
            if step == 4:
                assert contents[0] == self.placeholder and contents[-1] == LONG
                return LLMResponse(None, [ToolCall("short-2", "write_evidence", {"short": True})])
            if step == 5:
                assert LONG not in contents and contents[0] == self.placeholder
                folded_reads = [json.loads(text) for text in contents if text.startswith('{"status":')]
                assert len(folded_reads) == 2
                assert folded_reads[0]["read_content"] == folded_reads[1]["read_content"]
                return LLMResponse(None, [ToolCall("read-range", "read_content", {**self.reference,"start":2,"end":27})])
            assert step == 6 and contents[-1] == LONG[2:27]
            return LLMResponse("finished")
'''


def sources(directory: Path) -> None:
    """复用真实安装夹具，仅替换模型响应序列与工具正文。"""
    plugin = directory / 'reference_reader'
    shutil.copytree(Path(__file__).resolve().parents[2] / 'plugins/content_view', plugin,
                    ignore=shutil.ignore_patterns('__pycache__'))
    entry = plugin / 'plugin.py'
    entry.write_text(entry.read_text().replace('name = "content_view"', 'name = "reference_reader"'))
    provider = directory / 'test_provider/plugin.py'
    text = provider.read_text()
    start = text.index('        async def complete(self, request):')
    end = text.index('    descriptor = BoundModelDescriptor(', start)
    text = text[:start] + DRIVER + text[end:]
    text = text.replace('from contextlib import asynccontextmanager',
                        'import json\nfrom plugins.models.views import CONTENT_VIEWS, ContentViews\n'
                        + f'LONG = {LONG!r}\nfrom contextlib import asynccontextmanager')
    text = text.replace('await ctx.provide(MODEL_CONTENT, ContentOwner())',
                        'await ctx.provide(MODEL_CONTENT, ContentOwner())\n'
                        '    await ctx.provide(CONTENT_VIEWS, ContentViews(ctx))')
    text = text.replace('return Result("success", (ContentPart("text", "written"),))',
                        'return Result("success", (ContentPart("text", "small"),)) if args.get("short") else '
                        'Result("success", (ContentPart("text", LONG), ContentPart("text", "short result")))')
    # 原测试的固定工具 schema 只接受空参数；该场景显式增加 short。
    text = text.replace('parameters={"type":"object"}',
                        'parameters={"type":"object", "properties":{"short":{"type":"boolean"}}}')
    provider.write_text(text)


async def check(directory: Path) -> dict[str, object]:
    """先跑真实闭环，再从关闭后的磁盘数据检验重放和失败边界。"""
    statements: list[str] = []
    async with application(directory, replying=True, extra_sources=sources) as (log, host):
        log._connection.set_trace_callback(statements.append)
        async with live_root(host) as root:
            await root.context.require(CHANNEL_INPUT)(
                'test:room', 'u1', ChannelInboundMessage('test', 'user', 'room', 'read then read again',
                                                       datetime(2026, 9, 5, tzinfo=UTC), {}))
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader('test:room').snapshot()
                if any(isinstance(row.body, Output) and row.body.finish == 'complete' for row in rows):
                    return rows
        rows = await asyncio.wait_for(completed(), 30)
        async with live_root(host) as root:
            requests = list(root.context.require(ServiceKey('fixture.calls')))
            assert len(requests) == 6
        original = next(row for row in rows if isinstance(row.body, ToolResult))
        assert original.body.parts[0].value == LONG
        reads = [row for row in rows if isinstance(row.body, ToolResult)
                 and row.body.parts[0].kind == 'content_view.read']
        assert len(reads) == 2
        assert all(len(json.dumps(dict(row.body.parts[0].value))) < 250 for row in reads)
        first_output = next(row for row in rows if isinstance(row.body, Output))
        tool = ReadContent()
        call_source = CallSource(original.body.call_ref, rows)
        denied = await tool.prepare({'message_id':'other-session','part_index':0}, call_source)
        assert isinstance(denied, str)
        denied = await tool.prepare({'message_id':original.message_id,'part_index':0,'end':len(LONG)+1}, call_source)
        assert isinstance(denied, str)
        reread = await tool.prepare({'message_id':reads[0].message_id,'part_index':0,'start':3,'end':9}, call_source)
        assert reread == {'message_id':original.message_id,'part_index':0,'start':3,'end':9}
        # 严格冻结关联，不让恢复时重建的候选请求改变首次展示证据。
        for request in requests:
            restored = _decode_request(_encode_request(request))
            assert restored.content_refs == request.content_refs and restored.messages == request.messages
            assert restored.content_transformed == request.content_transformed
        saved_rows = rows
    assert (directory / 'effect.txt').read_text() == 'once\nonce\nonce\n'
    assert not any('messages' in statement.lower() and statement.lstrip().lower().startswith(
        ('update ', 'delete ', 'replace ')) for statement in statements)
    # 所有 runtime 已关闭；新读者从原数据库恢复，插件没有独立 seen 或归档状态。
    log = MessageLog(directory / 'sessions.db')
    store_path = next((directory / 'workspace').rglob('models.db'))
    store = ModelsStore(store_path, directory / 'reopen-backups')
    store.initialize()
    try:
        rows = log.reader('test:room').snapshot()
        assert rows == saved_rows
        model = SimpleNamespace(descriptor=SimpleNamespace(binding_id='fixture-model', capabilities=
                                                           SimpleNamespace(context_window=1000000)))
        def projection(*, tools=frozenset({'read_content'}), prepare=None):
            return MessageProjection(model, source=original.source, render_content=lambda p: render_content(p, artifacts={}),
                                     tool_name=lambda _: 'fixture-tool', read_call=store.read_call,
                                     check_summary=lambda _: None,
                                     prepare_content=prepare or partial(prepare_view, fold_after_chars=8192),
                                     tool_names=tools)
        restored = projection().render(rows, after_seq=-1)
        restored_text = [block.get('text') for row in restored.messages for block in row.get('content', ())]
        assert LONG not in restored_text and LONG[2:27] in restored_text
        # 无回读工具时不折叠，原始和回读内容都能完整展开。
        unfolded = projection(tools=frozenset()).render(rows, after_seq=-1)
        assert any(block.get('text') == LONG for row in unfolded.messages for block in row.get('content', ()))
        # 首次请求之后尚无成功 Output：多次估算/渲染不会抢先折叠。
        prefix = tuple(row for row in rows if row.seq <= original.seq)
        first = projection().render(prefix, after_seq=-1)
        again = projection().render(prefix, after_seq=-1)
        assert first.messages == again.messages and (original.message_id, 0) in first.content_refs
        assert any(block.get('text') == LONG for row in first.messages for block in row.get('content', ()))
        # 已有摘要覆盖的原文不得进入该次展示回执。
        tail = projection().render(rows, after_seq=original.seq, fresh=True)
        assert (original.message_id, 0) not in tail.content_refs
        # 纯投影替换也能服务其他内容，不依赖长结果插件的名称。
        def unrelated(messages, source, tools, seen):
            return lambda message, index: RenderedContent(({'type':'text','text':'alternate view'},)) if message == original else None
        changed = projection(prepare=unrelated).render(prefix, after_seq=-1)
        assert 'alternate view' in str(changed.messages) and (original.message_id, 0) not in changed.content_refs
        # 成功的旧请求建立在折叠视图上；卸载视图后绝不能接续其 opaque 状态。
        last = next(row for row in reversed(rows) if isinstance(row.body, Output))
        facts = next(part for part in last.body.parts if part.kind == 'model.facts')
        new_facts = ContentPart('model.facts', {**facts.value, 'content_transformed':True,
            'continuation':{'binding_id':'fixture-model','payload':{'opaque':'old-folded-view'}}})
        from dataclasses import replace
        opaque_rows = tuple(replace(row, body=Output(tuple(new_facts if part.kind == 'model.facts' else part
                                    for part in row.body.parts), row.body.finish)) if row == last else row for row in rows)
        removed = projection(prepare=lambda messages, source, tools, seen: lambda message, index: None)
        assert removed.render(opaque_rows, after_seq=-1).continuation is None
        # 视图存在时也不能把新展开的回读混入旧 opaque。
        assert projection().render(opaque_rows, after_seq=-1).continuation is None
        assert log.reader('test:room').snapshot() == rows
    finally:
        log.close()
        store.close()
    return {'requests':len(requests), 'messages':len(saved_rows), 'original_characters':len(LONG),
            'checks':['renamed installed plugin', 'first full exposure', 'stable fold', 'full readback',
                      'readback folds', 'range readback', 'no reference chains', 'session scope',
                      'invalid range', 'request freeze', 'disk reopen', 'no tool no fold',
                      'render is not exposure', 'summary boundary', 'unrelated content view', 'removed view resets opaque',
                      'original rows unchanged', 'external tool not rerun']}


async def check_failure(directory: Path) -> None:
    """真实 provider 调用失败后，已冻结的展示位置不能成为成功展示回执。"""
    def fail_sources(path):
        sources(path)
        provider = path / 'test_provider/plugin.py'
        provider.write_text(provider.read_text().replace('            if step == 2:',
                            '            if step == 2:\n                raise ValueError("planned provider failure")'))
    async with application(directory, replying=True, extra_sources=fail_sources) as (log, host):
        async with live_root(host) as root:
            await root.context.require(CHANNEL_INPUT)(
                'test:room', 'u1', ChannelInboundMessage('test', 'user', 'room', 'failure scenario',
                                                       datetime(2026, 9, 5, tzinfo=UTC), {}))
        async def failed():
            async for _ in log.catalog().follow():
                rows = log.reader('test:room').snapshot()
                if any(isinstance(row.body, Control) and row.body.action == 'failure' for row in rows):
                    return rows
        rows = await asyncio.wait_for(failed(), 30)
        original = next(row for row in rows if isinstance(row.body, ToolResult))
        async with live_root(host) as root:
            calls = root.context.require(ServiceKey('fixture.calls'))
            assert len(calls) == 2 and (original.message_id, 0) in calls[1].content_refs
            model = SimpleNamespace(descriptor=SimpleNamespace(binding_id='fixture-model', capabilities=
                                                               SimpleNamespace(context_window=1000000)))
            projection = MessageProjection(
                model, source=original.source, render_content=lambda p: render_content(p, artifacts={}),
                tool_name=lambda _: 'write_evidence', read_call=root.context.require(MODEL_CALLS),
                check_summary=lambda _: None,
                prepare_content=partial(prepare_view, fold_after_chars=8192), tool_names=frozenset({'read_content'}))
            request = projection.render(rows, after_seq=-1)
            assert (original.message_id, 0) in request.content_refs
            assert any(block.get('text') == LONG for row in request.messages for block in row.get('content', ()))


async def check_lifecycle(directory: Path) -> None:
    """真实 Root 上验证贡献者排空、关闭失效和冲突，不靠 sleep 猜测调度。"""
    root = CompositionRoot('content-views-lifecycle')
    def runtime(name):
        return PluginRuntime(name, name, directory, directory / name, directory, {})
    async def owner(ctx):
        await ctx.provide(CONTENT_VIEWS, ContentViews(ctx))
    def contributor(label):
        async def apply(ctx):
            await ctx.require(CONTENT_VIEWS).register(ctx, name='renderer', prepare=
                lambda messages, source, tools, seen: lambda message, index:
                    RenderedContent(({'type':'text', 'text':label},)))
        return apply
    try:
        await root.mount(owner, name='alternate-models', runtime=runtime('alternate-models'))
        plugin = await root.mount(contributor('one'), name='one', runtime=runtime('one'), inject=(CONTENT_VIEWS,))
        views = root.context.require(CONTENT_VIEWS)
        probe = Message(message_id='probe', session_id='session', seq=0, recorded_at=datetime.now(UTC),
                        author='user', source='source', body=Input((ContentPart('text','body'),)))
        async with views.bind() as prepare:
            render = prepare((probe,), 'source', frozenset(), frozenset())
            assert render(probe, 0).blocks[0]['text'] == 'one'
            disposing = asyncio.create_task(plugin.dispose())
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await barrier.wait()
            assert not disposing.done()
        await disposing
        try:
            render(probe, 0)
        except RuntimeError:
            pass
        else:
            raise AssertionError('closed view remained usable')
        for name in ('two', 'three'):
            await root.mount(contributor(name), name=name, runtime=runtime(name), inject=(CONTENT_VIEWS,))
        async with views.bind() as prepare:
            try:
                prepare((probe,), 'source', frozenset(), frozenset())(probe, 0)
            except ValueError as error:
                assert 'owner' in str(error)
            else:
                raise AssertionError('conflicting renderers silently replaced one another')
    finally:
        await root.dispose()


async def run(directory: Path) -> dict[str, object]:
    result = await check(directory / 'normal')
    await check_failure(directory / 'failure')
    await check_lifecycle(directory / 'lifecycle')
    result['checks'] += ['failed provider is not exposure', 'contributor drain', 'closed view', 'conflicting views']
    return result


if __name__ == '__main__':
    with TemporaryDirectory(prefix='akashic-content-view-') as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary))), ensure_ascii=False, indent=2))
