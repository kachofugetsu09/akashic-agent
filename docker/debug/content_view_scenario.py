"""实际插件装配与 Message/Tools/Models 回执链；全部输入和数据库在临时目录。"""
from __future__ import annotations

import asyncio
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from agent.plugin_composition import CompositionRoot, ServiceKey
from agent.plugin_composition.model import PluginRuntime
from agent.plugin_composition.channels import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, Control, Input, Message, Output, ToolCall, ToolResult, json_value
from plugins.models.contract import CONTENT_VIEWS, MODEL_CALLS, RenderedContent
from plugins.tools.contract import CallSource
from agent.host_bridge.filesystem import ListDirOperation
from agent.plugins.manager import PluginManager
from infra.channels.artifacts import ChannelAttachmentArtifactStore
from plugins.content_view.plugin import ReadContent, check_read, prepare_view
from plugins.models.content import render_content
from plugins.models.projection import MessageProjection
from plugins.models.store import ModelsStore
from plugins.models.views import ContentViews
from session.log import MessageLog
from agent.plugin_contracts.ui import MESSAGE_DISPLAY
from session.artifact_store import ArtifactStore
from tests.test_default_reply import application, live_root

DISPLAY_SAMPLE: Path | None = None

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
                message_id, part_index = next(ref for ref in request.content_refs
                                              if ref[0].startswith("tool-result:") and ref[1] == 0)
                self.reference = {"message_id": message_id, "part_index": part_index}
                return LLMResponse(None, [ToolCall("short-1", "write_evidence", {"short": True})])
            if step == 3:
                assert LONG in contents
                # 模拟旧会话已有的回读参数；新结果无需回读才能继续查看。
                return LLMResponse(None, [ToolCall("read-full", "read_content", self.reference)])
            if step == 4:
                assert contents[0] == LONG and contents[-1] == LONG
                return LLMResponse(None, [ToolCall("short-2", "write_evidence", {"short": True})])
            if step == 5:
                assert contents.count(LONG) == 2
                return LLMResponse(None, [ToolCall("read-range", "read_content", {**self.reference,"start":2,"end":27})])
            assert step == 6 and contents[-1] == LONG[2:27]
            return LLMResponse("finished")
'''


def sources(directory: Path) -> None:
    """复用真实安装夹具，仅替换模型响应序列与工具正文。"""
    shutil.copytree(Path(__file__).resolve().parents[2] / "plugins/ui", directory / "ui",
                    ignore=shutil.ignore_patterns("__pycache__"))
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
                        'from plugins.models.projection import MODEL_DISPLAY, display_facts\n'
                        'import json\nfrom plugins.models.views import CONTENT_VIEWS, ContentViews\n'
                        + f'LONG = {LONG!r}\nfrom contextlib import asynccontextmanager')
    text = text.replace('await ctx.provide(MODEL_CONTENT, ContentOwner())',
                        'await ctx.provide(MODEL_CONTENT, ContentOwner())\n'
                        '    await ctx.provide(CONTENT_VIEWS, ContentViews(ctx))\n'
                        '    await ctx.provide(MODEL_DISPLAY, display_facts)')
    text = text.replace('return Result("success", (ContentPart("text", "written"),))',
                        'return Result("success", (ContentPart("text", "small"),)) if args.get("short") else '
                        'Result("success", (ContentPart("text", LONG), ContentPart("text", "short result")))')
    # 场景 provider 也要关闭真实模型账本，不能让父进程锁阻止恢复进程加载。
    text = text.replace('    store.initialize()',
                        '    store.initialize()\n    _ = await ctx.effect(lambda: store.close, label="fixture-model-store")')
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
        tool = ReadContent()
        call_source = CallSource(original.body.call_ref, rows)
        denied = await tool.prepare({'message_id':'other-session','part_index':0}, call_source)
        assert isinstance(denied, str)
        denied = await tool.prepare({'message_id':original.message_id,'part_index':0,'end':len(LONG)+1}, call_source)
        assert isinstance(denied, str)
        reread = await tool.prepare({'message_id':reads[0].message_id,'part_index':0,'start':3,'end':9}, call_source)
        assert reread == {'message_id':original.message_id,'part_index':0,'start':3,'end':9}
        # 实际页面投影展开不在当前页中的原文，且不重跑工具或改写消息。
        async with live_root(host) as root:
            reader = log.reader('test:room')
            display_rows = await root.context.require(MESSAGE_DISPLAY)(reader.read_page(limit=50), display_only=True)
            tails = []
            for message, expected in zip(reads, (LONG, LONG[2:27])):
                page = reader.read_page(after_seq=message.seq - 1, through_seq=message.seq, limit=1)
                tail = await root.context.require(MESSAGE_DISPLAY)(page, display_only=True)
                assert tail[0]['body']['parts'][0]['rendered'] == expected
                tails.extend(tail)
            assert reader.snapshot() == rows
            if DISPLAY_SAMPLE is not None:
                DISPLAY_SAMPLE.write_text(json.dumps({'messages': display_rows, 'tail': tails}, ensure_ascii=False))
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
                                     prepare_content=prepare or prepare_view,
                                     tool_names=tools)
        restored = projection().render(rows, after_seq=-1)
        restored_text = [block.get('text') for row in restored.messages for block in row.get('content', ())]
        assert restored_text.count(LONG) == 2 and LONG[2:27] in restored_text
        # 无回读工具时，旧回读内容也必须完整展开。
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
            'checks':['renamed installed plugin', 'first full exposure', 'full text after successful outputs', 'full readback',
                      'readback stays visible', 'range readback', 'no reference chains', 'session scope',
                      'invalid range', 'request freeze', 'disk reopen', 'no tool no fold',
                      'render is not exposure', 'summary boundary', 'unrelated content view', 'removed view resets opaque',
                      'original rows unchanged', 'external tool not rerun', 'read-only display with paged reference']}


async def check_data_display(directory: Path) -> list[dict[str, object]]:
    """真实调用引用和消息存储经页面协议输出未知工具数据。"""
    async with application(directory, replying=True, extra_sources=sources) as (log, host):
        async with live_root(host) as root:
            await root.context.require(CHANNEL_INPUT)(
                'test:room', 'u1', ChannelInboundMessage('test', 'user', 'room', 'data display',
                                                       datetime(2026, 9, 5, tzinfo=UTC), {}))
        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader('test:room').snapshot()
                if any(isinstance(row.body, Output) and row.body.finish == 'complete' for row in rows):
                    return rows
        original = await asyncio.wait_for(completed(), 30)
        call = next(part for row in original if isinstance(row.body, Output)
                    for part in row.body.parts if isinstance(part, ToolCall))
        output = log.writer('test:room', author='assistant', source='display-fixture',
                            body_types=(Output,), content={}, check_call=lambda _: None)
        request = output.append('data-call', Output((call,), 'continue'))
        ref = CallRef(request.message_id, 0)
        writer = log.writer('test:room', author='tool', source='display-fixture', body_types=(ToolResult,),
                            content={'fixture.data': lambda _: ContentReferences()}, call_ref=ref)
        value = {'output': 'line one\n**literal** <script>literal</script>', 'empty': '',
                 'zero': 0, 'false': False, 'null': None, 'nested': {'items': [1, 2]}}
        result = writer.append('data-result', ToolResult(ref, 'error', (ContentPart('fixture.data', value),)))
        before = log.reader('test:room').snapshot()
        async with live_root(host) as root:
            page = log.reader('test:room').read_page(after_seq=request.seq - 1, limit=50)
            displayed = await root.context.require(MESSAGE_DISPLAY)(page, display_only=True)
        assert displayed[-1]['body']['outcome'] == 'error'
        assert displayed[-1]['body']['parts'][0]['value'] == value
        assert log.reader('test:room').snapshot() == before
        assert before[:len(original)] == original and json_value(result.body.parts[0].value) == value
        # 同一个真实展示入口覆盖缺失引用、跨 Session 引用与未来消息引用。
        foreign_call = log.writer('test:other', author='assistant', source='display-fixture',
                                  body_types=(Output,), content={}, check_call=lambda _: None).append(
                                      'foreign-call', Output((call,), 'continue'))
        foreign_ref = CallRef(foreign_call.message_id, 0)
        log.writer('test:other', author='tool', source='display-fixture', body_types=(ToolResult,),
                   content={'text': lambda _: ContentReferences()}, call_ref=foreign_ref).append(
                       'foreign-result', ToolResult(foreign_ref, 'success', (ContentPart('text', 'private'),)))
        for index, target in enumerate(('missing-result', 'foreign-result', 'future-result')):
            request = output.append(f'reference-call-{index}', Output((call,), 'continue'))
            ref = CallRef(request.message_id, 0)
            log.writer('test:room', author='tool', source='display-fixture', body_types=(ToolResult,),
                       content={'content_view.read': check_read}, call_ref=ref).append(
                           f'reference-result-{index}', ToolResult(ref, 'success', (ContentPart('content_view.read',
                           {'message_id': target, 'part_index': 0, 'start': 0, 'end': 3}),)))
        request = output.append('future-call', Output((call,), 'continue'))
        ref = CallRef(request.message_id, 0)
        log.writer('test:room', author='tool', source='display-fixture', body_types=(ToolResult,),
                   content={'text': lambda _: ContentReferences()}, call_ref=ref).append(
                       'future-result', ToolResult(ref, 'success', (ContentPart('text', 'future'),)))
        before = log.reader('test:room').snapshot()
        async with live_root(host) as root:
            page = log.reader('test:room').read_page(after_seq=result.seq, limit=50)
            references = await root.context.require(MESSAGE_DISPLAY)(page, display_only=True)
        failures = [row for row in references if row['id'].startswith('reference-result-')]
        assert len(failures) == 3
        assert all(row['body']['parts'][0]['rendered']['error'] == '引用的原始内容不可用' for row in failures)
        assert log.reader('test:room').snapshot() == before
        return displayed


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
                prepare_content=prepare_view, tool_names=frozenset({'read_content'}))
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
        async with views.bind() as bound:
            prepare = bound.prepare
            assert prepare is not None
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
        async with views.bind() as bound:
            prepare = bound.prepare
            assert prepare is not None
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
    data_rows = await check_data_display(directory / 'data-display')
    if DISPLAY_SAMPLE is not None:
        sample = json.loads(DISPLAY_SAMPLE.read_text())
        sample['data'] = data_rows
        DISPLAY_SAMPLE.write_text(json.dumps(sample, ensure_ascii=False))
    result['checks'] += ['generic data display', 'failed result keeps body', 'missing, foreign and future display references']
    await check_failure(directory / 'failure')
    await check_lifecycle(directory / 'lifecycle')
    result['checks'] += ['failed provider is not exposure', 'contributor drain', 'closed view', 'conflicting views']
    # 三个 runtime 均已结束；全新进程只加载原事实，不重建场景或重跑工具。
    for failed, name in ((False, 'normal'), (True, 'failure')):
        command = [sys.executable, str(Path(__file__).resolve()), '--restart', str(directory / name)]
        if failed:
            command.append('--failed')
        restarted = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, timeout=60)
        if restarted.returncode:
            raise RuntimeError(f'恢复进程失败 {name}: {restarted.stderr}')
        result['restart_failed' if failed else 'restart_complete'] = json.loads(restarted.stdout)
    return result


async def restart(directory: Path, *, failed: bool) -> dict[str, object]:
    """重开原场景并重放同 ID；完成和失败都不能自动重跑已结算工具。"""
    # 1. 只重开 run 创建的场景，绝不初始化一套替代消息或手工修复旧正文。
    if not (directory / 'sessions.db').is_file() or not (directory / 'effect.txt').is_file():
        raise ValueError('缺少已完成的场景状态')
    with sqlite3.connect(directory / 'sessions.db') as connection:
        before = connection.execute('SELECT * FROM messages ORDER BY session_key, seq').fetchall()
    effect = (directory / 'effect.txt').read_bytes()
    log = MessageLog(directory / 'sessions.db')
    artifacts = ArtifactStore(directory / 'sessions.db')
    host = PluginManager([directory / 'plugins'], workspace=directory / 'workspace',
                         installed_cache_root=directory / 'home/cache', message_log=log,
                         channel_attachment_store=ChannelAttachmentArtifactStore(
                             workspace=directory / 'workspace', metadata_store=artifacts))
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        calls = root.context.require(ServiceKey('fixture.calls'))
        original = log.reader('test:room').get('u1')
        assert original is not None
        repeated = await root.context.require(CHANNEL_INPUT)(
            'test:room', 'u1', ChannelInboundMessage('test', 'user', 'room',
                'failure scenario' if failed else 'read then read again',
                datetime(2026, 9, 5, tzinfo=UTC), {}))
        assert repeated == original
        # 2. 终止会排空已接纳任务；最后核对真实效果、消息和数据库完整性。
        await host.terminate_all()
        assert calls == []
        assert (directory / 'effect.txt').read_bytes() == effect
        with sqlite3.connect(directory / 'sessions.db') as connection:
            assert connection.execute('SELECT * FROM messages ORDER BY session_key, seq').fetchall() == before
            assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            assert connection.execute('PRAGMA foreign_key_check').fetchall() == []
        return {'messages': len(before), 'same_id_receipt': True, 'model_calls': 0,
                'original_rows_equal': True, 'tool_effect_equal': True,
                'failed': failed, 'fresh_process': True, 'integrity': 'ok', 'foreign_keys': []}
    finally:
        try:
            await host.terminate_all()
        finally:
            try:
                log.close()
            finally:
                artifacts.close()


async def check_directory(directory: Path) -> dict[str, object]:
    """用真实目录工具返回页跑同一个完整显示与回读闭环。"""
    global LONG
    producer = directory / 'directory'
    producer.mkdir()
    for index in range(220):
        (producer / (f'{index:06d}-' + 'x' * 150)).touch()
    page = await ListDirOperation(enable_bridge=False).execute(str(producer))
    assert isinstance(page, str) and 8192 < len(page) and len(page.encode()) <= 10000
    assert 'after=' in page and '还有条目' in page
    LONG = page
    result = await run(directory / 'consumer')
    result['directory_page'] = {'bytes': len(page.encode()), 'characters': len(page),
                                'continuation_preserved': True, 'producer_calls': 1}
    return result


def main() -> None:
    """默认跑原文场景；可选目录页，恢复模式只供已创建场景的子进程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--display-sample', type=Path)
    parser.add_argument('--directory-page', action='store_true')
    parser.add_argument('--restart', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--failed', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    global DISPLAY_SAMPLE
    DISPLAY_SAMPLE = args.display_sample
    if args.restart is not None:
        print(json.dumps(asyncio.run(restart(args.restart, failed=args.failed)), ensure_ascii=False))
        return
    with TemporaryDirectory(prefix='akashic-content-view-') as temporary:
        check = check_directory if args.directory_page else run
        print(json.dumps(asyncio.run(check(Path(temporary))), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
