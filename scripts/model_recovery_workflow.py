"""真实 HTTP 故障、ReAct、文件工具与两份持久账本的隔离工作流验证。"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager, contextmanager
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from agent.plugin_composition.models import BoundModelDescriptor, CapabilitySources, ModelCapabilities
from agent.plugin_composition.tasks import Tasks
from agent.plugin_contracts import ContentPart, Control, Input, Output, ToolResult
from agent.tool_catalog import normalize_tool_result
from core.net.http import HttpClient
from plugins.content.plugin import _decode_text, check_text
from plugins.context.api import check_summary
from plugins.context.plugin import ContextBuilder
from plugins.models.content import render_content
from plugins.models.projection import MessageProjection, check_facts, check_tool_rejection
from plugins.gemini.driver import _Chat as GeminiChat
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.openai_compatible.driver import _BoundChat as PhysicalChat, _ConnectionConfig, _ModelConfig
from plugins.react.plugin import react
from plugins.reply.status import ReplyState
from plugins.sources.session import SourceSession
from plugins.standard_tools.filesystem import WriteFileTool
from plugins.tools.execution import MessageReply, Result, ToolExecution
from plugins.tools.menu import ToolCallDecode
from scripts.check_model_retry import Credential, GeminiCredential
from session.log import MessageLog, SessionAttributes


async def run(folder: Path, *, cancel: bool = False, gemini: bool = False) -> dict:
    """一次输入先写文件，再经历生成失败与超时，最后自动给出正文。"""
    # 1. 真实 HTTP 对端先返回工具，再产生断流或畸形调用及超时。
    plan = deque(['tool', 'timeout', 'final'] if cancel else ['tool', 'broken', 'timeout', 'final'])
    requests, previews, invocations = [], [], []
    held = asyncio.Event()
    arrived = asyncio.Event()
    handlers = set()

    async def serve(reader, writer):
        """按顺序产生 SSE、断流与悬挂连接，记录实际收到的请求。"""
        current = asyncio.current_task()
        handlers.add(current)
        try:
            header = (await reader.readuntil(b'\r\n\r\n')).decode()
            size = int(next(line.split(':', 1)[1] for line in header.split('\r\n') if line.lower().startswith('content-length:')))
            request = json.loads(await reader.readexactly(size))
            requests.append(request)
            kind = plan.popleft()
            if kind == 'timeout':
                arrived.set()
                await held.wait()
                return
            call = {'index': 0, 'id': 'write-1', 'type': 'function', 'function': {
                'name': 'write_file', 'arguments': json.dumps({'path': str(folder / 'receipt.txt'), 'content': 'done\n'})}}
            chunks = [{'choices': [{'delta': {'reasoning_content': 'thinking'}}]}]
            chunks.append({'choices': [{'delta': {'tool_calls': [call]} if kind != 'final' else {'content': 'file verified'}}]})
            if kind != 'broken':
                chunks.append({'choices': [{'delta': {}, 'finish_reason': 'tool_calls' if kind == 'tool' else 'stop'}]})
            if gemini:
                part = {'text': 'file verified'} if kind == 'final' else {'functionCall': {
                    'name': 'write_file', 'args': {'path': str(folder / 'receipt.txt'), 'content': 'done\n'},
                    'id': 'write-1'}}
                chunks = [
                    {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'thinking', 'thought': True}]}}]},
                    {'candidates': [{'content': {'role': 'model', 'parts': [part]}}]},
                    {'candidates': [{'finishReason': 'MALFORMED_FUNCTION_CALL' if kind == 'broken' else 'STOP'}]},
                ]
            body = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
            if kind != 'broken':
                body += 'data: [DONE]\n\n'
            data = body.encode()
            writer.write(f'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len(data) + (100 if kind == "broken" and not gemini else 0)}\r\nConnection: close\r\n\r\n'.encode() + data)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(current)

    server = await asyncio.start_server(serve, '127.0.0.1', 0)
    endpoint = f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}'
    http = HttpClient(lambda: httpx.AsyncClient(base_url=endpoint, trust_env=False, timeout=0.15))
    descriptor = BoundModelDescriptor('model', 'scenario', 0, 'model', 'connection', 'gemini' if gemini else 'openai-compatible', '1',
        'scenario', 'scenario', 'agent', None, ModelCapabilities(context_window=10000), CapabilitySources(), 'scenario')
    store = ModelsStore(folder / 'models.db', folder / 'backups')
    store.initialize()
    native_http = httpx.AsyncClient(base_url=endpoint + '/', trust_env=False, timeout=0.15)
    physical = (GeminiChat(native_http, GeminiCredential(), descriptor) if gemini else
        PhysicalChat(_ConnectionConfig(endpoint, 1, 0.15, 0, False),
                     Credential(), descriptor, _ModelConfig(None, 16), http))
    model = _BoundChat(descriptor, physical, store, max_attempts=None)
    # 2. 使用真实文件工具、模型调用账与消息日志，目标限制在临时目录。
    log, tasks, status = MessageLog(folder / 'sessions.db'), Tasks(), ReplyState()
    log.ensure_session('scenario', SessionAttributes())
    log.save_binding('file', {'target': str(folder)})
    operation = WriteFileTool(folder, enable_bridge=False)

    def writer(body, ref=None):
        return log.writer('scenario', author='scenario', source='conversation', body_types=(body,),
            content={'text': check_text, 'model.facts': check_facts, 'model.tool_rejection': check_tool_rejection,
                     'context.summary': check_summary}, call_ref=ref, check_call=lambda call: None)

    class Target:
        idempotent = False
        async def prepare(self, arguments, source=None):
            return arguments
        async def query(self, key):
            return None
        async def invoke(self, key, arguments):
            invocations.append(key)
            actual = normalize_tool_result(await operation.execute(**dict(arguments)))
            return Result('error' if actual.is_error else 'success', (ContentPart('text', actual.text),))

    @asynccontextmanager
    async def open_tool(binding):
        assert binding == 'file'
        yield Target()

    async def authorize(binding, arguments):
        assert binding == 'file' and Path(arguments['path']) == folder / 'receipt.txt'
        return {'decision': 'allowed'}

    execution = ToolExecution(log.owner('tools'), tasks, open_tool, authorize, task_key='tools')
    class Menu:
        def __init__(self):
            self.schemas = ({'type': 'function', 'function': {'name': 'write_file', 'parameters': operation.parameters}},)
        def decode(self, call):
            assert call.name == 'write_file'
            return ToolCallDecode('file', call.arguments)
        def name(self, binding):
            return 'write_file'
        def parallel(self, binding):
            return False
        def check_start(self, transaction):
            if not task.active:
                raise asyncio.CancelledError
        async def execute(self, ref, *, commit_after=None):
            reply = MessageReply('result:' + ref.message_id, ref, log.reader('scenario'), writer(ToolResult, ref), self.check_start)
            try:
                return await execution.execute_call(reply, commit_after=commit_after)
            finally:
                reply.writer.expire()
    class Content:
        def check_metadata(self, metadata):
            assert not metadata
        async def decode(self, text, references):
            return await _decode_text(text, (), references)
    source = SourceSession(reader=log.reader('scenario'), inputs=writer(Input),
        controls=writer(Control), tasks=tasks)
    original = await source.accept('input', Input((ContentPart('text', 'write receipt then confirm'),)))
    async def program(active, reader, lane):
        """从已接纳 Input 运行生产 ReAct，观察 Reply 草稿和最终提交。"""
        nonlocal task
        task = active
        projection = MessageProjection(model, source=lane, check_summary=check_summary,
            render_content=lambda part: render_content(part, artifacts={}), tool_name=lambda binding: 'write_file',
            read_call=store.read_call)
        async def materials(snapshot):
            return {}
        with status.open(active, 'scenario', lane) as preview:
            @contextmanager
            def observe(message_id):
                with preview(message_id) as publish:
                    async def delta(value):
                        await publish(value)
                        previews.append(status.snapshot('scenario')[0].preview)
                    yield delta
            return await react(reader, writer(Output), model=model, context=ContextBuilder(), projection=projection,
                materials=materials, content=Content(), tools=Menu(), max_output_tokens=100, max_steps=4,
                preview=observe, state=log.owner('react'))
    # 3. 只提交一次 Input，观察自动完成及取消后恢复，核对原持久事实。
    task = None
    try:
        active = await source.start(program)
        if cancel:
            await asyncio.wait_for(arrived.wait(), 5)
            await source.pause('pause')
            try:
                await active.join()
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('pause 没有结束当前模型等待')
            assert len(requests) == 2 and len(invocations) == 1
            await source.resume('resume', 'input')
            active = await source.start(program)
        await asyncio.wait_for(active.join(), 20)
        rows = log.reader('scenario').snapshot()
        assert rows[0] == original and len(invocations) == 1
        assert (folder / 'receipt.txt').read_text() == 'done\n'
        assert sum(isinstance(row.body, ToolResult) for row in rows) == 1
        assert isinstance(rows[-1].body, Output) and rows[-1].body.finish == 'complete'
        assert 'file verified' in [part.value for part in rows[-1].body.parts if isinstance(part, ContentPart)]
        assert all(request == requests[1] for request in requests[2:]), '模型恢复改变了原冻结请求'
        if gemini:
            assert any('functionResponse' in part for row in requests[1]['contents'] for part in row['parts'])
            if not cancel:
                failed = next(call for call in store.read_calls('', 100)
                              if call['failure'] and 'MALFORMED_FUNCTION_CALL' in call['failure'])
                assert failed['response'] is None and failed['partial_response']
                assert failed['send_evidence'] is None and failed['next_attempt_at'] is not None
        else:
            assert any(message['role'] == 'tool' for message in requests[1]['messages'])
        if not cancel:
            assert any(preview.retry_status and not preview.text and not preview.thinking for preview in previews)
        expected_failure = 'CancelledError' if cancel else 'ModelTimeoutError' if gemini else 'ReadTimeout'
        assert any(expected_failure in call['failure'] for call in store.read_calls('', 100) if call['failure'])
        report = {'protocol': 'gemini' if gemini else 'openai-compatible', 'input_messages': 1, 'http_attempts': len(requests), 'file_effects': len(invocations),
                  'final_output': 'file verified', 'frozen_request_preserved': True, 'failed_draft_removed': True}
        log.close()
        log = MessageLog(folder / 'sessions.db')
        assert log.reader('scenario').snapshot() == rows
        return report
    finally:
        held.set()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers)
        await tasks.close()
        status.close()
        log.close()
        store.close()
        await operation.aclose()
        await http.aclose()
        await native_http.aclose()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='model-recovery-workflow-') as path:
        folder = Path(path)
        for gemini, cancel in ((False, False), (False, True), (True, False)):
            target = folder / ('gemini' if gemini else 'cancel' if cancel else 'automatic')
            target.mkdir()
            print(json.dumps(asyncio.run(run(target, cancel=cancel, gemini=gemini)), ensure_ascii=False, indent=2))
