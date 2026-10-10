"""用合成输入量测回复链和本地协议；诊断环境需 tiktoken 0.13.0。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--source', type=Path, required=True)
args_parser.add_argument('--output', type=Path, required=True)
args = args_parser.parse_args()
sys.path[:0] = [str(args.source), str(args.source / 'sdk/python/src')]
import httpx
import tiktoken
from agent.plugin_composition import BoundModelDescriptor, CapabilitySources, ModelCapabilities, ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from core.net.http import HttpClient
from plugins.codex.responses import CodexResponses
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.openai_compatible import driver as compatible
from plugins.opencode_go import driver as opencode
from session.message import Output
from tests.test_default_reply import application, live_root

ENCODING = tiktoken.get_encoding('cl100k_base')
REMINDER = 'SYNTHETIC_PREFIX_REMINDER_871'


def plain(value):
    """把冻结的请求值转换成普通 JSON，保留数组和对象顺序。"""
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def encoded(value):
    return json.dumps(plain(value), ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def measure(value):
    text = encoded(value)
    return {'utf8_bytes': len(text.encode()), 'cl100k_tokens': len(ENCODING.encode(text)), 'sha256': digest(value)}


def without_descriptions(value):
    if isinstance(value, dict):
        return {key: without_descriptions(item) for key, item in value.items() if key != 'description'}
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


def request_metrics(request):
    """分开统计 system、reminder、历史和 schema，不把 BPE 片段相加当真实账单。"""
    groups = {'system': [], 'reminder': [], 'history': []}
    for row in plain(request.messages):
        name = 'system' if row['role'] == 'system' else 'reminder' if REMINDER in encoded(row) else 'history'
        groups[name].append(row)
    tools = plain(request.tools)
    bare = without_descriptions(tools)
    return {**{key: measure(value) for key, value in groups.items()},
            'schemas': measure(tools), 'schemas_without_descriptions': measure(bare),
            'schema_description_json_bytes': len(encoded(tools).encode()) - len(encoded(bare).encode()),
            'tool_order': [tool['function']['name'] for tool in tools],
            'reminder_copies': encoded(request.messages).count(REMINDER),
            'message_roles': [row['role'] for row in request.messages]}


def source_changes(path, reverse):
    """只改变一次性夹具的注册顺序，并通过普通材料能力提供合成提醒。"""
    provider = path / 'test_provider/plugin.py'
    text = provider.read_text()
    text = text.replace('from contextlib import asynccontextmanager',
                        'from plugins.context.contract import MATERIALS\nfrom contextlib import asynccontextmanager')
    text = text.replace('inject = (TOOLS, TASKS, BINDINGS)', 'inject = (TOOLS, TASKS, BINDINGS, MATERIALS)')
    text = text.replace('    calls = []', f'''    calls = []
    async def material(messages, source):
        return {{"reminders": ({{"name":"stable", "text":{REMINDER!r}, "priority":1}},)}}
    await ctx.require(MATERIALS).register(ctx, name="prefix_probe", prepare=material, kind="context")''')
    block = '''    await ctx.require(TOOLS).register(ctx, name="write_evidence", description="record local test evidence",
        parameters={"type":"object"}, open=open)'''
    assert block in text
    names = ['inspect_evidence', 'write_evidence'] if reverse else ['write_evidence', 'inspect_evidence']
    text = text.replace(block, f'''    for tool_name in {names!r}:
        await ctx.require(TOOLS).register(ctx, name=tool_name, description="record local synthetic evidence with a fixed description",
            parameters={{"type":"object", "properties":{{"note":{{"type":"string", "description":"optional synthetic note"}}}}}}, open=open)''')
    if 'load-call' not in text:
        text = text.replace('if len(calls) == 1:', 'if len(calls) % 2 == 1:')
    provider.write_text(text)


async def collect(directory, *, reverse=False, discovery=False, turns=2):
    """从真实 PluginManager 装配、消息提交和 ReAct 取得最终 ModelRequest。"""
    async with application(directory, replying=True, discovery=discovery,
                           extra_sources=lambda path: source_changes(path, reverse)) as (log, host):
        for turn in range(turns):
            async with live_root(host) as root:
                await root.context.require(CHANNEL_INPUT)(
                    'test:room', f'input-{turn}', ChannelInboundMessage(
                        'test', 'user', 'room', f'synthetic turn {turn}', datetime(2026, 9, 5, tzinfo=UTC), {}))
            async def completed():
                async for _ in log.catalog().follow():
                    rows = log.reader('test:room').snapshot()
                    if sum(isinstance(row.body, Output) and row.body.finish == 'complete' for row in rows) >= turn + 1:
                        return
            await asyncio.wait_for(completed(), 10)
        async with live_root(host) as root:
            return list(root.context.require(ServiceKey('fixture.calls')))


class Credential:
    connection_id = auth_identity = 'usage-local-fixture'
    async def read(self):
        return {'api_key': 'local-only', 'driver':'codex', 'access_token':'local-only',
                'account_id':'local-only', 'expires_at':'2099-01-01T00:00:00+00:00'}
    async def refresh(self, payload):
        raise AssertionError('fixture must never refresh credentials')
    @asynccontextmanager
    async def exclusive(self):
        yield


def binding(protocol, endpoint):
    descriptor = BoundModelDescriptor(binding_id='usage-' + protocol, plugin_snapshot_id='fixture',
        model_revision=0, model_id='fixture', connection_id='usage-local-fixture', driver_id=protocol,
        driver_contract_version='fixture', auth_identity='usage-local-fixture', model='fixture', role='agent',
        reasoning_effort=None, capabilities=ModelCapabilities(), capability_sources=CapabilitySources(), capability_digest='fixture')
    http = HttpClient(lambda: httpx.AsyncClient(base_url=endpoint, trust_env=False, timeout=httpx.Timeout(2)))
    if protocol == 'codex':
        physical = CodexResponses(http=http, credential=Credential(), descriptor=descriptor, config={})
    elif protocol == 'compatible':
        physical = compatible._BoundChat(compatible._ConnectionConfig(endpoint, 2, 2, 0, False),
            Credential(), descriptor, compatible._ModelConfig(None, 16), http)
    else:
        physical = opencode._BoundChat(opencode._ConnectionConfig(endpoint, 2, 2, 0), Credential(), descriptor, http)
    return descriptor, physical, http


class Handler(BaseHTTPRequestHandler):
    """记录真实 POST body；响应只有公开协议形状和合成数字。"""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.records.append(body)
        usage = self.server.usage
        if self.server.protocol == 'codex':
            events = [{'type':'response.output_text.delta', 'delta':'done'},
                      {'type':'response.completed', 'response':{'usage':usage}}]
        else:
            events = [{'choices':[{'delta':{'content':'done'}, 'finish_reason':None}]},
                      {'choices':[], 'usage':usage}]
        if body.get('stream'):
            data = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
            data += 'data: [DONE]\n\n' if self.server.protocol != 'codex' else ''
            content_type = 'text/event-stream'
        else:
            data = json.dumps({'choices':[{'message':{'role':'assistant','content':'done'},'finish_reason':'stop'}], 'usage':usage})
            content_type = 'application/json'
        self.send_response(200)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data.encode())))
        self.end_headers()
        self.wfile.write(data.encode())
        self.wfile.flush()
    def log_message(self, *values):
        pass


async def wires(directory, requests):
    """核对三个真实 driver 的最终请求、真实 SQLite usage 回执与关库重开。"""
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    server.records = []
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    report = []
    try:
        for protocol in ('compatible','opencode','codex'):
            server.protocol = protocol
            native = protocol == 'codex'
            cases = [('nested', {'input_tokens':100,'output_tokens':30,'input_tokens_details':{'cached_tokens':80,'cache_write_tokens':7},
                                 'output_tokens_details':{'reasoning_tokens':12}} if native else
                                {'prompt_tokens':100,'completion_tokens':30,'prompt_tokens_details':{'cached_tokens':80,'cache_write_tokens':7},
                                 'completion_tokens_details':{'reasoning_tokens':12}})]
            if not native:
                cases += [('top_level', {'prompt_tokens':100,'completion_tokens':30,'cached_tokens':80}),
                          ('top_level_stream', {'prompt_tokens':100,'completion_tokens':30,'cached_tokens':80}),
                          ('top_level_zero', {'prompt_tokens':100,'completion_tokens':30,'cached_tokens':0}),
                          ('nested_zero_wins', {'prompt_tokens':100,'completion_tokens':30,'cached_tokens':80,'prompt_tokens_details':{'cached_tokens':0}}),
                          ('nested_wins', {'prompt_tokens':100,'completion_tokens':30,'cached_tokens':99,'prompt_tokens_details':{'cached_tokens':80}}),
                          ('deepseek', {'completion_tokens':30,'prompt_cache_hit_tokens':80,'prompt_cache_miss_tokens':20}),
                          ('deepseek_wins', {'prompt_tokens':100,'completion_tokens':30,'prompt_cache_hit_tokens':80,'cached_tokens':99}),
                          ('unknown_cache', {'prompt_tokens':100,'completion_tokens':30})]
            descriptor, physical, http = binding(protocol, f'http://127.0.0.1:{server.server_port}')
            database = directory / (protocol + '.db')
            store = ModelsStore(database, directory / 'backups')
            store.initialize()
            saved_usage = {}
            try:
                bound = _BoundChat(descriptor,physical,store)
                # 1. 发送真实装配的冻结请求，比较最终 provider body。
                for index, (name, request) in enumerate(requests):
                    server.usage = cases[0][1]
                    response = await bound.complete(replace(request,request_key=f'{protocol}-wire-{index}',on_delta=None))
                    assert response.content == 'done'
                    assert response.usage.input_tokens == 100 and response.usage.cached_input_tokens == 80
                    body = server.records[-1]
                    report.append({'protocol':protocol,'kind':'wire','scenario':name, 'wire':measure(body),
                                   'tools':measure(body.get('tools',[])), 'usage':asdict(response.usage)})
                # 2. 回执必须保存同一组用量，公开统计不能丢掉命中字段。
                for name, raw in cases:
                    server.usage = raw
                    key = protocol + '-usage-' + name
                    async def receive(delta):
                        pass
                    response = await bound.complete(replace(requests[0][1],request_key=key,
                        on_delta=receive if name == 'top_level_stream' else None))
                    expected_cache = None if name == 'unknown_cache' else 0 if name in {'nested_zero_wins','top_level_zero'} else 80
                    assert response.usage.input_tokens == 100 and response.usage.output_tokens == 30
                    assert response.usage.cached_input_tokens == expected_cache, (protocol,name,response.usage)
                    rows = store.calls_for_key(key)
                    assert len(rows) == 1 and rows[0]['state'] == 'success'
                    usage = asdict(response.usage)
                    assert plain(rows[0]['usage']) == usage
                    assert asdict(store.read_call_stats(response.call_record_id).usage) == usage
                    saved_usage[key] = usage
                    report.append({'protocol':protocol,'kind':'usage','shape':name,'raw_usage':raw,
                                   'normalized':asdict(response.usage),'call_id':response.call_record_id})
            finally:
                store.close()
                await http.aclose()
            reopened = ModelsStore(database, directory / 'reopen-backups')
            reopened.initialize()
            try:
                # 3. 关库重开后读取原账与窄统计，不重发请求。
                for key, usage in saved_usage.items():
                    row = reopened.calls_for_key(key)[0]
                    assert row['state'] == 'success' and plain(row['usage']) == usage
                    assert asdict(reopened.read_call_stats(row['id']).usage) == usage
            finally:
                reopened.close()
        assert len(server.records) == len(report)
        return report
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


async def run(directory):
    a = await collect(directory/'normal')
    b = await collect(directory/'reversed',reverse=True)
    cold = await collect(directory/'cold',turns=1)
    discovery = await collect(directory/'discovery',discovery=True,turns=1)
    assert len(a) == len(b) == 4 and len(cold)==2 and len(discovery)==3
    assert [request_metrics(row)['reminder_copies'] for row in a] == [1,1,2,2]
    assert all(digest(x.tools) == digest(y.tools) for x,y in zip(a,b))
    assert digest(a[0].tools) == digest(cold[0].tools)
    assert digest(a[0].messages[0]) == digest(cold[0].messages[0])
    assert len({digest(row.tools) for row in discovery}) == 1
    named = [(f'{group}-{index}',request) for group,rows in [('normal',a),('reversed',b),('cold',cold),('unlock',discovery)]
             for index,request in enumerate(rows)]
    wire_rows = await wires(directory,named)
    for protocol in ('compatible','opencode','codex'):
        rows = {item['scenario']:item for item in wire_rows if item['protocol']==protocol and item['kind']=='wire'}
        assert rows['normal-0']['wire']['sha256']==rows['reversed-0']['wire']['sha256']==rows['cold-0']['wire']['sha256']
        assert all(rows[f'normal-{i}']['tools']['sha256']==rows[f'reversed-{i}']['tools']['sha256'] for i in range(4))
        assert len({rows[f'unlock-{i}']['tools']['sha256'] for i in range(3)})==1
    result = {'source':str(args.source),'tokenizer':'tiktoken cl100k_base diagnostic only, not provider billing',
              'requests':[{ 'scenario':name,**request_metrics(request)} for name,request in named],
              'checks':['actual installed two Turns','cold system/tools stability','registration order does not reorder wire schemas',
                        'one reminder per Input identity','group unlock leaves direct schema menu stable'],
              'wire_cases':wire_rows,
              'provider_usage':'all numeric usage values are synthetic localhost responses; real cache hit rate/fees/paid provider unrun'}
    return result


with tempfile.TemporaryDirectory(prefix='model-request-costs-') as temporary:
    result = asyncio.run(asyncio.wait_for(run(Path(temporary)),60))
assert not args.output.exists()
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'requests':len(result['requests']),'http_cases':len(result['wire_cases']),
                  'checks':result['checks'],'report':str(args.output)},ensure_ascii=False))
