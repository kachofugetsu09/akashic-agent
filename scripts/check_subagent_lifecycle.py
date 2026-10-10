"""以真实消息、Task 与文件副作用验证子任务失败、取消和恢复。"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--crash-worker', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

from agent.plugin_composition import CompositionRoot, PluginRuntime
from plugins.ledger.contract import BINDINGS
from plugins.ledger.bindings import Bindings
from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
)
from agent.plugin_composition.tasks import TASKS, PluginTasks, TaskServiceClosed
from plugins.ledger.contract import ContentPart, Control, Input, Output
from plugins.content.plugin import Content
from plugins.conversation.plugin import check_origin
from plugins.subagent.inputs import CONTENT, CHECK_ORIGIN
from plugins.subagent.request import PROFILE_TOOLS, Request
from plugins.subagent.runtime import SUBAGENT_PROGRAM
from plugins.subagent.tools import Prepared, Spawn
from plugins.tools.execution import Result, ToolExecution
from plugins.ledger.log import MessageLog


class Fixture:
    """只控制故障时点，结算继续使用真实 owner 与持久存储。"""

    def __init__(self, directory: Path, mode: str):
        self.directory, self.mode = directory, mode
        self.entered, self.release, self.output_saved = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.calls = 0
        self.root = CompositionRoot('subagent-' + mode)
        self.log = MessageLog(directory / 'sessions.db')
        self.tasks = PluginTasks()
        self.bindings = Bindings(self.log, self.root.context)

    async def open(self):
        """装配真实服务，并固定一个已准备请求的合法 binding 描述符。"""
        async def core(ctx):
            for key, value in [(MESSAGE_CATALOG, self.log.catalog()), (MESSAGE_WRITERS, MessageWriters(self.log)),
                               (OWNER_STATE, OwnerState(self.log)), (SESSION_ADMISSION, SessionAdmission(self.log)),
                               (TASKS, self.tasks), (BINDINGS, self.bindings), (CHECK_ORIGIN, check_origin)]:
                await ctx.provide(key, value)
            await ctx.provide(CONTENT, Content(ctx))

        async def jobs(ctx):
            self.ctx = ctx
            await ctx.provide(SUBAGENT_PROGRAM, self.program)

        await self.root.mount(core, name='core')
        await self.root.mount(jobs, name='subagent', inject=(MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE,
                              SESSION_ADMISSION, TASKS, BINDINGS, CONTENT, CHECK_ORIGIN),
                              runtime=PluginRuntime('subagent', 'fixture', args.source / 'plugins/subagent',
                                                    self.directory / 'plugin-data', self.directory, {},
                                                    workspace_roots=('subagent-runs',),
                                                    workspace_files=('memory/spawn_trace.jsonl',)))
        # 1. 隔离准备后的执行边界；不把 fixture 当正式 Manager 安装验收。
        descriptor = {'version': 2, 'origins': {'subagent': 'fixture'},
                      'service': SUBAGENT_PROGRAM.name, 'metadata': {}}
        identity = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.log.save_binding(identity, descriptor)
        self.request = Request(job_id='1' * 32, label='lifecycle', profile='research', background=False,
                               retry_count=0, parent_session_id='parent', parent_message_id='parent-message',
                               parent_part_index=0, origin=None, sink=None, program_binding=identity,
                               tools={name: identity for name in PROFILE_TOOLS['research']})
        self.spawn = Spawn(self.ctx, {}, {})
        return self

    async def program(self, task, reader, request):
        """故障和取消都发生在实际 Task 内，最终正文由真实 writer 提交。"""
        self.calls += 1
        self.entered.set()
        if self.mode == 'error':
            raise RuntimeError('program failed before output')
        if self.mode == 'unexpected-cancel':
            assert task.active
            raise asyncio.CancelledError('program cancelled without owner request')
        if self.mode == 'wait':
            await self.release.wait()
        if self.mode in {'crash', 'recover-effect'}:
            await self.effect()
        writer = self.ctx.require(MESSAGE_WRITERS).bind(
            self.ctx, author='subagent', source='subagent', body_types=(Output,),
            content={'text': self.ctx.require(CONTENT).check_text})(reader.session_id)
        try:
            output = writer.append('final:' + request.job_id, Output((ContentPart('text', 'finished'),), 'complete'))
        finally:
            writer.expire()
        self.output_saved.set()
        if self.mode == 'output-cancel':
            raise asyncio.CancelledError('cancel after committed output')
        if self.mode == 'output-wait':
            await self.release.wait()
        return output

    async def effect(self):
        """真实非幂等文件写入后硬退出；恢复只读取原 started 回执。"""
        fixture = self

        class FileEffect:
            idempotent = False

            async def prepare(self, arguments, source=None):
                return arguments

            async def query(self, key):
                return None

            async def invoke(self, key, arguments):
                with (fixture.directory / 'external-effect.txt').open('a') as stream:
                    stream.write(key + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                assert fixture.mode == 'crash', '恢复不得再次进入非幂等 invoke'
                os._exit(23)

        @asynccontextmanager
        async def open_tool(_binding):
            yield FileEffect()

        async def authorize(_binding, _arguments):
            return {'fixture': 'local-file-only'}

        execution = ToolExecution(self.ctx.require(OWNER_STATE).open_scoped(self.ctx, 'effects'),
                                  self.ctx.require(TASKS).open(self.ctx), open_tool, authorize,
                                  task_key='fixture-effect')
        result = await execution.execute('one-effect', 'local-file', {})
        assert result.outcome == 'error'
        raise RuntimeError('non-idempotent effect interrupted; inspect original receipt')

    async def invoke(self):
        """真实父等待者返回工具结果或明确取消，并记录自身取消计数。"""
        async with self.ctx.runtime_scope():
            try:
                result = await self.spawn.invoke('effect', Prepared(task='local fixture', request=self.request).model_dump())
                return {'result': result.outcome}
            except asyncio.CancelledError:
                caller = asyncio.current_task()
                assert caller is not None
                return {'cancelled': True, 'caller_cancelling': caller.cancelling()}
            except TaskServiceClosed:
                if not args.baseline:
                    raise
                return {'service_closed': True}

    async def facts(self):
        """从原消息和 owner 指针读结果，不用 Task.done 代替验收。"""
        async with self.ctx.runtime_scope():
            found = self.spawn.jobs.read('effect')
            assert found is not None
            record, _, reader = found
            messages = reader.snapshot()
            return record.value['settled'], await self.spawn.jobs.outcome(reader), messages

    async def close(self):
        """先排空自己的 Task，再释放 Root 和数据库。"""
        await self.tasks.close()
        await self.root.dispose()
        self.log.close()


async def run(directory: Path):
    """确定性协调故障、停机、重开与提交竞争，不使用 sleep。"""
    checks = []
    # 1. 程序失败、自行取消和已提交输出后的取消。
    for mode, expected in [('error', 'failed'), ('unexpected-cancel', 'failed'), ('output-cancel', 'completed')]:
        path = directory / mode
        path.mkdir()
        fixture = await Fixture(path, mode).open()
        try:
            parent = await fixture.invoke()
            settled, outcome, messages = await fixture.facts()
            if mode == 'unexpected-cancel' and args.baseline:
                assert parent == {'cancelled': True, 'caller_cancelling': 0}
                assert outcome[0] == 'cancelled' and messages[-1].body.reason == '用户取消子任务'
            else:
                assert parent == {'result': 'success' if expected == 'completed' else 'error'}
                assert outcome[0] == expected
            assert settled and len(messages) == 2
            checks.append({'case': mode, 'parent': parent, 'outcome': outcome[0], 'settled': settled})
        finally:
            await fixture.close()

    # 2. 用户取消和调用者取消都先保存 pause，且实际程序资源已排空。
    for mode in ['owner-cancel', 'caller-cancel']:
        path = directory / mode
        path.mkdir()
        fixture = await Fixture(path, 'wait').open()
        waiter = asyncio.create_task(fixture.invoke())
        try:
            await fixture.entered.wait()
            if mode == 'owner-cancel':
                async with fixture.ctx.runtime_scope():
                    assert await fixture.spawn.jobs.cancel(fixture.request.job_id)
            else:
                waiter.cancel()
            parent = await waiter
            settled, outcome, messages = await fixture.facts()
            assert outcome[0] == 'cancelled' and settled
            assert messages[-1].body == Control('pause', messages[0].seq, '用户取消子任务')
            assert parent == ({'result': 'error'} if mode == 'owner-cancel' else
                              {'cancelled': True, 'caller_cancelling': 1})
            checks.append({'case': mode, 'parent': parent, 'outcome': outcome[0]})
        finally:
            await fixture.close()

    # 3. 子 owner 停机不伪造用户请求；新 Root 复用原 Input 和效果 key。
    for handled_cancel in [False, True]:
        path = directory / ('shutdown-handled-cancel' if handled_cancel else 'shutdown')
        path.mkdir()
        fixture = await Fixture(path, 'wait').open()

        async def wait_after_cancel():
            if handled_cancel:
                # asyncio 保留已处理的取消计数；它不是这一次用户请求的证据。
                caller = asyncio.current_task()
                assert caller is not None
                caller.cancel()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            return await fixture.invoke()

        waiter = asyncio.create_task(wait_after_cancel())
        await fixture.entered.wait()
        await fixture.tasks.close()
        parent = await waiter
        settled, outcome, prefix = await fixture.facts()
        if args.baseline:
            assert parent == {'service_closed': True} and outcome[0] == 'cancelled'
        else:
            assert parent == {'cancelled': True, 'caller_cancelling': int(handled_cancel)}
            assert not settled and outcome is None and len(prefix) == 1
        await fixture.close()
        if not args.baseline:
            fixture = await Fixture(path, 'success').open()
            try:
                assert await fixture.invoke() == {'result': 'success'}
                settled, outcome, messages = await fixture.facts()
                assert settled and outcome[0] == 'completed' and messages[:1] == prefix
                assert len(messages) == 2 and fixture.calls == 1
                assert await fixture.invoke() == {'result': 'success'} and fixture.calls == 1
            finally:
                await fixture.close()
        checks.append({'case': path.name + '/reopen', 'parent': parent, 'original_input_preserved': True})

    # 4. 最终 Output 已提交时，后来的取消不得生成第二个结果。
    path = directory / 'output-race'
    path.mkdir()
    fixture = await Fixture(path, 'output-wait').open()
    waiter = asyncio.create_task(fixture.invoke())
    try:
        await fixture.output_saved.wait()
        async with fixture.ctx.runtime_scope():
            assert not await fixture.spawn.jobs.cancel(fixture.request.job_id)
        fixture.release.set()
        assert await waiter == {'result': 'success'}
        settled, outcome, messages = await fixture.facts()
        assert settled and outcome[0] == 'completed' and len(messages) == 2
        checks.append({'case': 'output/cancel race', 'one_terminal_message': True})
    finally:
        await fixture.close()

    # 5. 真正子进程在非幂等效果后硬退出，重开不重复外部文件写入。
    path = directory / 'process-crash'
    path.mkdir()
    child = await asyncio.create_subprocess_exec(sys.executable, str(Path(__file__).resolve()),
                                                '--source', str(args.source.resolve()), '--crash-worker', str(path))
    assert await child.wait() == 23
    fixture = await Fixture(path, 'recover-effect').open()
    try:
        assert fixture.log._connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert await fixture.invoke() == {'result': 'error'}
        settled, outcome, messages = await fixture.facts()
        assert settled and outcome[0] == 'failed' and len(messages) == 2
        receipt = fixture.log.owner('plugin:subagent:effects').read('program:one-effect')
        assert receipt.value['phase'] == 'done' and receipt.value['result']['outcome'] == 'error'
        assert len((path / 'external-effect.txt').read_text().splitlines()) == 1
        assert await fixture.invoke() == {'result': 'error'} and fixture.calls == 1
        checks.append({'case': 'process crash after effect/reopen', 'external_writes': 1,
                       'tool_receipt': 'done/error', 'parent_result': 'error'})
    finally:
        await fixture.close()
    return checks


if args.crash_worker is not None:
    async def crash():
        fixture = await Fixture(args.crash_worker, 'crash').open()
        await fixture.invoke()
        raise AssertionError('worker must hard-exit at the real file effect')
    asyncio.run(crash())
else:
    with tempfile.TemporaryDirectory(prefix='akashic-subagent-lifecycle-') as temporary:
        with asyncio.Runner() as runner:
            checks = runner.run(asyncio.wait_for(run(Path(temporary)), 30))
        print(json.dumps({'source': str(args.source.resolve()), 'baseline': args.baseline,
                          'checks': checks, 'formal_workspace': 'not touched',
                          'full_plugin_manager_install': 'not run'}, ensure_ascii=False, indent=2))
