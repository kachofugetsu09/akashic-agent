"""用真实 SQLite 写锁和 localhost driver 验证 Models 账本等待。"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, replace
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Any

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

import httpx
from agent.plugin_composition import BoundModelDescriptor, CapabilitySources, ModelCapabilities, ModelRequest
from core.net.http import HttpClient
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.openai_compatible import driver
from core.common.file_io import run_file_io
from plugins.models.contract import (
    InvalidRequestError,
    ModelUnavailableError,
    ModelError,
)
from agent.plugin_composition.models import DriverChatModel, LLMResponse
from agent.plugin_composition import CompositionRoot
from plugins.models.contract import (
    CHAT_MODELS,
    MODEL_DRIVERS,
)
from plugins.models.settings import MODEL_SETTINGS, AddConnection, AddModel, CreateConnectionWithModel, SetDefaultModel
from plugins.models.state import ModelsState


class Handler(BaseHTTPRequestHandler):
    """真实本地 HTTP/SSE，仅测试记账与流消费，不评价生成质量。"""

    def do_GET(self):
        assert self.path.endswith('/models'), self.path
        body = json.dumps({'data': [{'id': 'scenario'}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.posts += 1
        rejected = request.get('messages') == [{'role': 'user', 'content': 'fixture-rejected'}]
        if rejected:
            body = json.dumps({'error': {'message': 'fixture provider rejection', 'type': 'invalid_request_error'}}).encode()
            kind = 'application/json'
        elif request.get('stream'):
            body = ('data: ' + json.dumps({'choices': [{'delta': {'content': 'local-result'}}]}) + '\n\n'
                    + 'data: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            body = json.dumps({'choices': [{'message': {'content': 'local-result'}, 'finish_reason': 'stop'}],
                               'usage': {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}}).encode()
            kind = 'application/json'
        self.send_response(400 if rejected else 200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class Credential:
    connection_id = auth_identity = 'scenario'

    def __init__(self):
        self._payload = {'api_key': 'local-fixture'}
        self._lock = asyncio.Lock()

    async def read(self) -> Mapping[str, str]:
        return dict(self._payload)

    async def refresh(self, payload: Mapping[str, str]) -> None:
        self._payload = dict(payload)

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        async with self._lock:
            yield


class LockedStore(ModelsStore):
    """锁住真实数据库，只在目标操作前安排独立线程释放。"""

    def __init__(self, path, backup_dir, phase):
        super().__init__(path, backup_dir)
        self.phase = phase
        self.loop = asyncio.get_running_loop()
        self.observation = None

    def blocked(self, phase, operation):
        """loop 回执决定释放；基线只能等独立线程的终止上限。"""
        if self.phase != phase or self.observation is not None:
            return operation()
        blocker = sqlite3.connect(self.path, check_same_thread=False)
        blocker.execute('BEGIN IMMEDIATE')
        heartbeat = threading.Event()
        started = time.perf_counter()
        observation = {'phase': phase}
        self.observation = observation

        def tick():
            observation['loop_lag_ms'] = (time.perf_counter() - started) * 1000
            heartbeat.set()

        def release():
            self.loop.call_soon_threadsafe(tick)
            observation['heartbeat_before_release'] = heartbeat.wait(1)
            blocker.rollback()
            blocker.close()

        releaser = threading.Thread(target=release)
        releaser.start()
        try:
            return operation()
        finally:
            releaser.join(2)
            assert not releaser.is_alive()
            observation['operation_ms'] = (time.perf_counter() - started) * 1000

    def resume_call(self, *values, **options):
        return self.blocked('begin', partial(super().resume_call, *values, **options))

    def finish_call(self, *values, **options):
        return self.blocked('finish', partial(super().finish_call, *values, **options))


class BarrierStore(ModelsStore):
    """只在一个真实存储阶段保留线程，供 loop 确定性取消或加入等待。"""

    def __init__(self, path, backup_dir, phase):
        super().__init__(path, backup_dir)
        self.phase, self.used = phase, False
        self.loop = asyncio.get_running_loop()
        self.entered, self.release = asyncio.Event(), threading.Event()

    def pause(self, phase):
        if self.phase == phase and not self.used:
            self.used = True
            self.loop.call_soon_threadsafe(self.entered.set)
            assert self.release.wait(5), 'worker 未获释放；不能伪装已排空'

    def calls_for_key(self, key):
        self.pause('read')
        return super().calls_for_key(key)

    def resume_call(self, *values, **options):
        self.pause('begin-before')
        identity = super().resume_call(*values, **options)
        self.pause('begin-after')
        return identity

    def finish_call(self, *values, **options):
        self.pause('finish-before')
        result = super().finish_call(*values, **options)
        self.pause('finish-after')
        return result


async def checkpoint():
    """让当前已排队的 loop 工作到达明确屏障，不依赖 sleep。"""
    result = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(result.set_result, None)
    await result


async def cancellation_checks(directory, server, descriptor, physical):
    """实际写入前后取消，核对账本、发送数和物理排空。"""
    observations = []
    async def delta(_value):
        pass
    for phase in ['begin-before', 'begin-after', 'finish-before', 'finish-after']:
        store = BarrierStore(directory / ('cancel-' + phase + '.db'), directory / 'backups', phase)
        store.initialize()
        request = ModelRequest([{'role': 'user', 'content': 'fixture'}], request_key='cancel-' + phase, on_delta=delta)
        before = server.posts
        bound = _BoundChat(descriptor, physical, store)
        caller = asyncio.create_task(bound.complete(request))
        try:
            await store.entered.wait()
            caller.cancel()
            await checkpoint()
            assert not caller.done(), '实际工作未结束时不能释放 owner'
            caller.cancel()
            await checkpoint()
            assert not caller.done(), '重复取消也不能放弃正在提交的线程'
            store.release.set()
            try:
                await caller
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('caller cancellation was lost')
            records = store.calls_for_key(request.request_key)
            assert len(records) == 1 and records[0]['state'] != 'started'
            expected = 'success' if phase.startswith('finish') else 'error'
            assert records[0]['state'] == expected
            if phase.startswith('begin'):
                assert records[0]['send_evidence'] == 'unsent' and server.posts == before
            else:
                assert server.posts == before + 1
                if expected == 'error':
                    assert records[0]['send_evidence'] is None and records[0]['next_attempt_at'] is None
            # 取消等待不抹掉真实成功；同 key 复用或拒绝，均不再 POST。
            try:
                replayed = await bound.complete(request)
            except (RuntimeError, TimeoutError) as _model_error:
                if not (ModelError.matches(_model_error, ModelUnavailableError)):
                    raise
                assert expected == 'error'
            else:
                assert expected == 'success' and replayed.content == 'local-result'
            assert server.posts == before + (0 if phase.startswith('begin') else 1)
            observations.append({'case': 'cancel-' + phase, 'state': expected,
                                 'posts': server.posts - before, 'physically_drained': True})
        finally:
            store.release.set()
            await asyncio.gather(caller, return_exceptions=True)
            store.close()
    return observations


async def shared_checks(directory, server, descriptor, physical):
    """同 key 在存储等待之前已有唯一 owner，取消 follower 不杀 owner。"""
    observations = []
    for cancel_follower in [False, True]:
        name = 'shared-cancel-follower' if cancel_follower else 'shared-key'
        store = BarrierStore(directory / (name + '.db'), directory / 'backups', 'read')
        store.initialize()
        bound = _BoundChat(descriptor, physical, store)
        request = ModelRequest([{'role': 'user', 'content': 'fixture'}], request_key=name)
        before = server.posts
        owner = asyncio.create_task(bound.complete(request))
        follower = None
        try:
            await store.entered.wait()
            follower = asyncio.create_task(bound.complete(request))
            await checkpoint()
            assert not owner.done() and not follower.done() and server.posts == before
            if cancel_follower:
                follower.cancel()
                try:
                    await follower
                except asyncio.CancelledError:
                    pass
                assert not owner.done()
            store.release.set()
            response = await owner
            if not cancel_follower:
                assert await follower is response
            assert response.content == 'local-result' and server.posts == before + 1
            assert len(store.calls_for_key(name)) == 1
            # 同 key 的两个等待者共享已结算响应；任一调用者不能改变另一个的事实。
            try:
                setattr(response, 'content', 'caller edit')
            except FrozenInstanceError:
                pass
            else:
                raise AssertionError('shared response accepted a caller edit')
            replayed = await bound.complete(request)
            assert replayed.content == response.content == 'local-result'
            assert replayed.call_record_id == response.call_record_id
            assert server.posts == before + 1
            observations.append({'case': name, 'posts': 1, 'call_rows': 1, 'shared_response_frozen': True})
        finally:
            store.release.set()
            await asyncio.gather(owner, *((follower,) if follower is not None else ()), return_exceptions=True)
            store.close()
    return observations


async def queue_check(directory, server, descriptor, physical):
    """占满四个现有磁盘名额，取消排队的调用不得创建请求或发送。"""
    store = ModelsStore(directory / 'queue.db', directory / 'backups')
    store.initialize()
    loop = asyncio.get_running_loop()
    entered, release, lock = asyncio.Event(), threading.Event(), threading.Lock()
    count = 0
    def occupy():
        nonlocal count
        with lock:
            count += 1
            if count == 4:
                loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return store.calls_for_key('empty')
    workers = [asyncio.create_task(run_file_io(occupy)) for _ in range(4)]
    caller = None
    before = server.posts
    try:
        await entered.wait()
        caller = asyncio.create_task(_BoundChat(descriptor, physical, store).complete(
            ModelRequest([], request_key='queued')))
        await checkpoint()
        assert not caller.done()
        caller.cancel()
        try:
            await caller
        except asyncio.CancelledError:
            pass
        assert not store.calls_for_key('queued') and server.posts == before
        return {'case': 'cancel-queued', 'occupied_slots': count, 'call_rows': 0, 'posts': 0}
    finally:
        release.set()
        await asyncio.gather(*workers, *((caller,) if caller is not None else ()), return_exceptions=True)
        store.close()


async def cross_store_checks(directory, server, descriptor, physical):
    """两个 Root 在空账本读完后竞争同 key，提交时仍只能产生一个效果。"""
    observations = []
    for kind in ['success', 'digest', 'binding', 'terminal', 'live']:
        name = 'cross-store-' + kind
        first_store = BarrierStore(directory / (name + '.db'), directory / 'backups', 'begin-before')
        second_store = BarrierStore(first_store.path, directory / 'backups', 'begin-before')
        first_store.initialize()
        second_store.initialize()
        sent, finish = asyncio.Event(), asyncio.Event()

        async def delta(value):
            if value.get('content_delta'):
                sent.set()
                await finish.wait()

        request = ModelRequest([{'role': 'user', 'content': 'fixture'}], request_key=name,
                               on_delta=delta if kind == 'live' else None)
        other_request = replace(request, messages=[{'role': 'user', 'content': 'different'}]) if kind == 'digest' else request
        other_descriptor = replace(descriptor, binding_id='different') if kind == 'binding' else descriptor
        before = server.posts
        # 终结失败在尚有额度时也不得重付；其他场景只有一个耐久名额。
        budget = 3 if kind == 'terminal' else 1
        first = asyncio.create_task(_BoundChat(descriptor, physical, first_store,
                                              max_attempts=budget).complete(request))
        second = asyncio.create_task(_BoundChat(other_descriptor, physical, second_store,
                                               max_attempts=budget).complete(other_request))
        try:
            await asyncio.gather(first_store.entered.wait(), second_store.entered.wait())
            assert not first_store.calls_for_key(name), '两个 loop 读取都应先于真实提交'
            if kind == 'terminal':
                first.cancel()
                await checkpoint()
            first_store.release.set()
            if kind == 'live':
                await sent.wait()
            elif kind == 'terminal':
                try:
                    await first
                except asyncio.CancelledError:
                    pass
            else:
                assert (await first).content == 'local-result'
            second_store.release.set()
            if kind == 'success':
                assert (await second).content == 'local-result'
            else:
                expected = ValueError if kind in ['digest', 'binding'] else RuntimeError
                try:
                    await second
                except expected as error:
                    if kind not in ['digest', 'binding']:
                        assert ModelError.matches(error, ModelUnavailableError)
                else:
                    raise AssertionError('过时准入不应再次调用 provider')
            finish.set()
            if kind == 'live':
                assert (await first).content == 'local-result'
            records = first_store.calls_for_key(name)
            posts = 0 if kind == 'terminal' else 1
            assert len(records) == 1 and records[0]['attempt'] == 0
            assert server.posts == before + posts
            observations.append({'case': name, 'max_attempts': budget,
                                 'posts': posts, 'call_rows': len(records), 'attempts': [0]})
        finally:
            first_store.release.set()
            second_store.release.set()
            finish.set()
            await asyncio.gather(first, second, return_exceptions=True)
            first_store.close()
            second_store.close()
    return observations


async def queued_settlement_check(directory, server, descriptor, physical: DriverChatModel, *, phase='begin', reject=False):
    """未发送、真实成功和拒绝的结算，排队时重复取消仍保留实际回执。"""
    name = 'queued-settlement-' + phase + ('-rejected' if reject else '')
    store = BarrierStore(directory / (name + '.db'), directory / 'backups',
                         'begin-after' if phase == 'begin' else 'disabled')
    store.initialize()
    if reject:
        with sqlite3.connect(store.path) as connection:
            connection.execute("CREATE TRIGGER reject_finish BEFORE UPDATE ON model_calls BEGIN SELECT RAISE(ABORT, 'fixture finish rejected'); END")
    loop = asyncio.get_running_loop()
    occupied, release, lock = asyncio.Event(), threading.Event(), threading.Lock()
    count = 0

    def occupy():
        nonlocal count
        with lock:
            count += 1
            if count == 4:
                loop.call_soon_threadsafe(occupied.set)
        assert release.wait(5)

    workers = []
    driver_failures: list[tuple[BaseException, ModelError, str]] = []

    class QueuedDriver:
        def estimate_context_tokens(
            self, messages: Sequence[Mapping[str, Any]],
            tools: Sequence[Mapping[str, Any]] = (),
        ) -> int:
            return physical.estimate_context_tokens(messages, tools)

        def estimate_appended_message_tokens(
            self, messages: Sequence[Mapping[str, Any]],
        ) -> int:
            return physical.estimate_appended_message_tokens(messages)

        @property
        def max_tool_schemas(self) -> int | None:
            return physical.max_tool_schemas

        async def complete(self, request: ModelRequest) -> LLMResponse:
            try:
                return await physical.complete(request)
            except (RuntimeError, TimeoutError) as error:
                failure = ModelError.read(error)
                assert failure is not None
                driver_failures.append((error, failure, failure.message))
                raise
            finally:
                workers.extend(asyncio.create_task(run_file_io(occupy)) for _ in range(4))
                await occupied.wait()

    before = server.posts
    bound = _BoundChat(descriptor, physical if phase == 'begin' else QueuedDriver(), store)
    request = ModelRequest([{'role': 'user', 'content': 'fixture-rejected'}] if phase == 'failure' else [],
                           request_key=name)
    caller = asyncio.create_task(bound.complete(request))
    try:
        if phase == 'begin':
            await store.entered.wait()
            workers.extend(asyncio.create_task(run_file_io(occupy)) for _ in range(4))
            caller.cancel()
            await checkpoint()
            store.release.set()
        await occupied.wait()
        await checkpoint()
        await checkpoint()
        caller.cancel()
        await checkpoint()
        caller.cancel()
        await checkpoint()
        await checkpoint()
        assert not caller.done(), '原 call 已提交时，排队的取消结算不能被再次取消遗弃'
        assert store.calls_for_key(name)[0]['state'] == 'started'
        release.set()
        await asyncio.gather(*workers)
        try:
            await caller
        except asyncio.CancelledError:
            assert not reject
        except BaseExceptionGroup as failures:
            assert reject or phase == 'failure'
            settlement = failures
            if phase == 'failure':
                provider_error, settlement_error = failures.exceptions
                assert ModelError.matches(provider_error, InvalidRequestError)
                assert (failure := ModelError.read(provider_error)) is not None and failure.send_evidence == 'rejected'
                original_error, original_value, original_message = driver_failures[0]
                assert ModelError.read(original_error) is original_value
                assert original_value.message == original_message
                assert failure is not original_value and '已尝试' not in original_message
                assert 'fixture provider rejection' in str(provider_error)
                if not reject:
                    assert isinstance(settlement_error, asyncio.CancelledError)
                else:
                    assert isinstance(settlement_error, BaseExceptionGroup)
                    settlement = settlement_error
            if reject:
                assert {type(error).__name__ for error in settlement.exceptions} == {'CancelledError', 'RuntimeError'}
                error = next(error for error in settlement.exceptions if isinstance(error, RuntimeError))
                assert isinstance(error.__cause__, sqlite3.IntegrityError)
        else:
            raise AssertionError('caller cancellation was lost')
        record = store.calls_for_key(name)[0]
        expected = 'success' if phase == 'success' else 'error'
        evidence = {'begin': 'unsent', 'failure': 'rejected', 'success': None}[phase]
        assert record['state'] == ('started' if reject else expected)
        assert record['send_evidence'] == (None if reject else evidence)
        if not reject:
            if phase == 'failure':
                assert record['failure'].startswith('InvalidRequestError: ')
                assert 'fixture provider rejection' in record['failure']
            else:
                assert record['failure'] == ('CancelledError' if phase == 'begin' else None)
            try:
                replayed = await bound.complete(request)
            except (RuntimeError, TimeoutError) as _model_error:
                if not (ModelError.matches(_model_error, ModelUnavailableError)):
                    raise
                assert expected == 'error'
            else:
                assert expected == 'success' and replayed.content == 'local-result'
                assert replayed.call_record_id == record['id']
                assert record['usage']['input_tokens'] == replayed.usage.input_tokens == 7
                assert record['usage']['output_tokens'] == replayed.usage.output_tokens == 3
        posts = 0 if phase == 'begin' else 1
        assert server.posts == before + posts
        return {'case': name, 'occupied_slots': count, 'posts': posts,
                'state': record['state'], 'send_evidence': record['send_evidence'],
                'provider_failure_reported': phase == 'failure',
                'provider_failure_preserved': phase != 'failure' or bool(driver_failures),
                'settlement_failure_reported': reject}
    finally:
        store.release.set()
        release.set()
        await asyncio.gather(caller, *workers, return_exceptions=True)
        store.close()


async def failure_checks(directory, server, descriptor, physical):
    """实际数据库拒写及提交后丢回执，不能被包装成持久成功或再次发送。"""
    observations = []
    store = BarrierStore(directory / 'begin-failure.db', directory / 'backups', 'begin-before')
    store.initialize()
    with sqlite3.connect(store.path) as connection:
        connection.execute("CREATE TRIGGER reject_begin BEFORE INSERT ON model_calls BEGIN SELECT RAISE(ABORT, 'fixture begin rejected'); END")
    before = server.posts
    caller = asyncio.create_task(_BoundChat(descriptor, physical, store).complete(ModelRequest([], request_key='begin-failure')))
    try:
        await store.entered.wait()
        caller.cancel()
        await checkpoint()
        assert not caller.done()
        store.release.set()
        try:
            await caller
        except BaseExceptionGroup as failures:
            kinds = {type(error).__name__ for error in failures.exceptions}
            assert kinds == {'CancelledError', 'IntegrityError'}
        else:
            raise AssertionError('cancel/write failure evidence was lost')
        assert not store.calls_for_key('begin-failure') and server.posts == before
        observations.append({'case': 'cancel-and-real-write-failure', 'errors': sorted(kinds), 'posts': 0})
    finally:
        store.release.set()
        await asyncio.gather(caller, return_exceptions=True)
        store.close()

    class LostReceiptStore(ModelsStore):
        def finish_call(self, *values, **options):
            super().finish_call(*values, **options)
            raise OSError('fixture response lost after real commit')
    store = LostReceiptStore(directory / 'lost-receipt.db', directory / 'backups')
    store.initialize()
    bound = _BoundChat(descriptor, physical, store)
    request = ModelRequest([], request_key='lost-receipt')
    before = server.posts
    try:
        try:
            await bound.complete(request)
        except OSError:
            pass
        else:
            raise AssertionError('commit acknowledgement loss was hidden')
        assert store.calls_for_key('lost-receipt')[0]['state'] == 'success'
    finally:
        store.close()
    reopened = ModelsStore(directory / 'lost-receipt.db', directory / 'backups')
    reopened.initialize()
    try:
        response = await _BoundChat(descriptor, physical, reopened).complete(request)
        assert response.content == 'local-result' and server.posts == before + 1
        with sqlite3.connect(reopened.path) as connection:
            assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        observations.append({'case': 'commit-ack-loss/reopen', 'posts': 1, 'durable_state': 'success'})
    finally:
        reopened.close()
    return observations


async def scope_checks(directory, server, endpoint):
    """公开 Models execution 与真实 Root 停机必须等账本线程排空才关 Store。"""
    observations = []
    for phase in ['begin-after', 'finish-after']:
        name = 'root-drain-' + phase
        root = CompositionRoot(name)
        store = BarrierStore(directory / (name + '.db'), directory / 'backups', 'disabled')
        states = []

        async def models(ctx):
            store.initialize()
            state = ModelsState(store, context=ctx)
            states.append(state)
            await ctx.effect(lambda: store.close, label='models-store')
            await ctx.provide(MODEL_DRIVERS, state.drivers)
            await ctx.provide(CHAT_MODELS, state.chat_models)
            await ctx.provide(MODEL_SETTINGS, state.settings)

        async def provider(ctx):
            await ctx.require(MODEL_DRIVERS).register(ctx, driver.definition())

        caller = closer = None
        try:
            await root.mount(models, name='models')
            await root.mount(provider, name='driver', inject=(MODEL_DRIVERS,))
            state = states[0]
            receipt = await state.settings.apply(CreateConnectionWithModel(
                AddConnection(0, name, name, driver.definition().driver_id,
                              endpoint, name, {'api_key': 'local-fixture'}),
                AddModel(0, name, name, 'chat', 'scenario',
                         ModelCapabilities(context_window=8192, supports_tool_calls=True),
                         CapabilitySources(context_window='fixture')),
            ))
            await state.settings.apply(SetDefaultModel(receipt.revision, 'default', name))
            store.phase = phase
            before = server.posts

            async def run():
                async with state.chat_models.execution() as execution:
                    return await execution.chat('default').complete(ModelRequest(
                        [{'role': 'user', 'content': 'fixture'}], request_key=name))

            caller = asyncio.create_task(run())
            await store.entered.wait()
            closer = asyncio.create_task(root.dispose())
            await checkpoint()
            caller.cancel()
            await checkpoint()
            assert not caller.done() and not closer.done() and store.holds_host_lock
            store.release.set()
            try:
                await caller
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('scope cancellation was lost')
            await closer
            assert not store.holds_host_lock
            records = store.calls_for_key(name)
            expected = 'success' if phase == 'finish-after' else 'error'
            posts = 1 if expected == 'success' else 0
            assert len(records) == 1 and records[0]['state'] == expected
            assert server.posts == before + posts
            observations.append({'case': name, 'state': expected, 'posts': posts,
                                 'root_waited_for_worker': True, 'host_lock_released': True})
        finally:
            store.release.set()
            await asyncio.gather(*[task for task in [caller, closer] if task is not None],
                                 return_exceptions=True)
            await root.dispose()
    return observations


async def check(directory, server, endpoint):
    """同一真实 driver 分别经过开始、首字和终结的 SQLite 锁。"""
    descriptor = BoundModelDescriptor(
        binding_id='scenario', plugin_snapshot_id='scenario', model_revision=0,
        model_id='scenario', connection_id='scenario', driver_id='openai-compatible',
        driver_contract_version='scenario', auth_identity='scenario', model='scenario',
        role='default', reasoning_effort=None, capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(), capability_digest='scenario',
    )
    http = HttpClient(lambda: httpx.AsyncClient(base_url=endpoint, trust_env=False))
    physical = driver._BoundChat(driver._ConnectionConfig(endpoint, 3, 3, 0, False),
                                Credential(), descriptor, driver._ModelConfig(None, 16), http)
    observations = []
    for phase in ['begin', 'finish']:
        store = LockedStore(directory / (phase + '.db'), directory / 'backups', phase)
        store.initialize()
        received = []

        async def delta(value):
            received.append(value)

        before = server.posts
        bound = _BoundChat(descriptor, physical, store)
        try:
            response = await bound.complete(ModelRequest([{'role': 'user', 'content': 'fixture'}],
                                                       request_key=phase, on_delta=delta))
            assert response.content == 'local-result'
            records = store.calls_for_key(phase)
            assert len(records) == 1 and records[0]['state'] == 'success'
            assert records[0]['first_token_ms'] is not None and server.posts == before + 1
            assert any(value.get('content_delta') for value in received)
            assert store.observation['heartbeat_before_release'] is (not args.baseline)
            with sqlite3.connect(store.path) as connection:
                assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            observations.append(store.observation)
        finally:
            store.close()
    if not args.baseline:
        observations += await cancellation_checks(directory, server, descriptor, physical)
        observations += await shared_checks(directory, server, descriptor, physical)
        observations += await cross_store_checks(directory, server, descriptor, physical)
        for phase in ['begin', 'success', 'failure']:
            for reject in [False, True]:
                observations.append(await queued_settlement_check(
                    directory, server, descriptor, physical, phase=phase, reject=reject))
        observations.append(await queue_check(directory, server, descriptor, physical))
        observations += await failure_checks(directory, server, descriptor, physical)
        observations += await scope_checks(directory, server, endpoint)
    return observations


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.posts = 0
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='akashic-model-ledger-') as temporary:
            result = asyncio.run(asyncio.wait_for(check(Path(temporary), server,
                                  f'http://127.0.0.1:{server.server_port}'), 20))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
    print(json.dumps({'source': str(args.source.resolve()), 'baseline': args.baseline,
                      'checks': result, 'posts': server.posts, 'formal_workspace': 'not touched'},
                     ensure_ascii=False, indent=2))
