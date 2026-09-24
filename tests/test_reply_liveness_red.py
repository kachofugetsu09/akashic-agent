"""持久回复状态失去后继时的红测试。

这些测试会故意失败，直到每个已提交状态都能到达终态，或留下可由新进程恢复的持久工作。
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelRequest,
)
from plugins.content.plugin import check_text
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.reply.follow import follow
from plugins.tools.abandon import follow_abandon, reject_start
from plugins.tools.api import MessageReply, result_message_id
from session.message import ContentPart, Control, Input, Output, ToolResult
from tests import test_message_react as react_fixtures
from tests.test_reply_follow import accept, output, running
from tests.test_tool_abandon import abandon, named_dialogue
from tests.test_tool_execution_receipts import dialogue, environment


@pytest.mark.asyncio
async def test_committed_input_recovers_when_its_wakeup_is_lost(tmp_path):
    """已提交 Input 不能永久依赖一次进程内唤醒。"""
    answered = asyncio.Event()

    async def program(ctx, task, reader, source):
        writer = output(ctx, task, reader, source)
        writer.append("answer", Output((ContentPart("text", "answer"),), "complete"))
        answered.set()

    class BrokenLoop:
        def call_soon_threadsafe(self, callback, *args):
            del callback, args
            raise RuntimeError("stale listener")

    async with running(tmp_path, program) as (log, host, _watcher):
        # 等 reply.follow 订阅后，把一个失效 listener 放到最前面。
        for _ in range(100):
            with log._lock:
                if log._listeners:
                    listeners = tuple(log._listeners.items())
                    break
            await asyncio.sleep(0)
        else:
            raise AssertionError("reply follower did not subscribe")

        poison = asyncio.Event()
        with log._lock:
            log._listeners.clear()
            log._listeners[poison] = cast(Any, BrokenLoop())
            log._listeners.update(listeners)
        try:
            # 已提交事务返回原结果；observer 通知失败不污染已提交的 Input。
            await accept(host, "one", "u1")
            assert log.reader("one").get("u1") is not None
            with log._lock:
                assert log._listeners[poison] is not None
        finally:
            with log._lock:
                log._listeners.pop(poison, None)

        # level-triggered sweep 必须在没有新写入时恢复持久 Input。
        await asyncio.wait_for(answered.wait(), 1)


@pytest.mark.asyncio
async def test_one_lane_admission_fault_does_not_kill_reply_follower():
    """一个 lane 的 open/start 故障不能移除其他 lane 的未来。"""
    updates: asyncio.Queue[dict[str, int]] = asyncio.Queue()
    heads: dict[str, int] = {}
    good_first = asyncio.Event()
    good_second = asyncio.Event()

    class Reader:
        def __init__(self, session_id):
            self.session_id = session_id

        def source_names(self):
            return ("conversation",)

        def head(self, *, source=None):
            del source
            return heads[self.session_id]

    class Catalog:
        async def follow(self):
            while True:
                heads.update(await updates.get())
                yield dict(heads)

        def reader(self, session_id):
            return Reader(session_id)

    class Session:
        def __init__(self, session_id):
            self.session_id = session_id

        async def start(self, program):
            del program
            if self.session_id == "broken":
                await good_first.wait()
                raise OSError("lane admission failed")
            if good_first.is_set():
                good_second.set()
            else:
                good_first.set()
            return None

        def needs_reply(self, reader, source):
            del reader, source
            # 持久停摆已提交后没有新事实，lane 如实退出而不影响其他 lane。
            return False

        async def record_failure(self, error, *, boundary=None):
            del error, boundary

    class Source:
        name = "conversation"
        context = None

        def open(self, session_id):
            return Session(session_id)

    class Sources:
        def __init__(self):
            self.source = Source()

        def entries(self):
            return (self.source,)

        def needs_reply(self, reader, source):
            return self.source.open("broken").needs_reply(reader, source)

        async def changes(self):
            await asyncio.Event().wait()
            yield ()

    class Context:
        @asynccontextmanager
        async def runtime_scope(self):
            yield

        def capture_runtime_scope(self):
            return self

        async def wait_admission_closed(self):
            await asyncio.Event().wait()

        async def close(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    Source.context = Context()

    async def program(task, reader, source):
        del task, reader, source

    watcher = asyncio.create_task(follow(
        cast(Any, Context()), cast(Any, Catalog()), cast(Any, Sources()), program
    ))
    try:
        await updates.put({"broken": 1, "good": 1})
        await asyncio.wait_for(good_first.wait(), 1)
        # 单个 lane 故障不能结束根 follower。
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(watcher), 0.1)
        await updates.put({"broken": 1, "good": 2})
        await asyncio.wait_for(good_second.wait(), 1)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


@pytest.mark.asyncio
async def test_reply_follower_faults_when_drive_makes_no_durable_progress():
    """持久 head 未变化时不能热循环同一个可能收费的程序。"""
    updates: asyncio.Queue[dict[str, int]] = asyncio.Queue()
    hold_repeat = asyncio.Event()
    starts = 0

    class Reader:
        def source_names(self):
            return ("conversation",)

        def head(self, source=None):
            # 真实持久边界：本测试没有消息追加，head 恒为 0。
            return 0

    class Catalog:
        async def follow(self):
            yield await updates.get()
            await asyncio.Event().wait()

        def reader(self, session_id):
            del session_id
            return Reader()

    class FailedTask:
        done = True
        active = False
        boundary_hint = 0

        def on_done(self, callback):
            callback()

        def cancel(self):
            pass

        async def join(self):
            await asyncio.sleep(0)
            raise OSError("failure receipt was not committed")

    class Session:
        def needs_reply(self, reader, source):
            # 失败没有持久停摆；同一 prefix 仍欠回复。
            return True

        async def start(self, program):
            nonlocal starts
            del program
            starts += 1
            if starts > 1:
                await hold_repeat.wait()
            return FailedTask()

    class Source:
        name = "conversation"
        context = None

        def open(self, session_id):
            del session_id
            return Session()

    class Sources:
        def __init__(self):
            self.source = Source()

        def entries(self):
            return (self.source,)

        def needs_reply(self, reader, source):
            return Session().needs_reply(reader, source)

        async def changes(self):
            await asyncio.Event().wait()
            yield ()

    class Context:
        @asynccontextmanager
        async def runtime_scope(self):
            yield

        def capture_runtime_scope(self):
            return self

        async def wait_admission_closed(self):
            await asyncio.Event().wait()

        async def close(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    Source.context = Context()

    async def program(task, reader, source):
        del task, reader, source

    watcher = asyncio.create_task(follow(
        cast(Any, Context()), cast(Any, Catalog()), cast(Any, Sources()), program
    ))
    await updates.put({"one": 1})
    try:
        with pytest.raises(ExceptionGroup) as caught:
            await asyncio.wait_for(watcher, 0.2)
        assert len(caught.value.exceptions) == 1
        assert isinstance(caught.value.exceptions[0], RuntimeError)
        assert "没有持久进展" in str(caught.value.exceptions[0])
        assert starts == 1
    finally:
        hold_repeat.set()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


@pytest.mark.asyncio
async def test_terminal_output_releases_lane_before_cleanup_finishes(tmp_path):
    """已提交业务终态不能与物理清理共享存活性。"""
    terminal = asyncio.Event()
    draining = asyncio.Event()
    release = asyncio.Event()
    next_done = asyncio.Event()

    async def program(ctx, task, reader, source):
        writer = output(ctx, task, reader, source)
        latest = reader.snapshot()[-1].message_id
        writer.append(
            "answer:" + latest,
            Output((ContentPart("text", latest),), "complete"),
        )
        if latest != "u1":
            next_done.set()
            return
        terminal.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            raise

    async with running(tmp_path, program) as (_log, host, _watcher):
        try:
            await accept(host, "one", "u1")
            await asyncio.wait_for(terminal.wait(), 1)
            await accept(host, "one", "u2")
            await asyncio.wait_for(draining.wait(), 1)
            await asyncio.wait_for(next_done.wait(), 0.2)
        finally:
            release.set()


def _descriptor() -> BoundModelDescriptor:
    return BoundModelDescriptor(
        binding_id="bound",
        plugin_snapshot_id="snapshot",
        model_revision=0,
        model_id="model",
        connection_id="connection",
        driver_id="driver",
        driver_contract_version="1",
        auth_identity="identity",
        model="model",
        role="agent",
        reasoning_effort=None,
        capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(),
        capability_digest="digest",
    )


@pytest.mark.asyncio
async def test_successful_model_response_is_durable_before_return(tmp_path):
    """Model owner 必须先持久化 response，才允许 React 使用。"""
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()

    class Driver:
        max_tool_schemas = None

        async def complete(self, request):
            del request
            return LLMResponse("durable answer")

    response = await _BoundChat(_descriptor(), cast(Any, Driver()), store).complete(
        ModelRequest(())
    )
    assert response.call_record_id is not None
    reopened = ModelsStore(store.path, store.backup_dir, writable=False)
    record = reopened.read_call(response.call_record_id)
    assert record["response"]["content"] == "durable answer"


@pytest.mark.asyncio
async def test_received_model_response_survives_crash_before_output(
    tmp_path, monkeypatch
):
    """provider 成功后崩溃只重放 materialize，不重做外部调用。"""

    class Crash(BaseException):
        pass

    calls = 0

    async def complete(request):
        nonlocal calls
        del request
        calls += 1
        return LLMResponse("answer")

    async def invoke(key, arguments):
        del key, arguments
        raise AssertionError("tool must not run")

    async def crash_decode(*args, **kwargs):
        del args, kwargs
        raise Crash("process died after provider response")

    with monkeypatch.context() as patch:
        patch.setattr(react_fixtures, "_decode_text", crash_decode)
        async with react_fixtures.runtime(
            tmp_path,
            complete,
            invoke,
        ) as (conversation, _log, _store, run):
            await conversation.accept("u1", Input((ContentPart("text", "hello"),)))
            with pytest.raises(Crash, match="provider response"):
                await (await conversation.start(run)).join()

    async with react_fixtures.runtime(
        tmp_path,
        complete,
        invoke,
    ) as (conversation, _log, _store, run):
        result = await asyncio.wait_for((await conversation.start(run)).join(), 1)
        assert isinstance(result.body, Output) and result.body.finish == "complete"
    assert calls == 1


@pytest.mark.asyncio
async def test_orphaned_started_model_call_reconciles_without_replay(
    tmp_path, monkeypatch
):
    """started 孤儿处于不确定窗口，不能授权一次全新请求。"""

    class Crash(BaseException):
        pass

    calls = 0

    async def complete(request):
        nonlocal calls
        del request
        calls += 1
        if calls == 1:
            raise Crash("process died in provider window")
        return LLMResponse("blind replay")

    async def invoke(key, arguments):
        del key, arguments
        raise AssertionError("tool must not run")

    async with react_fixtures.runtime(
        tmp_path,
        complete,
        invoke,
    ) as (conversation, _log, store, run):
        await conversation.accept("u1", Input((ContentPart("text", "hello"),)))

        def lose_settlement(*args, **kwargs):
            del args, kwargs
            raise OSError("settlement store unavailable")

        monkeypatch.setattr(store, "finish_call", lose_settlement)
        with pytest.raises(Crash, match="provider window"):
            await (await conversation.start(run)).join()
        assert store.read_calls("", 10)[0]["state"] == "started"

    async with react_fixtures.runtime(
        tmp_path,
        complete,
        invoke,
    ) as (conversation, _log, store, run):
        task = await conversation.start(run)
        assert task is not None
        try:
            await asyncio.wait_for(task.join(), 1)
        except Exception:
            pass
        assert all(row["state"] != "started" for row in store.read_calls("", 10))
    assert calls == 1


@pytest.mark.asyncio
async def test_tool_result_releases_waiter_before_cleanup_finishes(environment):
    """持久 ToolResult 先关闭业务等待，物理清理独立退出。"""
    log, _state, tasks, probe, _permissions, execution = environment
    reply = dialogue(log)
    cleanup_entered = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def open_tool(binding):
        assert binding == "fixed-A"
        try:
            yield probe
        finally:
            cleanup_entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue

    execution._open_tool = open_tool
    running_call = asyncio.create_task(execution.execute_call(reply))
    try:
        await asyncio.wait_for(cleanup_entered.wait(), 1)
        assert any(
            isinstance(message.body, ToolResult) for message in reply.reader.snapshot()
        )
        result = await asyncio.wait_for(asyncio.shield(running_call), 0.2)
        assert result.outcome == "success"
    finally:
        release.set()
        await asyncio.gather(running_call, return_exceptions=True)
        await tasks.close()


@pytest.mark.asyncio
async def test_abandon_releases_lane_when_model_ignores_cancel(tmp_path):
    """持久 abandon 先关闭逻辑等待，旧 provider 独立退出。"""
    provider_entered = asyncio.Event()
    release = asyncio.Event()
    next_model_call = asyncio.Event()
    calls = 0

    async def complete(request):
        nonlocal calls
        del request
        calls += 1
        if calls > 1:
            next_model_call.set()
            return LLMResponse("new answer")
        provider_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        return LLMResponse("late abandoned answer")

    async def invoke(key, arguments):
        del key, arguments
        raise AssertionError("tool must not run")

    async with react_fixtures.runtime(
        tmp_path,
        complete,
        invoke,
    ) as (conversation, log, _store, run):
        next_start = None
        try:
            await conversation.accept("u1", Input(()))
            old = await conversation.start(run)
            assert old is not None
            await asyncio.wait_for(provider_entered.wait(), 1)
            head = log.reader("s").head()
            await conversation.control(
                "abandon",
                Control("abandon", head),
                expected_head=head,
                handle=old.handle,
            )
            await conversation.accept("u2", Input(()))
            next_start = asyncio.create_task(conversation.start(run))
            await asyncio.wait_for(next_model_call.wait(), 0.2)
        finally:
            release.set()
            if next_start is not None:
                await asyncio.gather(next_start, return_exceptions=True)


@pytest.mark.asyncio
async def test_abandon_item_fault_isolated_and_later_calls_continue(environment):
    """单个持久 item 故障记录 incident，但不能杀死整个消费者。"""
    log, state, tasks, probe, _, _ = environment
    broken = named_dialogue(log, "broken", "broken-call")
    abandon(log, broken, identity="abandon-broken")
    incidents = []
    incident = asyncio.Event()

    @asynccontextmanager
    async def reply(reader, source, ref):
        if reader.session_id == "broken":
            raise OSError("broken archived binding")
        writer = log.writer(
            reader.session_id,
            author="tool",
            source=source,
            body_types=(ToolResult,),
            content={"text": check_text},
            call_ref=ref,
        )
        try:
            yield MessageReply(
                result_message_id(ref), ref, reader, writer, reject_start
            )
        finally:
            writer.expire()

    def report(kind, message):
        incidents.append((kind, message))
        incident.set()

    watcher = asyncio.create_task(
        follow_abandon(
            log.catalog(),
            state,
            tasks,
            reply,
            task_key="tools",
            report_incident=report,
        )
    )
    try:
        await asyncio.wait_for(incident.wait(), 0.2)
        good = named_dialogue(log, "good", "good-call")
        abandon(log, good, identity="abandon-good")

        async def settled():
            async for _ in good.reader.follow():
                result = good.reader.get(good.message_id)
                if result is not None:
                    return result.body
            raise AssertionError("tool result subscription ended")

        result = await asyncio.wait_for(settled(), 1)
        assert isinstance(result, ToolResult) and result.outcome == "denied"
        assert incidents and not watcher.done()
        assert broken.reader.get(broken.message_id) is None
        assert not probe.calls and probe.query_count == 0
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await tasks.close()
