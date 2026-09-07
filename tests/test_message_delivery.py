import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone

import pytest

from agent.plugin_composition.tasks import Tasks
from plugins.delivery.api import Receipt, Sink
from plugins.delivery.execution import Deliveries
from plugins.delivery.history import DeliveryHistory
from plugins.delivery.records import Delivery, DeliveryRecords, delivery_key
from session.log import MessageCatalog, MessageConflict, MessageLog, OwnerTransaction, SessionAttributes
from session.message import ContentPart, ContentReferences, Output


class Provider:
    idempotent = False

    def __init__(self):
        self.sent = []
        self.queries = []
        self.receipt = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error = None

    async def send(self, key, address, message):
        self.sent.append((key, address, message))
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        self.receipt = Receipt(status="delivered", provider_ids=("provider-1",))
        return self.receipt

    async def query(self, key, address):
        self.queries.append((key, address))
        return self.receipt


@pytest.fixture
def env(tmp_path):
    log = MessageLog(tmp_path / "state.db")
    state = log.owner("plugin:delivery")
    records = DeliveryRecords(state, "fixture")
    tasks = Tasks()
    provider = Provider()
    opened = []
    denied = set()

    @asynccontextmanager
    async def open_sender(binding_id):
        opened.append(binding_id)
        if binding_id in denied:
            raise PermissionError("original credential revoked")
        yield provider

    reader = log.reader("chat")
    writer = log.writer("chat", author="reply", source="conversation", body_types=(Output,),
                        content={"text": lambda part: ContentReferences()})
    message = writer.append("answer", Output((ContentPart("text", "unique original body"),), "complete"))
    sink = Sink(name="phone", binding_id="original-adapter", address="original-address")
    execution = Deliveries(records, MessageCatalog(log), tasks, open_sender, task_key="delivery")
    yield log, records, state, reader, writer, message, sink, tasks, provider, opened, denied, execution
    log.close()


@pytest.mark.asyncio
async def test_delivery_history_confirms_once_across_sinks_and_keeps_legacy_unknown(env):
    log, records, state, reader, writer, message, sink, tasks, provider, _, _, execution = env
    history = DeliveryHistory(lambda: state, MessageCatalog(log))
    since = datetime.now(timezone.utc) - timedelta(days=1)
    until = since + timedelta(days=2)
    try:
        records.prepare(reader, message, (sink,))
        assert history.recent(since=since, until=until, limit=10) == ()
        await execution.send(message.message_id, sink.name)
        first = history.recent(since=since, until=until, limit=10)
        assert len(first) == 1 and first[0].message == message
        assert first[0].confirmed_at == records.read(message.message_id, sink.name)[1].confirmed_at
        other = sink.model_copy(update={"name": "desktop"})
        records.add(message.message_id, other)
        await execution.send(message.message_id, other.name)
        assert history.recent(since=since, until=until, limit=10) == first
        assert len(provider.sent) == 2

        # 旧 schema 的 delivered 回执没有确认时间；读取和重复发送不能补造时间。
        legacy = writer.append("legacy", Output((ContentPart("text", "old"),), "complete"))
        records.prepare(reader, legacy, (sink,))
        row, _ = records.read(legacy.message_id, sink.name)
        state.transact(lambda tx: tx.save(delivery_key(legacy.message_id, sink.name),
            {"version": 1, "sink": sink.model_dump(mode="json"), "phase": "delivered",
             "receipt": Receipt(status="delivered").model_dump(mode="json")}, expected_version=row.version))
        await execution.send(legacy.message_id, sink.name)
        assert records.read(legacy.message_id, sink.name)[1].confirmed_at is None
        assert history.recent(since=since, until=until, limit=10) == first
        assert len(provider.sent) == 2
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_confirmation_index_failure_rolls_back_receipt_then_queries_original_effect(env, monkeypatch):
    log, records, state, reader, _, message, sink, tasks, provider, _, _, execution = env
    history = DeliveryHistory(lambda: state, MessageCatalog(log))
    since = datetime.now(timezone.utc) - timedelta(days=1)
    until = since + timedelta(days=2)
    original = OwnerTransaction.save

    def fail_index(self, key, value, **kwargs):
        if key.startswith("confirmed-time:"):
            raise OSError("index disk fault")
        return original(self, key, value, **kwargs)

    try:
        records.prepare(reader, message, (sink,))
        with monkeypatch.context() as patch:
            patch.setattr(OwnerTransaction, "save", fail_index)
            with pytest.raises(OSError, match="index disk fault"):
                await execution.send(message.message_id, sink.name)
        assert records.read(message.message_id, sink.name)[1].phase == "started"
        assert state.read("confirmed-message:" + message.message_id) is None
        assert history.recent(since=since, until=until, limit=10) == ()
        await execution.send(message.message_id, sink.name)
        assert len(provider.sent) == 1 and len(provider.queries) == 1
        assert history.recent(since=since, until=until, limit=10)[0].message == message
    finally:
        await tasks.close()


def test_history_pages_before_filtering_sources_and_stays_in_owner(env):
    log, records, state, reader, writer, message, sink, *_ = env
    history = DeliveryHistory(lambda: state, MessageCatalog(log))
    wake = log.writer("wake-chat", author="wake", source="wake", body_types=(Output,),
                      content={"text": lambda part: ContentReferences()})
    oldest = wake.append("notice", Output((ContentPart("text", "notice"),), "complete"))
    records.prepare(log.reader("wake-chat"), oldest, (sink,), passive=True)
    row, _ = records.read(oldest.message_id, sink.name)
    records.save(oldest.message_id, row, Delivery(sink=sink, phase="delivered", receipt=Receipt(status="delivered")))
    for index in range(130):
        item = writer.append(f"passive-{index}", Output((), "quiet"))
        records.prepare(reader, item, (sink,))
        row, _ = records.read(item.message_id, sink.name)
        records.save(item.message_id, row, Delivery(sink=sink, phase="delivered", receipt=Receipt(status="delivered")))
    log.ensure_session("child", SessionAttributes(visibility="internal", learning="excluded"))
    child = log.writer("child", author="subagent", source="subagent", body_types=(Output,), content={})
    hidden = child.append("internal-result", Output((), "complete"))
    records.prepare(log.reader("child"), hidden, (sink,), passive=False)
    row, _ = records.read(hidden.message_id, sink.name)
    records.save(hidden.message_id, row, Delivery(sink=sink, phase="delivered", receipt=Receipt(status="delivered")))
    # 其他 owner 的同形损坏记录不属于 Delivery 的读取权限。
    log.owner("unrelated").transact(lambda tx: tx.save("confirmed-time:9999", {}, expected_version=None))
    until = datetime.now(timezone.utc) + timedelta(seconds=1)
    result = history.recent(since=until - timedelta(days=1), until=until, limit=1,
                            excluded_sources=frozenset({"conversation"}), visibility="listed")
    assert len(result) == 1 and result[0].message == oldest
    assert history.recent(since=until - timedelta(days=1), until=result[0].confirmed_at, limit=10) == ()


def test_owner_snapshot_keeps_pages_and_messages_at_one_database_view(env, tmp_path):
    log, _, state, reader, _, *_ = env
    # 仅此一次性测试库启用 WAL，让第二连接在快照期间真实提交。
    with closing(sqlite3.connect(tmp_path / "state.db")) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    state.transact(lambda tx: tx.save("page:a", {"value": "original"}, expected_version=None))
    other = MessageLog(tmp_path / "state.db")
    try:
        def read_pages():
            first = state.scan(start="page:", stop="page:z", limit=1)
            # 独立 SQLite 连接在两页之间提交；读取必须保持第一条查询的快照。
            other.owner("plugin:delivery").transact(
                lambda tx: tx.save("page:0", {"value": "late"}, expected_version=None))
            writer = other.writer("chat", author="reply", source="conversation", body_types=(Output,), content={})
            writer.append("late-message", Output((), "complete"))
            assert state.scan(start="page:", stop=first[-1][0], limit=1) == ()
            assert reader.get("late-message") is None
        state.snapshot(read_pages)
        assert state.scan(start="page:", stop="page:a", limit=1)[0][0] == "page:0"
        assert reader.get("late-message") is not None
    finally:
        other.close()


def test_rollback_journal_writer_commits_after_owner_snapshot_closes(env, tmp_path):
    _, _, state, *_ = env
    other = MessageLog(tmp_path / "state.db")
    committing = threading.Event()
    other._connection.set_trace_callback(lambda sql: committing.set() if sql == "COMMIT" else None)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = None
            def read():
                nonlocal pending
                assert state.read("concurrent") is None
                pending = executor.submit(other.owner("plugin:delivery").transact,
                    lambda tx: tx.save("concurrent", {"value": "committed"}, expected_version=None))
                assert committing.wait(2)
                assert state.read("concurrent") is None
            state.snapshot(read)
            assert pending is not None
            assert pending.result(timeout=2).value == {"value": "committed"}
            assert state.read("concurrent").value == {"value": "committed"}
    finally:
        other.close()


@pytest.mark.asyncio
async def test_multisink_selection_is_fixed_before_io_and_never_reselected(env):
    log, records, state, reader, _, message, sink, tasks, provider, opened, _, execution = env
    other = Sink(name="desktop", binding_id="desktop-old", address="desktop-address")
    try:
        selection = records.consume(reader, message, (sink, other))
        assert records.cursor("chat") == message.seq
        assert records.read(message.message_id, "desktop")[1].phase == "prepared"
        assert records.consume(reader, message, ()) == selection
        encoded = json.dumps([(key, str(row.value)) for key, row in state.list()])
        assert "unique original body" not in encoded
        first = await execution.send(message.message_id, sink.name)
        assert await execution.send(message.message_id, sink.name) == first
        assert len(provider.sent) == 1
        assert provider.sent[0][1:] == (sink.address, message)
        assert opened == [sink.binding_id]
        assert records.read(message.message_id, "desktop")[1].phase == "prepared"
        assert reader.get(message.message_id) == message
    finally:
        await tasks.close()


def test_selection_receipts_and_cursor_roll_back_together(env, monkeypatch):
    _, records, state, reader, _, message, sink, *_ = env
    original = OwnerTransaction.save

    def fail_cursor(self, key, value, **kwargs):
        if key.startswith("cursor:"):
            raise OSError("disk fault")
        return original(self, key, value, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(OwnerTransaction, "save", fail_cursor)
        with pytest.raises(OSError, match="disk fault"):
            records.consume(reader, message, (sink,))
    assert state.list() == ()
    assert records.cursor("chat") == -1
    records.consume(reader, message, (sink,))
    assert records.pending() == ((message.message_id, sink.name),)


def test_cursor_cannot_skip_and_explicit_new_sink_cannot_rebind(env):
    _, records, _, reader, writer, message, sink, *_ = env
    second = writer.append("answer-2", Output((ContentPart("text", "next"),), "complete"))
    with pytest.raises(MessageConflict, match="跳过"):
        records.consume(reader, second, (sink,))
    selection = records.consume(reader, message, ())
    records.add(message.message_id, sink)
    assert records.selection(message.message_id) == selection
    with pytest.raises(MessageConflict, match="更换"):
        records.add(message.message_id, sink.model_copy(update={"address": "new-address"}))
    records.consume(reader, second, ())
    assert records.cursor("chat") == second.seq


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,found,idempotent,expected,sends", [
    ("prepared", False, False, "delivered", 1),
    ("started", True, False, "delivered", 0),
    ("started", False, False, "unknown", 0),
    ("started", False, True, "delivered", 1),
    ("unknown", True, False, "delivered", 0),
    ("unknown", False, False, "unknown", 0),
    ("delivered", True, False, "delivered", 0),
])
async def test_restart_preserves_original_effect_and_queries_before_retry(env, tmp_path, phase, found, idempotent, expected, sends):
    log, records, _, reader, _, message, sink, tasks, provider, opened, _, execution = env
    try:
        records.consume(reader, message, (sink,))
        row, old = records.read(message.message_id, sink.name)
        receipt = Receipt(status=phase) if phase in {"unknown", "delivered"} else None
        records.save(message.message_id, row, Delivery(sink=old.sink, phase=phase, receipt=receipt))
        if found:
            provider.receipt = Receipt(status="delivered", provider_ids=("already-sent",))
        provider.idempotent = idempotent
        # 真正关闭并重开数据库；不同实例不携带内存选择或发送状态。
        path = tmp_path / "state.db"
        log.close()
        reopened = MessageLog(path)
        try:
            restored = DeliveryRecords(reopened.owner("plugin:delivery"), "fixture")
            execution = Deliveries(restored, MessageCatalog(reopened), tasks, execution._open_sender, task_key="delivery")
            result = await execution.send(message.message_id, sink.name)
            assert result.status == expected
            assert len(provider.sent) == sends
            assert opened == ([] if phase == "delivered" else [sink.binding_id])
            if phase in {"started", "unknown"}:
                assert provider.queries == [(delivery_key(message.message_id, sink.name), sink.address)]
            if sends:
                assert provider.sent[0] == (delivery_key(message.message_id, sink.name), sink.address, message)
            assert restored.read(message.message_id, sink.name)[1].phase == expected
        finally:
            reopened.close()
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_credential_revocation_preserves_prepared_original_route(env):
    _, records, _, reader, _, message, sink, tasks, provider, opened, denied, execution = env
    try:
        records.consume(reader, message, (sink,))
        denied.add(sink.binding_id)
        with pytest.raises(PermissionError, match="revoked"):
            await execution.send(message.message_id, sink.name)
        assert not provider.sent
        assert records.read(message.message_id, sink.name)[1] == Delivery(sink=sink, phase="prepared")
        assert opened == [sink.binding_id]
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_timeout_is_unknown_and_does_not_resend_non_idempotent_provider(env):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    try:
        records.consume(reader, message, (sink,))
        provider.error = TimeoutError("ack lost")
        with pytest.raises(TimeoutError, match="ack lost"):
            await execution.send(message.message_id, sink.name)
        assert records.read(message.message_id, sink.name)[1].phase == "unknown"
        assert (await execution.send(message.message_id, sink.name)).status == "unknown"
        assert len(provider.sent) == 1
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_cancel_after_start_keeps_unknown_and_original_body(env):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    provider.release.clear()
    try:
        records.consume(reader, message, (sink,))
        waiter = asyncio.create_task(execution.send(message.message_id, sink.name))
        await provider.started.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert records.read(message.message_id, sink.name)[1].phase == "unknown"
        assert reader.get(message.message_id) == message
        assert len(provider.sent) == 1
    finally:
        provider.release.set()
        await tasks.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("domain_cancel", [False, True])
async def test_shutdown_preserves_prepared_but_explicit_cancel_prevents_future_send(env, domain_cancel):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    opening = asyncio.Event()
    released = asyncio.Event()
    ready = asyncio.Event()

    @asynccontextmanager
    async def slow_open(binding):
        assert binding == sink.binding_id
        opening.set()
        try:
            await ready.wait()
            yield provider
        finally:
            released.set()

    execution._open_sender = slow_open
    try:
        records.prepare(reader, message, (sink,))
        waiter = asyncio.create_task(execution.send(message.message_id, sink.name))
        await opening.wait()
        if domain_cancel:
            assert await execution.cancel_prepared(message.message_id, sink.name, "notification withdrawn")
        else:
            waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert released.is_set()
        assert not provider.sent
        assert records.read(message.message_id, sink.name)[1].phase == ("rejected" if domain_cancel else "prepared")
        ready.set()
        if domain_cancel:
            assert (await execution.send(message.message_id, sink.name)).status == "rejected"
            assert not provider.sent
            assert (await execution.retry(message.message_id, sink.name)).status == "delivered"
        else:
            assert (await execution.send(message.message_id, sink.name)).status == "delivered"
        assert len(provider.sent) == 1
        assert not await execution.cancel_prepared(message.message_id, sink.name, "too late")
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_provider_rejection_needs_explicit_retry_with_the_same_effect_key(env):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    original = provider.send

    async def reject(key, address, message):
        provider.sent.append((key, address, message))
        return Receipt(status="rejected", error="provider refused before send")

    provider.send = reject
    try:
        records.prepare(reader, message, (sink,))
        assert (await execution.send(message.message_id, sink.name)).status == "rejected"
        assert not records.pending()
        assert (await execution.send(message.message_id, sink.name)).status == "rejected"
        assert len(provider.sent) == 1
        provider.send = original
        assert (await execution.retry(message.message_id, sink.name)).status == "delivered"
        assert len(provider.sent) == 2
        assert provider.sent[0] == provider.sent[1]
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_duplicate_waiter_cannot_cancel_the_actual_sender(env):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    provider.release.clear()
    joined = asyncio.Event()
    original_admit = tasks.admit

    async def observe(key, callback):
        result = await original_admit(key, callback)
        if isinstance(result, tuple) and result[1] is False:
            joined.set()
        return result

    tasks.admit = observe
    try:
        records.prepare(reader, message, (sink,))
        owner = asyncio.create_task(execution.send(message.message_id, sink.name))
        await provider.started.wait()
        duplicate = asyncio.create_task(execution.send(message.message_id, sink.name))
        await joined.wait()
        duplicate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
        assert records.read(message.message_id, sink.name)[1].phase == "started"
        provider.release.set()
        assert (await owner).status == "delivered"
        assert len(provider.sent) == 1
    finally:
        provider.release.set()
        await tasks.close()


@pytest.mark.asyncio
async def test_explicit_retry_is_not_swallowed_by_a_rejected_task_still_closing(env):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    returned = asyncio.Event()
    close = asyncio.Event()
    retry_waiting = asyncio.Event()
    original = execution._send
    admit = tasks.admit

    async def slow_close(task, message_id, name, before_start):
        result = await original(task, message_id, name, before_start)
        if result.status == "rejected":
            returned.set()
            await close.wait()
        return result

    async def observe(key, callback):
        result = await admit(key, callback)
        if callback.__name__ == "rearm" and result is not None:
            retry_waiting.set()
        return result

    execution._send = slow_close
    tasks.admit = observe
    try:
        records.prepare(reader, message, (sink,))
        await execution.cancel_prepared(message.message_id, sink.name, "withdrawn")
        old = asyncio.create_task(execution.send(message.message_id, sink.name))
        await returned.wait()
        retry = asyncio.create_task(execution.retry(message.message_id, sink.name))
        await retry_waiting.wait()
        assert not provider.sent
        close.set()
        assert (await old).status == "rejected"
        assert (await retry).status == "delivered"
        assert len(provider.sent) == 1
    finally:
        close.set()
        await tasks.close()


@pytest.mark.asyncio
async def test_cancel_prepared_drains_resources_then_preserves_caller_cancellation(env):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    opening, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def open_sender(binding):
        opening.set()
        try:
            await asyncio.Event().wait()
            yield provider
        finally:
            closing.set()
            await release.wait()

    execution._open_sender = open_sender
    try:
        records.prepare(reader, message, (sink,))
        sending = asyncio.create_task(execution.send(message.message_id, sink.name))
        await opening.wait()
        cancelling = asyncio.create_task(execution.cancel_prepared(message.message_id, sink.name, "withdrawn"))
        await closing.wait()
        cancelling.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelling
        with pytest.raises(asyncio.CancelledError):
            await sending
        assert records.read(message.message_id, sink.name)[1].phase == "rejected"
        assert not provider.sent
    finally:
        release.set()
        await tasks.close()


@pytest.mark.asyncio
async def test_passive_reply_covers_sends_while_nonpassive_waits_in_order(env):
    _, records, _, reader, writer, reply, sink, tasks, provider, _, _, execution = env
    one = writer.append("first-notice", Output((ContentPart("text", "one"),), "complete"))
    two = writer.append("second-notice", Output((ContentPart("text", "two"),), "complete"))
    execution.prepare(reader, one, (sink,))
    execution.prepare(reader, two, (sink,))
    execution.prepare(reader, reply, (sink,), passive=True)
    entered = asyncio.Queue()
    exclusive = tasks.exclusive

    @asynccontextmanager
    async def observe(key, *, idle=False):
        entered.put_nowait(idle)
        async with exclusive(key, idle=idle):
            yield

    tasks.exclusive = observe
    try:
        with execution.activity(sink.name, sink.address):
            first = asyncio.create_task(execution.send(one.message_id, sink.name))
            assert await entered.get() is True
            second = asyncio.create_task(execution.send(two.message_id, sink.name))
            assert await entered.get() is True
            passive = asyncio.create_task(execution.send(reply.message_id, sink.name))
            assert await entered.get() is False
            await passive
            assert [item[2].message_id for item in provider.sent] == [reply.message_id]
            assert not first.done() and not second.done()
            assert records.read(one.message_id, sink.name)[1].phase == "prepared"
        await asyncio.gather(first, second)
        assert [item[2].message_id for item in provider.sent] == [reply.message_id, one.message_id, two.message_id]
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_passive_unknown_query_holds_idle_until_actual_probe_finishes(env):
    _, records, _, reader, _, reply, sink, tasks, provider, _, _, execution = env
    execution.prepare(reader, reply, (sink,), passive=True)
    row, value = records.read(reply.message_id, sink.name)
    records.save(reply.message_id, row, Delivery(sink=value.sink, phase="started"))
    probing, release, idle_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def query(key, address):
        probing.set()
        await release.wait()
        return None

    async def wait_idle():
        idle_waiting.set()
        await execution.wait_idle(sink.name, sink.address)

    provider.query = query
    try:
        probe = asyncio.create_task(execution.send(reply.message_id, sink.name))
        await probing.wait()
        idle = asyncio.create_task(wait_idle())
        await idle_waiting.wait()
        assert not idle.done()
        release.set()
        assert (await probe).status == "unknown"
        await idle
        assert records.read(reply.message_id, sink.name)[1].phase == "unknown"
        assert provider.sent == []
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_foreign_owner_cannot_join_an_active_send(env):
    log, records, _, reader, _, reply, sink, tasks, provider, _, _, execution = env
    execution.prepare(reader, reply, (sink,))
    provider.release.clear()
    active = asyncio.create_task(execution.send(reply.message_id, sink.name))
    await provider.started.wait()

    @asynccontextmanager
    async def open_sender(_binding):
        pytest.fail("foreign consumer must not open the provider")
        yield provider

    foreign = Deliveries(DeliveryRecords(log.owner("plugin:delivery"), "foreign"), log.catalog(),
                         tasks, open_sender, task_key="delivery")
    try:
        with pytest.raises(PermissionError, match="owner"):
            await foreign.send(reply.message_id, sink.name)
        assert not active.done()
        assert records.read(reply.message_id, sink.name)[1].phase == "started"
        provider.release.set()
        assert (await active).status == "delivered"
    finally:
        provider.release.set()
        await tasks.close()


def test_unowned_message_advances_only_policy_cursor_and_preserves_explicit_route(env, monkeypatch):
    log, records, state, reader, writer, message, sink, *_ = env
    policy = DeliveryRecords(state, "policy")
    assert policy.consume(reader, message, ()).sinks == ()
    output = log.writer(reader.session_id, author="assistant", source="background:one", body_types=(Output,),
                        content={"text": lambda part: ContentReferences()})
    report = output.append("report", Output((ContentPart("text", "main result"),), "complete"))
    save = OwnerTransaction.save
    def fail_cursor(tx, key, value, **kwargs):
        if key.startswith("cursor:"):
            raise OSError("cursor disk fault")
        return save(tx, key, value, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(OwnerTransaction, "save", fail_cursor)
        with pytest.raises(OSError, match="cursor disk fault"):
            policy.consume(reader, report, None)
    assert policy.cursor(reader.session_id) == message.seq
    assert policy.selection(report.message_id) is None
    assert policy.consume(reader, report, None) is None
    assert policy.consume(reader, report, None) is None
    assert policy.cursor(reader.session_id) == report.seq
    selection = records.prepare(reader, report, (sink,))
    assert selection.recovery_owner == "fixture"
    assert policy.consume(reader, report, ()) == selection
    assert records.read(report.message_id, sink.name)[1].sink == sink


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,idempotent,query_result,expected,sent", [
    ("prepared", True, None, "rejected", 0),
    ("started", True, None, "delivered", 1),
    ("started", False, None, "unknown", 0),
    ("unknown", False, "delivered", "delivered", 0),
])
async def test_before_start_rechecks_after_queue_but_never_rejects_unknown_effect(
    env, phase, idempotent, query_result, expected, sent,
):
    _, records, _, reader, _, message, sink, tasks, provider, _, _, execution = env
    checks = []
    try:
        records.prepare(reader, message, (sink,))
        if phase != "prepared":
            row, _ = records.read(message.message_id, sink.name)
            records.save(message.message_id, row, Delivery(sink=sink, phase=phase,
                receipt=Receipt(status="unknown") if phase == "unknown" else None))
        provider.idempotent = idempotent
        provider.receipt = Receipt(status=query_result) if query_result is not None else None
        expired = False
        queued = asyncio.Event()
        original_exclusive = tasks.exclusive
        @asynccontextmanager
        async def observe_queue(*args, **kwargs):
            queued.set()
            async with original_exclusive(*args, **kwargs):
                yield
        tasks.exclusive = observe_queue
        def before_start():
            checks.append(expired)
            return "expired before send" if expired else None
        with execution.activity(sink.name, sink.address):
            sending = asyncio.create_task(execution.send(message.message_id, sink.name, before_start=before_start))
            await queued.wait()
            assert not checks and not provider.sent
            expired = True
        result = await sending
        assert result.status == expected
        assert len(provider.sent) == sent
        assert checks == ([True] if phase == "prepared" else [])
        assert records.read(message.message_id, sink.name)[1].phase == expected
        if phase == "prepared":
            assert (await execution.send(message.message_id, sink.name)).status == "rejected"
            assert not provider.sent
    finally:
        await tasks.close()
