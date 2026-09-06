import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugin_composition.tasks import Tasks
from plugins.delivery.api import Sink
from plugins.delivery.execution import Deliveries
from plugins.delivery.records import DeliveryRecords
from plugins.delivery_policy.follow import follow
from plugins.delivery_policy.plugin import origin
from session.log import MessageLog, OwnerTransaction
from session.message import ContentPart, ContentReferences, Input, Output
from tests.test_default_reply import application
from tests.test_delivery_bindings import sources
from tests.test_message_delivery import Provider


@pytest.mark.asyncio
async def test_real_input_reply_and_archived_delivery_are_independent_consumers(tmp_path, monkeypatch):
    sources(tmp_path / "plugins")
    shutil.copytree(Path(__file__).parents[1] / "plugins/delivery_policy", tmp_path / "plugins/delivery_policy",
                    ignore=shutil.ignore_patterns("__pycache__"))
    delivered = asyncio.Event()
    original = OwnerTransaction.save

    def observe(self, key, value, **kwargs):
        result = original(self, key, value, **kwargs)
        if key.startswith("delivery:") and value["phase"] == "delivered":
            delivered.set()
        return result

    monkeypatch.setattr(OwnerTransaction, "save", observe)
    async with application(tmp_path, replying=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            accepted = await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "test:room", "u1", ChannelInboundMessage(
                    "test", "user", "room", "do the work", datetime(2026, 9, 6, tzinfo=UTC), {},
                ),
            )
        async with asyncio.timeout(10):
            await delivered.wait()
        messages = log.reader("test:room").snapshot()
        assert messages[0] == accepted
        final = [message for message in messages if isinstance(message.body, Output) and message.body.finish == "complete"]
        assert len(final) == 1
        import json
        effects = [json.loads(line) for line in next((tmp_path / "workspace").rglob("sent.jsonl")).read_text().splitlines()]
        assert len(effects) == 1
        assert effects[0][1:3] == ["room", final[0].message_id]
        records = DeliveryRecords(log.owner("plugin:delivery"), "delivery_policy")
        assert records.read(final[0].message_id, "test")[1].phase == "delivered"
        assert records.cursor("test:room") == final[0].seq


class Scope:
    @asynccontextmanager
    async def runtime_scope(self):
        yield


@pytest.mark.asyncio
async def test_failed_sink_does_not_cancel_other_sink_and_restart_keeps_original_set(tmp_path, monkeypatch):
    log = MessageLog(tmp_path / "state.db")
    records = DeliveryRecords(log.owner("delivery"), "fixture")
    tasks = Tasks()
    bad, good = Provider(), Provider()
    bad.error = TimeoutError("lost ACK")
    good.release.clear()
    finished = asyncio.Event()
    original = OwnerTransaction.save

    def observe(self, key, value, **kwargs):
        row = original(self, key, value, **kwargs)
        if key.startswith("delivery:") and value["phase"] == "delivered":
            finished.set()
        return row

    monkeypatch.setattr(OwnerTransaction, "save", observe)

    @asynccontextmanager
    async def open_sender(binding):
        yield {"bad-A": bad, "good-A": good}[binding]

    def execution():
        return Deliveries(records, log.catalog(), tasks, open_sender, task_key="delivery")

    writer = log.writer("chat", author="reply", source="conversation", body_types=(Output,),
                        content={"text": lambda part: ContentReferences()})
    message = writer.append("answer", Output((ContentPart("text", "original"),), "complete"))
    selected = (
        Sink(name="bad", binding_id="bad-A", address="bad-original"),
        Sink(name="good", binding_id="good-A", address="good-original"),
    )
    watcher = asyncio.create_task(follow(Scope(), log.catalog(), execution, lambda *_: selected))
    try:
        async with asyncio.timeout(3):
            await good.started.wait()
        assert len(bad.sent) == 1
        assert records.read(message.message_id, "good")[1].phase == "started"
        good.release.set()
        async with asyncio.timeout(3):
            await finished.wait()
        assert records.read(message.message_id, "bad")[1].phase == "unknown"
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher
        queried = asyncio.Event()
        query = bad.query

        async def observe_query(key, address):
            result = await query(key, address)
            queried.set()
            return result

        bad.query = observe_query

        def changed_policy(*_):
            pytest.fail("restart must not reselect the original message")

        watcher = asyncio.create_task(follow(Scope(), log.catalog(), execution, changed_policy))
        async with asyncio.timeout(3):
            await queried.wait()
        assert len(good.sent) == len(bad.sent) == 1
        assert records.selection(message.message_id).sinks == ("bad", "good")
    finally:
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher
        await tasks.close()
        log.close()


def test_policy_uses_original_input_prefix_and_excludes_intermediate_and_quiet(tmp_path):
    log = MessageLog(tmp_path / "state.db")
    try:
        content = {"text": lambda part: ContentReferences(), "channel.origin": lambda part: ContentReferences()}
        inputs = log.writer("chat", author="user", source="conversation", body_types=(Input,), content=content)
        outputs = log.writer("chat", author="reply", source="conversation", body_types=(Output,), content=content)
        inputs.append("u1", Input((ContentPart("channel.origin", {"channel": "test", "chat_id": "old", "sender": "user"}),)))
        intermediate = outputs.append("thinking", Output((ContentPart("text", "progress"),), "continue"))
        quiet = outputs.append("quiet", Output((), "quiet"))
        final = outputs.append("answer", Output((ContentPart("text", "final"),), "complete"))
        inputs.append("u2", Input((ContentPart("channel.origin", {"channel": "test", "chat_id": "new", "sender": "user"}),)))
        reader = log.reader("chat")
        assert origin(reader, intermediate, ("conversation",)) is None
        assert origin(reader, quiet, ("conversation",)) is None
        assert origin(reader, final, ("conversation",)) == ("test", "old")
    finally:
        log.close()


@pytest.mark.asyncio
async def test_restart_policy_cannot_send_a_scheduler_notification_cancelled_on_disk(tmp_path, monkeypatch):
    """取消文件已提交但 Delivery 仍 prepared；默认 follower 先启动也没有发送权。"""
    from plugins.scheduler.schedule import ScheduledJob
    from plugins.scheduler.store import JobStore

    store = JobStore(tmp_path / "schedules.json")
    job = ScheduledJob(trigger="after", tier="instant", fire_at=datetime.now(UTC),
                       channel="phone", chat_id="room", timezone="UTC", message="reminder")
    store.add("schedule", job, "created")
    fire = store.start_fire(job)
    log = MessageLog(tmp_path / "state.db")
    reader = log.reader("phone:room")
    writer = log.writer(reader.session_id, author="scheduler", source="scheduler", body_types=(Output,),
                        content={"text": lambda part: ContentReferences()})
    message = writer.append(fire.notification_id, Output((ContentPart("text", "reminder"),), "complete"))
    scheduler = DeliveryRecords(log.owner("delivery"), "scheduler")
    sink = Sink(name="phone", binding_id="original", address="room")
    scheduler.prepare(reader, message, (sink,))
    store.cancel("cancel", (job.id,))
    log.close()

    log = MessageLog(tmp_path / "state.db")
    policy = DeliveryRecords(log.owner("delivery"), "delivery_policy")
    scheduler = DeliveryRecords(log.owner("delivery"), "scheduler")
    tasks = Tasks()
    provider = Provider()
    consumed = asyncio.Event()
    save = OwnerTransaction.save

    def observe(self, key, value, **kwargs):
        result = save(self, key, value, **kwargs)
        if key.startswith("cursor:"):
            consumed.set()
        return result

    monkeypatch.setattr(OwnerTransaction, "save", observe)

    @asynccontextmanager
    async def open_sender(binding):
        yield provider

    def execution():
        return Deliveries(policy, log.catalog(), tasks, open_sender, task_key="delivery")

    def changed_policy(*_):
        pytest.fail("another owner already fixed this selection")

    watcher = asyncio.create_task(follow(Scope(), log.catalog(), execution, changed_policy))
    try:
        async with asyncio.timeout(3):
            await consumed.wait()
        assert policy.pending() == ()
        assert scheduler.pending() == ((message.message_id, sink.name),)
        for operation in (execution().send(message.message_id, sink.name),
                          execution().retry(message.message_id, sink.name),
                          execution().cancel_prepared(message.message_id, sink.name, "foreign")):
            with pytest.raises(PermissionError, match="owner"):
                await operation
        assert store.read().fires[fire.key].status == "cancelled"
        recovery = Deliveries(scheduler, log.catalog(), tasks, open_sender, task_key="delivery")
        assert await recovery.cancel_prepared(message.message_id, sink.name, "任务已被明确取消")
        assert scheduler.read(message.message_id, sink.name)[1].phase == "rejected"
        assert provider.sent == provider.queries == []
    finally:
        watcher.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watcher
        await tasks.close()
        log.close()


@pytest.mark.asyncio
async def test_actual_reply_keeps_target_busy_through_provider_send(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from agent.plugin_composition import ServiceKey

    fixture = ModuleType("delivery_coordination_fixture")
    sending, release = asyncio.Event(), asyncio.Event()

    async def send_started():
        sending.set()
        await release.wait()

    fixture.send_started = send_started
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    sources(tmp_path / "plugins")
    sender = tmp_path / "plugins/test_sender/plugin.py"
    sender.write_text(sender.read_text().replace(
        'async def send(self, key, address, message):',
        'async def send(self, key, address, message):\n            from delivery_coordination_fixture import send_started\n            await send_started()'))
    shutil.copytree(Path(__file__).parents[1] / "plugins/delivery_policy", tmp_path / "plugins/delivery_policy",
                    ignore=shutil.ignore_patterns("__pycache__"))
    async with application(tmp_path, replying=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            await root.require(CHANNEL_INPUT)("test:room", "u1", ChannelInboundMessage(
                "test", "user", "room", "do the work", datetime(2026, 9, 6, tzinfo=UTC), {},
            ))
            async with asyncio.timeout(5):
                await sending.wait()
            first = next(row for row in log.reader("test:room").snapshot()
                         if isinstance(row.body, Output) and row.body.finish == "complete")
            await root.require(CHANNEL_INPUT)("test:room", "u2", ChannelInboundMessage(
                "test", "user", "room", "next input", datetime(2026, 9, 6, tzinfo=UTC), {},
            ))
            async with asyncio.timeout(5):
                async for _ in log.catalog().follow():
                    completed = [row for row in log.reader("test:room").snapshot()
                                 if isinstance(row.body, Output) and row.body.finish == "complete"]
                    if len(completed) == 2:
                        break
            records = DeliveryRecords(log.owner("plugin:delivery"), "delivery_policy")
            assert records.read(first.message_id, "test")[1].phase == "started"
            delivery = root.require(ServiceKey("fixture.delivery"))()
            waiting = asyncio.Event()

            async def idle():
                waiting.set()
                await delivery.wait_idle("test", "room")

            pending = asyncio.create_task(idle())
            await waiting.wait()
            assert not pending.done()
            release.set()
            async with asyncio.timeout(5):
                await pending
            records = DeliveryRecords(log.owner("plugin:delivery"), "delivery_policy")
            assert records.pending() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", [False, True])
async def test_input_commit_blocks_idle_before_reply_program_starts(tmp_path, monkeypatch, pause):
    from agent.plugin_composition import ServiceKey

    sources(tmp_path / "plugins")
    shutil.copytree(Path(__file__).parents[1] / "plugins/delivery_policy", tmp_path / "plugins/delivery_policy",
                    ignore=shutil.ignore_patterns("__pycache__"))
    async with application(tmp_path, replying=True) as (log, host):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            from plugins.conversation.plugin import CONVERSATION
            kind = type(root.require(CONVERSATION)("test:room"))
            start = kind.start
            entering, release = asyncio.Event(), asyncio.Event()

            async def delayed(self, program):
                entering.set()
                await release.wait()
                return await start(self, program)

            monkeypatch.setattr(kind, "start", delayed)
            await root.require(CHANNEL_INPUT)("test:room", "u1", ChannelInboundMessage(
                "test", "user", "room", "do the work", datetime(2026, 9, 6, tzinfo=UTC), {},
            ))
            await entering.wait()
            assert root.require(ServiceKey("fixture.calls")) == []
            waiting = asyncio.Event()
            delivery = root.require(ServiceKey("fixture.delivery"))()

            async def idle():
                waiting.set()
                await delivery.wait_idle("test", "room")

            pending = asyncio.create_task(idle())
            await waiting.wait()
            assert not pending.done()
            if pause:
                await root.require(CHANNEL_INPUT)("test:room", "stop", ChannelInboundMessage(
                    "test", "user", "room", "/stop", datetime(2026, 9, 6, tzinfo=UTC), {},
                ))
                async with asyncio.timeout(5):
                    await pending
                assert root.require(ServiceKey("fixture.calls")) == []
            release.set()
            async with asyncio.timeout(5):
                await pending
            outputs = [row for row in log.reader("test:room").snapshot()
                       if isinstance(row.body, Output) and row.body.finish == "complete"]
            if pause:
                assert outputs == []
                return
            assert len(outputs) == 1
            records = DeliveryRecords(log.owner("plugin:delivery"), "delivery_policy")
            assert records.read(outputs[0].message_id, "test")[1].phase == "delivered"
