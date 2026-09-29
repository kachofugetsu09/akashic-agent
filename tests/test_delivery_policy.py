import asyncio
from datetime import UTC, datetime
from pathlib import Path
import shutil
import pytest
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from plugins.delivery.records import DeliveryRecords
from session.log import OwnerTransaction
from session.message import Output
from tests.test_default_reply import application, live_root
from tests.support.delivery_sources import sources

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
        async with live_root(host) as root:
            accepted = await root.context.require(CHANNEL_INPUT)(
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


@pytest.mark.asyncio
async def test_slow_destination_does_not_delay_next_fast_receipt(tmp_path):
    """O: real durable selections advance independently, with per-sink order."""
    from contextlib import asynccontextmanager
    from agent.plugin_composition import CompositionRoot
    from agent.plugin_composition.tasks import Tasks
    from plugins.delivery.api import Receipt
    from plugins.delivery.execution import Deliveries
    from plugins.delivery_policy.follow import follow
    from session.log import MessageLog
    from tests.test_message_log import writer

    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("delivery-isolation")
    tasks = Tasks()
    entered, release, fast = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sent = {"slow": [], "fast": []}

    class Sender:
        idempotent = True

        async def send(self, key, address, message):
            if address == "slow" and message.message_id == "first":
                entered.set()
                await release.wait()
            sent[address].append(message.message_id)
            if address == "fast" and message.message_id == "second":
                fast.set()
            return Receipt(status="delivered")

        async def query(self, key, address):
            return None

    @asynccontextmanager
    async def open_sender(binding):
        yield Sender()

    records = DeliveryRecords(log.owner("delivery"), "policy")
    delivery = Deliveries(records, log.catalog(), tasks, open_sender, task_key="delivery")
    sinks = tuple({"name": name, "binding_id": name, "address": name} for name in sent)
    for name in sent:
        log.save_binding(name, {"name": name})
    outputs = writer(log, author="assistant", bodies=(Output,))
    outputs.append("first", Output((), "complete"))
    job = asyncio.create_task(follow(root.context, log.catalog(), lambda: delivery, lambda reader, message: sinks))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        outputs.append("second", Output((), "complete"))
        await asyncio.wait_for(fast.wait(), 2)
        assert records.read("second", "fast")[1].phase == "delivered"
        assert records.read("first", "slow")[1].phase == "started"
        assert sent["fast"] == ["first", "second"]
        release.set()
        # Joining the original effect also observes its durable settlement.
        await delivery.send("first", "slow")
    finally:
        release.set()
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        await tasks.close()
        await root.dispose()
        log.close()
