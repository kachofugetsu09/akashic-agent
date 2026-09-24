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
