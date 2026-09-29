"""O: an Alert keeps its EventMail identity without waiting for Content work."""
import asyncio
from datetime import UTC, datetime
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot
from plugins.eventmail.plugin import _WakeServices
from plugins.eventmail.store import EventMailStore
from plugins.wake.admission import Duties
from plugins.wake.api import DriftWakeServices
from plugins.wake._boundary import SemanticInterest
from plugins.wake.state import WakeState


@pytest.mark.asyncio
async def test_due_alert_bypasses_running_content_score(tmp_path):
    mail = EventMailStore(tmp_path / "mail.db")
    mail.initialize()
    state = WakeState(tmp_path / "wake.db")
    root = CompositionRoot("wake-isolation")
    entered, release = asyncio.Event(), asyncio.Event()
    now = datetime.now(UTC)
    mail.submit("feed", "batch", [{"item_id": "one", "revision": "r1", "payload": {"title": "news"}}])

    class Interest:
        async def score(self, texts, *, cutoff):
            entered.set()
            await release.wait()
            return tuple(0.5 for _ in texts)

    class Drift:
        def snapshot(self, now):
            return {"proposals": ()}

    duties = Duties(_WakeServices(mail), cast(DriftWakeServices, Drift()), state, cast(SemanticInterest, Interest()))

    async def maintain():
        async with root.context.runtime_scope():
            return await duties.maintain(now)

    job = asyncio.create_task(maintain())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        receipt = mail.report_alert(source_id="monitor", event_id="alert", payload={"message": "attention"},
                                    observed_at=now)
        async with root.context.runtime_scope():
            admitted = await asyncio.wait_for(duties.check(now), 1)
        assert admitted.owner == "alert"
        assert admitted.pool.due_count == 1 and admitted.pool.scored_count == 0
        assert mail.peek_alert(now)["mail_id"] == receipt["mail_id"]
        assert mail.alert_status("monitor", "alert") == "pending"
        assert not job.done()
    finally:
        release.set()
        await job
        await root.dispose()
