import asyncio
from datetime import UTC, datetime

import pytest

from agent.plugin_composition.tasks import Tasks
from plugins.react.plugin import _settle
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore
from plugins.scheduler.tools import ScheduleTool
from session.message import CallRef


@pytest.mark.asyncio
async def test_cancel_self_commits_then_lets_original_fire_drain_its_tool(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    job = ScheduledJob(trigger="after", tier="soft", fire_at=datetime.now(UTC),
                       channel="test", chat_id="room", timezone="UTC", prompt="work")
    store.add("schedule", job, "created")
    fire = store.start_fire(job)
    tasks = Tasks()
    target = ScheduleTool(store, tasks, "cancel")
    prepared = await target.prepare({"id": job.id})
    results = []

    class Menu:
        async def execute(self, call):
            result = await target.invoke("cancel-self", prepared)
            results.append(result)
            return result

    async def run(task):
        await _settle(Menu(), CallRef("call", 0))

    try:
        task = await tasks.admit(("fire", fire.key), lambda slot: slot.start(run))
        async with asyncio.timeout(2):
            with pytest.raises(asyncio.CancelledError):
                await task.join()
        assert len(results) == 1 and results[0].outcome == "success"
        assert store.read().fires[fire.key].status == "cancelled"
        assert await target.query("cancel-self") == results[0]
        assert task.done
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_prepared_schedule_and_cancel_replay_the_original_ids_and_times(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    tasks = Tasks()
    now = datetime(2026, 9, 6, tzinfo=UTC)
    schedule = ScheduleTool(store, tasks, "schedule", now=lambda: now)
    cancel = ScheduleTool(store, tasks, "cancel")
    try:
        prepared = await schedule.prepare({"tier": "instant", "trigger": "after", "when": "1h",
            "channel": "test", "chat_id": "room", "timezone": "UTC", "message": "one", "name": "same"})
        result = await schedule.invoke("create-one", prepared)
        original = store.load()[0]
        cancellation = await cancel.prepare({"name": "same"})
        later = await schedule.prepare({"tier": "instant", "trigger": "after", "when": "2h",
            "channel": "test", "chat_id": "room", "timezone": "UTC", "message": "two", "name": "same"})
        await schedule.invoke("create-two", later)
        cancelled = await cancel.invoke("cancel-one", cancellation)
        assert await cancel.query("cancel-one") == cancelled
        assert await schedule.invoke("create-one", prepared) == result
        assert await schedule.query("create-one") == result
        assert len(store.load()) == 1 and store.load()[0].id != original.id
        assert store.load()[0].fire_at.hour == 2
    finally:
        await tasks.close()


@pytest.mark.asyncio
async def test_corrupt_store_is_not_reported_as_invalid_cancel_arguments(tmp_path):
    store = JobStore(tmp_path / "schedules.json")
    store.path.write_text('{"version":2,"jobs":"corrupt"}')
    tasks = Tasks()
    try:
        with pytest.raises(ValueError, match="schema"):
            await ScheduleTool(store, tasks, "cancel").prepare({"name": "valid"})
    finally:
        await tasks.close()
