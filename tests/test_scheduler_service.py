"""Tests for SchedulerService: tick, execution, misfire, rescheduling."""

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from agent.scheduler import (
    LatencyTracker,
    ScheduleCapacityError,
    SchedulerService,
    ScheduledJob,
)
from tests.conftest import drain_tasks, make_job

# ── Helpers ──────────────────────────────────────────────────────


def make_service(tmp_path, mock_push, mock_loop, now, tracker=None):
    return SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop=mock_loop,
        tracker=tracker or LatencyTracker(default=25.0),
        _now_fn=lambda: now,
    )


# ── Execution: INSTANT ───────────────────────────────────────────


async def test_instant_calls_push_not_ai(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(tier="instant", fire_at=fixed_now - timedelta(seconds=1))
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    mock_push.execute.assert_called_once()
    mock_loop.process_direct.assert_not_called()


async def test_instant_push_receives_correct_args(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        channel="telegram",
        chat_id="999",
        message="喝水了",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    mock_push.execute.assert_called_once_with(
        target_channel="telegram",
        target_chat_id="999",
        message="喝水了",
    )


# ── Execution: SOFT ──────────────────────────────────────────────


async def test_soft_calls_process_direct_not_push_directly(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    # fire_at - lead (25s) must be <= now; set fire_at far enough in past
    job = make_job(
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),
        channel="telegram",
        chat_id="123",
        message=None,
        prompt="查询北京天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    mock_loop.process_direct.assert_called_once()
    call_kwargs = mock_loop.process_direct.call_args
    assert call_kwargs.kwargs["content"] == "查询北京天气"
    assert call_kwargs.kwargs["channel"] == "scheduler"
    assert call_kwargs.kwargs["chat_id"] == job.id
    assert call_kwargs.kwargs["session_key"] == f"scheduler:{job.id}"
    assert call_kwargs.kwargs["busy_session_key"] == "telegram:123"
    assert call_kwargs.kwargs["stateless"] is True
    assert call_kwargs.kwargs["disabled_tools"] == [
        "message_push",
        "recall_memory",
        "memorize",
        "forget_memory",
    ]


async def test_soft_sends_ai_response_via_push(
    tmp_path, mock_push, mock_loop, fixed_now
):
    mock_loop.process_direct = AsyncMock(return_value="北京今天晴，15°C")
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),
        prompt="查询北京天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    mock_push.execute.assert_called_once_with(
        target_channel=job.channel,
        target_chat_id=job.chat_id,
        message="北京今天晴，15°C",
    )


async def test_soft_records_latency(tmp_path, mock_push, mock_loop, fixed_now):
    tracker = LatencyTracker(default=25.0)
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now, tracker)
    job = make_job(
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),
        prompt="天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert len(tracker._samples) == 1


# ── Timing: pre-trigger ──────────────────────────────────────────


async def test_soft_not_fired_before_pretrigger(
    tmp_path, mock_push, mock_loop, fixed_now
):
    tracker = LatencyTracker(default=30.0)
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now, tracker)
    # fire_at is 60s in future; pretrigger = fire_at - 30s = now+30s, not yet due
    job = make_job(
        tier="soft",
        fire_at=fixed_now + timedelta(seconds=60),
        prompt="天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    mock_loop.process_direct.assert_not_called()


async def test_instant_not_fired_before_fire_at(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(tier="instant", fire_at=fixed_now + timedelta(seconds=10))
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    mock_push.execute.assert_not_called()


# ── One-shot jobs retained as disabled after firing ──────────────


async def test_at_job_retained_disabled_after_fire(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="at", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert job.id not in svc._jobs
    assert svc.list_jobs() == []
    persisted = svc.store.load()
    assert len(persisted) == 1
    assert persisted[0].id == job.id
    assert persisted[0].enabled is False
    assert persisted[0].run_count == 1


async def test_after_job_retained_disabled_after_fire(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="after", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert job.id not in svc._jobs
    assert svc.store.load()[0].enabled is False
    assert svc.store.load()[0].run_count == 1


async def test_failed_one_shot_is_retained_disabled_without_run_count_increment(
    tmp_path, mock_push, mock_loop, fixed_now
):
    mock_push.execute.side_effect = RuntimeError("push failed")
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="after", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert svc._jobs == {}
    persisted = svc.store.load()
    assert persisted[0].enabled is False
    assert persisted[0].run_count == 0


# ── Every: rescheduling ───────────────────────────────────────────


async def test_every_job_rescheduled_after_fire(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=3600,
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    # Job should still exist
    assert job.id in svc._jobs
    # fire_at should have advanced to approximately now + 1h
    new_fire_at = svc._jobs[job.id].fire_at
    assert new_fire_at > fixed_now


async def test_every_run_count_increments(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=60,
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert svc._jobs[job.id].run_count == 1


def test_add_job_does_not_publish_memory_state_when_persistence_fails(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    svc.store.save = MagicMock(side_effect=RuntimeError("persist failed"))
    job = make_job()

    with pytest.raises(RuntimeError, match="persist failed"):
        svc.add_job(job)

    assert svc.list_jobs() == []


def test_add_job_rejects_eleventh_active_job_without_write(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    for index in range(SchedulerService.MAX_ACTIVE_JOBS):
        svc.add_job(make_job(name=f"job-{index}"))
    before = svc.store.path.read_bytes()
    svc.store.save = MagicMock(wraps=svc.store.save)

    with pytest.raises(ScheduleCapacityError) as exc_info:
        svc.add_job(make_job(name="overflow"))

    assert exc_info.value.code == "schedule_capacity_reached"
    assert exc_info.value.active_jobs == SchedulerService.MAX_ACTIVE_JOBS
    assert len(svc.list_jobs()) == SchedulerService.MAX_ACTIVE_JOBS
    assert svc.store.save.call_count == 0
    assert svc.store.path.read_bytes() == before


def test_add_job_replacement_does_not_consume_capacity(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    jobs = [
        make_job(name=f"job-{index}")
        for index in range(SchedulerService.MAX_ACTIVE_JOBS)
    ]
    for job in jobs:
        svc.add_job(job)

    replacement = make_job(name="replacement")
    replacement.id = jobs[0].id
    svc.add_job(replacement)

    assert len(svc.list_jobs()) == SchedulerService.MAX_ACTIVE_JOBS
    assert svc._jobs[replacement.id] is replacement


def test_disabled_job_does_not_consume_capacity(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    for index in range(SchedulerService.MAX_ACTIVE_JOBS):
        svc.add_job(make_job(name=f"job-{index}"))

    disabled = make_job(name="disabled")
    disabled.enabled = False
    svc.add_job(disabled)

    assert len(svc._jobs) == SchedulerService.MAX_ACTIVE_JOBS
    assert {job.id for job in svc.store.load()} == {
        *svc._jobs,
        disabled.id,
    }
    with pytest.raises(ScheduleCapacityError):
        svc.add_job(make_job(name="overflow"))


def test_legacy_disabled_job_survives_add(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    disabled = make_job(
        name="legacy-disabled",
        fire_at=fixed_now + timedelta(hours=1),
    )
    disabled.enabled = False
    svc.store.save({disabled.id: disabled})

    svc.load_and_recover()
    active = make_job(
        name="new-active",
        fire_at=fixed_now + timedelta(hours=2),
    )
    svc.add_job(active)

    assert set(svc._jobs) == {active.id}
    assert {job.id for job in svc.store.load()} == {disabled.id, active.id}
    assert svc.store.load()[0].enabled is False


def test_cancel_other_keeps_disabled_job(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    disabled = make_job(
        name="disabled-other",
        fire_at=fixed_now + timedelta(hours=1),
    )
    disabled.enabled = False
    target = make_job(
        name="cancel-target",
        fire_at=fixed_now + timedelta(hours=2),
    )
    svc.store.save({disabled.id: disabled, target.id: target})
    svc.load_and_recover()

    assert svc.cancel_job(target.id) is True

    persisted = svc.store.load()
    assert [job.id for job in persisted] == [disabled.id]
    assert persisted[0].enabled is False


def test_empty_persisted_set_does_not_revive_stale_active_job(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    stale = make_job(name="stale")
    svc._jobs[stale.id] = stale
    svc._commit_jobs({})

    fresh = make_job(name="fresh")
    svc.add_job(fresh)

    assert set(svc._jobs) == {fresh.id}
    assert [job.id for job in svc.store.load()] == [fresh.id]


def test_replace_and_cancel_disabled_job(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    disabled = make_job(
        name="replace-me",
        fire_at=fixed_now + timedelta(hours=1),
    )
    disabled.enabled = False
    svc.store.save({disabled.id: disabled})
    svc.load_and_recover()

    replacement = make_job(
        name="replacement",
        fire_at=fixed_now + timedelta(hours=2),
    )
    replacement.id = disabled.id
    svc.add_job(replacement)
    assert set(svc._jobs) == {disabled.id}
    assert svc.store.load()[0].enabled is True

    replacement_disabled = replace(replacement, enabled=False)
    svc.add_job(replacement_disabled)
    assert disabled.id not in svc._jobs
    assert svc.store.load()[0].enabled is False

    assert svc.cancel_job(disabled.id) is True
    assert svc.store.load() == []


def test_cancel_job_keeps_memory_state_when_persistence_fails(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job()
    svc._jobs[job.id] = job
    svc.store.save = MagicMock(side_effect=RuntimeError("persist failed"))

    with pytest.raises(RuntimeError, match="persist failed"):
        svc.cancel_job(job.id)

    assert svc._jobs[job.id] is job


async def test_every_soft_p90_updates_affect_next_trigger(
    tmp_path, mock_push, mock_loop, fixed_now
):
    tracker = LatencyTracker(default=25.0, window=5)
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now, tracker)
    job = make_job(
        trigger="every",
        tier="soft",
        fire_at=fixed_now - timedelta(seconds=30),
        interval_seconds=3600,
        prompt="天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    # P90 should now have a sample (from soft execution)
    assert len(tracker._samples) == 1


async def test_every_soft_cron_pretrigger_advances_past_current_boundary(
    tmp_path, mock_push, mock_loop
):
    """SOFT cron jobs should not re-fire the same nominal boundary."""

    now_ref = {"value": datetime(2025, 6, 1, 7, 59, 40, tzinfo=timezone.utc)}
    svc = SchedulerService(
        store_path=tmp_path / "jobs.json",
        push_tool=mock_push,
        agent_loop=mock_loop,
        tracker=LatencyTracker(default=25.0),
        _now_fn=lambda: now_ref["value"],
    )
    fire_at = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
    job = make_job(
        trigger="every",
        tier="soft",
        fire_at=fire_at,
        cron_expr="0 8 * * *",
        timezone_="UTC",
        message=None,
        prompt="查询北京天气",
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert mock_loop.process_direct.call_count == 1
    assert svc._jobs[job.id].fire_at > fire_at

    now_ref["value"] = datetime(2025, 6, 1, 7, 59, 46, tzinfo=timezone.utc)
    await svc._tick()
    await drain_tasks()

    assert mock_loop.process_direct.call_count == 1


async def test_cancel_inflight_every_job_does_not_reschedule(
    tmp_path, mock_push, mock_loop, fixed_now
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_push(**kwargs):
        started.set()
        await release.wait()
        return "文本已发送"

    mock_push.execute.side_effect = blocked_push
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=60,
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await started.wait()
    assert svc.cancel_job(job.id) is True
    release.set()
    await drain_tasks()

    assert job.id not in svc._jobs
    assert svc.store.load() == []


async def test_run_cancellation_cleans_up_inflight_tasks(
    tmp_path, mock_push, mock_loop, fixed_now
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_push(**kwargs):
        started.set()
        await release.wait()
        return "文本已发送"

    mock_push.execute.side_effect = blocked_push
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=60,
    )
    svc._jobs[job.id] = job
    svc.store.save({job.id: job})

    runner = asyncio.create_task(svc.run())
    await asyncio.sleep(0)
    svc._jobs[job.id].fire_at = fixed_now - timedelta(seconds=1)
    await svc._tick()
    await started.wait()
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert svc._tasks == set()
    assert svc._in_flight == set()
    assert job.id in svc._jobs
    assert svc.store.load()[0].id == job.id
    release.set()


def test_advance_every_jumps_over_long_misfire_window(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(days=365),
        interval_seconds=1,
    )

    assert svc._advance_every(job, fixed_now) == fixed_now + timedelta(seconds=1)


@pytest.mark.parametrize("interval_seconds", [0, None])
def test_advance_every_rejects_invalid_interval(
    tmp_path, mock_push, mock_loop, fixed_now, interval_seconds
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now,
        interval_seconds=interval_seconds,
    )

    with pytest.raises(ValueError, match="interval_seconds 必须为正数"):
        svc._advance_every(job, fixed_now)


@pytest.mark.asyncio
async def test_run_shutdown_logs_non_cancelled_background_failure(
    tmp_path, mock_push, mock_loop, fixed_now, caplog
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_push(**kwargs):
        started.set()
        await release.wait()
        return "文本已发送"

    mock_push.execute.side_effect = blocked_push
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=60,
    )
    svc._jobs[job.id] = job
    svc.store.save({job.id: job})

    runner = asyncio.create_task(svc.run())
    await asyncio.sleep(0)
    svc._jobs[job.id].fire_at = fixed_now - timedelta(seconds=1)
    svc.store.save = MagicMock(side_effect=RuntimeError("persist failed"))
    await svc._tick()
    await started.wait()
    caplog.set_level(logging.ERROR, logger="agent.scheduler")
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert caplog.text.count("scheduler 后台任务异常") == 1


# ── Misfire handling ─────────────────────────────────────────────


def test_misfire_within_grace_loaded(tmp_path, mock_push, mock_loop, fixed_now):
    """Jobs missed within 5min grace period are retained for execution."""
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="at",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=100),  # 100s ago < 300s grace
    )
    # Persist and recover
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id in svc._jobs


def test_misfire_beyond_grace_retained_disabled_after_restart(
    tmp_path, mock_push, mock_loop, fixed_now
):
    """Jobs missed beyond 5min grace remain as disabled terminal records."""
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="at",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=400),  # 400s ago > 300s grace
    )
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id not in svc._jobs
    assert svc.list_jobs() == []
    assert svc.match_job_ids(job.id[:8]) == [job.id]
    persisted = svc.store.load()
    assert len(persisted) == 1
    assert persisted[0].enabled is False

    restarted = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    restarted.load_and_recover()
    assert restarted._jobs == {}
    assert restarted.match_job_ids(job.id) == [job.id]
    assert restarted.store.load()[0].enabled is False


async def test_terminal_disabled_job_does_not_consume_capacity(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    terminal = make_job(
        trigger="at", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[terminal.id] = terminal

    await svc._tick()
    await drain_tasks()

    assert svc._jobs == {}
    assert svc.store.load()[0].enabled is False
    for index in range(SchedulerService.MAX_ACTIVE_JOBS):
        svc.add_job(make_job(name=f"active-{index}"))
    with pytest.raises(ScheduleCapacityError):
        svc.add_job(make_job(name="overflow"))
    assert terminal.id in svc.match_job_ids(terminal.id)


async def test_terminal_disabled_job_is_removed_only_by_explicit_cancel(
    tmp_path, mock_push, mock_loop, fixed_now
):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    terminal = make_job(
        trigger="at", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[terminal.id] = terminal

    await svc._tick()
    await drain_tasks()
    assert svc.store.load()[0].enabled is False

    assert svc.cancel_job(terminal.id) is True
    assert svc.store.load() == []


def test_every_misfire_advances_to_future(tmp_path, mock_push, mock_loop, fixed_now):
    """Recurring jobs missed on restart are advanced to next future fire."""
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        # Missed 3 hours ago, interval is 1h
        fire_at=fixed_now - timedelta(hours=3),
        interval_seconds=3600,
    )
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id in svc._jobs
    assert svc._jobs[job.id].fire_at > fixed_now
    assert svc.store.load()[0].fire_at == svc._jobs[job.id].fire_at


@pytest.mark.asyncio
async def test_run_propagates_scheduler_internal_background_failure(
    tmp_path, mock_push, mock_loop, fixed_now, monkeypatch
):
    real_sleep = asyncio.sleep
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    monkeypatch.setattr("agent.scheduler.asyncio.sleep", controlled_sleep)
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="every",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=1),
        interval_seconds=60,
    )
    svc._jobs[job.id] = job
    svc.store.save({job.id: job})

    runner = asyncio.create_task(svc.run())
    await sleep_started.wait()
    svc._jobs[job.id].fire_at = fixed_now - timedelta(seconds=1)
    svc.store.save = MagicMock(side_effect=RuntimeError("persist failed"))
    await svc._tick()
    await real_sleep(0)
    release_sleep.set()

    with pytest.raises(RuntimeError, match="persist failed"):
        await runner


@pytest.mark.asyncio
async def test_stop_does_not_start_an_extra_tick(
    tmp_path, mock_push, mock_loop, fixed_now, monkeypatch
):
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    monkeypatch.setattr("agent.scheduler.asyncio.sleep", controlled_sleep)
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    tick = AsyncMock()
    monkeypatch.setattr(svc, "_tick", tick)

    runner = asyncio.create_task(svc.run())
    await sleep_started.wait()
    svc.stop()
    release_sleep.set()
    await runner

    tick.assert_not_awaited()


# ── Cancel ───────────────────────────────────────────────────────


def test_cancel_job_by_id(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job()
    svc._jobs[job.id] = job

    result = svc.cancel_job(job.id)

    assert result is True
    assert job.id not in svc._jobs


def test_cancel_nonexistent_returns_false(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    assert svc.cancel_job("nonexistent-id") is False


def test_cancel_by_name(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    j1 = make_job(name="daily-weather")
    j2 = make_job(name="other")
    svc._jobs[j1.id] = j1
    svc._jobs[j2.id] = j2

    cancelled = svc.cancel_job_by_name("daily-weather")

    assert len(cancelled) == 1
    assert j1.id not in svc._jobs
    assert j2.id in svc._jobs
