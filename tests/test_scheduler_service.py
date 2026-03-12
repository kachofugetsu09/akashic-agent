"""Tests for SchedulerService: tick, execution, misfire, rescheduling."""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, call

import pytest

from agent.scheduler import LatencyTracker, SchedulerService, ScheduledJob
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
        channel="telegram", chat_id="999", message="喝水了"
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
    assert call_kwargs.kwargs["channel"] == "telegram"
    assert call_kwargs.kwargs["chat_id"] == "123"


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
        channel=job.channel, chat_id=job.chat_id, message="北京今天晴，15°C"
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


# ── One-shot jobs removed after firing ───────────────────────────


async def test_at_job_removed_after_fire(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="at", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert job.id not in svc._jobs


async def test_after_job_removed_after_fire(tmp_path, mock_push, mock_loop, fixed_now):
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="after", tier="instant", fire_at=fixed_now - timedelta(seconds=1)
    )
    svc._jobs[job.id] = job

    await svc._tick()
    await drain_tasks()

    assert job.id not in svc._jobs


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


def test_misfire_beyond_grace_discarded(tmp_path, mock_push, mock_loop, fixed_now):
    """Jobs missed beyond 5min grace are discarded on startup."""
    svc = make_service(tmp_path, mock_push, mock_loop, fixed_now)
    job = make_job(
        trigger="at",
        tier="instant",
        fire_at=fixed_now - timedelta(seconds=400),  # 400s ago > 300s grace
    )
    svc.store.save({job.id: job})
    svc.load_and_recover()

    assert job.id not in svc._jobs


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
