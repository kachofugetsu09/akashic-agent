import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agent.control.timer import AsyncioOneShotTimer, TimerReceipt, TimerStatus


def test_timer_receipt_normalizes_aware_times_and_status() -> None:
    deadline = datetime(2026, 8, 22, 20, 0, tzinfo=UTC)
    receipt = TimerReceipt(
        timer_id="timer:1",
        deadline=deadline,
        settled_at=deadline + timedelta(seconds=1),
        status=TimerStatus.FIRED,
    )

    assert receipt.deadline == deadline
    assert receipt.status is TimerStatus.FIRED


def test_timer_receipt_rejects_identity_and_naive_time() -> None:
    aware = datetime(2026, 8, 22, 20, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="timer_id"):
        TimerReceipt("", aware, aware, TimerStatus.CANCELLED)
    with pytest.raises(ValueError, match="时区"):
        TimerReceipt("timer:1", aware.replace(tzinfo=None), aware, TimerStatus.FIRED)


@pytest.mark.asyncio
async def test_asyncio_timer_fires_once_and_reuses_terminal_receipt() -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    handle = AsyncioOneShotTimer(clock=lambda: now, sleeper=sleep).schedule(
        now + timedelta(seconds=7)
    )

    first = await handle.result()
    second = await handle.cancel()

    assert delays == [7.0]
    assert first is second
    assert first.status is TimerStatus.FIRED


@pytest.mark.asyncio
async def test_asyncio_timer_cancel_and_cleanup_leave_no_wait() -> None:
    now = datetime(2026, 8, 22, 12, tzinfo=UTC)
    sleeping = asyncio.Event()

    async def sleep(_delay: float) -> None:
        sleeping.set()
        await asyncio.Future()

    handle = AsyncioOneShotTimer(clock=lambda: now, sleeper=sleep).schedule(
        now + timedelta(days=1)
    )
    await sleeping.wait()

    cancelled = await handle.cancel()
    await handle.cleanup()

    assert cancelled.status is TimerStatus.CANCELLED
    assert await handle.result() is cancelled


@pytest.mark.asyncio
async def test_asyncio_timer_rejects_naive_deadline_before_task_creation() -> None:
    timer = AsyncioOneShotTimer()

    with pytest.raises(ValueError, match="deadline"):
        timer.schedule(datetime(2026, 8, 22, 12))
