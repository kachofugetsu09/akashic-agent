from datetime import UTC, datetime, timedelta

import pytest

from agent.control.timer import TimerReceipt, TimerStatus


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
