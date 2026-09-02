"""Tests for fire_at and actual_trigger computation."""

from datetime import datetime, timedelta, timezone

import pytest

from agent.scheduler import (
    LatencyTracker,
    compute_actual_trigger,
    compute_fire_at,
)

# Fixed "now" for deterministic tests
_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_NOW_FN = lambda: _NOW  # noqa: E731


class TestComputeFireAt:
    def test_after_with_request_time(self):
        # request_time is T+0, tool called at T+20s — fire_at should be T+30s
        request_time = _NOW.isoformat()
        fire_at = compute_fire_at("after", "30s", "UTC", request_time, _NOW_FN)
        assert fire_at == _NOW + timedelta(seconds=30)

    def test_after_compensates_for_tool_call_delay(self):
        # request_time is 20s before tool is called
        request_time = (_NOW - timedelta(seconds=20)).isoformat()
        now_fn = lambda: _NOW  # noqa: E731  (tool called at _NOW)
        fire_at = compute_fire_at("after", "30s", "UTC", request_time, now_fn)
        # fire_at = request_time + 30s = (_NOW - 20s) + 30s = _NOW + 10s
        assert fire_at == _NOW + timedelta(seconds=10)

    def test_after_without_request_time_uses_now(self):
        fire_at = compute_fire_at("after", "5m", "UTC", None, _NOW_FN)
        assert fire_at == _NOW + timedelta(minutes=5)

    def test_at_absolute_iso(self):
        fire_at = compute_fire_at("at", "2025-06-01T14:00:00", "UTC", None, _NOW_FN)
        assert fire_at.hour == 14
        assert fire_at.minute == 0

    def test_every_interval_returns_now_plus_interval(self):
        fire_at = compute_fire_at("every", "1h", "UTC", None, _NOW_FN)
        assert fire_at == _NOW + timedelta(hours=1)

    def test_after_request_time_with_tz_offset_not_treated_as_utc(self):
        # request_time 包含 +08:00 时，不能被当作 UTC 处理
        # CST 15:48:40 = UTC 07:48:40
        request_time = "2025-06-01T15:48:40+08:00"
        now_fn = lambda: datetime(2025, 6, 1, 7, 48, 50, tzinfo=timezone.utc)
        fire_at = compute_fire_at("after", "30s", "UTC", request_time, now_fn)
        # fire_at 应是 CST 15:49:10 = UTC 07:49:10，不是 UTC 15:49:10
        assert fire_at.utctimetuple().tm_hour == 7
        assert fire_at.utctimetuple().tm_min == 49

    def test_unknown_trigger_raises(self):
        with pytest.raises(ValueError, match="未知触发类型"):
            compute_fire_at("sometime", "5m", "UTC", None, _NOW_FN)


class TestComputeActualTrigger:
    def test_instant_no_pretrigger(self):
        tracker = LatencyTracker(default=25.0)
        fire_at = _NOW + timedelta(minutes=5)
        actual = compute_actual_trigger(fire_at, "instant", tracker)
        assert actual == fire_at

    def test_soft_subtracts_lead(self):
        tracker = LatencyTracker(default=30.0)
        fire_at = _NOW + timedelta(minutes=5)
        actual = compute_actual_trigger(fire_at, "soft", tracker)
        assert actual == fire_at - timedelta(seconds=30)

    def test_soft_uses_adaptive_p90(self):
        tracker = LatencyTracker(default=25.0, window=20)
        for _ in range(20):
            tracker.record(10.0)  # stable 10s latency → P90 = 10s
        fire_at = _NOW + timedelta(minutes=5)
        actual = compute_actual_trigger(fire_at, "soft", tracker)
        # actual_trigger should be ≈ fire_at - 10s (not 25s default)
        diff = (fire_at - actual).total_seconds()
        assert 9 < diff < 12

    def test_soft_fire_at_in_past_returns_past_time(self):
        # Scheduler should still fire it; the tick logic will catch it
        tracker = LatencyTracker(default=30.0)
        fire_at = _NOW - timedelta(seconds=5)  # already past
        actual = compute_actual_trigger(fire_at, "soft", tracker)
        # actual_trigger = fire_at - 30s, even more in the past — that's fine
        assert actual < _NOW
