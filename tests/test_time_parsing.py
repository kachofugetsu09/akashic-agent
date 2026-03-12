"""Tests for duration and datetime parsing utilities."""

from datetime import datetime, timedelta, timezone

import pytest

from agent.scheduler import is_cron_expr, next_cron_fire, parse_duration, parse_when_at


class TestParseDuration:
    def test_seconds(self):
        assert parse_duration("30s") == timedelta(seconds=30)

    def test_minutes(self):
        assert parse_duration("5m") == timedelta(minutes=5)

    def test_hours(self):
        assert parse_duration("2h") == timedelta(hours=2)

    def test_days(self):
        assert parse_duration("1d") == timedelta(days=1)

    def test_compound_hours_minutes(self):
        assert parse_duration("1h30m") == timedelta(hours=1, minutes=30)

    def test_compound_all(self):
        assert parse_duration("1d2h30m15s") == timedelta(
            days=1, hours=2, minutes=30, seconds=15
        )

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="无效的时间间隔"):
            parse_duration("abc")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_duration("")


class TestParseWhenAt:
    def test_hhmm_future_today(self):
        # 13:00 is in the future relative to 12:00
        from zoneinfo import ZoneInfo

        tz = "Asia/Shanghai"
        ref = datetime(2025, 6, 1, 12, 0, 0, tzinfo=ZoneInfo(tz))
        result = parse_when_at("13:00", tz, _now_fn=lambda: ref)
        assert result.hour == 13
        assert result.minute == 0
        assert result.date() == ref.date()

    def test_hhmm_past_advances_to_tomorrow(self):
        from zoneinfo import ZoneInfo

        tz = "Asia/Shanghai"
        ref = datetime(2025, 6, 1, 14, 0, 0, tzinfo=ZoneInfo(tz))
        result = parse_when_at("09:00", tz, _now_fn=lambda: ref)
        assert result.day == 2  # tomorrow

    def test_iso_datetime(self):
        result = parse_when_at("2025-06-01T14:30:00", "UTC")
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 1
        assert result.hour == 14
        assert result.minute == 30

    def test_iso_with_offset(self):
        result = parse_when_at("2025-06-01T14:30:00+08:00", "UTC")
        assert result.utcoffset() is not None

    def test_timezone_applied_to_hhmm(self):
        from zoneinfo import ZoneInfo

        tz = "Asia/Shanghai"
        ref = datetime(2025, 6, 1, 8, 0, 0, tzinfo=ZoneInfo(tz))
        result = parse_when_at("09:00", tz, _now_fn=lambda: ref)
        assert result.tzinfo is not None

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="无法解析时间"):
            parse_when_at("not-a-time", "UTC")


class TestIsCronExpr:
    def test_valid_cron_5_fields(self):
        assert is_cron_expr("0 9 * * *") is True

    def test_valid_cron_with_wildcards(self):
        assert is_cron_expr("*/5 * * * *") is True

    def test_interval_string_not_cron(self):
        assert is_cron_expr("1h") is False

    def test_duration_not_cron(self):
        assert is_cron_expr("30s") is False

    def test_four_fields_not_cron(self):
        assert is_cron_expr("0 9 * *") is False


class TestNextCronFire:
    def test_fixed_daily_cron_returns_next_boundary(self):
        after = datetime(2025, 6, 1, 8, 0, 1, tzinfo=timezone.utc)
        result = next_cron_fire("0 9 * * *", "UTC", after)
        assert result == datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)

    def test_step_cron_advances_to_next_match(self):
        after = datetime(2025, 6, 1, 8, 1, 0, tzinfo=timezone.utc)
        result = next_cron_fire("*/5 * * * *", "UTC", after)
        assert result == datetime(2025, 6, 1, 8, 5, 0, tzinfo=timezone.utc)
