from datetime import UTC, datetime, timedelta

import pytest

from core.clock import ReplayClock, SystemClock, clock_from_env


def test_replay_clock_persists_and_advances(tmp_path) -> None:
    path = tmp_path / "clock.json"
    clock = ReplayClock(path, datetime(2026, 1, 2, 3, 4, tzinfo=UTC))

    assert clock.now() == datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    assert clock.advance(timedelta(minutes=30)) == datetime(
        2026, 1, 2, 3, 34, tzinfo=UTC
    )
    assert ReplayClock(path).now() == datetime(2026, 1, 2, 3, 34, tzinfo=UTC)


def test_replay_clock_rejects_naive_datetime(tmp_path) -> None:
    with pytest.raises(ValueError, match="时区"):
        ReplayClock(tmp_path / "clock.json").set(datetime(2026, 1, 2))


def test_clock_from_env_selects_replay_clock(tmp_path) -> None:
    path = tmp_path / "clock.json"
    ReplayClock(path, datetime(2026, 1, 2, tzinfo=UTC))

    assert isinstance(clock_from_env({}), SystemClock)
    selected = clock_from_env({"AKASHIC_REPLAY_CLOCK_FILE": str(path)})
    assert selected.now() == datetime(2026, 1, 2, tzinfo=UTC)
