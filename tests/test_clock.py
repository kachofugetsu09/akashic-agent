from concurrent.futures import ThreadPoolExecutor
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


def test_replay_clock_advance_is_atomic_within_instance(tmp_path) -> None:
    start = datetime(2026, 1, 2, tzinfo=UTC)
    clock = ReplayClock(tmp_path / "clock.json", start)
    worker_count = 8
    advances_per_worker = 50

    def advance_many(_: int) -> list[datetime]:
        return [
            clock.advance(timedelta(minutes=1))
            for _ in range(advances_per_worker)
        ]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        batches = executor.map(advance_many, range(worker_count))
        results = [value for batch in batches for value in batch]

    total_advances = worker_count * advances_per_worker
    assert len(set(results)) == total_advances
    assert clock.now() == start + timedelta(minutes=total_advances)


def test_clock_from_env_selects_replay_clock(tmp_path) -> None:
    path = tmp_path / "clock.json"
    ReplayClock(path, datetime(2026, 1, 2, tzinfo=UTC))

    assert isinstance(clock_from_env({}), SystemClock)
    selected = clock_from_env({"AKASHIC_REPLAY_CLOCK_FILE": str(path)})
    assert selected.now() == datetime(2026, 1, 2, tzinfo=UTC)
