from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from plugins.wake.pool import build_initial_score, measure_pool, rank_events

NOW = datetime(2026, 9, 11, 0, tzinfo=UTC)


def _event(
    index: int, interest: float, *, age: timedelta = timedelta()
) -> dict[str, object]:
    identity = f"content-{index}"
    return {
        "id": identity,
        "source_id": "feed",
        "published_at": (NOW - age).isoformat(),
        "_wake_admission_identity": identity,
        "_wake_initial_score": build_initial_score(
            interest, has_published_at=True, wake_eligible=True
        ),
        "_wake_semantic_interest": interest,
    }


def test_squared_interest_separates_low_and_high_quality_content() -> None:
    low = [_event(index, 0.1) for index in range(100)]
    low_result = measure_pool(low, now=NOW, new_item_ids={"content-0"})

    assert low_result.should_wake is False
    assert low_result.pool_mass == 0.0
    assert low_result.below_floor == 100

    high = _event(100, 0.8)
    high_result = measure_pool([high], now=NOW, new_item_ids={"content-100"})

    assert high_result.should_wake is True
    assert high_result.pool_mass > high_result.threshold


def test_freshness_halves_squared_mass_after_eighteen_hours() -> None:
    fresh = rank_events([_event(1, 0.8)], now=NOW)[0]
    aged = rank_events([_event(1, 0.8, age=timedelta(hours=18))], now=NOW)[0]

    fresh_mass = fresh["_wake_rank_features"]["admission_mass"]
    aged_mass = aged["_wake_rank_features"]["admission_mass"]
    assert aged_mass == pytest.approx(fresh_mass / 2)


def test_admission_mass_includes_the_whole_active_pool() -> None:
    events = [_event(index, 0.2) for index in range(40)]

    result = measure_pool(events, now=NOW, new_item_ids={"content-39"})

    single_mass = rank_events([events[0]], now=NOW)[0]["_wake_rank_features"][
        "admission_mass"
    ]
    assert result.pool_mass == pytest.approx(single_mass * len(events))
    assert result.should_wake is True
