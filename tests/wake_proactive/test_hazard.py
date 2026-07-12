from datetime import UTC, datetime, timedelta

from plugins.wake_proactive.hazard import advance_hazard, rank_events


def test_hazard_uses_semantic_interest_without_changing_title_order():
    now = datetime(2026, 7, 12, tzinfo=UTC)
    result = advance_hazard(
        [
            {
                "id": "feed:1",
                "published_at": now.isoformat(),
                "preprocess_score": 0.0,
                "_wake_interest_score": 0.9,
            }
        ],
        now=now,
        hazard=0.0,
        threshold=0.8,
        updated_at=None,
        last_wake_at=None,
    )

    assert result.evidence > 2.0
    assert result.driver_item_id == "feed:1"


def test_hazard_continuously_downweights_stale_missing_and_ineligible_backfill():
    now = datetime(2026, 7, 12, tzinfo=UTC)
    result = advance_hazard(
        [
            {"id": "missing", "preprocess_score": 0.99},
            {
                "id": "stale",
                "published_at": (now - timedelta(days=10)).isoformat(),
                "preprocess_score": 0.99,
            },
            {
                "id": "backfill",
                "published_at": now.isoformat(),
                "preprocess_score": 0.99,
                "wake_eligible": False,
            },
        ],
        now=now,
        hazard=0.3,
        threshold=0.4,
        updated_at=now - timedelta(hours=5),
        last_wake_at=None,
    )

    fresh = advance_hazard(
        [
            {
                "id": "fresh",
                "published_at": now.isoformat(),
                "preprocess_score": 0.99,
            }
        ],
        now=now,
        hazard=0.3,
        threshold=10.0,
        updated_at=now - timedelta(hours=5),
        last_wake_at=None,
    )

    assert result.should_wake is False
    assert result.evidence > 0
    assert result.evidence < fresh.evidence * 0.05
    assert result.hazard_after < 0.3


def test_ranking_uses_source_diversity_decay_instead_of_source_quota():
    now = datetime(2026, 7, 12, tzinfo=UTC)
    ranked = rank_events(
        [
            {
                "id": "a1",
                "source_id": "a",
                "published_at": now.isoformat(),
                "preprocess_score": 0.9,
            },
            {
                "id": "a2",
                "source_id": "a",
                "published_at": now.isoformat(),
                "preprocess_score": 0.89,
            },
            {
                "id": "b1",
                "source_id": "b",
                "published_at": now.isoformat(),
                "preprocess_score": 0.7,
            },
        ],
        now=now,
    )

    assert [event["id"] for event in ranked] == ["a1", "b1", "a2"]
    assert ranked[2]["_wake_rank_features"]["source_diversity"] < 1.0


def test_weak_stale_signal_reaches_low_steady_state_instead_of_waiting_out_gate():
    now = datetime(2026, 7, 12, tzinfo=UTC)
    event = {
        "id": "weak",
        "published_at": (now - timedelta(days=5)).isoformat(),
        "preprocess_score": 0.1,
    }
    hazard = 0.0
    for hour in range(48):
        result = advance_hazard(
            [event],
            now=now + timedelta(hours=hour),
            hazard=hazard,
            threshold=1.0,
            updated_at=now + timedelta(hours=hour - 1),
            last_wake_at=None,
        )
        hazard = result.hazard_after

    assert result.should_wake is False
    assert hazard < 0.2


def test_time_causal_interest_continuously_increases_wake_pressure():
    now = datetime(2026, 7, 12, tzinfo=UTC)
    result = advance_hazard(
        [
            {
                "id": "gpt-sol-release",
                "published_at": now.isoformat(),
                "preprocess_score": 0.5,
                "_wake_semantic_interest": 0.9,
                "_wake_interest_score": 0.95,
            }
        ],
        now=now,
        hazard=0.0,
        threshold=0.8,
        updated_at=now - timedelta(minutes=5),
        last_wake_at=None,
    )

    assert result.hazard_after < result.threshold
    assert result.should_wake is True

    unrelated = advance_hazard(
        [
            {
                "id": "generic-release",
                "published_at": now.isoformat(),
                "preprocess_score": 0.5,
                "_wake_semantic_interest": 0.1,
                "_wake_interest_score": 0.55,
            }
        ],
        now=now,
        hazard=0.0,
        threshold=1.0,
        updated_at=now - timedelta(minutes=5),
        last_wake_at=None,
    )

    assert unrelated.should_wake is False
