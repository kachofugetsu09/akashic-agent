from datetime import UTC, datetime, timedelta

from plugins.wake_proactive.hazard import advance_hazard, rank_events


def test_new_content_event_can_trigger_from_semantic_interest() -> None:
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
        new_item_ids={"feed:1"},
        random_draw=0.1,
        last_wake_at=None,
    )

    assert result.should_wake is True
    assert result.rate > 0.1
    assert result.driver_item_id == "feed:1"


def test_content_trigger_uses_every_active_source() -> None:
    now = datetime(2026, 7, 12, tzinfo=UTC)
    events = [
        {
            "id": f"feed:{index}",
            "source_id": f"source:{index}",
            "published_at": now.isoformat(),
            "preprocess_score": 0.4,
        }
        for index in range(12)
    ]
    result = advance_hazard(
        events,
        now=now,
        new_item_ids={"feed:11"},
        random_draw=1.0,
        last_wake_at=None,
    )
    first_eight = advance_hazard(
        events[:8],
        now=now,
        new_item_ids={"feed:7"},
        random_draw=1.0,
        last_wake_at=None,
    )

    assert result.evidence > first_eight.evidence


def test_old_pool_never_triggers_without_a_new_event() -> None:
    now = datetime(2026, 7, 12, tzinfo=UTC)
    result = advance_hazard(
        [
            {
                "id": "old",
                "published_at": now.isoformat(),
                "preprocess_score": 0.99,
            }
        ],
        now=now,
        new_item_ids=set(),
        random_draw=0.0,
        last_wake_at=None,
    )

    assert result.should_wake is False
    assert result.rate == 0.0


def test_recent_judgement_softly_reduces_event_probability() -> None:
    now = datetime(2026, 7, 12, tzinfo=UTC)
    event = {
        "id": "new",
        "published_at": now.isoformat(),
        "preprocess_score": 0.9,
    }
    recent = advance_hazard(
        [event],
        now=now,
        new_item_ids={"new"},
        random_draw=1.0,
        last_wake_at=now - timedelta(minutes=5),
    )
    rested = advance_hazard(
        [event],
        now=now,
        new_item_ids={"new"},
        random_draw=1.0,
        last_wake_at=now - timedelta(hours=8),
    )

    assert 0.0 < recent.rate < rested.rate


def test_ranking_geometrically_saturates_each_source() -> None:
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
    assert ranked[2]["_wake_rank_features"]["source_diversity"] == 0.5
