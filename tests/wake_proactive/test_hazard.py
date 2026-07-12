from datetime import UTC, datetime

from plugins.wake_proactive.hazard import advance_hazard


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
        threshold=10.0,
        updated_at=None,
        last_wake_at=None,
    )

    assert result.evidence > 2.0
    assert result.driver_item_id == "feed:1"
