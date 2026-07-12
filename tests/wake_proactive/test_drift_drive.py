from datetime import UTC, datetime, timedelta

from plugins.wake_proactive.drift_drive import advance_drift_drive


NOW = datetime(2026, 7, 12, 12, tzinfo=UTC)


def test_low_content_evidence_accumulates_with_idle_time() -> None:
    early = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=0.8,
        updated_at=NOW - timedelta(hours=4),
        last_user_at=NOW - timedelta(minutes=10),
        last_drift_at=None,
        content_evidence=0.05,
    )
    idle = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=0.8,
        updated_at=NOW - timedelta(hours=4),
        last_user_at=NOW - timedelta(hours=12),
        last_drift_at=None,
        content_evidence=0.05,
    )

    assert early.decision == "idle"
    assert idle.decision == "attempt"
    assert idle.idle_drive > early.idle_drive
    assert idle.hazard_after > early.hazard_after
    assert "leisure_ready" in idle.reasons


def test_high_content_evidence_continuously_reduces_drift_rate() -> None:
    baseline = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=2.0,
        updated_at=NOW - timedelta(hours=2),
        last_user_at=NOW - timedelta(hours=12),
        last_drift_at=None,
        content_evidence=0.0,
    )
    result = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=2.0,
        updated_at=NOW - timedelta(hours=2),
        last_user_at=NOW - timedelta(hours=12),
        last_drift_at=None,
        content_evidence=0.8,
    )

    assert result.decision == "idle"
    assert result.content_suppression == 0.8
    assert 0 < result.rate < baseline.rate
    assert "content_evidence" in result.reasons


def test_busy_sleeping_and_in_game_decay_rate_without_hard_block() -> None:
    baseline = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=2.0,
        updated_at=NOW - timedelta(hours=2),
        last_user_at=NOW - timedelta(hours=12),
        last_drift_at=None,
        content_evidence=0.0,
    )
    for flag in ("busy", "sleeping", "in_game"):
        result = advance_drift_drive(
            now=NOW,
            hazard=0.0,
            threshold=2.0,
            updated_at=NOW - timedelta(hours=2),
            last_user_at=NOW - timedelta(hours=12),
            last_drift_at=None,
            content_evidence=0.0,
            **{flag: True},
        )

        assert result.decision == "idle"
        assert 0 < result.rate < baseline.rate
        assert flag in result.reasons


def test_recent_drift_and_repetition_reduce_rate() -> None:
    baseline = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=2.0,
        updated_at=NOW - timedelta(hours=1),
        last_user_at=NOW - timedelta(hours=12),
        last_drift_at=None,
        content_evidence=0.0,
    )
    suppressed = advance_drift_drive(
        now=NOW,
        hazard=0.0,
        threshold=2.0,
        updated_at=NOW - timedelta(hours=1),
        last_user_at=NOW - timedelta(hours=12),
        last_drift_at=NOW - timedelta(minutes=10),
        content_evidence=0.0,
        repetition=0.8,
    )

    assert suppressed.rate < baseline.rate * 0.1
    assert "recent_drift" in suppressed.reasons
    assert "repetition" in suppressed.reasons
