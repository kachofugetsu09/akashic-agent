from plugins.drift import drive as drift_drive
from plugins.wake import hazard
from plugins.wake_proactive import drift_drive as legacy_drift_drive
from plugins.wake_proactive import hazard as legacy_hazard


def test_legacy_and_v3_wake_share_exact_pure_policy_functions() -> None:
    assert legacy_hazard.rank_events is hazard.rank_events
    assert legacy_hazard.advance_hazard is hazard.advance_hazard
    assert legacy_drift_drive.advance_drift_drive is drift_drive.advance_drift_drive
    assert (
        legacy_drift_drive.sample_drift_delay_hours
        is drift_drive.sample_drift_delay_hours
    )
