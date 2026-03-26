from __future__ import annotations

import sys
from pathlib import Path


def _import_health_event_v2():
    module_dir = Path(__file__).resolve().parents[1] / "scripts" / "fitbit-monitor"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import health_event_v2  # type: ignore

    return health_event_v2


def test_update_persists_metrics_into_pending(tmp_path):
    mod = _import_health_event_v2()
    runtime = mod.HealthEventV2Runtime(state_path=tmp_path / "health_event_v2_state.json")

    class _FakeEngine:
        def process(self, rows):
            return [
                mod.V2Event(
                    event_id="evt-001",
                    type="recovery_debt",
                    severity="high",
                    confidence=0.9,
                    message="测试事件",
                    created_at="2099-01-01 00:00:00",
                    metrics={"sleep_hours": 5.1, "debt_hours": 2.3},
                )
            ]

    original_engine = mod.HealthEventV2Engine
    try:
        mod.HealthEventV2Engine = _FakeEngine
        runtime.update(
            log_entry={"poll_time": "2099-01-01 00:00:00"},
            history=[{"poll_time": "2099-01-01 00:00:00"}],
        )
        events = runtime.get_pending_events()
    finally:
        mod.HealthEventV2Engine = original_engine

    assert len(events) == 1
    assert events[0]["id"] == "evt-001"
    assert events[0]["metrics"] == {"sleep_hours": 5.1, "debt_hours": 2.3}


def test_spo2_no_longer_emits_persistent_low_oxygen():
    mod = _import_health_event_v2()
    engine = mod.HealthEventV2Engine()

    rows = [
        {
            "poll_time": f"2099-01-01 12:{minute:02d}:00",
            "state": "awake",
            "spo2": 88.5,
            "data_lag_min": 5,
            "signals": {"zero_steps_count": 20},
        }
        for minute in (0, 5, 10, 15, 20, 25)
    ]

    events = engine.process(rows)

    assert not any(e.type == "persistent_low_oxygen" for e in events)
