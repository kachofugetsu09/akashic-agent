from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_runtime_has_no_tool_loop_guard_control_word() -> None:
    paths = (
        ROOT / "agent/core/passive_turn.py",
        ROOT / "plugins/subagent/plugin.py",
        ROOT / "docker/debug/plugin_composition_v3_gate.py",
        ROOT / "docker/debug/plugin_v3_e2_gate.py",
        ROOT / "docker/debug/plugin_v3_e4_gate.py",
        ROOT / "docker/debug/plugin_v3_fleet_gate.py",
        ROOT / "docker/debug/plugin-composition-v3.lock.json",
        ROOT / "docker/debug/plugin-v3-fleet.lock.json",
    )

    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in paths
        if "tool_loop_guard" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_scoped_turn_core_api_has_no_source_business_words() -> None:
    source = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "agent/control/scoped_turn.py",
            "agent/control/timer.py",
        )
    )

    assert "scheduler" not in source.lower()
    assert "subagent" not in source.lower()
    assert "proactive" not in source.lower()
