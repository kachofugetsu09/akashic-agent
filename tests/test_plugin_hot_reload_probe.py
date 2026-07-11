from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_PROBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docker/debug/plugin_hot_reload_probe.py"
)
_SPEC = importlib.util.spec_from_file_location("plugin_hot_reload_probe", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def test_integrity_gate_fails_when_any_check_fails() -> None:
    checks = [
        probe.CheckResult("read_only", True, {}),
        probe.CheckResult("repositories_unchanged", False, {}),
    ]

    assert probe._gate_status(checks) == "failed"


def test_controller_rejects_protected_sandbox() -> None:
    root = Path("/workspace").resolve()

    assert probe._sandbox_is_protected(root / "gate", [root])
    assert not probe._sandbox_is_protected(Path("/tmp/gate"), [root])


def test_system_gate_propagates_subgate_failure() -> None:
    baseline = {
        "build_returncode": 0,
        "integrity_returncode": 0,
        "smoke_passed": True,
        "cleanup_returncode": 0,
        "unchanged": True,
        "controller_error": "",
    }
    assert probe._controller_gate_passed(**baseline)

    for key, value in (
        ("build_returncode", 1),
        ("integrity_returncode", 1),
        ("smoke_passed", False),
        ("cleanup_returncode", 1),
        ("unchanged", False),
        ("controller_error", "boom"),
    ):
        failed = {**baseline, key: value}
        assert not probe._controller_gate_passed(**failed)
