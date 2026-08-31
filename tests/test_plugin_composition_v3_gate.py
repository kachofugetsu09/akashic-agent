from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_GATE_PATH = Path(__file__).resolve().parents[1] / "docker/debug/plugin_composition_v3_gate.py"
_SPEC = importlib.util.spec_from_file_location("plugin_composition_v3_gate", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)


def test_lock_pins_exact_trial_plugin_set() -> None:
    plugins = gate._load_lock(gate.DEFAULT_LOCK)

    assert tuple(plugin.id for plugin in plugins) == gate.EXPECTED_PLUGIN_IDS
    assert all(plugin.requested_ref == plugin.resolved_sha for plugin in plugins)
    assert all(plugin.change_source_pr_head == plugin.resolved_sha for plugin in plugins)


def test_lock_rejects_floating_revision(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["plugins"][0]["requested_ref"] = "main"
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="完整 SHA"):
        gate._load_lock(lock)


def test_lock_rejects_plugin_order_drift(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["plugins"].reverse()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="集合或顺序"):
        gate._load_lock(lock)
