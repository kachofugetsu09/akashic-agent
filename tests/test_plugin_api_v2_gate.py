from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_GATE_PATH = Path(__file__).resolve().parents[1] / "docker/debug/plugin_api_v2_gate.py"
_SPEC = importlib.util.spec_from_file_location("plugin_api_v2_gate", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)


def test_release_lock_pins_every_external_plugin() -> None:
    release = gate._load_lock(gate.DEFAULT_LOCK)

    assert {plugin.id for plugin in release.plugins} == gate.EXPECTED_PLUGIN_IDS
    assert release.contract.id == "plugin-contracts"
    assert all(len(plugin.commit) == 40 for plugin in release.plugins)


def test_release_lock_rejects_missing_plugin(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["plugins"].pop()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="插件集合错误"):
        gate._load_lock(lock)


def test_release_lock_rejects_floating_reference(tmp_path: Path) -> None:
    raw = json.loads(gate.DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw["plugins"][0]["commit"] = "main"
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="完整 SHA"):
        gate._load_lock(lock)


def test_runtime_gate_uses_business_named_phases() -> None:
    assert gate.RUNTIME_PHASES == ("atomic-reload", "all-plugins", "fitbit")
