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


def test_gate_requires_prepare_then_authorizers() -> None:
    assert gate.EXPECTED_LISTENERS == (
        "transform:tool.input.prepare[akashic.tool-input.v1]:shell_restore",
        "serial:tool.execution.authorize[bail=akashic.tool-deny-reason.v1]:shell_safety",
    )


def test_scenario_catalog_has_stable_profile_and_digest() -> None:
    assert gate.SCENARIO_PROFILE == "plugin-tool-v3-v1"
    assert tuple(case.id for case in gate.SCENARIO_CATALOG) == (
        "plain-rm",
        "sudo-cluster",
        "sudo-preserve-env",
        "sudo-mode-denied",
        "repeat-1",
        "repeat-2",
        "repeat-3",
    )
    assert all(case.expected_status == "success" for case in gate.SCENARIO_CATALOG[-3:])
    assert len(gate._scenario_catalog_sha256()) == 64


def test_gate_pins_protocol_source_and_version() -> None:
    assert gate.GATE_VERSION == 1
    assert gate.PROTOCOL_SOURCE_REPOSITORY == (
        "https://github.com/kachofugetsu09/akashic-agent.git"
    )
    assert gate.PROTOCOL_SOURCE_COMMIT == (
        "0940e9e74a62efef54470f11a7064a99ca5e9acc"
    )
    evidence = gate._protocol_source_evidence()
    assert evidence["commit"] == gate.PROTOCOL_SOURCE_COMMIT
    assert [item["path"] for item in evidence["files"]] == list(
        gate.PROTOCOL_SOURCE_PATHS
    )
    assert all(len(item["sha256"]) == 64 for item in evidence["files"])


def test_ci_fetches_protocol_source_history() -> None:
    workflow = (gate.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    gate_job = workflow.split("  plugin-composition-v3-gate:\n", 1)[1].split(
        "\n  check-and-test:",
        1,
    )[0]
    check_job = workflow.split("  check-and-test:\n", 1)[1].split(
        "\n  docker-control-gate:",
        1,
    )[0]

    assert "  workflow_dispatch:\n" in workflow
    assert "fetch-depth: 0" in gate_job
    assert "fetch-depth: 0" in check_job
