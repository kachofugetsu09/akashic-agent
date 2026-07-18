from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]


def _gate_module() -> ModuleType:
    path = ROOT / "docker" / "debug" / "gate.py"
    spec = importlib.util.spec_from_file_location("akashic_change_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalog_has_no_unmapped_executable_files() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    report = gate.audit_catalog(catalog, check_baseline=False)

    assert report["unmappedExecutableFiles"] == []
    assert report["issues"] == []
    assert report["stateContracts"]["plugin_data"] == {
        "normalChange": "plugin_owned_update",
        "destructiveOwner": "explicit_plugin_data_deletion",
        "protectedTables": [],
    }


def test_unknown_executable_file_is_not_silently_accepted() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    mutant_path = "new_subsystem/ownerless_runtime.py"

    assert gate._is_executable(mutant_path, catalog)
    assert gate._groups_for_path(mutant_path, catalog) == set()


def test_every_p0_oracle_declares_a_known_wrong_mutant() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    p0_scenarios = {
        scenario_id
        for group in catalog.groups.values()
        if group.priority == "p0"
        for scenario_id in group.scenarios
    }

    assert p0_scenarios
    assert {
        scenario_id: catalog.scenarios[scenario_id].mutants
        for scenario_id in sorted(p0_scenarios)
        if not catalog.scenarios[scenario_id].mutants
    } == {}


def test_public_plan_contains_no_private_provider_identity() -> None:
    gate = _gate_module()
    if not gate.BASELINE_PATH.is_file() or not (ROOT / ".git").exists():
        return
    plan = cast(dict[str, Any], gate.build_plan("HEAD"))

    assert "providers" not in plan
    assert "providerIds" not in plan
    assert isinstance(plan["privateGateRequired"], bool)


def test_state_contract_loads_destructive_policy_fields() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    plugin_data = catalog.states["plugin_data"]

    assert plugin_data.normal_change == "plugin_owned_update"
    assert plugin_data.destructive_owner == "explicit_plugin_data_deletion"
    assert "PLG-010" in plugin_data.requirements


def test_state_contract_rejects_plugin_uninstall_as_data_owner() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    plugin_data = catalog.states["plugin_data"]
    catalog.states["plugin_data"] = replace(
        plugin_data,
        destructive_owner="plugin_uninstall",
    )

    issues = gate._validate_state_contracts(catalog, gate._requirement_ids())

    assert any("普通卸载不得拥有 plugin-data 删除权" in issue for issue in issues)


def test_gate_build_and_scenario_have_independent_timeouts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    gate = _gate_module()
    commands: list[tuple[list[str], int | None]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        commands.append((command, kwargs.get("timeout")))
        return gate.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate.subprocess, "run", run)
    build_report = tmp_path / "build-report"
    build_report.mkdir()

    build = gate._build_change_gate_image("test-run", build_report)

    sandbox = tmp_path / "scenario"
    sandbox.mkdir()
    monkeypatch.setattr(gate, "_prepare_sandbox", lambda *_args, **_kwargs: sandbox)
    scenario = gate.Scenario(
        id="timeout_contract",
        requirements=("TST-001",),
        groups=("tooling",),
        environment="public_clean_workspace",
        timeout_seconds=17,
        command=("python", "-V"),
        observes=("process_exit",),
        mutants=(),
    )

    result = gate._run_scenario(
        scenario,
        run_id="test-run",
        report_dir=tmp_path / "scenario-report",
    )

    build_command, build_timeout = commands[0]
    scenario_command, scenario_timeout = commands[1]
    assert build["status"] == result["status"] == "passed"
    assert build_command[-2:] == ["build", "change-gate"]
    assert build_timeout == gate.IMAGE_BUILD_TIMEOUT_SECONDS
    assert "--build" not in scenario_command
    assert scenario_command[-4:-1] == ["--no-deps", "change-gate", "python"]
    assert scenario_timeout == scenario.timeout_seconds
