from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

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
    assert report["baselineStatus"] == "unchecked"
    assert report["issues"] == []
    assert report["stateContracts"]["plugin_data"] == {
        "normalChange": "plugin_owned_update",
        "destructiveOwner": "explicit_plugin_data_deletion",
        "protectedTables": [],
        "retention": "plugin-data survives ordinary uninstall",
        "physicalReductionOwner": "explicit_plugin_data_deletion",
        "recoveryEvidence": ["candidate_generation", "rollback"],
        "failureSemantics": ["unit_failed", "cleanup_degraded", "runtime_fatal"],
    }


def test_requirement_catalog_accepts_two_and_three_letter_owners() -> None:
    gate = _gate_module()

    requirements = gate._requirement_ids()

    assert {"SH-001", "FS-001", "RUN-001"}.issubset(requirements)


def test_unknown_executable_file_is_not_silently_accepted() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    mutant_path = "new_subsystem/ownerless_runtime.py"

    assert gate._is_executable(mutant_path, catalog)
    assert gate._groups_for_path(mutant_path, catalog) == set()


def test_change_classes_allow_single_class_and_reject_mixed_contract_changes() -> None:
    gate = _gate_module()

    protected_only = gate.classify_change_paths(
        [
            "docs/projectneed.md",
            "scripts/measure_production_sloc.py",
            "tests_scenarios/contracts/impact.toml",
        ]
    )
    production_only = gate.classify_change_paths(
        ["agent/core/passive_turn.py", "migrations/20260722_example/migration.py"]
    )
    mixed = gate.classify_change_paths(
        ["agent/core/passive_turn.py", "tests/semantic/test_change_gate.py"]
    )
    production_with_test = gate.classify_change_paths(
        ["agent/core/passive_turn.py", "tests/test_agent_core_foundation.py"]
    )
    migration_with_test = gate.classify_change_paths(
        ["migrations/20260722_example/migration.py", "tests/test_migration_runner.py"]
    )

    assert protected_only == {
        "productionSourcePaths": [],
        "protectedContractPaths": [
            "docs/projectneed.md",
            "scripts/measure_production_sloc.py",
            "tests_scenarios/contracts/impact.toml",
        ],
    }
    assert production_only == {
        "productionSourcePaths": [
            "agent/core/passive_turn.py",
            "migrations/20260722_example/migration.py",
        ],
        "protectedContractPaths": [],
    }
    assert mixed["productionSourcePaths"] == ["agent/core/passive_turn.py"]
    assert mixed["protectedContractPaths"] == [
        "tests/semantic/test_change_gate.py"
    ]
    assert production_with_test["protectedContractPaths"] == []
    assert migration_with_test["protectedContractPaths"] == []
    assert not gate.is_protected_contract_path(
        "migrations/20260722_example/migration.py"
    )


def test_build_plan_runs_full_public_gate_for_production_and_contract_mixes(
    monkeypatch: Any,
) -> None:
    gate = _gate_module()
    monkeypatch.setattr(
        gate,
        "_changed_paths",
        lambda _base: [
            "agent/core/passive_turn.py",
            "tests/semantic/test_change_gate.py",
        ],
    )
    monkeypatch.setattr(gate, "_resolve_commit", lambda _base: "base-commit")
    monkeypatch.setattr(gate, "_dirty_status", lambda: [])
    monkeypatch.setattr(gate, "_load_baseline", lambda: {"acceptedGaps": []})
    monkeypatch.setattr(gate, "source_digest", lambda: "source-digest")
    monkeypatch.setattr(gate, "catalog_digest", lambda: "catalog-digest")
    monkeypatch.setattr(
        gate,
        "audit_catalog",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        gate,
        "_run_git",
        lambda *args, **_kwargs: gate.subprocess.CompletedProcess(
            ["git", *args], 0, b"head-commit\n", b""
        ),
    )

    plan = gate.build_plan("origin/main")

    assert plan["status"] == "planned"
    assert plan["full"] is True
    assert plan["productionSourcePaths"] == ["agent/core/passive_turn.py"]
    assert plan["protectedContractPaths"] == [
        "tests/semantic/test_change_gate.py"
    ]


def test_mixed_commands_run_public_scenarios(
    monkeypatch: Any,
) -> None:
    gate = _gate_module()
    plan = {
        "status": "planned",
        "base": "base",
        "head": "head",
        "dirtyStatus": [],
        "sourceDigest": "source",
        "impactCatalogDigest": "catalog",
        "planDigest": "plan",
        "changedPaths": [
            "agent/core/passive_turn.py",
            "tests/semantic/test_change_gate.py",
        ],
        "productionSourcePaths": ["agent/core/passive_turn.py"],
        "protectedContractPaths": ["tests/semantic/test_change_gate.py"],
        "affectedGroups": ["runtime"],
        "selectedScenarios": ["expensive"],
    }
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(gate, "build_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        gate,
        "load_catalog",
        lambda: SimpleNamespace(scenarios={"expensive": SimpleNamespace()}),
    )
    monkeypatch.setattr(gate, "_new_run_id", lambda: "run")
    report_dir = gate.ROOT / "docker" / "debug" / "reports" / "change-gate" / "test"
    monkeypatch.setattr(gate, "_report_dir", lambda _run_id: report_dir)
    monkeypatch.setattr(gate, "_print_plan", lambda _plan: None)
    monkeypatch.setattr(
        gate,
        "_atomic_json",
        lambda _path, payload: writes.append(payload),
    )
    monkeypatch.setattr(
        gate,
        "_build_change_gate_image",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        gate,
        "_run_scenario",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "residualResources": {"containers": [], "networks": [], "volumes": []},
        },
    )
    args = Namespace(base="origin/main", full=False)

    assert gate.command_plan(args) == 0
    assert gate.command_run(args) == 0
    assert writes[-1]["status"] == "passed"
    assert len(cast(list[object], writes[-1]["checks"])) == 1


def test_gate_temp_root_uses_explicit_existing_directory(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    monkeypatch.setenv("AKASHIC_CHANGE_GATE_TMPDIR", str(tmp_path))

    assert gate._gate_temp_root() == tmp_path.resolve()


def test_gate_temp_root_rejects_missing_directory(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    gate = _gate_module()
    missing = tmp_path / "missing"
    monkeypatch.setenv("AKASHIC_CHANGE_GATE_TMPDIR", str(missing))

    with pytest.raises(gate.GateError, match="不是已存在目录"):
        gate._gate_temp_root()


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


def test_public_plan_contains_no_private_provider_identity(
    monkeypatch: Any,
) -> None:
    gate = _gate_module()
    if not gate.BASELINE_PATH.is_file() or not (ROOT / ".git").exists():
        return
    monkeypatch.setattr(gate, "audit_catalog", lambda _catalog: {"status": "passed"})
    plan = cast(dict[str, Any], gate.build_plan("HEAD"))

    assert "providers" not in plan
    assert "providerIds" not in plan


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


def test_companion_group_rejects_wrong_path_owner() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    catalog.groups["companion_tool_context"] = replace(
        catalog.groups["companion_tool_context"],
        paths=("agent/does-not-exist.py",),
    )

    report = gate.audit_catalog(catalog, check_baseline=False)

    assert any(
        "groups.companion_tool_context 路径规则无匹配" in issue
        for issue in report["issues"]
    )


def test_deleted_path_requires_frozen_base_match(tmp_path: Path, monkeypatch: Any) -> None:
    gate = _gate_module()
    baseline = json.loads(gate.BASELINE_PATH.read_text(encoding="utf-8"))
    baseline_path = tmp_path / "coverage-baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(gate, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(gate, "_commit_tree_paths", lambda _base: [])

    report = gate.audit_catalog(gate.load_catalog())

    assert any("deleted_paths 在 frozen base 无匹配" in issue for issue in report["issues"])


def test_companion_scenario_requires_mutant_node_in_command() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    scenario = catalog.scenarios["companion_tool_context_contract"]
    catalog.scenarios[scenario.id] = replace(
        scenario,
        command=tuple(
            item
            for item in scenario.command
            if "test_tool_context_rejects_origin_override_mutant" not in item
        ),
    )

    issues = gate._validate_references(catalog, gate._requirement_ids())

    assert any("command 未运行 mutant test node" in issue for issue in issues)


def test_deleted_peer_path_selects_peer_group_in_plan(monkeypatch: Any) -> None:
    gate = _gate_module()
    monkeypatch.setattr(
        gate,
        "_changed_paths",
        lambda _base: ["agent/peer_agent/registry.py"],
    )
    monkeypatch.setattr(gate, "_resolve_commit", lambda _base: "base-commit")
    monkeypatch.setattr(gate, "_dirty_status", lambda: [])
    monkeypatch.setattr(gate, "_load_baseline", lambda: {"acceptedGaps": []})
    monkeypatch.setattr(gate, "source_digest", lambda: "source-digest")
    monkeypatch.setattr(gate, "catalog_digest", lambda: "catalog-digest")
    monkeypatch.setattr(
        gate,
        "audit_catalog",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    monkeypatch.setattr(
        gate,
        "_run_git",
        lambda *args, **kwargs: gate.subprocess.CompletedProcess(
            ["git", *args], 0, b"head-commit\n", b""
        ),
    )

    plan = gate.build_plan("origin/main")

    assert "companion_peer_removal" in plan["affectedGroups"]


def test_state_contract_rejects_protocol_owner_mismatch() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    state = catalog.states["mobile_receipts"]
    catalog.states[state.id] = replace(
        state,
        physical_reduction_owner="other.owner",
    )

    issues = gate._validate_state_contracts(catalog, gate._requirement_ids())

    assert any("protocol_owner 必须等于 physical_reduction_owner" in issue for issue in issues)


def test_state_contract_rejects_not_applicable_physical_owner() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()
    state = catalog.states["runtime_workspace_boundary"]
    catalog.states[state.id] = replace(
        state,
        physical_reduction_owner="runtime.workspace_selection",
    )

    issues = gate._validate_state_contracts(catalog, gate._requirement_ids())

    assert any("not_applicable 必须使用 not_applicable physical owner" in issue for issue in issues)


def test_baseline_rejects_non_sha_base() -> None:
    gate = _gate_module()
    baseline = json.loads(gate.BASELINE_PATH.read_text(encoding="utf-8"))
    baseline["base"] = "HEAD"

    issues = gate._validate_baseline(baseline, gate.load_catalog())

    assert any("base 必须是完整 40 位 SHA" in issue for issue in issues)


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
