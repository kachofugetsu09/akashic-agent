from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
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


def test_deleted_path_contract_maps_live_and_removed_peer_files() -> None:
    gate = _gate_module()
    catalog = gate.load_catalog()

    group = catalog.groups["companion_peer_removal"]

    assert group.deleted_paths == ("agent/peer_agent/**",)
    assert "companion_peer_removal" in gate._groups_for_path(
        "agent/peer_agent/tool.py", catalog
    )


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


def test_build_plan_marks_production_and_protected_contract_mixes(
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

    assert plan["status"] == "protected_contract_mixed"
    assert plan["productionSourcePaths"] == ["agent/core/passive_turn.py"]
    assert plan["protectedContractPaths"] == [
        "tests/semantic/test_change_gate.py"
    ]


def test_mixed_commands_fail_without_building_or_running_scenarios(
    monkeypatch: Any,
) -> None:
    gate = _gate_module()
    plan = {
        "status": "protected_contract_mixed",
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
        "privateGateRequired": True,
    }
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(gate, "build_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(gate, "load_catalog", lambda: SimpleNamespace(scenarios={}))
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mixed Gate 不得构建镜像")
        ),
    )
    monkeypatch.setattr(
        gate,
        "_run_scenario",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mixed Gate 不得运行场景")
        ),
    )
    args = Namespace(base="origin/main", full=False)

    assert gate.command_plan(args) == 1
    assert gate.command_run(args) == 1
    assert writes[-1]["status"] == "protected_contract_mixed"
    assert writes[-1]["checks"] == []


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
