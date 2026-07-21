#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NoReturn, Sequence, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_production_sloc import is_production_source_path  # noqa: E402


CONTRACTS_DIR = ROOT / "tests_scenarios" / "contracts"
IMPACT_PATH = CONTRACTS_DIR / "impact.toml"
STATE_PATH = CONTRACTS_DIR / "state-contracts.toml"
SCENARIO_PATH = CONTRACTS_DIR / "scenarios.toml"
BASELINE_PATH = CONTRACTS_DIR / "coverage-baseline.json"
REPORT_ROOT = ROOT / "docker" / "debug" / "reports" / "change-gate"
COMPOSE_PATH = ROOT / "docker" / "debug" / "docker-compose.change-gate.yml"
REQUIREMENT_PATTERN = re.compile(r"\b[A-Z]{3}-[0-9]{3}\b")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
IMAGE_BUILD_TIMEOUT_SECONDS = 600
PROTECTED_CONTRACT_PATHS = (
    "docs/projectneed.md",
    "docs/decisions/**",
    "tests/semantic/**",
    "tests_scenarios/contracts/**",
    "docker/debug/gate.py",
    "scripts/measure_production_sloc.py",
    "scripts/check_migrations_append_only.py",
    ".github/workflows/**",
)


class GateError(RuntimeError):
    """表示 Gate 无法安全继续的显式失败。"""


@dataclass(frozen=True)
class Group:
    id: str
    priority: str
    requirements: tuple[str, ...]
    paths: tuple[str, ...]
    depends_on: tuple[str, ...]
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    id: str
    requirements: tuple[str, ...]
    groups: tuple[str, ...]
    environment: str
    timeout_seconds: int
    command: tuple[str, ...]
    observes: tuple[str, ...]
    mutants: tuple[str, ...]


@dataclass(frozen=True)
class StateContract:
    id: str
    priority: str
    requirements: tuple[str, ...]
    owner: str
    normal_change: str
    destructive_owner: str
    writers: tuple[str, ...]
    consumers: tuple[str, ...]
    protected_tables: tuple[str, ...]
    oracles: tuple[str, ...]


@dataclass(frozen=True)
class Catalog:
    groups: dict[str, Group]
    scenarios: dict[str, Scenario]
    states: dict[str, StateContract]
    baseline_scenarios: tuple[str, ...]
    executable_suffixes: frozenset[str]
    executable_names: frozenset[str]
    private_groups: frozenset[str]
    always_full_paths: tuple[str, ...]
    mutant_tests: dict[str, str]


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
    )


def _load_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise GateError(f"缺少 Gate 合同文件: {path.relative_to(ROOT)}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if data.get("version") != 1:
        raise GateError(f"不支持的 Gate schema version: {path.relative_to(ROOT)}")
    return cast(dict[str, object], data)


def _table(data: dict[str, object], key: str, *, source: Path) -> dict[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise GateError(f"{source.relative_to(ROOT)} 缺少 [{key}] table")
    return cast(dict[str, object], value)


def _strings(data: dict[str, object], key: str, *, owner: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise GateError(f"{owner}.{key} 必须是非空字符串数组")
    return tuple(cast(list[str], value))


def _optional_strings(
    data: dict[str, object], key: str, *, owner: str
) -> tuple[str, ...]:
    if key not in data:
        return ()
    return _strings(data, key, owner=owner)


def _possibly_empty_strings(
    data: dict[str, object], key: str, *, owner: str
) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise GateError(f"{owner}.{key} 必须是字符串数组")
    return tuple(cast(list[str], value))


def _string_mapping(
    data: dict[str, object], key: str, *, source: Path
) -> dict[str, str]:
    table = _table(data, key, source=source)
    if not all(
        isinstance(mapping_key, str)
        and mapping_key
        and isinstance(value, str)
        and value
        for mapping_key, value in table.items()
    ):
        raise GateError(f"{source.relative_to(ROOT)} [{key}] 必须只含非空字符串映射")
    return cast(dict[str, str], table)


def _require_id(value: str, *, owner: str) -> str:
    if not ID_PATTERN.fullmatch(value):
        raise GateError(f"{owner} id 必须使用 snake_case: {value!r}")
    return value


def _load_groups(data: dict[str, object]) -> dict[str, Group]:
    groups: dict[str, Group] = {}
    for group_id, raw in sorted(_table(data, "groups", source=IMPACT_PATH).items()):
        _require_id(group_id, owner="group")
        if not isinstance(raw, dict):
            raise GateError(f"groups.{group_id} 必须是 table")
        item = cast(dict[str, object], raw)
        priority = item.get("priority")
        if priority not in {"p0", "p1", "p2"}:
            raise GateError(f"groups.{group_id}.priority 必须是 p0/p1/p2")
        groups[group_id] = Group(
            id=group_id,
            priority=cast(str, priority),
            requirements=_strings(item, "requirements", owner=f"groups.{group_id}"),
            paths=_strings(item, "paths", owner=f"groups.{group_id}"),
            depends_on=_optional_strings(
                item, "depends_on", owner=f"groups.{group_id}"
            ),
            scenarios=_strings(item, "scenarios", owner=f"groups.{group_id}"),
        )
    return groups


def _load_scenarios(data: dict[str, object]) -> dict[str, Scenario]:
    scenarios: dict[str, Scenario] = {}
    for scenario_id, raw in sorted(
        _table(data, "scenarios", source=SCENARIO_PATH).items()
    ):
        _require_id(scenario_id, owner="scenario")
        if not isinstance(raw, dict):
            raise GateError(f"scenarios.{scenario_id} 必须是 table")
        item = cast(dict[str, object], raw)
        environment_value = item.get("environment")
        timeout = item.get("timeout_seconds")
        if environment_value != "public_clean_workspace":
            raise GateError(f"scenarios.{scenario_id} 必须使用 public_clean_workspace")
        if not isinstance(timeout, int) or timeout <= 0:
            raise GateError(f"scenarios.{scenario_id}.timeout_seconds 必须大于 0")
        scenarios[scenario_id] = Scenario(
            id=scenario_id,
            requirements=_strings(
                item, "requirements", owner=f"scenarios.{scenario_id}"
            ),
            groups=_strings(item, "groups", owner=f"scenarios.{scenario_id}"),
            environment=cast(str, environment_value),
            timeout_seconds=timeout,
            command=_strings(item, "command", owner=f"scenarios.{scenario_id}"),
            observes=_strings(item, "observes", owner=f"scenarios.{scenario_id}"),
            mutants=_optional_strings(
                item, "mutants", owner=f"scenarios.{scenario_id}"
            ),
        )
    return scenarios


def _load_states(data: dict[str, object]) -> dict[str, StateContract]:
    states: dict[str, StateContract] = {}
    for state_id, raw in sorted(_table(data, "states", source=STATE_PATH).items()):
        _require_id(state_id, owner="state")
        if not isinstance(raw, dict):
            raise GateError(f"states.{state_id} 必须是 table")
        item = cast(dict[str, object], raw)
        priority = item.get("priority")
        owner = item.get("owner")
        normal_change = item.get("normal_change")
        destructive_owner = item.get("destructive_owner")
        if priority not in {"p0", "p1", "p2"}:
            raise GateError(f"states.{state_id}.priority 必须是 p0/p1/p2")
        if not isinstance(owner, str) or not owner:
            raise GateError(f"states.{state_id}.owner 必须是非空字符串")
        if normal_change not in {
            "insert_only",
            "metadata_upsert",
            "select_isolated_root",
            "plugin_owned_update",
        }:
            raise GateError(f"states.{state_id}.normal_change 使用未知协议")
        if destructive_owner not in {
            "explicit_user_data_management",
            "explicit_plugin_data_deletion",
            "not_applicable",
        }:
            raise GateError(f"states.{state_id}.destructive_owner 使用未知 owner")
        states[state_id] = StateContract(
            id=state_id,
            priority=cast(str, priority),
            requirements=_strings(item, "requirements", owner=f"states.{state_id}"),
            owner=owner,
            normal_change=cast(str, normal_change),
            destructive_owner=cast(str, destructive_owner),
            writers=_strings(item, "writers", owner=f"states.{state_id}"),
            consumers=_strings(item, "consumers", owner=f"states.{state_id}"),
            protected_tables=_possibly_empty_strings(
                item, "protected_tables", owner=f"states.{state_id}"
            ),
            oracles=_strings(item, "oracles", owner=f"states.{state_id}"),
        )
    return states


def load_catalog() -> Catalog:
    """严格加载公开能力、状态和场景合同。"""

    # 1. 分别解析三份权威清单，拒绝隐式默认 schema。
    impact = _load_toml(IMPACT_PATH)
    scenario_data = _load_toml(SCENARIO_PATH)
    scenarios = _load_scenarios(scenario_data)
    states = _load_states(_load_toml(STATE_PATH))
    defaults = _table(impact, "defaults", source=IMPACT_PATH)

    # 2. 组装不可变运行视图，引用关系由 audit 统一验证。
    mutant_tests = _string_mapping(scenario_data, "mutant_tests", source=SCENARIO_PATH)
    for mutant_id in mutant_tests:
        _require_id(mutant_id, owner="mutant")
    return Catalog(
        groups=_load_groups(impact),
        scenarios=scenarios,
        states=states,
        baseline_scenarios=_strings(defaults, "baseline_scenarios", owner="defaults"),
        executable_suffixes=frozenset(
            _strings(defaults, "executable_suffixes", owner="defaults")
        ),
        executable_names=frozenset(
            _strings(defaults, "executable_names", owner="defaults")
        ),
        private_groups=frozenset(
            _strings(defaults, "private_groups", owner="defaults")
        ),
        always_full_paths=_strings(defaults, "always_full_paths", owner="defaults"),
        mutant_tests=mutant_tests,
    )


def _tracked_and_untracked_files() -> list[str]:
    inventory_path = os.environ.get("AKASHIC_GATE_INVENTORY")
    if inventory_path:
        payload = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(
            isinstance(path, str)
            and path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            for path in payload
        ):
            raise GateError("AKASHIC_GATE_INVENTORY 必须是安全的相对路径数组")
        return sorted(cast(list[str], payload))
    output = _run_git(
        "ls-files", "-z", "--cached", "--others", "--exclude-standard"
    ).stdout
    return sorted(
        path.decode("utf-8")
        for path in output.split(b"\0")
        if path and not path.startswith(b"private_runtime/")
    )


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern)


def is_protected_contract_path(path: str) -> bool:
    """判断路径是否拥有受保护的合同或策略。"""

    return any(_matches(path, pattern) for pattern in PROTECTED_CONTRACT_PATHS)


def _path_sort(paths: Iterable[str]) -> list[str]:
    return sorted(paths, key=lambda path: path.encode("utf-8"))


def classify_change_paths(changed: Sequence[str]) -> dict[str, list[str]]:
    """把变更路径分成生产源码与受保护合同两组。"""

    return {
        "productionSourcePaths": _path_sort(
            path for path in changed if is_production_source_path(path)
        ),
        "protectedContractPaths": _path_sort(
            path for path in changed if is_protected_contract_path(path)
        ),
    }


def _groups_for_path(path: str, catalog: Catalog) -> set[str]:
    return {
        group.id
        for group in catalog.groups.values()
        if any(_matches(path, pattern) for pattern in group.paths)
    }


def _is_executable(path: str, catalog: Catalog) -> bool:
    name = Path(path).name
    return (
        Path(path).suffix in catalog.executable_suffixes
        or name in catalog.executable_names
        or path == "private_runtime"
    )


def _requirement_ids() -> set[str]:
    text = (ROOT / "docs" / "projectneed.md").read_text(encoding="utf-8")
    return set(REQUIREMENT_PATTERN.findall(text))


def catalog_digest() -> str:
    digest = hashlib.sha256()
    for path in (IMPACT_PATH, STATE_PATH, SCENARIO_PATH):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_digest(files: Sequence[str] | None = None) -> str:
    """计算当前候选源码摘要，包括未忽略的未跟踪文件。"""

    # 1. 路径和内容共同进入摘要，避免重命名被当成相同候选。
    digest = hashlib.sha256()
    for relative in files or _tracked_and_untracked_files():
        digest.update(relative.encode("utf-8"))
        path = ROOT / relative
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8"))
        elif path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(_run_git("ls-files", "--stage", "--", relative).stdout)
    return digest.hexdigest()


def _validate_references(catalog: Catalog, requirements: set[str]) -> list[str]:
    issues: list[str] = []
    for scenario_id in catalog.baseline_scenarios:
        if scenario_id not in catalog.scenarios:
            issues.append(f"baseline scenario 不存在: {scenario_id}")
    for private_group in catalog.private_groups:
        if private_group not in catalog.groups:
            issues.append(f"private group 不存在: {private_group}")
    for group in catalog.groups.values():
        for dependency in group.depends_on:
            if dependency not in catalog.groups:
                issues.append(f"groups.{group.id} 引用未知依赖: {dependency}")
        for scenario in group.scenarios:
            if scenario not in catalog.scenarios:
                issues.append(f"groups.{group.id} 引用未知场景: {scenario}")
        for requirement in group.requirements:
            if requirement not in requirements:
                issues.append(f"groups.{group.id} 引用未知条款: {requirement}")
    for scenario in catalog.scenarios.values():
        for group_id in scenario.groups:
            if group_id not in catalog.groups:
                issues.append(f"scenarios.{scenario.id} 引用未知 group: {group_id}")
        for requirement in scenario.requirements:
            if requirement not in requirements:
                issues.append(f"scenarios.{scenario.id} 引用未知条款: {requirement}")
        for mutant in scenario.mutants:
            if mutant not in catalog.mutant_tests:
                issues.append(f"scenarios.{scenario.id} 引用未知 mutant: {mutant}")
    for mutant_id, test_node in catalog.mutant_tests.items():
        test_path = ROOT / test_node.split("::", 1)[0]
        if not test_path.is_file():
            issues.append(f"mutant_tests.{mutant_id} 测试文件不存在: {test_node}")
    return issues


def _validate_state_contracts(catalog: Catalog, requirements: set[str]) -> list[str]:
    issues: list[str] = []
    for state in catalog.states.values():
        if state.normal_change == "insert_only" and not state.protected_tables:
            issues.append(f"states.{state.id} 的 insert_only 合同缺少 protected_tables")
        if state.destructive_owner == "not_applicable" and state.protected_tables:
            issues.append(
                f"states.{state.id} 声明 destructive_owner 不适用却保护持久表"
            )
        if "PLG-010" in state.requirements and (
            state.destructive_owner != "explicit_plugin_data_deletion"
        ):
            issues.append(
                f"states.{state.id} 违反 PLG-010：普通卸载不得拥有 plugin-data 删除权"
            )
        for group_id in (*state.writers, *state.consumers):
            if group_id not in catalog.groups:
                issues.append(f"states.{state.id} 引用未知 group: {group_id}")
        for oracle in state.oracles:
            scenario = catalog.scenarios.get(oracle)
            if scenario is None:
                issues.append(f"states.{state.id} 引用未知 oracle: {oracle}")
            elif state.priority == "p0" and not scenario.mutants:
                issues.append(f"states.{state.id} 的 P0 oracle 缺少 mutant: {oracle}")
            elif (
                state.normal_change == "insert_only"
                and state.protected_tables
                and "sqlite_write_set" not in scenario.observes
            ):
                issues.append(
                    f"states.{state.id} 的 insert_only oracle 未观察 sqlite_write_set: {oracle}"
                )
        for requirement in state.requirements:
            if requirement not in requirements:
                issues.append(f"states.{state.id} 引用未知条款: {requirement}")
    return issues


def _validate_baseline(payload: dict[str, object], catalog: Catalog) -> list[str]:
    """验证 baseline 只接受非 P0 缺口，且完整记录当前 P0 场景。"""
    issues: list[str] = []
    expected_p0 = {
        group.id: list(group.scenarios)
        for group in catalog.groups.values()
        if group.priority == "p0"
    }
    if payload.get("coveredP0") != expected_p0:
        issues.append("coverage baseline 未完整覆盖当前 P0 场景")
    raw_gaps = payload.get("acceptedGaps")
    if not isinstance(raw_gaps, list):
        return [*issues, "coverage baseline acceptedGaps 必须是数组"]
    seen: set[str] = set()
    for raw_gap in raw_gaps:
        if not isinstance(raw_gap, dict):
            issues.append("coverage baseline gap 必须是 object")
            continue
        gap = cast(dict[str, object], raw_gap)
        gap_id = gap.get("id")
        priority = gap.get("priority")
        if not isinstance(gap_id, str) or not gap_id:
            issues.append("coverage baseline gap 缺少 id")
        elif gap_id in seen:
            issues.append(f"coverage baseline gap id 重复: {gap_id}")
        else:
            seen.add(gap_id)
        if priority not in {"p1", "p2"}:
            issues.append(f"coverage baseline 不得接受 P0 缺口: {gap_id}")
    return issues


def audit_catalog(
    catalog: Catalog, *, check_baseline: bool = True
) -> dict[str, object]:
    """审计全仓可执行文件、合同引用和 P0 mutant 覆盖。"""

    # 1. 建立完整文件和条款清单，逐个验证 owner 映射。
    files = _tracked_and_untracked_files()
    requirements = _requirement_ids()
    issues = _validate_references(catalog, requirements)
    issues.extend(_validate_state_contracts(catalog, requirements))
    path_owners = {path: sorted(_groups_for_path(path, catalog)) for path in files}
    unmapped = [
        path
        for path, owners in path_owners.items()
        if _is_executable(path, catalog) and not owners
    ]
    issues.extend(f"未映射可执行文件: {path}" for path in unmapped)

    # 2. 验证每条路径规则有效，并要求每个 P0 场景自证能抓住已知错误。
    for group in catalog.groups.values():
        for pattern in group.paths:
            if not any(_matches(path, pattern) for path in files):
                issues.append(f"groups.{group.id} 路径规则无匹配: {pattern}")
        if group.priority == "p0":
            for scenario_id in group.scenarios:
                scenario = catalog.scenarios.get(scenario_id)
                if scenario is not None and not scenario.mutants:
                    issues.append(
                        f"groups.{group.id} 的 P0 场景缺少 mutant: {scenario_id}"
                    )

    # 3. baseline 一旦建立只能由评审过的显式 diff 更新。
    baseline_status = "missing"
    if BASELINE_PATH.is_file():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict) or baseline.get("version") != 1:
            issues.append("coverage-baseline.json schema 无效")
            baseline_status = "invalid"
        elif check_baseline and baseline.get("catalogDigest") != catalog_digest():
            issues.append("coverage baseline 与当前 catalog digest 不一致")
            baseline_status = "stale"
        else:
            baseline_issues = _validate_baseline(
                cast(dict[str, object], baseline), catalog
            )
            issues.extend(baseline_issues)
            baseline_status = "current" if not baseline_issues else "invalid"
    return {
        "version": 1,
        "status": "passed" if not issues else "failed",
        "catalogDigest": catalog_digest(),
        "fileCount": len(files),
        "executableFileCount": sum(_is_executable(path, catalog) for path in files),
        "pathOwners": path_owners,
        "stateContracts": {
            state.id: {
                "normalChange": state.normal_change,
                "destructiveOwner": state.destructive_owner,
                "protectedTables": list(state.protected_tables),
            }
            for state in catalog.states.values()
        },
        "unmappedExecutableFiles": unmapped,
        "baselineStatus": baseline_status,
        "issues": issues,
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _report_dir(run_id: str) -> Path:
    path = REPORT_ROOT / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def command_init(args: argparse.Namespace) -> int:
    """执行一次性全仓审计，并在零阻塞时建立 coverage baseline。"""

    if BASELINE_PATH.exists():
        raise GateError("coverage baseline 已存在；请使用 audit，禁止重新 init 覆盖")
    catalog = load_catalog()
    run_id = _new_run_id()
    report_dir = _report_dir(run_id)
    inventory = audit_catalog(catalog, check_baseline=False)
    _atomic_json(report_dir / "inventory.json", inventory)
    if inventory["status"] != "passed":
        print(f"INIT BLOCKED report={report_dir.relative_to(ROOT)}")
        for issue in cast(list[str], inventory["issues"]):
            print(f"- {issue}")
        return 1

    # 2. 实际运行所有已登记 fault injection，不能只相信清单中的名称。
    mutant_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *catalog.mutant_tests.values()],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    mutant_report: dict[str, object] = {
        "status": "passed" if mutant_result.returncode == 0 else "failed",
        "tests": list(catalog.mutant_tests.values()),
        "exitCode": mutant_result.returncode,
        "stdout": mutant_result.stdout,
        "stderr": mutant_result.stderr,
    }
    _atomic_json(report_dir / "mutants.json", mutant_report)
    if mutant_result.returncode != 0:
        print(f"INIT BLOCKED mutant tests failed report={report_dir.relative_to(ROOT)}")
        return 1

    # 3. baseline 落盘前，在全新 Docker workspace 中跑完全部公开场景。
    scenario_checks = [
        _run_scenario(scenario, run_id=run_id, report_dir=report_dir)
        for scenario in catalog.scenarios.values()
    ]
    if any(check["status"] != "passed" for check in scenario_checks):
        print(
            f"INIT BLOCKED public scenarios failed report={report_dir.relative_to(ROOT)}"
        )
        return 1

    # 4. P0 只有独立 oracle、mutant 和干净场景都通过时才形成基线。
    covered_p0 = {
        group.id: list(group.scenarios)
        for group in catalog.groups.values()
        if group.priority == "p0"
    }
    baseline: dict[str, object] = {
        "version": 1,
        "catalogDigest": catalog_digest(),
        "base": _resolve_commit(args.base),
        "coveredP0": covered_p0,
        "acceptedGaps": [],
    }
    _atomic_json(BASELINE_PATH, baseline)
    print(f"INIT PASSED baseline={BASELINE_PATH.relative_to(ROOT)}")
    print(f"report={report_dir.relative_to(ROOT)}")
    return 0


def command_audit(_args: argparse.Namespace) -> int:
    """审计当前索引，不修改任何权威合同。"""
    catalog = load_catalog()
    run_id = _new_run_id()
    report_dir = _report_dir(run_id)
    report = audit_catalog(catalog)
    _atomic_json(report_dir / "inventory.json", report)
    print(
        f"AUDIT {str(report['status']).upper()} report={report_dir.relative_to(ROOT)}"
    )
    for issue in cast(list[str], report["issues"]):
        print(f"- {issue}")
    return 0 if report["status"] == "passed" else 1


def _resolve_commit(reference: str) -> str:
    result = _run_git("rev-parse", "--verify", f"{reference}^{{commit}}", check=False)
    if result.returncode != 0:
        raise GateError(f"无法解析 base commit: {reference}")
    return result.stdout.decode("utf-8").strip()


def _changed_paths(base_commit: str) -> list[str]:
    changed = _run_git("diff", "--name-only", "-z", base_commit, "--").stdout
    untracked = _run_git("ls-files", "-z", "--others", "--exclude-standard").stdout
    return sorted(
        {
            path.decode("utf-8")
            for path in (*changed.split(b"\0"), *untracked.split(b"\0"))
            if path and not path.startswith(b"private_runtime/")
        }
    )


def _dirty_status() -> list[str]:
    output = _run_git("status", "--porcelain=v1", "-z").stdout
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def _dependency_closure(initial: set[str], catalog: Catalog) -> set[str]:
    affected = set(initial)
    while True:
        expanded = set(affected)
        for group_id in affected:
            expanded.update(catalog.groups[group_id].depends_on)
        for group in catalog.groups.values():
            if affected.intersection(group.depends_on):
                expanded.add(group.id)
        if expanded == affected:
            return affected
        affected = expanded


def _load_baseline() -> dict[str, object]:
    if not BASELINE_PATH.is_file():
        raise GateError("coverage baseline 尚未建立；先运行 gate.py init")
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise GateError("coverage-baseline.json schema 无效")
    return cast(dict[str, object], payload)


def _touched_gaps(
    changed: Sequence[str], baseline: dict[str, object], catalog: Catalog
) -> list[str]:
    raw_gaps = baseline.get("acceptedGaps")
    if not isinstance(raw_gaps, list):
        raise GateError("coverage-baseline.json acceptedGaps 必须是数组")
    touched: list[str] = []
    for raw_gap in raw_gaps:
        if not isinstance(raw_gap, dict):
            raise GateError("coverage-baseline.json gap 必须是 object")
        gap = cast(dict[str, object], raw_gap)
        gap_id = gap.get("id")
        groups = gap.get("groups")
        paths = gap.get("paths")
        if (
            not isinstance(gap_id, str)
            or not isinstance(groups, list)
            or not isinstance(paths, list)
        ):
            raise GateError("coverage-baseline.json gap 缺少 id/groups/paths")
        group_paths = [
            pattern
            for group_id in groups
            if isinstance(group_id, str) and group_id in catalog.groups
            for pattern in catalog.groups[group_id].paths
        ]
        patterns = [*cast(list[str], paths), *group_paths]
        if any(_matches(path, pattern) for path in changed for pattern in patterns):
            touched.append(gap_id)
    return sorted(touched)


def build_plan(base: str, *, full: bool = False) -> dict[str, object]:
    """根据 Git diff 和公开索引生成不含 provider 身份的计划。"""

    # 1. 先要求 catalog 与 baseline 自洽，再读取候选 diff。
    catalog = load_catalog()
    audit = audit_catalog(catalog)
    if audit["status"] != "passed":
        raise GateError("catalog audit 失败；先运行 gate.py audit 查看问题")
    baseline = _load_baseline()
    base_commit = _resolve_commit(base)
    changed = _changed_paths(base_commit)
    change_classes = classify_change_paths(changed)
    production_paths = change_classes["productionSourcePaths"]
    protected_paths = change_classes["protectedContractPaths"]
    protected_contract_mixed = bool(production_paths and protected_paths)

    # 2. 计算直接命中、依赖闭包和未知可执行改动。
    direct: set[str] = set()
    reasons: list[str] = []
    unmapped: list[str] = []
    for path in changed:
        owners = _groups_for_path(path, catalog)
        direct.update(owners)
        if owners:
            reasons.append(f"{path} -> {','.join(sorted(owners))}")
        elif _is_executable(path, catalog):
            unmapped.append(path)
    force_full = (
        full
        or bool(unmapped)
        or any(
            _matches(path, pattern)
            for path in changed
            for pattern in catalog.always_full_paths
        )
    )
    affected = (
        set(catalog.groups) if force_full else _dependency_closure(direct, catalog)
    )
    selected = set(catalog.baseline_scenarios if affected else ())
    for group_id in affected:
        selected.update(catalog.groups[group_id].scenarios)
    touched_gaps = _touched_gaps(changed, baseline, catalog)

    # 3. 公开计划只暴露能力分组，provider 选择留给 private companion。
    status = "planned"
    if protected_contract_mixed:
        status = "protected_contract_mixed"
    elif not changed and not full:
        status = "not_affected"
    elif unmapped:
        status = "unmapped_change"
    elif touched_gaps:
        status = "baseline_gap_touched"
    payload: dict[str, object] = {
        "version": 1,
        "status": status,
        "base": base_commit,
        "head": _run_git("rev-parse", "HEAD").stdout.decode("utf-8").strip(),
        "dirtyStatus": _dirty_status(),
        "sourceDigest": source_digest(),
        "impactCatalogDigest": catalog_digest(),
        "changedPaths": changed,
        "productionSourcePaths": production_paths,
        "protectedContractPaths": protected_paths,
        "affectedGroups": sorted(affected),
        "selectedScenarios": sorted(selected),
        "privateGateRequired": bool(affected.intersection(catalog.private_groups)),
        "unmappedChanges": unmapped,
        "touchedBaselineGaps": touched_gaps,
        "full": force_full,
        "reasons": reasons,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["planDigest"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _print_plan(plan: dict[str, object]) -> None:
    print(f"Status: {plan['status']}")
    print("Changed:")
    for path in cast(list[str], plan["changedPaths"]):
        print(f"  {path}")
    print("Production source paths:")
    for path in cast(list[str], plan["productionSourcePaths"]):
        print(f"  {path}")
    print("Protected contract/policy paths:")
    for path in cast(list[str], plan["protectedContractPaths"]):
        print(f"  {path}")
    print("Affected groups:")
    for group in cast(list[str], plan["affectedGroups"]):
        print(f"  {group}")
    print("Selected public scenarios:")
    for scenario in cast(list[str], plan["selectedScenarios"]):
        print(f"  {scenario}")
    print(f"Private gate required: {str(plan['privateGateRequired']).lower()}")
    print(f"Plan digest: {plan['planDigest']}")


def command_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args.base, full=args.full)
    report_dir = _report_dir(_new_run_id())
    _atomic_json(report_dir / "plan.json", plan)
    _print_plan(plan)
    print(f"report={report_dir.relative_to(ROOT)}")
    return (
        1
        if plan["status"]
        in {"unmapped_change", "baseline_gap_touched", "protected_contract_mixed"}
        else 0
    )


def _copy_candidate_source(destination: Path) -> None:
    """把当前候选源码复制进只读容器挂载目录。"""
    for relative in _tracked_and_untracked_files():
        source = ROOT / relative
        if not source.is_file() and not source.is_symlink():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)


def _write_sandbox_config(path: Path) -> None:
    path.write_text(
        '[runtime]\nworkspace = "/sandbox/workspace"\n',
        encoding="utf-8",
    )


def _compose_command(project: str, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_PATH),
        "-p",
        project,
        *args,
    ]


def _compose_env(sandbox: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "AKASHIC_CHANGE_GATE_SANDBOX": str(sandbox),
            "UID": str(os.getuid()),
            "GID": str(os.getgid()),
        }
    )
    return env


def _residual_resources(project: str) -> dict[str, list[str]]:
    resources: dict[str, list[str]] = {}
    for kind, command in (
        ("containers", ["docker", "ps", "-a", "-q", "--filter"]),
        ("networks", ["docker", "network", "ls", "-q", "--filter"]),
        ("volumes", ["docker", "volume", "ls", "-q", "--filter"]),
    ):
        result = subprocess.run(
            [*command, f"label=com.docker.compose.project={project}"],
            check=True,
            capture_output=True,
            text=True,
        )
        resources[kind] = [line for line in result.stdout.splitlines() if line]
    return resources


def _prepare_sandbox(run_id: str, scenario_id: str) -> Path:
    sandbox = Path(
        tempfile.mkdtemp(
            prefix=f"akashic-change-gate-{run_id}-{scenario_id}-",
            dir="/tmp",
        )
    )
    for name in ("app", "workspace", "plugin-home", "home", "reports", "static"):
        (sandbox / name).mkdir()
    _copy_candidate_source(sandbox / "app")
    (sandbox / "app" / "static").mkdir(exist_ok=True)
    _write_sandbox_config(sandbox / "config.toml")
    (sandbox / "source-inventory.json").write_text(
        json.dumps(_tracked_and_untracked_files(), ensure_ascii=False),
        encoding="utf-8",
    )
    return sandbox


def _build_change_gate_image(run_id: str, report_dir: Path) -> dict[str, object]:
    """构建一次 Gate 镜像，并把构建失败作为可审计结果返回。"""

    # 1. Compose 解析配置需要隔离路径，但构建阶段不会挂载或运行它。
    with tempfile.TemporaryDirectory(
        prefix=f"akashic-change-gate-{run_id}-image-build-",
        dir="/tmp",
    ) as sandbox_name:
        project = f"akashic-change-gate-build-{uuid.uuid4().hex[:12]}"
        env = _compose_env(Path(sandbox_name))
        started = time.monotonic()
        result: subprocess.CompletedProcess[str] | None = None
        timeout_error = ""

        # 2. 镜像构建不占用场景自己的执行超时。
        try:
            result = subprocess.run(
                _compose_command(project, "build", "change-gate"),
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=IMAGE_BUILD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            timeout_error = f"Gate 镜像构建超过 {IMAGE_BUILD_TIMEOUT_SECONDS}s"

    status = "passed"
    if timeout_error or result is None or result.returncode != 0:
        status = "failed"
    record: dict[str, object] = {
        "status": status,
        "durationSeconds": round(time.monotonic() - started, 3),
        "exitCode": None if result is None else result.returncode,
        "stdout": "" if result is None else result.stdout,
        "stderr": timeout_error if result is None else result.stderr,
    }
    _atomic_json(report_dir / "image-build.json", record)
    return record


def _run_scenario(
    scenario: Scenario, *, run_id: str, report_dir: Path
) -> dict[str, object]:
    """在独立 Docker workspace 中运行一个公开场景并审计清理。"""

    # 1. 每个场景建立全新的 workspace、plugin home 和 Compose project。
    sandbox = _prepare_sandbox(run_id, scenario.id)
    project = f"akashic-change-gate-{uuid.uuid4().hex[:12]}"
    env = _compose_env(sandbox)
    started = time.monotonic()
    result: subprocess.CompletedProcess[str] | None = None
    timeout_error = ""
    cleanup_result: subprocess.CompletedProcess[str] | None = None

    # 2. 在只读源码挂载中执行声明命令，失败输出仍写入公开报告。
    try:
        try:
            result = subprocess.run(
                _compose_command(
                    project,
                    "run",
                    "--rm",
                    "--no-deps",
                    "change-gate",
                    *scenario.command,
                ),
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=scenario.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            timeout_error = f"场景超过 {scenario.timeout_seconds}s"
    finally:
        cleanup_result = subprocess.run(
            _compose_command(project, "down", "--remove-orphans", "--volumes"),
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    # 3. cleanup 后检查 Docker label，确认没有跨场景状态残留。
    residual = _residual_resources(project)
    status = "passed"
    if timeout_error or result is None or result.returncode != 0:
        status = "failed"
    if cleanup_result.returncode != 0 or any(residual.values()):
        status = "failed"
    record: dict[str, object] = {
        "id": scenario.id,
        "status": status,
        "command": list(scenario.command),
        "durationSeconds": round(time.monotonic() - started, 3),
        "exitCode": None if result is None else result.returncode,
        "stdout": "" if result is None else result.stdout,
        "stderr": timeout_error if result is None else result.stderr,
        "cleanupExitCode": cleanup_result.returncode,
        "residualResources": residual,
        "observes": list(scenario.observes),
    }
    _atomic_json(report_dir / "public" / f"{scenario.id}.json", record)
    shutil.rmtree(sandbox)
    return record


def command_run(args: argparse.Namespace) -> int:
    """生成计划，并在干净 Docker workspace 中执行所选公开场景。"""

    # 1. 计划先落盘；公开 Gate 不读取 private runtime 或插件清单。
    catalog = load_catalog()
    plan = build_plan(args.base, full=args.full)
    run_id = _new_run_id()
    report_dir = _report_dir(run_id)
    _atomic_json(report_dir / "plan.json", plan)
    _print_plan(plan)

    # 2. 镜像只构建一次；每个场景的超时只约束场景本身。
    selected_scenarios = cast(list[str], plan["selectedScenarios"])
    image_build: dict[str, object] | None = None
    if selected_scenarios and plan["status"] != "protected_contract_mixed":
        image_build = _build_change_gate_image(run_id, report_dir)
    checks = []
    if plan["status"] != "protected_contract_mixed" and (
        image_build is None or image_build["status"] == "passed"
    ):
        checks = [
            _run_scenario(
                catalog.scenarios[scenario_id], run_id=run_id, report_dir=report_dir
            )
            for scenario_id in selected_scenarios
        ]

    # 3. 未知映射、baseline gap、构建失败或场景失败都必须阻断。
    status = "passed"
    if plan["status"] == "not_affected":
        status = "not_affected"
    elif plan["status"] in {
        "unmapped_change",
        "baseline_gap_touched",
        "protected_contract_mixed",
    }:
        status = cast(str, plan["status"])
    elif image_build is not None and image_build["status"] != "passed":
        status = "failed"
    elif any(check["status"] != "passed" for check in checks):
        status = "failed"
    residual = {
        resource: sorted(
            {
                item
                for check in checks
                for item in cast(dict[str, list[str]], check["residualResources"])[
                    resource
                ]
            }
        )
        for resource in ("containers", "networks", "volumes")
    }
    gate: dict[str, object] = {
        "version": 1,
        "status": status,
        "base": plan["base"],
        "head": plan["head"],
        "dirtyStatus": plan["dirtyStatus"],
        "sourceDigest": plan["sourceDigest"],
        "impactCatalogDigest": plan["impactCatalogDigest"],
        "planDigest": plan["planDigest"],
        "productionSourcePaths": plan["productionSourcePaths"],
        "protectedContractPaths": plan["protectedContractPaths"],
        "affectedGroups": plan["affectedGroups"],
        "selectedScenarios": plan["selectedScenarios"],
        "privateGateRequired": plan["privateGateRequired"],
        "imageBuild": image_build,
        "checks": checks,
        "residualResources": residual,
    }
    _atomic_json(report_dir / "gate.json", gate)
    print(f"GATE {status.upper()} report={report_dir.relative_to(ROOT)}")
    if plan["privateGateRequired"]:
        print("private-contract-gate: required (provider identity stays private)")
    return 0 if status in {"passed", "not_affected"} else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Akashic 变更影响契约 Gate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (
        ("init", command_init),
        ("audit", command_audit),
        ("plan", command_plan),
        ("run", command_run),
    ):
        subparser = subparsers.add_parser(name)
        if name != "audit":
            subparser.add_argument("--base", default="origin/main")
        if name in {"plan", "run"}:
            subparser.add_argument("--full", action="store_true")
        subparser.set_defaults(handler=handler)
    return parser


def _fail(message: str) -> NoReturn:
    print(f"GATE ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    args = _parser().parse_args()
    try:
        return cast(int, args.handler(args))
    except (GateError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        _fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
