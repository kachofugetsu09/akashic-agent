#!/usr/bin/env python3
"""Verify exact Content-source revisions and replay their owned fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import tomllib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.plugins.generation_activity_host import ActivityHost
from agent.plugins.generation_job_host import BackgroundJobActivityAdapter
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.eventmail.store import EventMailStore

DEFAULT_LOCK = Path(__file__).with_name("content-source-interop.lock.json")
DEFAULT_REPORT = (
    Path(__file__).parent / "reports" / "content-source-interop" / "gate.json"
)
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_PROACTIVE_MARKERS = (
    "PROACTIVE_COMPONENTS",
    "ProactiveSourceSpec",
    "ProactiveModuleDefinition",
    "get_proactive_events",
    "acknowledge_events",
    "take_proactive_events",
)


class GateError(RuntimeError):
    """Represent one actionable interoperability gate failure."""


class _MountOnlyConversationRuntime:
    async def start_turn(self, *args: object, **kwargs: object) -> object:
        return _unexpected_programmatic_call(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class PluginContract:
    id: str
    repository: str
    branch: str
    resolved_sha: str
    pull_request: str | None
    role: str
    atoms: tuple[str, ...]
    test_cwd: str
    cases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CrossRepoContract:
    id: str
    plugin_ids: tuple[str, ...]
    python_plugin_id: str
    cases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InteropContract:
    core_contract: str
    core_cases: tuple[str, ...]
    coexistence: tuple[dict[str, object], ...]
    cross_repo: tuple[CrossRepoContract, ...]
    plugins: tuple[PluginContract, ...]
    pending: tuple[dict[str, object], ...]
    retired: tuple[dict[str, object], ...]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_path_map(values: list[str], option: str) -> dict[str, Path]:
    """Parse repeated id=path CLI values without inventing default roots."""

    parsed: dict[str, Path] = {}
    for value in values:
        plugin_id, separator, raw_path = value.partition("=")
        if not separator or not plugin_id or not raw_path:
            raise GateError(f"{option} 必须使用 id=/absolute/path")
        if plugin_id in parsed:
            raise GateError(f"{option} 重复 id: {plugin_id}")
        path = Path(raw_path)
        if not path.is_absolute():
            raise GateError(f"{option} 必须是绝对路径: {value}")
        parsed[plugin_id] = path
    return parsed


def _load_contract(path: Path) -> InteropContract:
    """Strictly parse the revision and fixture contract at the file boundary."""

    raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "core_contract",
        "core_cases",
        "coexistence",
        "cross_repo",
        "plugins",
        "pending",
        "retired",
    }
    if not isinstance(raw_value, dict):
        raise GateError("interop lock 根结构无效")
    raw = cast(dict[str, object], raw_value)
    if set(raw) != expected:
        raise GateError("interop lock 根结构无效")
    if raw["schema_version"] != 1:
        raise GateError("interop lock schema_version 不受支持")
    core_contract = raw["core_contract"]
    if (
        not isinstance(core_contract, str)
        or SHA_PATTERN.fullmatch(core_contract) is None
    ):
        raise GateError("core_contract 必须是完整 SHA")
    core_cases = _string_tuple(raw["core_cases"], "core_cases")
    coexistence = _mapping_tuple(
        raw["coexistence"],
        {"plugin_id", "config_toml", "expected_content_rows"},
        "coexistence",
    )
    cross_repo_value = raw["cross_repo"]
    if not isinstance(cross_repo_value, list):
        raise GateError("cross_repo 必须是数组")
    cross_repo = tuple(
        _parse_cross_repo(item) for item in cast(list[object], cross_repo_value)
    )
    plugins_value = raw["plugins"]
    if not isinstance(plugins_value, list):
        raise GateError("plugins 必须是数组")
    plugins_raw = cast(list[object], plugins_value)
    plugins = tuple(_parse_plugin(item) for item in plugins_raw)
    ids = tuple(item.id for item in plugins)
    if len(ids) != len(set(ids)):
        raise GateError("plugins 不得重复")
    pending = _parse_pending(raw["pending"])
    retired = _parse_retired(raw["retired"])
    plugin_ids = {plugin.id for plugin in plugins}
    unknown_coexistence = [
        item["plugin_id"] for item in coexistence if item["plugin_id"] not in plugin_ids
    ]
    if unknown_coexistence:
        raise GateError(f"coexistence 引用未知插件: {unknown_coexistence}")
    for suite in cross_repo:
        unknown = set(suite.plugin_ids) - plugin_ids
        if unknown or suite.python_plugin_id not in plugin_ids:
            raise GateError(
                f"cross_repo 引用未知插件: {suite.id} "
                f"{sorted(unknown | {suite.python_plugin_id} - plugin_ids)}"
            )
    return InteropContract(
        core_contract,
        core_cases,
        coexistence,
        cross_repo,
        plugins,
        pending,
        retired,
    )


def _parse_plugin(raw: object) -> PluginContract:
    """Parse one exact external plugin identity and its owned fixture list."""

    fields = {
        "id",
        "repository",
        "branch",
        "resolved_sha",
        "pull_request",
        "role",
        "atoms",
        "test_cwd",
        "cases",
    }
    if not isinstance(raw, dict):
        raise GateError(f"plugin contract 字段无效: {raw}")
    item = cast(dict[str, object], raw)
    if set(item) != fields:
        raise GateError(f"plugin contract 字段无效: {raw}")
    strings: dict[str, str] = {}
    for field in ("id", "repository", "branch", "resolved_sha", "role", "test_cwd"):
        value = item[field]
        if not isinstance(value, str) or not value:
            raise GateError(f"plugin {field} 必须是非空字符串")
        strings[field] = value
    pull_request = item["pull_request"]
    if pull_request is not None and (
        not isinstance(pull_request, str) or not pull_request
    ):
        raise GateError("plugin pull_request 必须是非空字符串或 null")
    if SHA_PATTERN.fullmatch(strings["resolved_sha"]) is None:
        raise GateError(f"plugin resolved_sha 必须是完整 SHA: {strings['id']}")
    if strings["test_cwd"] not in {".", "tests"}:
        raise GateError(f"plugin test_cwd 不受支持: {strings['id']}")
    return PluginContract(
        **strings,
        pull_request=pull_request,
        atoms=_string_tuple(item["atoms"], f"{strings['id']}.atoms"),
        cases=_string_tuple(item["cases"], f"{strings['id']}.cases"),
    )


def _parse_cross_repo(raw: object) -> CrossRepoContract:
    """Parse one source-neutral suite that combines multiple exact checkouts."""

    fields = {"id", "plugin_ids", "python_plugin_id", "cases"}
    if not isinstance(raw, dict):
        raise GateError(f"cross_repo 条目字段无效: {raw}")
    item = cast(dict[str, object], raw)
    if set(item) != fields:
        raise GateError(f"cross_repo 条目字段无效: {raw}")
    suite_id = item["id"]
    python_plugin_id = item["python_plugin_id"]
    if not isinstance(suite_id, str) or not suite_id:
        raise GateError("cross_repo id 必须是非空字符串")
    if not isinstance(python_plugin_id, str) or not python_plugin_id:
        raise GateError("cross_repo python_plugin_id 必须是非空字符串")
    return CrossRepoContract(
        id=suite_id,
        plugin_ids=_string_tuple(item["plugin_ids"], f"{suite_id}.plugin_ids"),
        python_plugin_id=python_plugin_id,
        cases=_string_tuple(item["cases"], f"{suite_id}.cases"),
    )


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise GateError(f"{field} 必须是非空字符串数组")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) or not item for item in raw_items):
        raise GateError(f"{field} 必须是非空字符串数组")
    result = tuple(cast(list[str], raw_items))
    if len(result) != len(set(result)):
        raise GateError(f"{field} 不得重复")
    return result


def _mapping_tuple(
    value: object,
    fields: set[str],
    name: str,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise GateError(f"{name} 必须是数组")
    result: list[dict[str, object]] = []
    for raw_item in cast(list[object], value):
        if not isinstance(raw_item, dict):
            raise GateError(f"{name} 条目字段无效: {raw_item}")
        item = cast(dict[str, object], raw_item)
        if set(item) != fields:
            raise GateError(f"{name} 条目字段无效: {item}")
        result.append(item)
    return tuple(result)


def _parse_pending(value: object) -> tuple[dict[str, object], ...]:
    pending = _mapping_tuple(value, {"id", "reason"}, "pending")
    for item in pending:
        if any(not isinstance(item[field], str) or not item[field] for field in item):
            raise GateError(f"pending 字段必须是非空字符串: {item}")
    return pending


def _parse_retired(value: object) -> tuple[dict[str, object], ...]:
    retired = _mapping_tuple(
        value,
        {"id", "canonical_sha", "disposition", "evidence"},
        "retired",
    )
    for item in retired:
        for field in ("id", "disposition"):
            if not isinstance(item[field], str) or not item[field]:
                raise GateError(f"retired {field} 必须是非空字符串")
        revision = item["canonical_sha"]
        if not isinstance(revision, str) or SHA_PATTERN.fullmatch(revision) is None:
            raise GateError("retired canonical_sha 必须是完整 SHA")
        evidence = item["evidence"]
        if not isinstance(evidence, list):
            raise GateError("retired evidence 必须是非空字符串数组")
        evidence_items = cast(list[object], evidence)
        if not evidence_items or any(
            not isinstance(entry, str) or not entry for entry in evidence_items
        ):
            raise GateError("retired evidence 必须是非空字符串数组")
    return retired


def _validate_execution_mode(
    *,
    identity_only: bool,
    allow_pending: bool,
    expected_ids: set[str],
    python_ids: set[str],
) -> None:
    """Require explicit runtimes for behavioral evidence and narrow pending bypass."""

    if allow_pending and not identity_only:
        raise GateError("--allow-pending 只能与 --identity-only 一起使用")
    unknown = python_ids - expected_ids
    if unknown:
        raise GateError(f"未知 plugin Python: {sorted(unknown)}")
    if not identity_only and python_ids != expected_ids:
        raise GateError(
            "完整 Gate 必须为每个插件显式绑定 --plugin-python: "
            f"missing={sorted(expected_ids - python_ids)}"
        )


def _verify_core(contract: InteropContract) -> dict[str, object]:
    """Prove the current stack still descends from the approved Core contract."""

    head = _git(ROOT, "rev-parse", "HEAD")
    ancestry = subprocess.run(
        ("git", "merge-base", "--is-ancestor", contract.core_contract, head),
        cwd=ROOT,
        check=False,
    ).returncode
    if ancestry != 0:
        raise GateError(
            f"当前 Core 不包含批准合同: {contract.core_contract} head={head}"
        )
    missing = [case for case in contract.core_cases if not (ROOT / case).is_file()]
    if missing:
        raise GateError(f"Core fixture 缺失: {missing}")
    return {
        "head": head,
        "contract": contract.core_contract,
        "cases": contract.core_cases,
    }


def _source_identity(root: Path) -> dict[str, object]:
    """Read the exact Git source identity observed around one fixture."""

    return {
        "head": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "status": tuple(_git(root, "status", "--porcelain").splitlines()),
    }


def _python_receipt(python: Path) -> dict[str, object]:
    """Execute a controlled probe and prove the selected path is a Python runtime."""

    if not python.is_file() or not os.access(python, os.X_OK):
        raise GateError(f"fixture Python 不可执行: {python}")
    code = (
        "import json,sys;"
        "print(json.dumps({'executable':sys.executable,"
        "'implementation':sys.implementation.name,"
        "'version':[sys.version_info.major,sys.version_info.minor,sys.version_info.micro]}))"
    )
    result = subprocess.run(
        (str(python), "-I", "-c", code),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GateError(f"fixture Python probe 失败: {python} {result.stderr[-400:]}")
    try:
        decoded: object = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(f"fixture executable 不是 Python: {python}") from error
    if not isinstance(decoded, dict):
        raise GateError(f"fixture executable 不是 Python: {python}")
    receipt = cast(dict[str, object], decoded)
    if set(receipt) != {"executable", "implementation", "version"}:
        raise GateError(f"fixture Python receipt 无效: {python}")
    executable = receipt["executable"]
    implementation = receipt["implementation"]
    version = receipt["version"]
    version_items = cast(list[object], version) if isinstance(version, list) else []
    if (
        not isinstance(executable, str)
        or Path(executable).resolve() != python.resolve()
        or implementation != "cpython"
        or not isinstance(version, list)
        or len(version_items) != 3
        or any(not isinstance(part, int) for part in version_items)
    ):
        raise GateError(f"fixture Python identity 不匹配: {python} {receipt}")
    return {
        "requested": str(python),
        "realpath": str(python.resolve()),
        "implementation": implementation,
        "version": version,
    }


def _verify_plugin(plugin: PluginContract, root: Path) -> dict[str, object]:
    """Verify exact source identity, public manifest, and removed island seams."""

    if not root.is_dir():
        raise GateError(f"plugin root 不存在: {plugin.id}={root}")
    head = _git(root, "rev-parse", "HEAD")
    if head != plugin.resolved_sha:
        raise GateError(
            f"plugin SHA 不匹配: {plugin.id} expected={plugin.resolved_sha} actual={head}"
        )
    dirty = tuple(_git(root, "status", "--porcelain").splitlines())
    if dirty:
        raise GateError(f"plugin checkout 非 clean: {plugin.id} {dirty}")
    manifest_path = root / "akashic.plugin.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("api_version") != 3:
        raise GateError(f"plugin 不是 pure v3: {plugin.id}")
    if manifest.get("name") != plugin.id:
        raise GateError(
            f"plugin manifest identity 不匹配: {plugin.id} != {manifest.get('name')}"
        )
    missing_cases = [case for case in plugin.cases if not (root / case).is_file()]
    if missing_cases:
        raise GateError(f"plugin fixture 缺失: {plugin.id} {missing_cases}")
    findings = _forbidden_markers(root)
    if findings:
        raise GateError(f"plugin 仍含 proactive-only seam: {plugin.id} {findings}")
    return {
        **asdict(plugin),
        "root": str(root),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "status": "verified",
    }


def _forbidden_markers(root: Path) -> tuple[str, ...]:
    findings: list[str] = []
    ignored = {".git", ".venv", ".akashic-core", ".plugin-contracts", "tests"}
    for path in sorted(root.rglob("*.py")):
        if ignored.intersection(path.relative_to(root).parts):
            continue
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_PROACTIVE_MARKERS:
            if marker in text:
                findings.append(f"{path.relative_to(root)}:{marker}")
    return tuple(findings)


async def _run_coexistence_probe(
    contract: dict[str, object],
    plugin_root: Path,
) -> dict[str, object]:
    """Mount a real non-Content plugin beside Content and prove zero mailbox writes."""

    plugin_id = contract["plugin_id"]
    config_toml = contract["config_toml"]
    expected_rows = contract["expected_content_rows"]
    if (
        not isinstance(plugin_id, str)
        or not isinstance(config_toml, str)
        or not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows < 0
    ):
        raise GateError(f"coexistence contract 无效: {contract}")

    source_before = _source_identity(plugin_root)
    core_before = _source_identity(ROOT)
    with tempfile.TemporaryDirectory(prefix="akashic-content-source-interop-") as raw:
        root = Path(raw)
        plugins = root / "plugins"
        content_dir = plugins / "content"
        staged_plugin = plugins / plugin_id
        _ = shutil.copytree(ROOT / "plugins" / "eventmail", content_dir)
        _ = shutil.copytree(
            plugin_root,
            staged_plugin,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                ".akashic-core",
                ".plugin-contracts",
                ".pytest_cache",
                "__pycache__",
                "tests",
            ),
        )
        workspace = root / "workspace"
        data_root = workspace / "plugin-data" / f"{plugin_id}-builtin"
        data_root.mkdir(parents=True)
        _ = (data_root / "config.local.toml").write_text(
            config_toml,
            encoding="utf-8",
        )
        content_path = workspace / "plugin-data" / "eventmail-builtin" / "eventmail.sqlite3"
        baseline = PluginManager(
            plugin_dirs=[content_dir],
            event_bus=EventBus(),
            workspace=workspace,
            installed_cache_root=root / "baseline-cache",
        )
        try:
            await baseline.load_all()
        finally:
            await baseline.terminate_all()
        content_before = _content_logical_state(content_path)
        event_bus = EventBus()
        manager = PluginManager(
            plugin_dirs=[content_dir, staged_plugin],
            event_bus=event_bus,
            workspace=workspace,
            installed_cache_root=root / "cache",
        )
        mount_only_runtime = _MountOnlyConversationRuntime()
        manager.bind_conversation_runtime(
            mount_only_runtime,
            programmatic_session_creator=_unexpected_programmatic_call,
            programmatic_session_reader=_unexpected_programmatic_call,
        )
        manager.bind_activity_host(
            ActivityHost(
                (
                    BackgroundJobActivityAdapter(
                        manager.snapshot_store,
                        workspace=str(workspace),
                        conversation_runtime=mount_only_runtime,
                        programmatic_session_creator=_unexpected_programmatic_call,
                        programmatic_session_reader=_unexpected_programmatic_call,
                    ),
                )
            )
        )
        row_count = -1
        try:
            await manager.load_all()
            store = EventMailStore(content_path)
            row_count = sum(store.state_counts().values())
            if row_count != expected_rows:
                raise GateError(
                    f"coexistence Content rows 漂移: {plugin_id} "
                    f"expected={expected_rows} actual={row_count}"
                )
        finally:
            await manager.terminate_all()
        content_after = _content_logical_state(content_path)
        changed_tables = [
            table
            for table in content_before
            if content_before[table] != content_after[table]
        ]
        if changed_tables:
            raise GateError(
                f"coexistence 改写 Content logical state: {plugin_id} {changed_tables}"
            )
        receipt: dict[str, object] = {
            "plugin_id": plugin_id,
            "content_rows": row_count,
            "content_before": content_before,
            "content_after": content_after,
            "changed_tables": changed_tables,
        }
    source_after = _source_identity(plugin_root)
    core_after = _source_identity(ROOT)
    if source_after != source_before:
        raise GateError(
            f"coexistence fixture 改写 source checkout: {plugin_root} "
            f"before={source_before} after={source_after}"
        )
    if core_after != core_before:
        raise GateError(
            "coexistence fixture 改写 Core checkout: "
            f"before={core_before} after={core_after}"
        )
    return {
        **receipt,
        "source_before": source_before,
        "source_after": source_after,
        "core_before": core_before,
        "core_after": core_after,
    }


def _content_logical_state(path: Path) -> dict[str, object]:
    """Read every authoritative Content row needed to detect even empty submissions."""

    state: dict[str, object] = {}
    with closing(sqlite3.connect(path)) as connection:
        for table in ("items", "submissions", "content_state"):
            columns = tuple(
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
            rows = tuple(
                tuple(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} ORDER BY rowid"
                ).fetchall()
            )
            state[table] = {"columns": columns, "rows": rows}
    return state


def _unexpected_programmatic_call(*args: object, **kwargs: object) -> object:
    """Fail if a mount-only coexistence probe starts dispatching real work."""

    del args, kwargs
    raise GateError("mount-only coexistence probe 不得执行 programmatic Turn")


def _run_cases(
    plugin_python: Path,
    cwd: Path,
    cases: tuple[str, ...],
    source_root: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run owner tests with Core pytest and export an artifact service interpreter."""

    # 1. Pytest owns Core imports; only a distinct artifact interpreter runs services.
    pytest_python = Path(sys.executable)
    pytest_identity = _python_receipt(pytest_python)
    plugin_identity = _python_receipt(plugin_python)
    # venv launchers may resolve to Core Python; their requested path still selects the venv.
    has_artifact_runtime = plugin_python.absolute() != pytest_python.absolute()
    source_before = _source_identity(source_root)
    core_before = _source_identity(ROOT)
    selected = cases
    if cwd.name == "tests":
        selected = tuple(Path(case).name for case in cases)
    env = os.environ.copy()
    env["AKASHIC_AGENT_ROOT"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    _ = env.pop("AKASHIC_PLUGIN_FIXTURE_PYTHON", None)
    if has_artifact_runtime:
        env["AKASHIC_PLUGIN_FIXTURE_PYTHON"] = str(plugin_python)
    if extra_env is not None:
        env.update(extra_env)
    pythonpath = [str(ROOT), str(source_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    command = (
        str(pytest_python),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *selected,
    )
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    # 2. Freeze both repositories and publish the two interpreter identities.
    source_after = _source_identity(source_root)
    core_after = _source_identity(ROOT)
    if source_after != source_before:
        raise GateError(
            f"fixture 改写 source checkout: {source_root} "
            f"before={source_before} after={source_after}"
        )
    if core_after != core_before:
        raise GateError(
            f"fixture 改写 Core checkout: before={core_before} after={core_after}"
        )
    receipt: dict[str, object] = {
        "command": command,
        "cwd": str(cwd),
        "pytestInterpreter": pytest_identity,
        "pluginFixtureInterpreter": (plugin_identity if has_artifact_runtime else None),
        "source_before": source_before,
        "source_after": source_after,
        "core_before": core_before,
        "core_after": core_after,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode != 0:
        raise GateError(
            f"fixture 失败: cwd={cwd} returncode={result.returncode}\n"
            f"{result.stdout[-1200:]}\n{result.stderr[-1200:]}"
        )
    return receipt


def _run_cross_repo_suite(
    suite: CrossRepoContract,
    roots: dict[str, Path],
    pythons: dict[str, Path],
) -> dict[str, object]:
    """Run one real multi-plugin fixture while freezing every participating source."""

    missing = [case for case in suite.cases if not (ROOT / case).is_file()]
    if missing:
        raise GateError(f"cross_repo fixture 缺失: {suite.id} {missing}")
    external_before = {
        plugin_id: _source_identity(roots[plugin_id]) for plugin_id in suite.plugin_ids
    }
    receipt = _run_cases(
        pythons[suite.python_plugin_id],
        ROOT,
        suite.cases,
        ROOT,
        extra_env={
            "AKASHIC_INTEROP_PLUGIN_ROOTS": json.dumps(
                {plugin_id: str(roots[plugin_id]) for plugin_id in suite.plugin_ids},
                sort_keys=True,
            )
        },
    )
    external_after = {
        plugin_id: _source_identity(roots[plugin_id]) for plugin_id in suite.plugin_ids
    }
    changed = [
        plugin_id
        for plugin_id in suite.plugin_ids
        if external_before[plugin_id] != external_after[plugin_id]
    ]
    if changed:
        raise GateError(f"cross_repo fixture 改写 external checkout: {changed}")
    return {
        "id": f"cross_repo:{suite.id}",
        "plugin_ids": suite.plugin_ids,
        "python_plugin_id": suite.python_plugin_id,
        "external_before": external_before,
        "external_after": external_after,
        **receipt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 Content/Wake/Drift 与真实来源插件的 exact-revision 互操作"
    )
    _ = parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    _ = parser.add_argument("--plugin-root", action="append", default=[])
    _ = parser.add_argument("--plugin-python", action="append", default=[])
    _ = parser.add_argument("--identity-only", action="store_true")
    _ = parser.add_argument("--allow-pending", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Verify identities first, then replay Core and plugin-owned behavior fixtures."""

    args = _parse_args()
    report: dict[str, object] = {"status": "failed"}
    try:
        lock_path = cast(Path, args.lock).resolve()
        report_path = cast(Path, args.report)
        contract = _load_contract(lock_path)
        roots = _parse_path_map(cast(list[str], args.plugin_root), "--plugin-root")
        pythons = _parse_path_map(
            cast(list[str], args.plugin_python), "--plugin-python"
        )
        expected_ids = {plugin.id for plugin in contract.plugins}
        if set(roots) != expected_ids:
            raise GateError(
                f"plugin roots 必须精确覆盖 lock: missing={sorted(expected_ids - set(roots))} "
                f"extra={sorted(set(roots) - expected_ids)}"
            )
        _validate_execution_mode(
            identity_only=bool(args.identity_only),
            allow_pending=bool(args.allow_pending),
            expected_ids=expected_ids,
            python_ids=set(pythons),
        )

        core = _verify_core(contract)
        plugins = [
            _verify_plugin(plugin, roots[plugin.id]) for plugin in contract.plugins
        ]
        receipts: list[dict[str, object]] = []
        if not bool(args.identity_only):
            receipts.append(
                {
                    "id": "core",
                    **_run_cases(
                        Path(sys.executable),
                        ROOT,
                        contract.core_cases,
                        ROOT,
                    ),
                }
            )
            for coexistence in contract.coexistence:
                plugin_id = cast(str, coexistence["plugin_id"])
                receipts.append(
                    {
                        "id": f"coexistence:{plugin_id}",
                        **asyncio.run(
                            _run_coexistence_probe(
                                coexistence,
                                roots[plugin_id],
                            )
                        ),
                    }
                )
            for suite in contract.cross_repo:
                receipts.append(_run_cross_repo_suite(suite, roots, pythons))
            for plugin in contract.plugins:
                plugin_root = roots[plugin.id]
                receipts.append(
                    {
                        "id": plugin.id,
                        **_run_cases(
                            pythons[plugin.id],
                            plugin_root / plugin.test_cwd,
                            plugin.cases,
                            plugin_root,
                        ),
                    }
                )
        if contract.pending and not (
            bool(args.identity_only) and bool(args.allow_pending)
        ):
            pending_ids = [str(item["id"]) for item in contract.pending]
            raise GateError(f"interop 调查仍 pending: {pending_ids}")
        report = {
            "status": "passed",
            "core": core,
            "plugins": plugins,
            "receipts": receipts,
            "pending": contract.pending,
            "retired": contract.retired,
        }
    except (GateError, OSError, ValueError, subprocess.CalledProcessError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report_path = cast(Path, args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "passed":
        print(report["error"], file=sys.stderr)
        return 1
    print(f"content source interop gate passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
