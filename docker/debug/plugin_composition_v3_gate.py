from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.plugins.manager import PluginManager  # noqa: E402
from agent.plugins.snapshot import (  # noqa: E402
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tools.events import (  # noqa: E402
    ToolExecutionRequest,
    ToolExecutionResult,
)
from agent.tools.executor import ToolExecutor  # noqa: E402
from bus.event_bus import EventBus  # noqa: E402


DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-composition-v3.lock.json"
DEFAULT_REPORT = (
    ROOT / "docker" / "debug" / "reports" / "plugin-composition-v3" / "gate.json"
)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
GATE_VERSION = 1
PROTOCOL_SOURCE_REPOSITORY = "https://github.com/kachofugetsu09/akashic-agent.git"
PROTOCOL_SOURCE_COMMIT = "0940e9e74a62efef54470f11a7064a99ca5e9acc"
PROTOCOL_SOURCE_PATHS = (
    "agent/tools/events.py",
    "agent/tools/executor.py",
)
EXPECTED_PLUGIN_IDS = ("shell_restore", "shell_safety")
SCENARIO_PROFILE = "plugin-tool-v3-v1"
EXPECTED_LISTENERS = (
    "transform:tool.input.prepare[akashic.tool-input.v1]:shell_restore",
    "serial:tool.execution.authorize[bail=akashic.tool-deny-reason.v1]:shell_safety",
)
ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class PluginLock:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str


@dataclass(frozen=True)
class PluginEvidence:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str
    tree: str


@dataclass(frozen=True)
class ScenarioEvidence:
    id: str
    status: str
    final_command: str
    invoked: bool


@dataclass(frozen=True)
class ScenarioCase:
    id: str
    session: str
    command: str
    expected_status: str
    expected_invoked: bool


@dataclass(frozen=True)
class CleanupEvidence:
    listeners: tuple[str, ...]
    effects: tuple[str, ...]


SCENARIO_CATALOG = (
    ScenarioCase("plain-rm", "plain", "rm /tmp/plain.txt", "success", True),
    ScenarioCase(
        "sudo-cluster",
        "cluster",
        "sudo -nE rm /tmp/cluster.txt",
        "success",
        True,
    ),
    ScenarioCase(
        "sudo-preserve-env",
        "env",
        "sudo -n --preserve-env=HOME rm /tmp/env.txt",
        "success",
        True,
    ),
    ScenarioCase(
        "sudo-mode-denied",
        "mode",
        "sudo -n -s rm /tmp/mode.txt",
        "denied",
        False,
    ),
    ScenarioCase("repeat-1", "repeat", "rm /tmp/repeat.txt", "success", True),
    ScenarioCase("repeat-2", "repeat", "rm /tmp/repeat.txt", "success", True),
    ScenarioCase("repeat-3", "repeat", "rm /tmp/repeat.txt", "success", True),
)


def main() -> None:
    """Checkout exact plugins and verify their composed stable runtime behavior."""

    args = _parse_args()
    core_status = _git_output(ROOT, "status", "--porcelain").splitlines()
    if args.require_clean_core and core_status:
        raise RuntimeError(f"核心工作树不干净: {core_status}")
    locks = _load_lock(args.lock.resolve())

    with tempfile.TemporaryDirectory(prefix="akashic-plugin-composition-v3-") as raw:
        sandbox = Path(raw)
        providers = sandbox / "providers"
        providers.mkdir()
        plugin_evidence = tuple(
            _checkout_locked_plugin(lock, providers / lock.id) for lock in locks
        )
        listeners, scenarios, invocations, cleanup = asyncio.run(
            _verify_composition(providers, sandbox)
        )

    report = {
        "status": "passed",
        "gate_version": GATE_VERSION,
        "checked_at": datetime.now(UTC).isoformat(),
        "core": {
            "head": _git_output(ROOT, "rev-parse", "HEAD"),
            "tree": _git_output(ROOT, "rev-parse", "HEAD^{tree}"),
            "dirty_status": core_status,
        },
        "lock": str(args.lock.resolve().relative_to(ROOT)),
        "lock_sha256": _sha256(args.lock.resolve()),
        "protocol_source": _protocol_source_evidence(),
        "plugins": [asdict(item) for item in plugin_evidence],
        "topology_listeners": list(listeners),
        "scenario_profile": SCENARIO_PROFILE,
        "scenario_catalog_sha256": _scenario_catalog_sha256(),
        "scenario_catalog": [asdict(item) for item in SCENARIO_CATALOG],
        "scenarios": [asdict(item) for item in scenarios],
        "invocations": invocations,
        "cleanup": asdict(cleanup),
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"plugin composition v3 gate passed: {report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证固定 v3 插件的组合执行合同")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-clean-core", action="store_true")
    return parser.parse_args()


def _load_lock(path: Path) -> tuple[PluginLock, ...]:
    """Strictly load the immutable cross-repository plugin set."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "plugins"}:
        raise ValueError("v3 插件组合锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的 v3 插件组合锁版本: {raw['schema_version']}")
    raw_plugins = raw["plugins"]
    if not isinstance(raw_plugins, list):
        raise ValueError("v3 插件组合锁 plugins 必须是列表")
    plugins = tuple(_parse_plugin_lock(item) for item in raw_plugins)
    if tuple(item.id for item in plugins) != EXPECTED_PLUGIN_IDS:
        raise ValueError("v3 插件组合锁的插件集合或顺序错误")
    return plugins


def _parse_plugin_lock(raw: object) -> PluginLock:
    expected = {
        "id",
        "repository",
        "requested_ref",
        "resolved_sha",
        "change_source_pr_head",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"v3 插件组合锁字段无效: {raw}")
    item = cast(dict[str, object], raw)
    values = {name: _required_string(item, name) for name in expected}
    repository = values["repository"]
    if not repository.startswith("https://github.com/") or not repository.endswith(".git"):
        raise ValueError(f"插件仓库必须是公开 GitHub HTTPS Git 地址: {repository}")
    for field in ("requested_ref", "resolved_sha", "change_source_pr_head"):
        if COMMIT_PATTERN.fullmatch(values[field]) is None:
            raise ValueError(f"{field} 必须是完整 SHA: {values[field]}")
    if len({values[field] for field in ("requested_ref", "resolved_sha", "change_source_pr_head")}) != 1:
        raise ValueError(f"试点插件必须把三个 revision 固定到同一提交: {values['id']}")
    return PluginLock(
        id=values["id"],
        repository=repository,
        requested_ref=values["requested_ref"],
        resolved_sha=values["resolved_sha"],
        change_source_pr_head=values["change_source_pr_head"],
    )


def _required_string(item: dict[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"v3 插件组合锁字段必须是非空字符串: {name}")
    return value


def _checkout_locked_plugin(lock: PluginLock, checkout: Path) -> PluginEvidence:
    """Fetch only one declared public Git object into a fresh repository."""

    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", lock.repository), cwd=checkout)
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", lock.resolved_sha),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    if _git_output(checkout, "rev-parse", "HEAD") != lock.resolved_sha:
        raise RuntimeError(f"插件检出提交与锁不一致: {lock.id}")
    if _git_output(checkout, "status", "--porcelain"):
        raise RuntimeError(f"插件检出后工作树不干净: {lock.id}")
    return PluginEvidence(
        id=lock.id,
        repository=lock.repository,
        requested_ref=lock.requested_ref,
        resolved_sha=lock.resolved_sha,
        change_source_pr_head=lock.change_source_pr_head,
        tree=_git_output(checkout, "rev-parse", "HEAD^{tree}"),
    )


async def _verify_composition(
    providers: Path,
    sandbox: Path,
) -> tuple[
    tuple[str, ...],
    tuple[ScenarioEvidence, ...],
    list[dict[str, object]],
    CleanupEvidence,
]:
    """Load one stable Root and execute the migration interaction matrix."""

    workspace = sandbox / "workspace"
    manager = PluginManager(
        plugin_dirs=[providers],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=workspace,
        installed_cache_root=sandbox / "plugin-home" / "cache",
    )
    root = None
    result: tuple[
        tuple[str, ...],
        tuple[ScenarioEvidence, ...],
        list[dict[str, object]],
    ] | None = None
    try:
        await manager.load_all()
        snapshot = manager.current_snapshot
        if snapshot is None or snapshot.composition_root is None:
            raise RuntimeError("正式 snapshot 缺少 v3 CompositionRoot")
        root = snapshot.composition_root
        topology = root.topology_view()
        if topology.listeners != EXPECTED_LISTENERS:
            raise RuntimeError(f"v3 listener 顺序不符合组合合同: {topology.listeners}")

        restore = manager.generation("shell_restore")
        if restore is None:
            raise RuntimeError("正式 snapshot 缺少 shell_restore generation")
        restore_dir = restore.data_dir / "restore"
        executor = ToolExecutor()
        invocations: list[dict[str, object]] = []

        async def invoke(tool_name: str, arguments: dict[str, Any]) -> str:
            invocations.append({"tool_name": tool_name, "arguments": dict(arguments)})
            return "invoked"

        lease = manager.snapshot_store.lease()
        token = bind_runtime_snapshot(lease)
        try:
            scenarios = await _run_scenarios(
                executor,
                invoke,
                restore_dir,
                invocations,
            )
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

        expected_invocations = sum(case.expected_invoked for case in SCENARIO_CATALOG)
        if len(invocations) != expected_invocations:
            raise RuntimeError(
                "真实 invoker 调用次数错误: "
                f"expected={expected_invocations} actual={len(invocations)}"
            )
        result = topology.listeners, scenarios, invocations
    finally:
        await manager.terminate_all()
    if root is None or result is None:
        raise AssertionError("组合 Gate 成功路径没有保留正式 Root 结果")
    cleanup = CleanupEvidence(
        listeners=root.topology_view().listeners,
        effects=root.receipt().effects,
    )
    if cleanup.listeners or cleanup.effects:
        raise RuntimeError(f"正式 Root 终止后仍有组合资源: {cleanup}")
    return *result, cleanup


async def _run_scenarios(
    executor: ToolExecutor,
    invoker: ToolInvoker,
    restore_dir: Path,
    invocations: list[dict[str, object]],
) -> tuple[ScenarioEvidence, ...]:
    evidence: list[ScenarioEvidence] = []
    for index, case in enumerate(SCENARIO_CATALOG):
        before = len(invocations)
        result = await executor.execute(
            ToolExecutionRequest(
                call_id=f"gate-{index}",
                tool_name="shell",
                arguments={"command": case.command},
                source="passive",
                session_key=case.session,
            ),
            invoker,
        )
        invoked = len(invocations) == before + 1
        if invoked is not case.expected_invoked:
            raise RuntimeError(
                f"场景 {case.id} invoker 状态错误: "
                f"expected={case.expected_invoked} actual={invoked}"
            )
        final_command = str(result.final_arguments.get("command", ""))
        _assert_scenario(
            case.id,
            result,
            final_command,
            restore_dir,
            case.expected_status,
        )
        evidence.append(
            ScenarioEvidence(
                id=case.id,
                status=result.status,
                final_command=final_command,
                invoked=invoked,
            )
        )
    return tuple(evidence)


def _assert_scenario(
    case_id: str,
    result: ToolExecutionResult,
    final_command: str,
    restore_dir: Path,
    expected_status: str,
) -> None:
    if result.status != expected_status:
        raise RuntimeError(f"场景 {case_id} 状态错误: {result.status} {result.output}")
    if expected_status == "success":
        if " mv " not in f" {final_command} " or str(restore_dir) not in final_command:
            raise RuntimeError(f"场景 {case_id} 未把 rm 改写到插件数据根: {final_command}")
    elif case_id == "sudo-mode-denied":
        if "普通命令执行" not in str(result.output):
            raise RuntimeError(f"场景 {case_id} 未由 Safety 拒绝 sudo mode: {result.output}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_catalog_sha256() -> str:
    encoded = json.dumps(
        [asdict(item) for item in SCENARIO_CATALOG],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protocol_source_evidence() -> dict[str, object]:
    files: list[dict[str, str]] = []
    for path in PROTOCOL_SOURCE_PATHS:
        blob = _git_output(ROOT, "rev-parse", f"{PROTOCOL_SOURCE_COMMIT}:{path}")
        content = subprocess.run(
            ("git", "cat-file", "blob", blob),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        files.append(
            {
                "path": path,
                "git_blob": blob,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "repository": PROTOCOL_SOURCE_REPOSITORY,
        "commit": PROTOCOL_SOURCE_COMMIT,
        "files": files,
    }


def _git_output(cwd: Path, *args: str) -> str:
    return _run(("git", *args), cwd=cwd).stdout.strip()


def _run(command: tuple[str, ...], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


if __name__ == "__main__":
    main()
