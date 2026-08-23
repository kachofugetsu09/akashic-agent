#!/usr/bin/env python3
"""Verify exact proactive-source revisions and replay their owned fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import tomllib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from plugins.content.store import ContentStore


DEFAULT_LOCK = Path(__file__).with_name("proactive-source-interop.lock.json")
DEFAULT_REPORT = (
    Path(__file__).parent / "reports" / "proactive-source-interop" / "gate.json"
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


@dataclass(frozen=True, slots=True)
class PluginContract:
    id: str
    repository: str
    resolved_sha: str
    pull_request: str | None
    role: str
    atoms: tuple[str, ...]
    test_cwd: str
    cases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InteropContract:
    core_contract: str
    core_cases: tuple[str, ...]
    coexistence: tuple[dict[str, object], ...]
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
    if not isinstance(core_contract, str) or SHA_PATTERN.fullmatch(core_contract) is None:
        raise GateError("core_contract 必须是完整 SHA")
    core_cases = _string_tuple(raw["core_cases"], "core_cases")
    coexistence = _mapping_tuple(
        raw["coexistence"],
        {"plugin_id", "config_toml", "expected_content_rows"},
        "coexistence",
    )
    plugins_value = raw["plugins"]
    if not isinstance(plugins_value, list):
        raise GateError("plugins 必须是数组")
    plugins_raw = cast(list[object], plugins_value)
    plugins = tuple(_parse_plugin(item) for item in plugins_raw)
    ids = tuple(item.id for item in plugins)
    if len(ids) != len(set(ids)):
        raise GateError("plugins 不得重复")
    pending = _mapping_tuple(raw["pending"], {"id", "reason"}, "pending")
    retired = _mapping_tuple(
        raw["retired"],
        {"id", "canonical_sha", "disposition", "evidence"},
        "retired",
    )
    plugin_ids = {plugin.id for plugin in plugins}
    unknown_coexistence = [
        item["plugin_id"]
        for item in coexistence
        if item["plugin_id"] not in plugin_ids
    ]
    if unknown_coexistence:
        raise GateError(f"coexistence 引用未知插件: {unknown_coexistence}")
    return InteropContract(
        core_contract,
        core_cases,
        coexistence,
        plugins,
        pending,
        retired,
    )


def _parse_plugin(raw: object) -> PluginContract:
    """Parse one exact external plugin identity and its owned fixture list."""

    fields = {
        "id",
        "repository",
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
    for field in ("id", "repository", "resolved_sha", "role", "test_cwd"):
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
    return {"head": head, "contract": contract.core_contract, "cases": contract.core_cases}


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

    with tempfile.TemporaryDirectory(prefix="akashic-proactive-interop-") as raw:
        root = Path(raw)
        plugins = root / "plugins"
        content_dir = plugins / "content"
        staged_plugin = plugins / plugin_id
        _ = shutil.copytree(ROOT / "plugins" / "content", content_dir)
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
        manager = PluginManager(
            plugin_dirs=[content_dir, staged_plugin],
            event_bus=EventBus(),
            workspace=workspace,
            installed_cache_root=root / "cache",
        )
        row_count = -1
        try:
            await manager.load_all()
            content_path = (
                workspace
                / "plugin-data"
                / "content-builtin"
                / "content.sqlite3"
            )
            store = ContentStore(content_path)
            row_count = sum(store.state_counts().values())
            if row_count != expected_rows:
                raise GateError(
                    f"coexistence Content rows 漂移: {plugin_id} "
                    f"expected={expected_rows} actual={row_count}"
                )
        finally:
            await manager.terminate_all()
        return {
            "plugin_id": plugin_id,
            "content_rows": row_count,
            "formal_content_write_set": [],
        }


def _run_cases(
    python: Path,
    cwd: Path,
    cases: tuple[str, ...],
    plugin_root: Path | None = None,
) -> dict[str, object]:
    """Run one owner's unmodified fixture selection and capture its receipt."""

    if not python.is_file():
        raise GateError(f"fixture Python 不存在: {python}")
    selected = cases
    if cwd.name == "tests":
        selected = tuple(Path(case).name for case in cases)
    env = os.environ.copy()
    env["AKASHIC_AGENT_ROOT"] = str(ROOT)
    pythonpath = [str(ROOT)]
    if plugin_root is not None:
        pythonpath.append(str(plugin_root))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    command = (str(python), "-m", "pytest", "-q", *selected)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    receipt: dict[str, object] = {
        "command": command,
        "cwd": str(cwd),
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
        roots = _parse_path_map(
            cast(list[str], args.plugin_root), "--plugin-root"
        )
        pythons = _parse_path_map(
            cast(list[str], args.plugin_python), "--plugin-python"
        )
        expected_ids = {plugin.id for plugin in contract.plugins}
        if set(roots) != expected_ids:
            raise GateError(
                f"plugin roots 必须精确覆盖 lock: missing={sorted(expected_ids - set(roots))} "
                f"extra={sorted(set(roots) - expected_ids)}"
            )
        if set(pythons) - expected_ids:
            raise GateError(f"未知 plugin Python: {sorted(set(pythons) - expected_ids)}")

        core = _verify_core(contract)
        plugins = [
            _verify_plugin(plugin, roots[plugin.id]) for plugin in contract.plugins
        ]
        receipts: list[dict[str, object]] = []
        if not bool(args.identity_only):
            receipts.append(
                {
                    "id": "core",
                    **_run_cases(Path(sys.executable), ROOT, contract.core_cases),
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
            for plugin in contract.plugins:
                plugin_root = roots[plugin.id]
                receipts.append(
                    {
                        "id": plugin.id,
                        **_run_cases(
                            pythons.get(plugin.id, Path(sys.executable)),
                            plugin_root / plugin.test_cwd,
                            plugin.cases,
                            plugin_root,
                        ),
                    }
                )
        if contract.pending and not bool(args.allow_pending):
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
    print(f"proactive source interop gate passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
