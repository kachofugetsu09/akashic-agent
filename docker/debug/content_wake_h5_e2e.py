#!/usr/bin/env python3
"""Compose exact installation and existing fixture receipts into one H5 index."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docker.debug.wake_v3_provider_e2e import snapshot_protected_workspace

DEFAULT_MANIFEST = Path(__file__).with_name("content-wake-h5.manifest.json")
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FIXTURE_PACKAGES = (
    "pytest==9.0.3",
    "pytest-asyncio==1.3.0",
    "iniconfig==2.3.0",
    "packaging==26.1",
    "pluggy==1.6.0",
    "pygments==2.20.0",
)
PROTECTED_REQUIRED_FILES = frozenset(
    {
        "sessions.db",
        "proactive.db",
        "wake_proactive.db",
        "drift/drift.db",
        "PROACTIVE_CONTEXT.md",
        "proactive_pending.md",
        "proactive_quota.json",
    }
)
PROTECTED_SQLITE_TABLES = {
    "sessions.db": "messages",
    "proactive.db": "deliveries",
    "wake_proactive.db": "wake_runs",
    "drift/drift.db": "proposals",
}


class H5Error(RuntimeError):
    """Represent one evidence-composition failure with an actionable boundary."""


@dataclass(frozen=True, slots=True)
class PluginSource:
    id: str
    repository: str
    revision: str


@dataclass(frozen=True, slots=True)
class Suite:
    id: str
    cases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Manifest:
    lock: Path
    suites: tuple[Suite, ...]
    real_provider: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_manifest(path: Path) -> Manifest:
    """Validate the thin suite catalog and resolve repository-owned paths."""

    # 1. Validate only fields consumed by this composer.
    raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise H5Error("H5 manifest 必须是 JSON object")
    raw = cast(dict[str, object], raw_value)
    if set(raw) != {"schema_version", "interop_lock", "suites", "real_provider"}:
        raise H5Error("H5 manifest 字段无效")
    if raw["schema_version"] != 1:
        raise H5Error("H5 manifest schema_version 必须是 1")
    lock_value = raw["interop_lock"]
    suites_value = raw["suites"]
    provider = raw["real_provider"]
    if not isinstance(lock_value, str) or not lock_value:
        raise H5Error("interop_lock 必须是非空路径")
    if not isinstance(suites_value, list) or not suites_value:
        raise H5Error("suites 必须是非空数组")
    if not isinstance(provider, dict) or provider.get("status") != "PENDING":
        raise H5Error("real_provider 必须保持 PENDING")

    # 2. Every suite delegates to explicit existing pytest cases.
    suites: list[Suite] = []
    seen: set[str] = set()
    for item_value in cast(list[object], suites_value):
        if not isinstance(item_value, dict):
            raise H5Error("suite 必须是 object")
        item = cast(dict[str, object], item_value)
        if set(item) != {"id", "cases"}:
            raise H5Error("suite 字段无效")
        suite_id = item["id"]
        cases_value = item["cases"]
        if not isinstance(suite_id, str) or not suite_id or suite_id in seen:
            raise H5Error("suite id 必须唯一且非空")
        if (
            not isinstance(cases_value, list)
            or not cases_value
            or any(
                not isinstance(case, str) or not case
                for case in cast(list[object], cases_value)
            )
        ):
            raise H5Error(f"suite cases 无效: {suite_id}")
        cases = tuple(cast(list[str], cases_value))
        missing = [
            case for case in cases if not (ROOT / case.split("::", 1)[0]).is_file()
        ]
        if missing:
            raise H5Error(f"suite fixture 缺失: {suite_id} {missing}")
        seen.add(suite_id)
        suites.append(Suite(suite_id, cases))
    lock = Path(lock_value)
    if not lock.is_absolute():
        lock = ROOT / lock
    lock = lock.resolve()
    if not lock.is_file():
        raise H5Error(f"interop lock 不存在: {lock}")
    return Manifest(lock, tuple(suites), cast(dict[str, object], provider))


def _load_plugin_sources(lock: Path) -> tuple[PluginSource, ...]:
    """Read only source identities needed to invoke the trusted installer."""

    raw_value: object = json.loads(lock.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise H5Error("interop lock 必须是 JSON object")
    plugins_value = cast(dict[str, object], raw_value).get("plugins")
    if not isinstance(plugins_value, list) or not plugins_value:
        raise H5Error("interop lock plugins 必须是非空数组")
    sources: list[PluginSource] = []
    seen: set[str] = set()
    for value in cast(list[object], plugins_value):
        if not isinstance(value, dict):
            raise H5Error("interop plugin 必须是 object")
        item = cast(dict[str, object], value)
        plugin_id = item.get("id")
        repository = item.get("repository")
        revision = item.get("resolved_sha")
        if (
            not isinstance(plugin_id, str)
            or not plugin_id
            or plugin_id in seen
            or not isinstance(repository, str)
            or not repository
            or not isinstance(revision, str)
            or SHA_PATTERN.fullmatch(revision) is None
        ):
            raise H5Error("interop plugin source identity 无效")
        seen.add(plugin_id)
        sources.append(PluginSource(plugin_id, repository, revision))
    return tuple(sources)


def _run(
    command: tuple[str, ...], *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _command_report(
    report_path: Path,
    *,
    command: tuple[str, ...],
    result: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    report = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": list(command),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    _write_json(report_path, report)
    if result.returncode != 0:
        raise H5Error(f"command failed: {report_path.name}")
    return report


def _install(
    *,
    sources: tuple[PluginSource, ...],
    run_root: Path,
    env: dict[str, str],
) -> tuple[dict[str, object], dict[str, Path]]:
    """Install the exact batch and bind source ids only from its receipt paths."""

    # 1. The generated batch contains the same immutable revisions as the lock.
    batch_path = run_root / "reports" / "trusted-batch.json"
    _write_json(
        batch_path,
        {
            "schema_version": 1,
            "plugins": [
                {
                    "source": source.repository,
                    "marketplace": "h5-e2e",
                    "ref": source.revision,
                }
                for source in sources
            ],
        },
    )
    command = (
        sys.executable,
        str(ROOT / "main.py"),
        "plugin-install-trusted-batch",
        "--workspace",
        str(run_root / "workspace"),
        "--plugins-home",
        str(run_root / "plugin-home"),
        "--batch",
        str(batch_path),
        "--confirm-trusted",
        "--json",
    )
    result = _run(command, env=env)
    receipt_path = run_root / "reports" / "trusted-install.json"
    if result.returncode != 0:
        _command_report(receipt_path, command=command, result=result)
    receipt_value: object = json.loads(result.stdout)
    if not isinstance(receipt_value, dict):
        raise H5Error("trusted install receipt 必须是 object")
    receipt = cast(dict[str, object], receipt_value)
    plugins_value = receipt.get("plugins")
    if (
        receipt.get("mode") != "operator_trusted_offline_batch"
        or receipt.get("programmaticValidation") != "bypassed_by_operator_trust"
        or not isinstance(plugins_value, list)
        or len(plugins_value) != len(sources)
    ):
        raise H5Error("trusted install receipt 合同不匹配")
    roots: dict[str, Path] = {}
    for source, item_value in zip(
        sources, cast(list[object], plugins_value), strict=True
    ):
        if not isinstance(item_value, dict):
            raise H5Error("trusted install plugin receipt 无效")
        item = cast(dict[str, object], item_value)
        installed = item.get("installedPath")
        if item.get("sourceRevision") != source.revision or not isinstance(
            installed, str
        ):
            raise H5Error(f"installed revision 不匹配: {source.id}")
        root = Path(installed).resolve()
        if not root.is_dir() or not root.is_relative_to(run_root / "plugin-home"):
            raise H5Error(f"installedPath 不在隔离 plugin home: {source.id}")
        roots[source.id] = root
    _write_json(receipt_path, receipt)
    return receipt, roots


def _fixture_python(root: Path) -> Path:
    candidates = sorted(root.glob("**/.venv/bin/python"))
    if not candidates:
        return Path(sys.executable)
    return candidates[0]


def _fixture_support(run_root: Path, env: dict[str, str]) -> tuple[Path, Path, Path]:
    """Install the fixed test-only layer and expose only that layer to artifacts."""

    # 1. Install the complete fixed pytest dependency set without Core packages.
    layer = run_root / "home" / "fixture-layer"
    bootstrap = run_root / "home" / "fixture-bootstrap"
    layer.mkdir()
    bootstrap.mkdir()
    command = (
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        "--target",
        str(layer),
        *FIXTURE_PACKAGES,
    )
    install_env = {**env}
    install_env.pop("PYTHONPATH", None)
    result = _run(command, env=install_env)
    report = run_root / "reports" / "fixture-support.json"
    _command_report(report, command=command, result=result)

    # 2. Artifact interpreters append only the dedicated fixture layer.
    bootstrap.joinpath("sitecustomize.py").write_text(
        "import sys\n" f"sys.path.append({str(layer)!r})\n",
        encoding="utf-8",
    )
    return bootstrap, layer, report


def _verify_fixture_runtimes(
    *, roots: dict[str, Path], run_root: Path, env: dict[str, str], layer: Path
) -> Path:
    """Prove artifact dependencies win while only dedicated pytest is shared."""

    # 1. Establish one package that exists only in the Core development runtime.
    core_sentinel = importlib.util.find_spec("black")
    if core_sentinel is None or core_sentinel.origin is None:
        raise H5Error("Core-only black sentinel 不存在")

    # 2. Probe every artifact-owned interpreter without importing plugin code.
    receipts: list[dict[str, object]] = []
    code = (
        "import importlib.util,json,pytest\n"
        "def origin(name):\n"
        " spec=importlib.util.find_spec(name)\n"
        " return None if spec is None else spec.origin\n"
        "print(json.dumps({'pytest':pytest.__file__,'black':origin('black'),"
        "'mcp':origin('mcp'),'requests':origin('requests')}))\n"
    )
    for plugin_id, root in roots.items():
        python = _fixture_python(root)
        if python == Path(sys.executable):
            continue
        result = _run((str(python), "-c", code), env=env)
        if result.returncode != 0:
            raise H5Error(f"fixture runtime probe failed: {plugin_id}")
        value: object = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise H5Error(f"fixture runtime receipt 无效: {plugin_id}")
        receipt = cast(dict[str, object], value)
        pytest_path = Path(cast(str, receipt["pytest"])).resolve()
        dependency_value = receipt.get("mcp") or receipt.get("requests")
        if not isinstance(dependency_value, str):
            raise H5Error(f"artifact runtime dependency 缺失: {plugin_id}")
        dependency_path = Path(dependency_value).resolve()
        if (
            not pytest_path.is_relative_to(layer)
            or receipt.get("black") is not None
            or not dependency_path.is_relative_to(root)
        ):
            raise H5Error(f"fixture runtime path 污染: {plugin_id}")
        receipts.append(
            {
                "plugin_id": plugin_id,
                "python": str(python),
                "pytest_path": str(pytest_path),
                "core_only_black": "unavailable",
                "artifact_dependency_path": str(dependency_path),
            }
        )
    if not receipts:
        raise H5Error("没有 artifact-owned interpreter 可验证")

    # 3. Publish path identities so the index can attest the isolation boundary.
    report = run_root / "reports" / "fixture-runtime-bindings.json"
    _write_json(
        report,
        {
            "status": "passed",
            "core_sentinel": {"module": "black", "origin": core_sentinel.origin},
            "runtimes": receipts,
        },
    )
    return report


def _run_interop(
    *,
    manifest: Manifest,
    roots: dict[str, Path],
    run_root: Path,
    env: dict[str, str],
) -> Path:
    report = run_root / "reports" / "content-source-interop.json"
    bindings: list[str] = []
    for plugin_id, root in roots.items():
        bindings.extend(("--plugin-root", f"{plugin_id}={root}"))
        bindings.extend(("--plugin-python", f"{plugin_id}={_fixture_python(root)}"))
    command = (
        sys.executable,
        str(ROOT / "docker/debug/content_source_interop_gate.py"),
        "--lock",
        str(manifest.lock),
        "--report",
        str(report),
        *bindings,
    )
    result = _run(command, env=env)
    if result.returncode != 0:
        raise H5Error(f"Content source interop failed: {result.stderr[-1000:]}")
    return report


def _run_suites(
    *, manifest: Manifest, run_root: Path, env: dict[str, str]
) -> tuple[Path, ...]:
    reports: list[Path] = []
    for suite in manifest.suites:
        report = run_root / "reports" / f"fixture-{suite.id}.json"
        command = (sys.executable, "-m", "pytest", "-q", *suite.cases)
        result = _run(command, env=env)
        _command_report(report, command=command, result=result)
        reports.append(report)
    return tuple(reports)


def _report_entry(run_root: Path, report: Path) -> dict[str, object]:
    payload: object = json.loads(report.read_text(encoding="utf-8"))
    status = payload.get("status") if isinstance(payload, dict) else None
    if (
        status is None
        and isinstance(payload, dict)
        and payload.get("mode") == "operator_trusted_offline_batch"
        and payload.get("programmaticValidation") == "bypassed_by_operator_trust"
    ):
        status = "passed"
    return {
        "path": str(report.relative_to(run_root)),
        "sha256": _sha256(report),
        "status": status,
    }


def _seed_sqlite(path: Path, table: str, value: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, value TEXT NOT NULL)'
        )
        connection.execute(f'INSERT INTO "{table}" (value) VALUES (?)', (value,))
        connection.commit()
    finally:
        connection.close()


def _seed_protected_fixture(path: Path) -> None:
    """Create a nonempty historical workspace fixture with durable rows."""

    # 1. The explicit fixture target must exist and start empty.
    if not path.is_dir():
        raise H5Error("protected fixture 目录不存在")
    if any(path.iterdir()):
        raise H5Error("protected fixture seed 只接受空目录")

    # 2. Freeze Session and old-island file shapes without formal data.
    _seed_sqlite(path / "sessions.db", "messages", "fixture-session-message")
    _seed_sqlite(path / "proactive.db", "deliveries", "fixture-delivery")
    _seed_sqlite(path / "wake_proactive.db", "wake_runs", "fixture-wake-run")
    drift = path / "drift"
    drift.mkdir()
    _seed_sqlite(drift / "drift.db", "proposals", "fixture-drift-proposal")
    (path / "PROACTIVE_CONTEXT.md").write_text(
        "# Historical proactive context\n\nFixture archive bytes.\n",
        encoding="utf-8",
    )
    (path / "proactive_pending.md").write_text(
        "# Historical pending document\n\nFixture pending bytes.\n",
        encoding="utf-8",
    )
    (path / "proactive_quota.json").write_text(
        '{"used":1,"version":1,"window":"fixture"}\n', encoding="utf-8"
    )


def _validate_protected_snapshot(snapshot: dict[str, object]) -> None:
    """Require nonempty protected files and readable SQLite rows."""

    # 1. Require both Session state and historical island state.
    files = cast(dict[str, object], snapshot["files"])
    missing = PROTECTED_REQUIRED_FILES - set(files)
    if missing:
        raise H5Error(f"protected workspace 缺少非空fixture: {sorted(missing)}")
    for relative in PROTECTED_REQUIRED_FILES:
        item = cast(dict[str, object], files[relative])
        inode = item.get("inode")
        size = item.get("size")
        digest = item.get("sha256")
        if (
            not isinstance(inode, int)
            or inode <= 0
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise H5Error(f"protected workspace 文件为空: {relative}")

    # 2. Every authoritative database must retain its required table and rows.
    sqlite_state = cast(dict[str, object], snapshot["sqlite"])
    missing_sqlite = set(PROTECTED_SQLITE_TABLES) - set(sqlite_state)
    if missing_sqlite:
        raise H5Error(f"protected workspace 缺少SQLite: {sorted(missing_sqlite)}")
    for relative, required_table in PROTECTED_SQLITE_TABLES.items():
        state = cast(dict[str, object], sqlite_state[relative])
        rows_value = state.get("rows")
        rows = (
            cast(dict[str, object], rows_value) if isinstance(rows_value, dict) else {}
        )
        row_count = rows.get(required_table)
        if (
            state.get("integrity") != "ok"
            or state.get("quick_check") != "ok"
            or not isinstance(row_count, int)
            or row_count <= 0
        ):
            raise H5Error(f"protected workspace SQLite fixture 无效: {relative}")


def run(
    *,
    run_root: Path,
    protected_workspace: Path,
    manifest_path: Path,
    seed_protected_fixture: bool = False,
) -> Path:
    """Run existing owners in order and publish their immutable evidence index."""

    # 1. Establish one explicit isolated root and immutable source identities.
    if not run_root.is_absolute() or not protected_workspace.is_absolute():
        raise H5Error("run root 与 protected workspace 必须是绝对路径")
    run_root.mkdir(parents=True, exist_ok=False)
    for name in ("workspace", "plugin-home", "reports", "home"):
        (run_root / name).mkdir()
    manifest = _load_manifest(manifest_path.resolve())
    sources = _load_plugin_sources(manifest.lock)
    core_before = {
        "head": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "dirty": _git("status", "--porcelain").splitlines(),
    }
    if seed_protected_fixture:
        _seed_protected_fixture(protected_workspace)
    protected_before = snapshot_protected_workspace(protected_workspace)
    _validate_protected_snapshot(protected_before)
    env = {
        **os.environ,
        "HOME": str(run_root / "home"),
        "AKASHIC_WORKSPACE": str(run_root / "workspace"),
        "AKASHIC_PLUGIN_HOME": str(run_root / "plugin-home"),
        "PYTHONPATH": str(ROOT),
    }
    env.pop("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", None)

    # 2. Delegate installation, interop, and every behavior fixture to its owner.
    receipt, roots = _install(sources=sources, run_root=run_root, env=env)
    bootstrap, layer, support_report = _fixture_support(run_root, env)
    fixture_env = {**env, "PYTHONPATH": str(bootstrap)}
    runtime_report = _verify_fixture_runtimes(
        roots=roots,
        run_root=run_root,
        env=fixture_env,
        layer=layer,
    )
    interop_report = _run_interop(
        manifest=manifest, roots=roots, run_root=run_root, env=env
    )
    fixture_reports = _run_suites(manifest=manifest, run_root=run_root, env=env)

    # 3. Refuse source/protected-state drift, then combine report identities only.
    core_after = {
        "head": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "dirty": _git("status", "--porcelain").splitlines(),
    }
    protected_after = snapshot_protected_workspace(protected_workspace)
    _validate_protected_snapshot(protected_after)
    if core_before != core_after:
        raise H5Error("Core source changed during H5 run")
    if protected_before != protected_after:
        raise H5Error("protected workspace changed during H5 run")
    reports = (
        run_root / "reports" / "trusted-install.json",
        support_report,
        runtime_report,
        interop_report,
        *fixture_reports,
    )
    index = {
        "schema_version": 1,
        "status": "deterministic_passed",
        "core": core_after,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path.resolve()),
        },
        "interop_lock": {
            "path": str(manifest.lock),
            "sha256": _sha256(manifest.lock),
        },
        "trusted_batch": {
            "mode": receipt["mode"],
            "programmaticValidation": receipt["programmaticValidation"],
            "installed": [
                {
                    "id": source.id,
                    "revision": source.revision,
                    "installedPath": str(roots[source.id]),
                }
                for source in sources
            ],
        },
        "reports": [_report_entry(run_root, report) for report in reports],
        "protected_workspace": {
            "path": str(protected_workspace),
            "before": protected_before,
            "after": protected_after,
            "status": "unchanged",
        },
        "real_provider": manifest.real_provider,
    }
    index_path = run_root / "reports" / "h5-index.json"
    _write_json(index_path, index)
    return index_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose deterministic H5 evidence")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protected-workspace", type=Path, required=True)
    parser.add_argument("--seed-protected-fixture", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        index = run(
            run_root=cast(Path, args.run_root),
            protected_workspace=cast(Path, args.protected_workspace),
            manifest_path=cast(Path, args.manifest),
            seed_protected_fixture=bool(args.seed_protected_fixture),
        )
    except (H5Error, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"H5 evidence passed: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
