from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-v3-fleet.lock.json"
DEFAULT_REPORT = ROOT / "docker" / "debug" / "reports" / "plugin-v3-fleet" / "gate.json"
GATE_VERSION = 1
LOCK_SCHEMA_VERSION = 1
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
)
STATIC_MANIFEST_FILENAME = "akashic.plugin.toml"
MANIFEST_ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "api_version",
        "entrypoint",
        "python",
        "validation",
        "mcp",
        "processes",
        "channel_credentials",
    }
)
EXPECTED_PLUGIN_IDS: tuple[str, ...] = (
    "citation",
    "meme",
    "shell_restore",
    "shell_safety",
    "calendar-mcp",
    "emotion",
    "plugin_undo",
    "observe",
    "setup_helper",
    "status_commands",
    "feed-mcp",
    "feishu",
    "fitbit-mcp",
    "steam-mcp",
    "qqbot",
    "proactive_feedback",
    "huayue-skills",
    "github_watch",
)
EXCLUDED_PLUGIN_IDS = ("computer-use-linux", "context_pressure")
FORBIDDEN_PLUGIN_IDS = EXCLUDED_PLUGIN_IDS
FORBIDDEN_V2_MODULES = frozenset(
    {
        "agent.plugins",
        "agent.plugins.base",
        "agent.plugins.context",
        "agent.plugins.decorators",
        "agent.plugins.manager",
        "agent.plugins.registry",
    }
)
FORBIDDEN_V2_CLASS_BASES = frozenset({"Plugin", f"Plugin{'Context'}"})
FORBIDDEN_V2_FIXED_METHODS = frozenset(
    {
        "static_semantic_checks",
        "readiness_semantic_checks",
        "skill_roots",
        "drift_skill_roots",
        "mcp_servers",
        "managed_services",
        "proactive_sources",
        "before_turn_modules",
        "before_reasoning_modules",
        "prompt_render_modules",
        "before_step_modules",
        "after_step_modules",
        "after_reasoning_modules",
        "after_turn_modules",
        "proactive_modules",
        "proactive_lifecycles",
        "proactive_module_factories",
        "proactive_runtime_factories",
        "telegram_bot_commands",
        "mobile_bot_commands",
        "mobile_ui_available",
        "mobile_ui_query",
    }
)
E2E_NOT_RUN_REASON = (
    "static Gate 第一阶段不执行 runtime E2E；需最终 Core/plugin 组合与受控环境"
)
GIT_COMMAND_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class PluginLock:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str


@dataclass(frozen=True, slots=True)
class E2ECase:
    id: str
    title: str
    required_plugins: tuple[str, ...]
    oracle: tuple[str, ...]


E2E_CATALOG = (
    E2ECase(
        "E1",
        "Passive/Data/Mobile",
        (
            "akasha",
            "citation",
            "meme",
            "emotion",
            "observe",
            "proactive_feedback",
            "plugin_undo",
        ),
        (
            "prompt/recall/metadata/media",
            "bounded mobile query and lease",
            "append-only SessionDB write set",
        ),
    ),
    E2ECase(
        "E2",
        "Tool/MCP/Process",
        (
            "shell_restore",
            "shell_safety",
            "calendar-mcp",
            "feed-mcp",
            "fitbit-mcp",
            "steam-mcp",
        ),
        (
            "transform/authorize/invoke",
            "MCP and process readiness",
            "cancel and process cleanup",
            "controlled external read-only calls",
        ),
    ),
    E2ECase(
        "E3",
        "Fleet/Channel/Proactive",
        (
            "setup_helper",
            "status_commands",
            "feishu",
            "qqbot",
            "emotion",
            "calendar-mcp",
            "feed-mcp",
            "fitbit-mcp",
            "steam-mcp",
            "huayue-skills",
            "github_watch",
        ),
        (
            "full boot and catalog",
            "candidate discard and promotion",
            "loopback channel recording",
            "fixed-clock background-job restart",
            "controlled repository probe",
        ),
    ),
    E2ECase(
        "E4",
        "Production Rehearsal",
        ("E1", "E2", "E3"),
        (
            "copied-workspace database integrity",
            "complete write set",
            "artifact/pointer and restart",
            "stop cleanup and restore evidence",
        ),
    ),
)


class GateError(RuntimeError):
    """A reproducible static Gate input or evidence failure."""


def main() -> int:
    """Lock, inspect, and report the pure-v3 fleet without running E1-E4."""

    # 1. Freeze Core identity and validate the immutable fleet input.
    args = _parse_args()
    report_path = args.report.resolve()
    core = _core_evidence()
    errors: list[str] = []
    plugins: list[dict[str, object]] = []
    try:
        locks = _load_lock(args.lock.resolve())
        if args.require_clean_core and not cast(bool, core["clean"]):
            raise GateError(f"Core 工作树不干净: {core['dirty_status']}")
        if args.require_full_core_history and core["history"] != "full":
            raise GateError("Core checkout 必须是完整历史，当前是 shallow")
    except (GateError, OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"lock/core: {type(error).__name__}: {error}")
        report = _build_report(
            args.lock.resolve(),
            core,
            (),
            errors,
        )
        _write_report(report_path, report)
        _print_failure(report_path, errors)
        return 1

    # 2. Shallow-checkout each exact provider and inspect only static source.
    with tempfile.TemporaryDirectory(prefix="akashic-plugin-v3-fleet-") as raw:
        providers = Path(raw) / "providers"
        providers.mkdir()
        for lock in locks:
            try:
                checkout = _checkout_locked_plugin(lock, providers / lock.id)
                static = _inspect_static_plugin(providers / lock.id, lock.id)
                evidence = {
                    **asdict(checkout),
                    "status": "passed" if checkout.clean else "failed",
                    "static": static,
                }
                if static["status"] != "passed":
                    errors.append(f"{lock.id}: static v3 inspection failed")
                plugins.append(evidence)
            except (
                GateError,
                OSError,
                ValueError,
                subprocess.CalledProcessError,
            ) as error:
                errors.append(f"{lock.id}: {type(error).__name__}: {error}")
                plugins.append(_failed_plugin_evidence(lock, error))

    # 3. Persist auditable static evidence and explicit not-run E2E states.
    report = _build_report(args.lock.resolve(), core, plugins, errors)
    _write_report(report_path, report)
    if errors:
        _print_failure(report_path, errors)
        return 1
    print(f"plugin v3 fleet static gate passed: {report_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 pure-v3 external plugin fleet 静态合同"
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-clean-core", action="store_true")
    parser.add_argument("--require-full-core-history", action="store_true")
    return parser.parse_args()


def _load_lock(path: Path) -> tuple[PluginLock, ...]:
    """Strictly parse the exact locked plugin fleet and its hard exclusions."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "plugins"}:
        raise ValueError("pure-v3 fleet lock 根结构无效")
    if raw["schema_version"] != LOCK_SCHEMA_VERSION:
        raise ValueError(f"不支持的 pure-v3 fleet lock 版本: {raw['schema_version']}")
    raw_plugins = raw["plugins"]
    if not isinstance(raw_plugins, list):
        raise ValueError("pure-v3 fleet lock plugins 必须是数组")
    plugins = tuple(_parse_plugin_lock(item) for item in raw_plugins)
    ids = tuple(item.id for item in plugins)
    if len(ids) != len(set(ids)):
        raise ValueError("pure-v3 fleet lock 不得包含重复插件")
    excluded = sorted(set(ids) & set(EXCLUDED_PLUGIN_IDS))
    if excluded:
        raise ValueError(f"pure-v3 fleet lock 硬排除插件仍存在: {excluded}")
    if ids != EXPECTED_PLUGIN_IDS:
        missing = sorted(set(EXPECTED_PLUGIN_IDS) - set(ids))
        extra = sorted(set(ids) - set(EXPECTED_PLUGIN_IDS))
        raise ValueError(
            f"pure-v3 fleet lock 插件集合或顺序错误: missing={missing} extra={extra}"
        )
    return plugins


def _parse_plugin_lock(raw: object) -> PluginLock:
    """Validate one immutable repository identity at the trust boundary."""

    expected = {
        "id",
        "repository",
        "requested_ref",
        "resolved_sha",
        "change_source_pr_head",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"pure-v3 fleet lock 插件字段无效: {raw}")
    item = cast(dict[str, object], raw)
    values: dict[str, str] = {}
    for field in expected:
        value = item[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pure-v3 fleet lock 字段必须是非空字符串: {field}")
        values[field] = value
    if REPOSITORY_PATTERN.fullmatch(values["repository"]) is None:
        raise ValueError(f"插件仓库必须是 GitHub HTTPS 地址: {values['repository']}")
    revisions = ("requested_ref", "resolved_sha", "change_source_pr_head")
    for field in revisions:
        if COMMIT_PATTERN.fullmatch(values[field]) is None:
            raise ValueError(f"{field} 必须是完整 40 位 SHA: {values[field]}")
    if len({values[field] for field in revisions}) != 1:
        raise ValueError(f"三个 revision 必须固定到同一提交: {values['id']}")
    return PluginLock(**values)


@dataclass(frozen=True, slots=True)
class CheckoutEvidence:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str
    tree: str
    clean: bool
    dirty_status: tuple[str, ...]
    remote_ref: str | None
    history: str
    is_shallow: bool


def _checkout_locked_plugin(lock: PluginLock, checkout: Path) -> CheckoutEvidence:
    """Fetch one advertised exact commit into a fresh shallow checkout."""

    # 1. Prove the public remote is reachable and advertises the locked object.
    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", lock.repository), cwd=checkout)
    advertised = _run(
        ("git", "ls-remote", "--refs", "origin"),
        cwd=checkout,
    ).stdout
    refs = _matching_refs(advertised, lock.resolved_sha)

    # 2. Fetch only the exact ref when possible; direct object fetch is a strict fallback.
    remote_ref: str | None = refs[0] if refs else None
    if remote_ref is not None:
        _run(
            (
                "git",
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                f"{remote_ref}:refs/remotes/origin/fleet-lock",
            ),
            cwd=checkout,
        )
        checkout_ref = "refs/remotes/origin/fleet-lock"
    else:
        try:
            _run(
                ("git", "fetch", "--quiet", "--depth=1", "origin", lock.resolved_sha),
                cwd=checkout,
            )
        except GateError as error:
            raise GateError(
                f"远端未找到可达的锁定 SHA: {lock.id}@{lock.resolved_sha}"
            ) from error
        checkout_ref = "FETCH_HEAD"
    _run(("git", "checkout", "--quiet", "--detach", checkout_ref), cwd=checkout)

    # 3. Re-read the checked-out identity and cleanliness from Git.
    actual = _git_output(checkout, "rev-parse", "HEAD")
    if actual != lock.resolved_sha:
        raise GateError(
            f"插件检出提交与锁不一致: {lock.id} expected={lock.resolved_sha} actual={actual}"
        )
    dirty_status = tuple(_git_output(checkout, "status", "--porcelain").splitlines())
    history = (
        "shallow"
        if _git_output(checkout, "rev-parse", "--is-shallow-repository") == "true"
        else "full"
    )
    return CheckoutEvidence(
        id=lock.id,
        repository=lock.repository,
        requested_ref=lock.requested_ref,
        resolved_sha=lock.resolved_sha,
        change_source_pr_head=lock.change_source_pr_head,
        tree=_git_output(checkout, "rev-parse", "HEAD^{tree}"),
        clean=not dirty_status,
        dirty_status=dirty_status,
        remote_ref=remote_ref,
        history=history,
        is_shallow=history == "shallow",
    )


def _matching_refs(output: str, sha: str) -> tuple[str, ...]:
    refs: list[str] = []
    for line in output.splitlines():
        revision, separator, ref = line.partition("\t")
        if separator and revision == sha:
            refs.append(ref)
    return tuple(refs)


def _inspect_static_plugin(root: Path, plugin_id: str) -> dict[str, object]:
    """Inspect manifest, v3 namespace, and generic v2 imports without importing code."""

    # 1. Parse the import-free manifest and choose its declared entrypoint.
    manifest, manifest_errors = _inspect_manifest(root)
    entrypoint_name = str(manifest.get("entrypoint", "plugin.py"))
    entrypoint = root / entrypoint_name

    # 2. Parse the namespace AST and enforce api_version=3/apply(ctx, config).
    namespace = _inspect_namespace(root, entrypoint)

    # 3. Scan production Python sources for generic v2 import and class edges.
    forbidden = _find_forbidden_v2_imports(root)
    forbidden_classes = _find_forbidden_v2_classes(root)
    errors = [*manifest_errors, *cast(list[str], namespace["errors"])]
    manifest_name = manifest.get("name")
    namespace_name = namespace.get("name")
    if (
        isinstance(manifest_name, str)
        and isinstance(namespace_name, str)
        and manifest_name != namespace_name
    ):
        errors.append(
            f"manifest/module name 不一致: {manifest_name!r} != {namespace_name!r}"
        )
    if manifest.get("api_version") == 3 and namespace.get("api_version") != 3:
        errors.append("manifest/module api_version 不一致")
    if forbidden:
        errors.append("发现 generic v2 import")
    if forbidden_classes:
        errors.append("发现 legacy v2 Plugin class/fixed methods")
    return {
        "status": "passed" if not errors else "failed",
        "plugin_id": plugin_id,
        "manifest": manifest,
        "namespace": namespace,
        "forbidden_v2_imports": forbidden,
        "forbidden_v2_classes": forbidden_classes,
        "errors": errors,
    }


def _inspect_manifest(root: Path) -> tuple[dict[str, object], list[str]]:
    manifest_path = root / STATIC_MANIFEST_FILENAME
    evidence: dict[str, object] = {
        "path": STATIC_MANIFEST_FILENAME,
        "status": "missing",
    }
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return evidence, [f"缺少静态 manifest: {manifest_path}"]
    try:
        raw = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        return evidence, [f"静态 manifest 无法解析: {error}"]
    errors: list[str] = []
    if not isinstance(raw, dict):
        return evidence, ["静态 manifest 根必须是对象"]
    unknown = sorted(set(raw) - MANIFEST_ALLOWED_KEYS)
    if unknown:
        errors.append(f"静态 manifest 包含未知字段: {unknown}")
    for field in ("schema_version", "name", "version", "api_version", "entrypoint"):
        if field not in raw:
            errors.append(f"静态 manifest 缺少字段: {field}")
    if raw.get("schema_version") != 1:
        errors.append("静态 manifest schema_version 必须为 1")
    if raw.get("api_version") != 3:
        errors.append("静态 manifest api_version 必须为 3")
    if not isinstance(raw.get("name"), str) or not str(raw.get("name", "")).strip():
        errors.append("静态 manifest name 必须是非空字符串")
    if (
        not isinstance(raw.get("version"), str)
        or not str(raw.get("version", "")).strip()
    ):
        errors.append("静态 manifest version 必须是非空字符串")
    entrypoint = raw.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        errors.append("静态 manifest entrypoint 必须是非空字符串")
    elif not _safe_relative_path(entrypoint):
        errors.append(f"静态 manifest entrypoint 必须位于 artifact 内: {entrypoint}")
    elif (root / entrypoint).is_symlink() or not (root / entrypoint).is_file():
        errors.append(f"静态 manifest entrypoint 不存在或是 symlink: {entrypoint}")
    evidence.update(
        {
            "status": "passed" if not errors else "failed",
            "name": raw.get("name"),
            "version": raw.get("version"),
            "api_version": raw.get("api_version"),
            "entrypoint": entrypoint,
            "sha256": _sha256(manifest_path),
        }
    )
    return evidence, errors


def _inspect_namespace(root: Path, entrypoint: Path) -> dict[str, object]:
    evidence: dict[str, object] = {
        "path": _relative_or_name(entrypoint, root),
        "status": "failed",
        "errors": [],
    }
    errors = cast(list[str], evidence["errors"])
    if not entrypoint.is_file() or entrypoint.is_symlink():
        errors.append(f"v3 entrypoint 不存在或是 symlink: {entrypoint}")
        return evidence
    try:
        tree = ast.parse(
            entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint)
        )
    except (OSError, SyntaxError) as error:
        errors.append(f"v3 entrypoint 无法解析: {error}")
        return evidence
    api_version = _top_level_literal(tree, "api_version")
    name = _top_level_literal(tree, "name")
    apply_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "apply"
    ]
    apply_ok = False
    if len(apply_nodes) == 1:
        args = apply_nodes[0].args
        positional = [*args.posonlyargs, *args.args]
        apply_ok = (
            [item.arg for item in positional] == ["ctx", "config"]
            and args.vararg is None
            and args.kwarg is None
            and not args.kwonlyargs
            and not args.defaults
            and not args.kw_defaults
        )
    if api_version != 3:
        errors.append(f"namespace api_version 必须为 3: {api_version!r}")
    if not isinstance(name, str) or not name.strip():
        errors.append("namespace name 必须是非空字符串")
    if not apply_ok:
        errors.append("namespace 必须提供精确 apply(ctx, config)")
    evidence.update(
        {
            "status": "passed" if not errors else "failed",
            "api_version": api_version,
            "name": name,
            "apply_signature": "apply(ctx, config)" if apply_ok else None,
        }
    )
    return evidence


def _top_level_literal(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = node.value
        if value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError):
                return None
    return None


def _find_forbidden_v2_imports(root: Path) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for source in sorted(root.rglob("*.py")):
        if any(
            part in {".git", ".venv", "__pycache__", "scripts", "tests"}
            for part in source.parts
        ):
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as error:
            violations.append(
                {
                    "path": _relative_or_name(source, root),
                    "line": 1,
                    "error": str(error),
                }
            )
            continue
        for node in ast.walk(tree):
            module: str | None = None
            names: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
                if any(item in FORBIDDEN_V2_MODULES for item in imported):
                    module = next(
                        item for item in imported if item in FORBIDDEN_V2_MODULES
                    )
                    names = imported
            elif (
                isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_V2_MODULES
            ):
                module = node.module
                names = tuple(alias.name for alias in node.names)
            if module is not None:
                violations.append(
                    {
                        "path": _relative_or_name(source, root),
                        "line": node.lineno,
                        "module": module,
                        "names": names,
                    }
                )
    return violations


def _find_forbidden_v2_classes(root: Path) -> list[dict[str, object]]:
    """Find retired Plugin class contracts without importing artifact code."""

    violations: list[dict[str, object]] = []
    for source in sorted(root.rglob("*.py")):
        if any(
            part in {".git", ".venv", "__pycache__", "scripts", "tests"}
            for part in source.parts
        ):
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = tuple(_ast_name(base) for base in node.bases)
            legacy_bases = tuple(
                name
                for name in base_names
                if name is not None
                and name.rsplit(".", 1)[-1] in FORBIDDEN_V2_CLASS_BASES
            )
            method_names = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            fixed_methods = tuple(sorted(method_names & FORBIDDEN_V2_FIXED_METHODS))
            api_version = _class_literal(node, "api_version")
            if not legacy_bases and not fixed_methods and api_version != 2:
                continue
            violations.append(
                {
                    "path": _relative_or_name(source, root),
                    "line": node.lineno,
                    "class": node.name,
                    "legacy_bases": legacy_bases,
                    "fixed_methods": fixed_methods,
                    "api_version": api_version,
                }
            )
    return violations


def _ast_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _ast_name(node.value)
        return node.attr if prefix is None else f"{prefix}.{node.attr}"
    return None


def _class_literal(node: ast.ClassDef, name: str) -> object:
    for item in node.body:
        value: ast.AST | None = None
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in item.targets
        ):
            value = item.value
        elif (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == name
        ):
            value = item.value
        if value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError):
                return None
    return None


def _build_report(
    lock_path: Path,
    core: Mapping[str, object],
    plugins: tuple[dict[str, object], ...] | list[dict[str, object]],
    errors: list[str],
) -> dict[str, object]:
    """Build one report whose runtime E2E entries cannot claim execution."""

    e2e = _e2e_report()
    return {
        "status": "passed" if not errors else "failed",
        "phase": "static",
        "gate_version": GATE_VERSION,
        "checked_at": datetime.now(UTC).isoformat(),
        "core": dict(core),
        "lock": _relative_or_name(lock_path, ROOT),
        "lock_sha256": _sha256(lock_path) if lock_path.is_file() else None,
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "expected_plugin_ids": list(EXPECTED_PLUGIN_IDS),
        "excluded_plugin_ids": list(EXCLUDED_PLUGIN_IDS),
        "plugins": list(plugins),
        "static": {
            "status": "passed" if not errors else "failed",
            "error_count": len(errors),
        },
        "e2e": e2e,
        "errors": list(errors),
    }


def _e2e_report() -> dict[str, object]:
    catalog = [
        {
            **asdict(case),
            "required_plugins": list(case.required_plugins),
            "oracle": list(case.oracle),
            "status": "not_run",
            "executed": False,
            "reason": E2E_NOT_RUN_REASON,
        }
        for case in E2E_CATALOG
    ]
    return {
        "status": "not_run",
        "catalog_sha256": _json_sha256(catalog),
        "catalog": catalog,
        "reason": E2E_NOT_RUN_REASON,
    }


def _core_evidence() -> dict[str, object]:
    dirty_status = tuple(_git_output(ROOT, "status", "--porcelain").splitlines())
    commit = _git_output(ROOT, "rev-parse", "HEAD")
    tree = _git_output(ROOT, "rev-parse", "HEAD^{tree}")
    history = (
        "shallow"
        if _git_output(ROOT, "rev-parse", "--is-shallow-repository") == "true"
        else "full"
    )
    return {
        "commit": commit,
        "head": commit,
        "tree": tree,
        "dirty": list(dirty_status),
        "dirty_status": list(dirty_status),
        "clean": not dirty_status,
        "is_dirty": bool(dirty_status),
        "history": history,
        "is_shallow": history == "shallow",
    }


def _failed_plugin_evidence(
    lock: PluginLock, error: BaseException
) -> dict[str, object]:
    return {
        **asdict(lock),
        "status": "failed",
        "tree": None,
        "clean": None,
        "dirty_status": [],
        "remote_ref": None,
        "history": None,
        "is_shallow": None,
        "static": {"status": "not_run"},
        "error": f"{type(error).__name__}: {error}",
    }


def _write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_failure(path: Path, errors: list[str]) -> None:
    print(f"plugin v3 fleet static gate failed: {path}", file=sys.stderr)
    for error in errors:
        print(f"- {error}", file=sys.stderr)


def _safe_relative_path(value: str) -> bool:
    candidate = Path(value)
    return (
        not candidate.is_absolute()
        and ".." not in candidate.parts
        and "" not in candidate.parts
        and "\\" not in value
    )


def _relative_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_output(cwd: Path, *args: str) -> str:
    return _run(("git", *args), cwd=cwd).stdout.strip()


def _run(command: tuple[str, ...], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired as error:
        raise GateError(
            f"命令超时 {' '.join(command)} after={GIT_COMMAND_TIMEOUT_SECONDS}s"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if isinstance(error.stderr, str) else ""
        suffix = f": {detail}" if detail else ""
        raise GateError(f"命令失败 {' '.join(command)}{suffix}") from error


if __name__ == "__main__":
    raise SystemExit(main())
