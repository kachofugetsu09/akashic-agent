from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-v3-mobile.lock.json"
DEFAULT_REPORT = ROOT / "docker" / "debug" / "reports" / "plugin-v3-mobile" / "gate.json"
UI_CONTRACT_RUNNER = ROOT / "docker" / "debug" / "mobile_plugin_ui_contract.mjs"
CONTRACT_REPOSITORY = "https://github.com/akashic-plugins/plugin-contracts"
CONTRACT_SHA = "4dd69dd621e029e51e99aa428443fa3a4ec1f6cf"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
REPOSITORY_PATTERN = re.compile(
    r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?"
)
ALLOWED_SLOTS = frozenset(
    {
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    }
)
EXPECTED_PLUGIN_IDS = (
    "akasha",
    "observe",
    "fitbit-mcp",
    "proactive_feedback",
    "emotion",
    "status_commands",
)
EXTERNAL_PLUGIN_IDS = EXPECTED_PLUGIN_IDS[1:]
FORBIDDEN_LOCK_FIELDS = frozenset({"plugin_class", "query_methods", "mobile_ui_query"})
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


class GateError(RuntimeError):
    """报告可复核的输入、静态合同或行为证据失败。"""


@dataclass(frozen=True, slots=True)
class PluginContract:
    id: str
    source: str
    path: str
    entrypoint: str
    module: str
    stylesheet: str
    navigation: bool
    slots: tuple[str, ...]
    node_test: str
    node_setup: str
    repository: str | None = None
    requested_ref: str | None = None
    resolved_sha: str | None = None
    change_source_pr_head: str | None = None


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    command: str
    status: str
    returncode: int
    output_sha256: str


def main() -> int:
    """验证 exact pure-v3 Mobile fleet，并把静态、行为和清理证据写入报告。"""

    # 1. 固定 Core、锁和 disposable 运行边界。
    args = _parse_args()
    report_path = args.report.resolve()
    core = _core_evidence()
    report: dict[str, object] = {
        "status": "failed",
        "gate_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "lock": _display_path(args.lock.resolve()),
        "lock_sha256": None,
        "core": core,
        "protocol_source": None,
        "plugins": [],
        "python_contract": None,
        "errors": [],
        "cleanup": {
            "temporary_root_created": False,
            "temporary_root_removed": False,
            "formal_workspace_touched": False,
        },
    }
    temporary_root: Path | None = None
    try:
        lock_path = args.lock.resolve()
        report["lock_sha256"] = _sha256(lock_path)
        contracts = _load_lock(lock_path)
        if args.require_clean_core and not cast(bool, core["clean"]):
            raise GateError(f"Core 工作树不干净: {core['dirty_status']}")
        if not UI_CONTRACT_RUNNER.is_file():
            raise GateError(f"Core-owned JS contract runner 缺失: {UI_CONTRACT_RUNNER}")

        # 2. 获取 exact external source，并保留每个 checkout 的身份。
        with tempfile.TemporaryDirectory(
            dir=args.tmp_root,
            prefix="akashic-plugin-v3-mobile-",
        ) as raw_temp:
            temporary_root = Path(raw_temp)
            cast(dict[str, object], report["cleanup"])["temporary_root_created"] = True
            roots, source_evidence, source_errors = _resolve_sources(
                contracts,
                temporary_root,
                _parse_plugin_roots(args.plugin_root),
                offline=args.offline,
            )
            report["plugins"] = source_evidence
            if source_errors:
                cast(list[str], report["errors"]).extend(source_errors)
                raise GateError("至少一个 exact plugin source 无法获取")

            # 3. AST、manifest、资产与 Core-owned Node runner/测试逐项执行。
            plugin_errors: list[str] = []
            for contract in contracts:
                root = roots[contract.id]
                try:
                    evidence = _verify_plugin(contract, root)
                except (
                    GateError,
                    OSError,
                    SyntaxError,
                    ValueError,
                    tomllib.TOMLDecodeError,
                    subprocess.CalledProcessError,
                ) as error:
                    detail = f"{contract.id}: {type(error).__name__}: {error}"
                    plugin_errors.append(detail)
                    _replace_plugin_evidence(
                        report,
                        contract.id,
                        {
                            "status": "failed",
                            "error": detail,
                            "test_commands": [],
                        },
                    )
                else:
                    _replace_plugin_evidence(report, contract.id, evidence)

            # 4. 用锁定的 plugin-contracts 子进程验证全部 Python v3 入口。
            contract_evidence = _run_python_contract(contracts, roots, temporary_root)
            report["python_contract"] = contract_evidence
            report["protocol_source"] = contract_evidence["source"]
            if contract_evidence["status"] != "passed":
                raise GateError("plugin-contracts Python v3 合同失败")
            if plugin_errors:
                raise GateError("Mobile plugin 行为或静态合同失败: " + "; ".join(plugin_errors))
        cast(dict[str, object], report["cleanup"])["temporary_root_removed"] = (
            temporary_root is not None and not temporary_root.exists()
        )
        if not cast(bool, cast(dict[str, object], report["cleanup"])["temporary_root_removed"]):
            raise GateError("临时 plugin checkout 清理失败")
        report["status"] = "passed"
    except (
        GateError,
        OSError,
        SyntaxError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        cast(list[str], report["errors"]).append(f"{type(error).__name__}: {error}")
    finally:
        cleanup = cast(dict[str, object], report["cleanup"])
        if temporary_root is not None and not temporary_root.exists():
            cleanup["temporary_root_removed"] = True
        _write_report(report_path, report)

    if report["status"] != "passed":
        print(f"plugin v3 mobile gate failed: {report_path}", file=sys.stderr)
        for error in cast(list[str], report["errors"]):
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"plugin v3 mobile gate passed: {report_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 pure-v3 Mobile UI/query fleet")
    _ = parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    _ = parser.add_argument("--tmp-root", type=Path)
    _ = parser.add_argument(
        "--plugin-root",
        action="append",
        default=[],
        metavar="PLUGIN_ID=PATH",
        help="提供已核对的 external checkout；不改变锁定 source",
    )
    _ = parser.add_argument("--offline", action="store_true")
    _ = parser.add_argument("--require-clean-core", action="store_true")
    return parser.parse_args()


def _load_lock(path: Path) -> tuple[PluginContract, ...]:
    """严格解析只包含 v3 Mobile 字段的 exact lock。"""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "plugins"}:
        raise ValueError("v3 Mobile lock 根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的 v3 Mobile lock 版本: {raw['schema_version']}")
    raw_plugins = raw["plugins"]
    if not isinstance(raw_plugins, list):
        raise ValueError("v3 Mobile lock plugins 必须是数组")
    contracts = tuple(_parse_plugin(raw_item) for raw_item in raw_plugins)
    ids = tuple(item.id for item in contracts)
    if ids != EXPECTED_PLUGIN_IDS:
        expected_ids = set(cast(tuple[str, ...], EXPECTED_PLUGIN_IDS))
        actual_ids = set(ids)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"v3 Mobile lock 插件集合或顺序错误: missing={missing} extra={extra}")
    return contracts


def _parse_plugin(raw: object) -> PluginContract:
    if not isinstance(raw, dict):
        raise ValueError(f"v3 Mobile lock 插件项必须是对象: {raw!r}")
    if FORBIDDEN_LOCK_FIELDS & set(raw):
        raise ValueError(
            "v3 Mobile lock 禁止 v2 字段: "
            + ", ".join(sorted(FORBIDDEN_LOCK_FIELDS & set(raw)))
        )
    required = {
        "id",
        "source",
        "entrypoint",
        "module",
        "stylesheet",
        "navigation",
        "slots",
        "node_test",
        "node_setup",
    }
    source = raw.get("source")
    expected = required | (
        {"path"} if source == "in-tree" else
        {"repository", "requested_ref", "resolved_sha", "change_source_pr_head"}
    )
    if set(raw) != expected:
        raise ValueError(f"v3 Mobile lock 字段无效: {raw}")
    item = cast(dict[str, object], raw)
    strings = {
        field: _required_string(item, field)
        for field in ("id", "entrypoint", "module", "stylesheet", "node_test")
    }
    if source not in {"in-tree", "external"}:
        raise ValueError(f"v3 Mobile source 无效: {source!r}")
    path = _required_string(item, "path") if source == "in-tree" else "."
    repository: str | None = None
    revisions: tuple[str, ...] = ()
    if source == "external":
        repository = _required_string(item, "repository")
        if REPOSITORY_PATTERN.fullmatch(repository) is None:
            raise ValueError(f"插件 repository 必须是 GitHub HTTPS 地址: {repository}")
        revisions = tuple(_required_string(item, field) for field in (
            "requested_ref", "resolved_sha", "change_source_pr_head",
        ))
        if any(COMMIT_PATTERN.fullmatch(value) is None for value in revisions):
            raise ValueError(f"插件 revision 必须是完整 SHA: {strings['id']}")
        if len(set(revisions)) != 1:
            raise ValueError(f"插件 revision 必须固定到同一 SHA: {strings['id']}")
    _require_relative_path(path)
    _require_relative_path(strings["entrypoint"])
    _require_relative_path(strings["module"])
    _require_relative_path(strings["stylesheet"])
    _require_relative_path(strings["node_test"])
    navigation = item["navigation"]
    if not isinstance(navigation, bool):
        raise ValueError(f"插件 navigation 必须是布尔值: {strings['id']}")
    slots = _string_tuple(item, "slots")
    if len(set(slots)) != len(slots) or any(slot not in ALLOWED_SLOTS for slot in slots):
        raise ValueError(f"插件 slots 无效: {strings['id']}")
    node_setup = item["node_setup"]
    if node_setup not in {"none", "npm-ci"}:
        raise ValueError(f"插件 node_setup 无效: {strings['id']}")
    return PluginContract(
        id=strings["id"],
        source=cast(str, source),
        path=path,
        entrypoint=strings["entrypoint"],
        module=strings["module"],
        stylesheet=strings["stylesheet"],
        navigation=navigation,
        slots=slots,
        node_test=strings["node_test"],
        node_setup=cast(str, node_setup),
        repository=repository,
        requested_ref=revisions[0] if revisions else None,
        resolved_sha=revisions[1] if revisions else None,
        change_source_pr_head=revisions[2] if revisions else None,
    )


def _parse_plugin_roots(raw: list[str]) -> dict[str, Path]:
    """解析外部 exact checkout 路径，不接受未声明插件。"""

    roots: dict[str, Path] = {}
    for value in raw:
        plugin_id, separator, path = value.partition("=")
        if not separator or not plugin_id.strip() or not path.strip():
            raise GateError(f"--plugin-root 必须是 PLUGIN_ID=PATH: {value!r}")
        if plugin_id in roots:
            raise GateError(f"--plugin-root 重复: {plugin_id}")
        roots[plugin_id] = Path(path).expanduser().resolve(strict=False)
    unknown = sorted(set(roots) - set(EXTERNAL_PLUGIN_IDS))
    if unknown:
        raise GateError(f"--plugin-root 只允许 external Mobile plugin: {unknown}")
    return roots


def _resolve_sources(
    contracts: tuple[PluginContract, ...],
    temporary_root: Path,
    provided: dict[str, Path],
    *,
    offline: bool,
) -> tuple[dict[str, Path], list[dict[str, object]], list[str]]:
    roots: dict[str, Path] = {}
    evidence: list[dict[str, object]] = []
    errors: list[str] = []
    for contract in contracts:
        try:
            if contract.source == "in-tree":
                root = (ROOT / contract.path).resolve(strict=False)
                item = _local_source_evidence(contract, root)
            elif contract.id in provided:
                root = provided[contract.id]
                item = _local_source_evidence(contract, root)
            elif offline:
                raise GateError(
                    f"offline 模式没有 external checkout: {contract.id}@{contract.resolved_sha}"
                )
            else:
                root = temporary_root / contract.id
                item = _checkout_external(contract, root)
        except (
            GateError,
            OSError,
            ValueError,
            subprocess.CalledProcessError,
        ) as error:
            detail = f"{contract.id}: {type(error).__name__}: {error}"
            errors.append(detail)
            item = {
                "id": contract.id,
                "source": contract.source,
                "status": "failed",
                "source_sha": contract.resolved_sha,
                "error": detail,
            }
        else:
            roots[contract.id] = root
        evidence.append(item)
    return roots, evidence, errors


def _local_source_evidence(contract: PluginContract, root: Path) -> dict[str, object]:
    """核对 in-tree 或显式提供的 source SHA/tree/clean 状态。"""

    if not root.is_dir() or root.is_symlink():
        raise GateError(f"plugin source 不是实体目录: {contract.id}: {root}")
    sha = _git_output(root, "rev-parse", "HEAD")
    dirty_command = (
        ("git", "status", "--porcelain", "--", contract.path)
        if contract.source == "in-tree"
        else ("git", "status", "--porcelain")
    )
    dirty = tuple(
        _git_output(
            ROOT if contract.source == "in-tree" else root,
            *dirty_command[1:],
        ).splitlines()
    )
    if contract.source == "external" and sha != contract.resolved_sha:
        raise GateError(
            f"external checkout SHA 不匹配: {contract.id} expected={contract.resolved_sha} actual={sha}"
        )
    if dirty:
        raise GateError(f"plugin source 工作树不干净: {contract.id}: {dirty}")
    evidence: dict[str, object] = {
        "id": contract.id,
        "source": contract.source,
        "source_sha": sha,
        "tree": (
            _git_output(ROOT, "rev-parse", f"HEAD:{contract.path}")
            if contract.source == "in-tree"
            else _git_output(root, "rev-parse", "HEAD^{tree}")
        ),
        "clean": True,
        "dirty_status": [],
        "path": str(root),
        "status": "passed",
    }
    if contract.repository is not None:
        evidence.update(
            {
                "repository": contract.repository,
                "requested_ref": contract.requested_ref,
                "resolved_sha": contract.resolved_sha,
                "change_source_pr_head": contract.change_source_pr_head,
            }
        )
    return evidence


def _checkout_external(contract: PluginContract, checkout: Path) -> dict[str, object]:
    """从公开仓库取出一个不可变 external commit。"""

    if contract.repository is None or contract.resolved_sha is None:
        raise GateError(f"external lock 缺少 source identity: {contract.id}")
    _run_checked(("git", "init", "--quiet", str(checkout)), ROOT)
    _run_checked(("git", "remote", "add", "origin", contract.repository), checkout)
    _run_checked(
        ("git", "fetch", "--quiet", "--depth=1", "origin", contract.resolved_sha),
        checkout,
    )
    _run_checked(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), checkout)
    return _local_source_evidence(contract, checkout) | {
        "repository": contract.repository,
        "requested_ref": contract.requested_ref,
        "resolved_sha": contract.resolved_sha,
        "change_source_pr_head": contract.change_source_pr_head,
        "history": _git_output(checkout, "rev-parse", "--is-shallow-repository"),
    }


def _verify_plugin(contract: PluginContract, root: Path) -> dict[str, object]:
    """静态验证一个 pure-v3 Mobile source，并执行 Core runner 与真实 Node test。"""

    # 1. Manifest/namespace 与 AST UI seam 必须先闭合。
    entrypoint = _inside(root, contract.entrypoint)
    static = _inspect_static_source(contract, root, entrypoint)
    if static["status"] != "passed":
        raise GateError(f"Mobile static contract failed: {contract.id}: {static['errors']}")

    # 2. Core 只读取并哈希插件静态资产。
    assets = _asset_evidence(root, contract)
    commands: list[dict[str, object]] = []
    module_path = _inside(root, contract.module)
    runner_command = (
        "node",
        str(UI_CONTRACT_RUNNER),
        str(module_path),
        str(contract.navigation).lower(),
        json.dumps(contract.slots, ensure_ascii=False),
    )
    commands.append(asdict(_run_command(runner_command, root)))
    if contract.node_setup == "npm-ci":
        setup_command = ("npm", "ci", "--ignore-scripts")
        commands.append(asdict(_run_command(setup_command, root)))
    node_test = _node_test_path(root, contract)
    node_command = ("node", "--test", str(node_test))
    commands.append(asdict(_run_command(node_command, root)))
    return {
        "id": contract.id,
        "source": contract.source,
        "status": "passed",
        "static": static,
        "assets": assets,
        "test_commands": commands,
        "entrypoint": contract.entrypoint,
    }


def _inspect_static_source(
    contract: PluginContract,
    root: Path,
    entrypoint: Path,
) -> dict[str, object]:
    """检查 manifest、pure-v3 namespace、UI_SLOTS 注入和 apply 注册。"""

    errors: list[str] = []
    manifest: dict[str, object]
    if contract.source == "external":
        manifest, manifest_errors = _inspect_manifest(root)
        errors.extend(manifest_errors)
        if manifest.get("entrypoint") != contract.entrypoint:
            errors.append(
                "manifest entrypoint 与 lock 不一致: "
                f"{manifest.get('entrypoint')!r} != {contract.entrypoint!r}"
            )
    else:
        manifest = {"status": "in-tree", "path": None}
    try:
        tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    except (OSError, SyntaxError) as error:
        errors.append(f"entrypoint 无法解析: {error}")
        return {"status": "failed", "manifest": manifest, "errors": errors}
    errors.extend(_find_forbidden_v2_imports(root))
    namespace = _inspect_namespace(tree, contract)
    errors.extend(cast(list[str], namespace["errors"]))
    return {
        "status": "passed" if not errors else "failed",
        "manifest": manifest,
        "namespace": namespace,
        "errors": errors,
        "source_sha256": _sha256(entrypoint),
    }


def _inspect_manifest(root: Path) -> tuple[dict[str, object], list[str]]:
    path = root / "akashic.plugin.toml"
    if not path.is_file() or path.is_symlink():
        return {"status": "missing", "path": "akashic.plugin.toml"}, [
            f"缺少静态 manifest: {path}"
        ]
    errors: list[str] = []
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {"status": "failed", "path": path.name}, ["manifest 根必须是对象"]
    allowed = {
        "schema_version", "name", "version", "api_version", "entrypoint",
        "python", "validation", "mcp", "processes", "channel_credentials",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        errors.append(f"manifest 包含未知字段: {unknown}")
    for field in ("schema_version", "name", "version", "api_version", "entrypoint"):
        if field not in raw:
            errors.append(f"manifest 缺少字段: {field}")
    if raw.get("schema_version") != 1:
        errors.append("manifest schema_version 必须为 1")
    if raw.get("api_version") != 3:
        errors.append("manifest api_version 必须为 3")
    entrypoint = raw.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip() or not _safe_relative_path(entrypoint):
        errors.append(f"manifest entrypoint 必须是 artifact 内相对路径: {entrypoint!r}")
    elif not (root / entrypoint).is_file() or (root / entrypoint).is_symlink():
        errors.append(f"manifest entrypoint 不存在或是 symlink: {entrypoint}")
    return {
        "status": "passed" if not errors else "failed",
        "path": path.name,
        "name": raw.get("name"),
        "version": raw.get("version"),
        "api_version": raw.get("api_version"),
        "entrypoint": entrypoint,
        "sha256": _sha256(path),
    }, errors


def _inspect_namespace(tree: ast.Module, contract: PluginContract) -> dict[str, object]:
    errors: list[str] = []
    api_version = _top_level_literal(tree, "api_version")
    name = _top_level_literal(tree, "name")
    inject = _top_level_name_tuple(tree, "inject")
    if api_version != 3:
        errors.append(f"api_version 必须为 3: {api_version!r}")
    if not isinstance(name, str) or not name.strip():
        errors.append("name 必须是非空字符串")
    if inject is None or inject.count("UI_SLOTS") != 1:
        errors.append("inject 必须恰好包含 UI_SLOTS")
    apply_nodes = [
        node for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "apply"
    ]
    if len(apply_nodes) != 1:
        errors.append("必须提供唯一 async apply(ctx, config)")
        return _namespace_result(api_version, name, inject, errors)
    apply = apply_nodes[0]
    positional = [*apply.args.posonlyargs, *apply.args.args]
    if (
        [item.arg for item in positional] != ["ctx", "config"]
        or apply.args.vararg is not None
        or apply.args.kwarg is not None
        or apply.args.kwonlyargs
        or apply.args.defaults
        or apply.args.kw_defaults
    ):
        errors.append("apply 签名必须是 apply(ctx, config)")
    errors.extend(_inspect_mobile_registration(apply, contract))
    return _namespace_result(api_version, name, inject, errors)


def _inspect_mobile_registration(
    apply: ast.AsyncFunctionDef,
    contract: PluginContract,
) -> list[str]:
    errors: list[str] = []
    slot_names: set[str] = set()
    for node in ast.walk(apply):
        if not isinstance(node, ast.Call) or _attribute_name(node.func) != "require":
            continue
        if len(node.args) == 1 and _name_of(node.args[0]) == "UI_SLOTS":
            parent = _parent_assignment_name(apply, node)
            if parent is not None:
                slot_names.add(parent)
    registrations = [
        node for node in ast.walk(apply)
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "register_mobile"
    ]
    if len(registrations) != 1:
        errors.append("apply 必须包含唯一 register_mobile")
        return errors
    registration = registrations[0]
    receiver = registration.func.value if isinstance(registration.func, ast.Attribute) else None
    direct = (
        isinstance(receiver, ast.Call)
        and _attribute_name(receiver.func) == "require"
        and len(receiver.args) == 1
        and _name_of(receiver.args[0]) == "UI_SLOTS"
    )
    bound = isinstance(receiver, ast.Name) and receiver.id in slot_names
    if not direct and not bound:
        errors.append("register_mobile 必须由 ctx.require(UI_SLOTS) 所得服务调用")
    registration_keywords = {
        item.arg for item in registration.keywords if item.arg is not None
    }
    if "query" not in registration_keywords:
        errors.append("register_mobile 必须显式提供 v3 query handler")
    definitions = [
        node for node in registration.args
        if isinstance(node, ast.Call) and _attribute_name(node.func) == "MobileUiDefinition"
    ]
    if len(definitions) != 1:
        errors.append("register_mobile 必须接收 MobileUiDefinition")
        return errors
    definition = definitions[0]
    keywords = {item.arg: item.value for item in definition.keywords if item.arg is not None}
    if set(keywords) - {"module", "stylesheet", "navigation", "slots"}:
        errors.append("MobileUiDefinition 包含未声明字段")
    module = _literal_string(keywords.get("module"))
    stylesheet = _literal_string(keywords.get("stylesheet"))
    if module != contract.module:
        errors.append(f"MobileUiDefinition module 不匹配: {module!r}")
    if stylesheet != contract.stylesheet:
        errors.append(f"MobileUiDefinition stylesheet 不匹配: {stylesheet!r}")
    slots = _literal_strings(keywords.get("slots")) if "slots" in keywords else ()
    if slots != contract.slots:
        errors.append(f"MobileUiDefinition slots 不匹配: {slots!r}")
    navigation = keywords.get("navigation")
    has_navigation = navigation is not None
    if has_navigation != contract.navigation:
        errors.append(f"MobileUiDefinition navigation 存在性不匹配: {has_navigation}")
    if has_navigation:
        if not _is_navigation_call(navigation):
            errors.append("MobileUiDefinition navigation 必须是 MobileUiNavigation")
        else:
            navigation_keywords = {
                item.arg for item in navigation.keywords if item.arg is not None
            }
            if navigation_keywords != {"label", "description"}:
                errors.append("MobileUiNavigation 必须包含 label/description")
            if any(
                _literal_string(item.value) is None
                for item in navigation.keywords
                if item.arg in {"label", "description"}
            ):
                errors.append("MobileUiNavigation label/description 必须是字符串")
    return errors


def _run_python_contract(
    contracts: tuple[PluginContract, ...],
    roots: dict[str, Path],
    temporary_root: Path,
) -> dict[str, object]:
    """在独立进程运行已锁定的 plugin-contracts checker，不导入插件到 Core。"""

    contract_root = temporary_root / "plugin-contracts"
    source = _checkout_contract_source(contract_root)
    paths = tuple(str(_inside(roots[item.id], item.entrypoint)) for item in contracts)
    command = (
        sys.executable,
        "-m",
        "akashic_plugin_contracts",
        "check",
        *paths,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(contract_root)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = result.stdout
    status = "passed" if result.returncode == 0 else "failed"
    if result.returncode == 0:
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as error:
            raise GateError(f"plugin-contracts 输出不是 JSON: {error}") from error
        if not isinstance(parsed, dict) or parsed.get("passed") is not True:
            raise GateError("plugin-contracts 返回 passed=false")
        reports = parsed.get("reports")
        if not isinstance(reports, list) or len(reports) != len(contracts):
            raise GateError("plugin-contracts report 数量与 exact fleet 不一致")
    return {
        "status": status,
        "source": source,
        "command": shlex.join(command),
        "returncode": result.returncode,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def _checkout_contract_source(checkout: Path) -> dict[str, object]:
    _run_checked(("git", "init", "--quiet", str(checkout)), ROOT)
    _run_checked(("git", "remote", "add", "origin", CONTRACT_REPOSITORY), checkout)
    _run_checked(("git", "fetch", "--quiet", "--depth=1", "origin", CONTRACT_SHA), checkout)
    _run_checked(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), checkout)
    actual = _git_output(checkout, "rev-parse", "HEAD")
    if actual != CONTRACT_SHA:
        raise GateError(f"plugin-contracts SHA 不匹配: expected={CONTRACT_SHA} actual={actual}")
    dirty = _git_output(checkout, "status", "--porcelain")
    if dirty:
        raise GateError(f"plugin-contracts checkout 不干净: {dirty}")
    return {
        "repository": CONTRACT_REPOSITORY,
        "source_sha": actual,
        "tree": _git_output(checkout, "rev-parse", "HEAD^{tree}"),
        "clean": True,
    }


def _asset_evidence(root: Path, contract: PluginContract) -> dict[str, object]:
    module = _inside(root, contract.module)
    stylesheet = _inside(root, contract.stylesheet)
    module_bytes = module.read_bytes()
    stylesheet_bytes = stylesheet.read_bytes()
    if not module_bytes or not stylesheet_bytes:
        raise GateError(f"Mobile asset 不能为空: {contract.id}")
    if len(module_bytes) + len(stylesheet_bytes) > 240 * 1024:
        raise GateError(f"Mobile asset 超过 240 KiB: {contract.id}")
    return {
        "module": {
            "path": contract.module,
            "bytes": len(module_bytes),
            "sha256": hashlib.sha256(module_bytes).hexdigest(),
        },
        "stylesheet": {
            "path": contract.stylesheet,
            "bytes": len(stylesheet_bytes),
            "sha256": hashlib.sha256(stylesheet_bytes).hexdigest(),
        },
        "total_bytes": len(module_bytes) + len(stylesheet_bytes),
    }


def _node_test_path(root: Path, contract: PluginContract) -> Path:
    if contract.source == "in-tree":
        path = (ROOT / contract.node_test).resolve(strict=True)
        if not path.is_relative_to(ROOT) or path.is_symlink():
            raise GateError(f"in-tree Node test path 无效: {contract.node_test}")
        return path
    return _inside(root, contract.node_test)


def _run_command(command: tuple[str, ...], cwd: Path) -> CommandEvidence:
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_sha = hashlib.sha256(result.stdout.encode()).hexdigest()
    evidence = CommandEvidence(
        command=shlex.join(command),
        status="passed" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        output_sha256=output_sha,
    )
    if result.returncode != 0:
        raise GateError(
            f"命令失败 ({result.returncode}): {evidence.command}; output_sha256={output_sha}"
        )
    return evidence


def _core_evidence() -> dict[str, object]:
    dirty = tuple(_git_output(ROOT, "status", "--porcelain").splitlines())
    history = _git_output(ROOT, "rev-parse", "--is-shallow-repository")
    return {
        "head": _git_output(ROOT, "rev-parse", "HEAD"),
        "tree": _git_output(ROOT, "rev-parse", "HEAD^{tree}"),
        "dirty_status": list(dirty),
        "clean": not dirty,
        "history": history,
    }


def _replace_plugin_evidence(
    report: dict[str, object],
    plugin_id: str,
    evidence: dict[str, object],
) -> None:
    plugins = cast(list[dict[str, object]], report["plugins"])
    for index, item in enumerate(plugins):
        if item.get("id") == plugin_id:
            plugins[index] = {**item, **evidence}
            return
    raise GateError(f"报告缺少 plugin source evidence: {plugin_id}")


def _namespace_result(
    api_version: object,
    name: object,
    inject: tuple[str, ...] | None,
    errors: list[str],
) -> dict[str, object]:
    return {
        "status": "passed" if not errors else "failed",
        "api_version": api_version,
        "name": name,
        "inject": list(inject or ()),
        "apply_signature": "apply(ctx, config)" if not errors else None,
        "errors": errors,
    }


def _find_forbidden_v2_imports(root: Path) -> list[str]:
    violations: list[str] = []
    for source in sorted(root.rglob("*.py")):
        if any(part in {".git", ".venv", "__pycache__", "scripts", "tests"} for part in source.parts):
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError) as error:
            violations.append(f"{source}: parse failed: {error}")
            continue
        for node in ast.walk(tree):
            module: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module in FORBIDDEN_V2_MODULES:
                        violations.append(f"{source}: import {module}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module in FORBIDDEN_V2_MODULES:
                    violations.append(f"{source}: from {module} import ...")
    return violations


def _top_level_literal(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            value = node.value
        if value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError):
                return None
    return None


def _top_level_name_tuple(tree: ast.Module, name: str) -> tuple[str, ...] | None:
    for node in tree.body:
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            value = node.value
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        names = tuple(item.id for item in value.elts if isinstance(item, ast.Name))
        if len(names) != len(value.elts):
            return None
        return names
    return None


def _parent_assignment_name(root: ast.AST, target: ast.Call) -> str | None:
    for node in ast.walk(root):
        if isinstance(node, ast.Assign) and node.value is target:
            names = [item.id for item in node.targets if isinstance(item, ast.Name)]
            if len(names) == 1:
                return names[0]
        if isinstance(node, ast.AnnAssign) and node.value is target and isinstance(node.target, ast.Name):
            return node.target.id
    return None


def _is_navigation_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and _attribute_name(node.func) == "MobileUiNavigation"


def _attribute_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _name_of(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _literal_string(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _literal_strings(node: ast.expr | None) -> tuple[str, ...] | None:
    if node is None:
        return ()
    if not isinstance(node, ast.Tuple):
        return None
    values = tuple(_literal_string(item) for item in node.elts)
    if any(value is None for value in values):
        return None
    return cast(tuple[str, ...], values)


def _required_string(item: dict[str, object], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"v3 Mobile lock 字段必须是非空字符串: {name}")
    return value


def _string_tuple(item: dict[str, object], name: str) -> tuple[str, ...]:
    value = item.get(name)
    if not isinstance(value, list) or any(not isinstance(entry, str) or not entry for entry in value):
        raise ValueError(f"v3 Mobile lock {name} 必须是字符串数组")
    return tuple(cast(list[str], value))


def _require_relative_path(value: str) -> None:
    if not _safe_relative_path(value):
        raise ValueError(f"路径必须是 source 内相对路径: {value}")


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and bool(value.strip())


def _inside(root: Path, relative: str) -> Path:
    raw = root / relative
    if raw.is_symlink():
        raise GateError(f"source path 是 symlink: {relative}")
    path = raw.resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise GateError(f"source path 逃逸或是 symlink: {relative}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _run_checked(command: tuple[str, ...], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        raise GateError(f"命令失败 ({result.returncode}): {shlex.join(command)}")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
