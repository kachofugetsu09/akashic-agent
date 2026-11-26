"""独立采集真实用户交互入口并校验 Self-CLI Operation 覆盖（Parity 门禁）。

入口分母直接从插件源码 AST 独立枚举（HTTP/WS 路由、WebChat WS 帧、Plugin UI RPC、
Slash 命令、Gateway 方法），不依赖 Operation Registry 自报。
每个规范化入口必须有且只有一种归宿：
  - mapped_to_operation: 已映射到 Self-CLI operation
  - explicit_exemption:  显式豁免（禁止通配符，须注明类型与理由）
  - legacy_backlog:      迁移期冻结旧欠账（只许减少，新增未归属入口即失败）
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs/interaction-parity.json"
ALLOWED_EXEMPTION_KINDS = frozenset({"transport", "view_only", "user_only", "auth_security"})
HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete", "websocket"})


class MappedEntry(TypedDict):
    operation_id: str
    argument_mapping: str


class ExemptionEntry(TypedDict):
    kind: Literal["transport", "view_only", "user_only", "auth_security"]
    reason: str
    decision: str


class ParityManifest(TypedDict):
    version: int
    mapped_to_operation: dict[str, MappedEntry]
    explicit_exemption: dict[str, ExemptionEntry]
    legacy_backlog: list[str]


def _eval_str(expr: ast.expr, consts: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name) and expr.id in consts:
        return consts[expr.id]
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left, right = _eval_str(expr.left, consts), _eval_str(expr.right, consts)
        if left is not None and right is not None:
            return left + right
    return None


def _string_literals(expr: ast.expr) -> list[str]:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, (ast.Set, ast.Tuple, ast.List)):
        return [e.value for e in expr.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def collect_entries(plugins_dir: Path) -> set[str]:
    """从插件目录独立采集所有用户可触发入口，返回规范化 entry_id 集合。"""
    entries: set[str] = set()
    py_files = sorted(plugins_dir.rglob("*.py"))
    ui_plugins = {
        py.relative_to(plugins_dir).parts[0]
        for py in py_files
        if "register_plugin_ui" in py.read_text(encoding="utf-8")
    }
    for py in py_files:
        rel_parts = py.relative_to(plugins_dir).parts
        if not rel_parts:
            continue
        plugin = rel_parts[0]
        text = py.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        consts = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            # 1. FastAPI / Starlette HTTP & WebSocket 路由
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in HTTP_VERBS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("app", "router")
                and node.args
            ):
                route = _eval_str(node.args[0], consts)
                if route is not None:
                    entries.add(f"http:{plugin}:{node.func.attr.upper()} {route}")
            # 2. WebSocket 入站帧分派 & Plugin UI RPC method 分派
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
                if node.left.id == "frame_type":
                    for comp in node.comparators:
                        for name in _string_literals(comp):
                            entries.add(f"ws:{plugin}/{name}")
                elif node.left.id == "method" and plugin in ui_plugins:
                    for comp in node.comparators:
                        for name in _string_literals(comp):
                            if "." in name:
                                entries.add(f"rpc:{plugin}/{name}")
            # 3. Slash 命令注册 (CommandDefinition)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "CommandDefinition"
            ):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    entries.add(f"slash:{node.args[0].value}")
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        entries.add(f"slash:{kw.value.value}")
            # 4. Conversation 接纳快路径 (/stop, /abandon)
            if (
                plugin == "conversation"
                and isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id in ("command", "text", "stripped")
            ):
                for comp in node.comparators:
                    for name in _string_literals(comp):
                        if name.startswith("/") and len(name) > 1 and " " not in name:
                            entries.add(f"slash:{name.lstrip('/')}")
            # 5. Gateway 动态 RPC 注册 RpcMethod.key("...")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "key"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "RpcMethod"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                entries.add(f"gateway:{node.args[0].value}")
    return entries


def validate_parity(actual: set[str], manifest: ParityManifest) -> list[str]:
    """校验集合完整性、互斥性、无通配豁免与无陈旧条目，返回错误列表。"""
    errors: list[str] = []
    mapped = manifest["mapped_to_operation"]
    exempt = manifest["explicit_exemption"]
    backlog_list = manifest["legacy_backlog"]
    backlog = set(backlog_list)

    if len(backlog_list) != len(backlog):
        errors.append("legacy_backlog 存在重复项")

    for entry_id in (*mapped, *exempt, *backlog):
        if "*" in entry_id:
            errors.append(f"禁止通配符条目: {entry_id}")

    for entry_id, info in exempt.items():
        if info["kind"] not in ALLOWED_EXEMPTION_KINDS:
            errors.append(f"豁免类型无效 ({entry_id}): {info['kind']}")
        if not info["reason"].strip():
            errors.append(f"豁免缺少理由: {entry_id}")

    for entry_id, info in mapped.items():
        if not info["operation_id"].strip():
            errors.append(f"映射缺少 operation_id: {entry_id}")

    # 互斥性检查
    overlap_me = set(mapped) & set(exempt)
    overlap_mb = set(mapped) & backlog
    overlap_eb = set(exempt) & backlog
    for label, group in (
        ("mapped 与 exempt 重复", overlap_me),
        ("mapped 与 backlog 重复", overlap_mb),
        ("exempt 与 backlog 重复", overlap_eb),
    ):
        for item in sorted(group):
            errors.append(f"{label}: {item}")

    classified = set(mapped) | set(exempt) | backlog
    for missing in sorted(actual - classified):
        errors.append(f"未归属的用户交互入口: {missing}")
    for stale in sorted(classified - actual):
        errors.append(f"已消失的陈旧清单条目: {stale}")
    return errors


def load_manifest(path: Path) -> ParityManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ParityManifest(
        version=int(raw["version"]),
        mapped_to_operation=dict(raw.get("mapped_to_operation", {})),
        explicit_exemption=dict(raw.get("explicit_exemption", {})),
        legacy_backlog=list(raw.get("legacy_backlog", [])),
    )


def run_self_test(actual: set[str], manifest: ParityManifest) -> int:
    """负例防漏门：注入未登记假入口与通配豁免，验证检查器必报错。"""
    probe = "ws:akashic_clients/__unmapped_probe__"
    err_unmapped = validate_parity(actual | {probe}, manifest)
    if not any(probe in msg for msg in err_unmapped):
        print("SELF-TEST FAILED: 未检出注入的未归属入口", file=sys.stderr)
        return 1
    bad_manifest: ParityManifest = {
        "version": manifest["version"],
        "mapped_to_operation": dict(manifest["mapped_to_operation"]),
        "explicit_exemption": {
            **manifest["explicit_exemption"],
            "rpc:models/*": {"kind": "view_only", "reason": "wildcard probe", "decision": ""},
        },
        "legacy_backlog": list(manifest["legacy_backlog"]),
    }
    err_wildcard = validate_parity(actual, bad_manifest)
    if not any("禁止通配符条目" in msg for msg in err_wildcard):
        print("SELF-TEST FAILED: 未拦截通配符豁免", file=sys.stderr)
        return 1
    print(f"SELF-TEST PASSED: 成功拦截未归属入口 ({probe}) 与通配符豁免")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查用户交互入口与 Self-CLI Operation 的 Parity 覆盖")
    parser.add_argument("command", nargs="?", default="check", choices=("check", "report", "self-test"))
    parser.add_argument("--plugins-dir", type=Path, default=ROOT / "plugins")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    actual = collect_entries(args.plugins_dir)
    manifest = load_manifest(args.manifest)

    if args.command == "self-test":
        return run_self_test(actual, manifest)

    errors = validate_parity(actual, manifest)
    if args.command == "report":
        mapped, exempt, backlog = (
            len(manifest["mapped_to_operation"]),
            len(manifest["explicit_exemption"]),
            len(manifest["legacy_backlog"]),
        )
        print(f"总入口: {len(actual)} | mapped={mapped} | exempt={exempt} | legacy_backlog={backlog}")
        if errors:
            print(f"存在 {len(errors)} 项差异:")
            for err in errors:
                print(f"  - {err}")
            return 1
        return 0

    if errors:
        print(f"Interaction Parity 检查失败（共 {len(errors)} 项）：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(
        f"Interaction Parity 检查通过：总入口={len(actual)} "
        f"(mapped={len(manifest['mapped_to_operation'])}, "
        f"exempt={len(manifest['explicit_exemption'])}, "
        f"legacy_backlog={len(manifest['legacy_backlog'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
