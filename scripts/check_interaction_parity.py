"""用户界面入口与 Self-CLI operation 的一对一门禁。

两侧独立取证，都从插件源码 AST 采集，不信任运行时自报：
  - 用户入口：HTTP/WS 路由、WebChat WS 帧、Plugin UI RPC 方法、Slash 命令与接纳快路径、Gateway 动态 RPC；
  - Agent 入口：每个 `OperationSpec(operation_id=..., mirrors=(...))` 与 `QueryMirror(...)` 声明。
每个用户入口必须恰好被一个 operation 镜像，或写进 explicit_exemption（禁止通配，须写类型与理由）；
每个 operation 必须镜像至少一个用户入口，或写进 agent_only（须写理由）。
legacy_backlog 只是迁移期冻结欠账，`--strict` 要求为空。
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Literal, TypedDict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts/interaction_parity_manifest.json"
EXTRA_MANIFEST = ROOT / "interaction_parity.extra.json"
ALLOWED_EXEMPTION_KINDS = frozenset({"transport", "view_only", "user_only", "auth_security"})
HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete", "websocket"})
# 契约里的通用注册函数动态拼 OperationSpec，具体声明由调用方的 QueryMirror 提供。
GENERIC_SPEC_FILES = frozenset({"operations/contract.py"})


class ExemptionEntry(TypedDict):
    kind: Literal["transport", "view_only", "user_only", "auth_security"]
    reason: str
    decision: str


class AgentOnlyEntry(TypedDict):
    reason: str


class ParityManifest(TypedDict):
    version: int
    explicit_exemption: dict[str, ExemptionEntry]
    agent_only: dict[str, AgentOnlyEntry]
    legacy_backlog: list[str]


@dataclass(slots=True)
class Surface:
    """两侧采集结果：用户入口集合与 operation → 镜像入口。"""

    entries: set[str] = field(default_factory=set)
    operations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


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


def _keyword(node: ast.Call, name: str, position: int | None = None) -> ast.expr | None:
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    if position is not None and len(node.args) > position:
        return node.args[position]
    return None


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# 用户入口：按协议采集一个 AST 节点里出现的入口 ID。
def _user_entries(node: ast.AST, plugin: str, consts: dict[str, str], ui_plugin: bool) -> list[str]:
    found: list[str] = []
    # 1. FastAPI / Starlette HTTP 与 WebSocket 路由
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in HTTP_VERBS and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("app", "router") and node.args
    ):
        route = _eval_str(node.args[0], consts)
        if route is not None:
            found.append(f"http:{plugin}:{node.func.attr.upper()} {route}")
    # 2. WebSocket 入站帧分派与 Plugin UI RPC method 分派
    if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
        names = [name for comp in node.comparators for name in _string_literals(comp)]
        if node.left.id == "frame_type":
            found.extend(f"ws:{plugin}/{name}" for name in names)
        elif node.left.id == "method" and ui_plugin:
            found.extend(f"rpc:{plugin}/{name}" for name in names if "." in name)
        # 3. Conversation 接纳快路径 (/stop, /abandon)
        elif plugin == "conversation" and node.left.id in ("command", "text", "stripped"):
            found.extend(
                f"slash:{name.lstrip('/')}" for name in names
                if name.startswith("/") and len(name) > 1 and " " not in name
            )
    if isinstance(node, ast.Call):
        name = _call_name(node)
        # 4. Slash 命令注册
        if name == "CommandDefinition":
            value = _keyword(node, "name", 0)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.append(f"slash:{value.value}")
        # 5. Gateway 动态 RPC
        if (
            name == "key" and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "RpcMethod"
            and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
        ):
            found.append(f"gateway:{node.args[0].value}")
    return found


# Agent 入口：OperationSpec / QueryMirror 声明 → (operation_id, mirrors)。
def _operation_decl(node: ast.AST, plugin: str, rel: str, errors: list[str]) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(node, ast.Call):
        return None
    name = _call_name(node)
    if name == "OperationSpec":
        if rel in GENERIC_SPEC_FILES:
            return None
        op_value = _keyword(node, "operation_id", 0)
        if not (isinstance(op_value, ast.Constant) and isinstance(op_value.value, str)):
            errors.append(f"{rel}:{node.lineno} OperationSpec.operation_id 必须是字面量")
            return None
        mirrors_value = _keyword(node, "mirrors")
        mirrors = tuple(_string_literals(mirrors_value)) if mirrors_value is not None else ()
        if mirrors_value is not None and not mirrors and not (
            isinstance(mirrors_value, ast.Tuple) and not mirrors_value.elts
        ):
            errors.append(f"{rel}:{node.lineno} OperationSpec.mirrors 必须是字符串字面量元组")
        return op_value.value, mirrors
    if name == "QueryMirror":
        method = _keyword(node, "method", 0)
        if not (isinstance(method, ast.Constant) and isinstance(method.value, str)):
            errors.append(f"{rel}:{node.lineno} QueryMirror.method 必须是字面量")
            return None
        entry = _keyword(node, "entry")
        entry_id = entry.value if isinstance(entry, ast.Constant) and isinstance(entry.value, str) and entry.value else f"rpc:{plugin}/{method.value}"
        return f"{plugin}.{method.value}", (entry_id,)
    return None


def collect_surface(plugins_dir: Path) -> Surface:
    """从插件目录独立采集两侧入口。"""
    surface = Surface()
    py_files = sorted(plugins_dir.rglob("*.py"))
    ui_plugins = {
        py.relative_to(plugins_dir).parts[0]
        for py in py_files
        if "register_plugin_ui" in py.read_text(encoding="utf-8")
    }
    for py in py_files:
        rel_parts = py.relative_to(plugins_dir).parts
        if len(rel_parts) < 2:
            continue
        plugin, rel = rel_parts[0], "/".join(rel_parts)
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        consts = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        }
        for node in ast.walk(tree):
            surface.entries.update(_user_entries(node, plugin, consts, plugin in ui_plugins))
            decl = _operation_decl(node, plugin, rel, surface.errors)
            if decl is None:
                continue
            op_id, mirrors = decl
            if op_id in surface.operations:
                surface.errors.append(f"operation 重复声明: {op_id}")
            surface.operations[op_id] = mirrors
    return surface


def validate_parity(surface: Surface, manifest: ParityManifest, *, strict: bool = False) -> list[str]:
    """一对一核对：入口恰好一个归宿、operation 必有对应入口或 agent_only 理由。"""
    errors = list(surface.errors)
    exempt, agent_only = manifest["explicit_exemption"], manifest["agent_only"]
    backlog_list = manifest["legacy_backlog"]
    backlog = set(backlog_list)

    # 1. 清单自身：无重复、无通配、理由齐全。
    if len(backlog_list) != len(backlog):
        errors.append("legacy_backlog 存在重复项")
    if strict and backlog:
        errors.append(f"legacy_backlog 必须清零（剩余 {len(backlog)} 项）")
    for entry_id in (*exempt, *backlog, *agent_only):
        if "*" in entry_id:
            errors.append(f"禁止通配符条目: {entry_id}")
    for entry_id, info in exempt.items():
        if info["kind"] not in ALLOWED_EXEMPTION_KINDS:
            errors.append(f"豁免类型无效 ({entry_id}): {info['kind']}")
        if not info["reason"].strip():
            errors.append(f"豁免缺少理由: {entry_id}")
    for op_id, info in agent_only.items():
        if not info["reason"].strip():
            errors.append(f"agent_only 缺少理由: {op_id}")

    # 2. 用户入口 → operation：恰好一个镜像，且不与豁免/欠账重叠。
    mirrored: dict[str, list[str]] = {}
    for op_id, mirrors in surface.operations.items():
        for entry_id in mirrors:
            mirrored.setdefault(entry_id, []).append(op_id)
    for entry_id, ops in sorted(mirrored.items()):
        if len(ops) > 1:
            errors.append(f"入口被多个 operation 镜像（须一对一）: {entry_id} ← {', '.join(sorted(ops))}")
        if entry_id not in surface.entries:
            errors.append(f"operation 镜像了不存在的入口: {', '.join(ops)} → {entry_id}")
        if entry_id in exempt or entry_id in backlog:
            errors.append(f"已镜像入口不应再豁免或列欠账: {entry_id}")
    for label, group in (("exempt 与 backlog 重复", set(exempt) & backlog),):
        errors.extend(f"{label}: {item}" for item in sorted(group))
    classified = set(mirrored) | set(exempt) | backlog
    errors.extend(f"未归属的用户交互入口: {item}" for item in sorted(surface.entries - classified))
    errors.extend(
        f"已消失的陈旧清单条目: {item}" for item in sorted((set(exempt) | backlog) - surface.entries)
    )

    # 3. operation → 用户入口：没有镜像的必须是有理由的 agent_only。
    for op_id, mirrors in sorted(surface.operations.items()):
        if not mirrors and op_id not in agent_only:
            errors.append(f"operation 没有对应的用户入口，也未登记 agent_only: {op_id}")
        if mirrors and op_id in agent_only:
            errors.append(f"已镜像用户入口的 operation 不应登记 agent_only: {op_id}")
    errors.extend(
        f"agent_only 指向不存在的 operation: {op_id}"
        for op_id in sorted(set(agent_only) - set(surface.operations))
    )
    return errors


def _load_one(path: Path) -> ParityManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ParityManifest(
        version=int(raw["version"]),
        explicit_exemption=dict(raw.get("explicit_exemption", {})),
        agent_only=dict(raw.get("agent_only", {})),
        legacy_backlog=list(raw.get("legacy_backlog", [])),
    )


def load_manifest(paths: list[Path]) -> ParityManifest:
    """合并 Core 与 extra 的清单；同一条目在两份清单出现即视为冲突。"""
    merged = ParityManifest(version=2, explicit_exemption={}, agent_only={}, legacy_backlog=[])
    for path in paths:
        part = _load_one(path)
        for key in ("explicit_exemption", "agent_only"):
            clash = set(merged[key]) & set(part[key])
            if clash:
                raise SystemExit(f"清单条目重复 ({path}): {sorted(clash)}")
            merged[key].update(part[key])
        merged["legacy_backlog"].extend(part["legacy_backlog"])
    return merged


def run_self_test(surface: Surface, manifest: ParityManifest) -> int:
    """负例防漏门：每种违规注入一次，检查器必须精确报出。"""
    probe_entry = "ws:akashic_clients/__unmapped_probe__"
    some_entry = next(iter(sorted(surface.entries)), probe_entry)
    cases: list[tuple[str, Surface, ParityManifest, str]] = [
        ("未归属入口", Surface(surface.entries | {probe_entry}, dict(surface.operations)), manifest, probe_entry),
        ("operation 无对应入口", Surface(set(surface.entries), {**surface.operations, "probe.orphan": ()}), manifest, "probe.orphan"),
        ("入口被两个 operation 镜像", Surface(set(surface.entries), {**surface.operations, "probe.a": (some_entry,), "probe.b": (some_entry,)}), manifest, "须一对一"),
        ("通配豁免", surface, {**manifest, "explicit_exemption": {**manifest["explicit_exemption"], "rpc:models/*": {"kind": "view_only", "reason": "probe", "decision": ""}}}, "禁止通配符条目"),
    ]
    for label, probe_surface, probe_manifest, needle in cases:
        if not any(needle in msg for msg in validate_parity(probe_surface, probe_manifest)):
            print(f"SELF-TEST FAILED: 未检出「{label}」", file=sys.stderr)
            return 1
    print(f"SELF-TEST PASSED: {len(cases)} 类违规均被拦截")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查用户界面入口与 Self-CLI operation 一对一")
    parser.add_argument("command", nargs="?", default="check", choices=("check", "report", "self-test"))
    parser.add_argument("--plugins-dir", type=Path, default=ROOT / "plugins")
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--strict", action="store_true", help="要求 legacy_backlog 清零")
    args = parser.parse_args(argv)

    paths: list[Path] = args.manifest or [p for p in (DEFAULT_MANIFEST, EXTRA_MANIFEST) if p.is_file()]
    surface = collect_surface(args.plugins_dir)
    manifest = load_manifest(paths)
    if args.command == "self-test":
        return run_self_test(surface, manifest)

    errors = validate_parity(surface, manifest, strict=args.strict)
    mirrored = {entry for mirrors in surface.operations.values() for entry in mirrors}
    stats = (
        f"用户入口={len(surface.entries)} operation={len(surface.operations)} "
        f"(镜像入口={len(mirrored)}, exempt={len(manifest['explicit_exemption'])}, "
        f"agent_only={len(manifest['agent_only'])}, legacy_backlog={len(manifest['legacy_backlog'])})"
    )
    if args.command == "report":
        for op_id, mirrors in sorted(surface.operations.items()):
            print(f"{op_id}: {', '.join(mirrors) or '(agent_only)'}")
    if errors:
        print(f"Interaction Parity 检查失败（共 {len(errors)} 项）：{stats}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"Interaction Parity 检查通过：{stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
