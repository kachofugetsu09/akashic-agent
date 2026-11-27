"""用户界面入口与 Self-CLI operation 的一对一门禁。

两侧独立取证，都从插件源码 AST 采集，不信任运行时自报：
  - 用户入口：HTTP/WS 路由、WebChat WS 帧、Plugin UI RPC 方法、Slash 命令与接纳快路径、Gateway 动态 RPC；
  - Agent 入口：每个 `OperationSpec(operation_id=..., mirrors=(...))` 与 `QueryMirror(...)` 声明。
每个用户入口必须恰好被一个 operation 镜像，或写进 explicit_exemption（禁止通配，须写类型与理由）；
每个 operation 必须镜像至少一个用户入口，或写进 agent_only（须写理由）。
模型工具（`TOOLS.register(name=..., parameters=...)`）绕开 self 直接给模型用，每个都要写进 model_tools 并写明理由，
防止界面上的动作改写成孤立工具后 self 里没有（#2287）。
没有欠账清单：新入口在同一个 PR 里镜像或豁免，否则 CI 失败。
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import cast, Literal, TypedDict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts/interaction_parity_manifest.json"
EXTRA_MANIFEST = ROOT / "interaction_parity.extra.json"
ALLOWED_EXEMPTION_KINDS = frozenset({"transport", "view_only", "user_only", "auth_security"})
HTTP_VERBS = frozenset({"get", "post", "put", "patch", "delete", "websocket"})
# 多动作端点的判别字段名。
DISCRIMINATOR_FIELDS = frozenset({"type", "kind"})
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
    model_tools: dict[str, AgentOnlyEntry]


_DYNAMIC_RPC = "dynamic-rpc:"


@dataclass(slots=True)
class Surface:
    """两侧采集结果：用户入口集合与 operation → 镜像入口。"""

    entries: set[str] = field(default_factory=set)
    operations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    rpc_tables: set[str] = field(default_factory=set)
    tools: set[str] = field(default_factory=set)


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


# Slash 命令入口：无子动作为 slash:<name>；有子动作时每个子动作一个 slash:<name>#<sub>。
def _slash_entries(name: str, subs: ast.expr | None, errors: list[str]) -> list[str]:
    if subs is None:
        return [f"slash:{name}"]
    names = _string_literals(subs)
    if not isinstance(subs, ast.Tuple) or len(names) != len(subs.elts):
        errors.append(f"/{name} 的 CommandDefinition.subcommands 必须是字符串字面量元组")
    return [f"slash:{name}#{sub}" for sub in names] or [f"slash:{name}"]


# `app.get("/x")` / `router.post("/x")` 这类路由注册调用 → (动词, 调用节点)。
def _route_call(node: ast.AST) -> tuple[str, ast.Call] | None:
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in HTTP_VERBS and isinstance(node.func.value, ast.Name)
        and node.func.value.id in ("app", "router") and node.args
    ):
        return node.func.attr.upper(), node
    return None


# 判别联合 `Annotated[A | B, Field(discriminator="type")]`：别名 → 各成员类判别字段的 Literal 值。
def _discriminated_unions(tree: ast.Module, classes: dict[str, ast.ClassDef]) -> dict[str, set[str]]:
    unions: dict[str, set[str]] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call) and _call_name(n) == "Field"]
        fields = [v.value for c in calls if isinstance(v := _keyword(c, "discriminator"), ast.Constant)]
        field_name = next((f for f in fields if f in DISCRIMINATOR_FIELDS), None)
        if field_name is None:
            continue
        members = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name) and n.id in classes}
        unions[node.targets[0].id] = {
            value for member in members for stmt in classes[member].body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == field_name
            for literal in ast.walk(stmt.annotation) if isinstance(literal, ast.Subscript)
            and isinstance(literal.value, ast.Name) and literal.value.id == "Literal"
            for value in _string_literals(literal.slice)
        }
    return unions


def _discriminator_read(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Subscript):
        return isinstance(expr.slice, ast.Constant) and expr.slice.value in DISCRIMINATOR_FIELDS
    if isinstance(expr, ast.Attribute):
        return expr.attr in DISCRIMINATOR_FIELDS
    return (
        isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "get"
        and bool(expr.args) and isinstance(expr.args[0], ast.Constant) and expr.args[0].value in DISCRIMINATOR_FIELDS
    )


# 多动作端点：路由处理函数按 type/kind 判别字段分派子动作时，入口按值展开为 `<路由>#<值>`。
# 判别来源两种：处理函数里对判别字段的字面量比较；处理函数（经同模块类的基类/泛型实参）用到的判别联合。
# 只出现一个值是校验分支（如「只能置顶会话」），不算子动作。
def _route_actions(tree: ast.Module) -> dict[int, list[str]]:
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    unions = _discriminated_unions(tree, classes)

    def union_values(name: str, seen: set[str]) -> set[str]:
        if name in unions or name in seen or name not in classes:
            return set(unions.get(name, ()))
        seen.add(name)
        bases = [n.id for base in classes[name].bases for n in ast.walk(base) if isinstance(n, ast.Name)]
        return {value for base in bases for value in union_values(base, seen)}

    actions: dict[int, list[str]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        routes = [route[1] for d in fn.decorator_list if (route := _route_call(d)) is not None]
        if not routes:
            continue
        values = {
            value for node in ast.walk(fn) if isinstance(node, ast.Compare) and _discriminator_read(node.left)
            for comp in node.comparators for value in _string_literals(comp)
        }
        values |= {value for node in ast.walk(fn) if isinstance(node, ast.Name) for value in union_values(node.id, set())}
        for call in routes:
            actions[id(call)] = sorted(values) if len(values) > 1 else []
    return actions


# 用户入口：按协议采集一个 AST 节点里出现的入口 ID。
def _user_entries(
    node: ast.AST, plugin: str, consts: dict[str, str], ui_plugin: bool, actions: dict[int, list[str]],
    errors: list[str], tables: dict[str, list[str]] | None = None,
) -> list[str]:
    tables = tables or {}
    found: list[str] = []
    # 1. FastAPI / Starlette HTTP 与 WebSocket 路由；多动作端点只计子动作，不再计父入口。
    if (route_call := _route_call(node)) is not None:
        verb, call = route_call
        route = _eval_str(call.args[0], consts)
        if route is not None:
            entry = f"http:{plugin}:{verb} {route}"
            found.extend([f"{entry}#{value}" for value in actions.get(id(call), ())] or [entry])
    # 2. WebSocket 入站帧分派与 Plugin UI RPC method 分派
    if isinstance(node, ast.Compare):
        names = [name for comp in node.comparators for name in _string_literals(comp)]
        left = node.left.id if isinstance(node.left, ast.Name) else None
        if left == "frame_type":
            found.extend(f"ws:{plugin}/{name}" for name in names)
        elif left == "method" and ui_plugin:
            found.extend(f"rpc:{plugin}/{name}" for name in names if "." in name)
        # 3. Conversation 接纳快路径：任何与 "/xxx" 字面量的比较都是不经命令目录的文本命令（如 /stop）
        elif plugin == "conversation":
            found.extend(
                f"slash:{name.lstrip('/')}" for name in names
                if name.startswith("/") and len(name) > 1 and " " not in name
            )
    if isinstance(node, ast.Call):
        name = _call_name(node)
        # 4. Slash 命令注册；声明了 subcommands 的命令按子动作逐个计数。
        if name == "CommandDefinition":
            value = _keyword(node, "name", 0)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.extend(_slash_entries(value.value, _keyword(node, "subcommands"), errors))
        # 5. Gateway 动态 RPC：provide(RpcMethod.key(<字面量或模块常量>))；变量名由 rpc_methods() 表另行采集。
        if name == "provide" and node.args and isinstance(key := node.args[0], ast.Call) and (
            isinstance(key.func, ast.Attribute) and key.func.attr == "key"
            and isinstance(key.func.value, ast.Name) and key.func.value.id == "RpcMethod" and key.args
        ):
            # 只认大写模块常量：小写名多半是循环变量（与插件清单里的 name 等同名会误解析）。
            arg = key.args[0]
            method = None if isinstance(arg, ast.Name) and not arg.id.isupper() else _eval_str(arg, consts)
            if method is not None:
                found.append(f"gateway:{method}")
            elif not isinstance(key.args[0], ast.Name):
                errors.append(f"{plugin}: RpcMethod.key 的参数无法静态解析（line {node.lineno}）")
            else:
                found.append(f"{_DYNAMIC_RPC}{plugin}:{node.lineno}")
    # 6. `def rpc_methods(...)` 返回的方法表：字典字面量的键就是 Gateway 方法名。
    if isinstance(node, ast.FunctionDef) and node.name == "rpc_methods":
        names = _rpc_table(node, tables)
        if names is None:
            errors.append(f"{plugin}: rpc_methods() 须直接返回字典字面量，或对模块级字典字面量 .items() 的推导（line {node.lineno}）")
        found.extend(f"gateway:{name}" for name in names or ())
    return found


# rpc_methods() 本层（不进嵌套函数）的 return：字典字面量的键，或 `{k: ... for k, v in TABLE.items()}` 里 TABLE 的键。
def _rpc_table(fn: ast.FunctionDef, tables: dict[str, list[str]]) -> list[str] | None:
    pending: list[ast.AST] = list(fn.body)
    names: list[str] = []
    while pending:
        item = pending.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(item, ast.Return) and item.value is not None:
            value = item.value
            if isinstance(value, ast.Dict) and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys):
                names += [cast(str, cast(ast.Constant, k).value) for k in value.keys]
                continue
            gen = value.generators[0].iter if isinstance(value, ast.DictComp) and len(value.generators) == 1 else None
            if (isinstance(gen, ast.Call) and isinstance(gen.func, ast.Attribute) and gen.func.attr == "items"
                    and isinstance(gen.func.value, ast.Name) and gen.func.value.id in tables):
                names += tables[gen.func.value.id]
                continue
            return None
        pending.extend(ast.iter_child_nodes(item))
    return names


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
    if name == "RpcMirror":
        method = _keyword(node, "method", 0)
        if not (isinstance(method, ast.Constant) and isinstance(method.value, str)):
            errors.append(f"{rel}:{node.lineno} RpcMirror.method 必须是字面量")
            return None
        return f"{plugin}.{method.value}", (f"gateway:{method.value}",)
    if name == "QueryMirror":
        method = _keyword(node, "method", 0)
        if not (isinstance(method, ast.Constant) and isinstance(method.value, str)):
            errors.append(f"{rel}:{node.lineno} QueryMirror.method 必须是字面量")
            return None
        entry = _keyword(node, "entry")
        entry_id = entry.value if isinstance(entry, ast.Constant) and isinstance(entry.value, str) and entry.value else f"rpc:{plugin}/{method.value}"
        return f"{plugin}.{method.value}", (entry_id,)
    return None


# 模型工具注册：带 name= 与 parameters= 的 .register(...) → tool:<插件>/<名字>；名字不是字面量时记源码表达式。
def _tool_decl(node: ast.AST, plugin: str, consts: dict[str, str]) -> str | None:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register"):
        return None
    keywords = {kw.arg: kw.value for kw in node.keywords}
    name = keywords.get("name")
    if name is None or "parameters" not in keywords:
        return None
    literal = _eval_str(name, consts)
    return f"tool:{plugin}/{literal if literal is not None else ast.unparse(name)}"


# operation 的命令路径（字面量元组）；拿不到时返回 None，由运行期注册兜底。
def _op_path(node: ast.Call) -> tuple[str, ...] | None:
    name = _call_name(node)
    value = _keyword(node, "path", 1) if name in ("QueryMirror", "RpcMirror") else _keyword(node, "path") \
        if name == "OperationSpec" else None
    if not isinstance(value, ast.Tuple):
        return None
    parts = _string_literals(value)
    return tuple(parts) if len(parts) == len(value.elts) else None


def collect_surface(plugins_dir: Path) -> Surface:
    """从插件目录独立采集两侧入口。"""
    surface = Surface()
    paths: dict[tuple[str, ...], str] = {}
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
        actions = _route_actions(tree)
        if any(isinstance(n, ast.FunctionDef) and n.name == "rpc_methods" for n in tree.body):
            surface.rpc_tables.add(plugin)
        tables = {
            node.target.id: [cast(str, cast(ast.Constant, k).value) for k in node.value.keys]
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and isinstance(node.value, ast.Dict)
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.value.keys)
        } | {
            node.targets[0].id: [cast(str, cast(ast.Constant, k).value) for k in node.value.keys]
            for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Dict)
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.value.keys)
        }
        for node in ast.walk(tree):
            surface.entries.update(_user_entries(node, plugin, consts, plugin in ui_plugins, actions, surface.errors, tables))
            tool = _tool_decl(node, plugin, consts)
            if tool is not None:
                surface.tools.add(tool)
            decl = _operation_decl(node, plugin, rel, surface.errors)
            if decl is None:
                continue
            op_id, mirrors = decl
            if op_id in surface.operations:
                surface.errors.append(f"operation 重复声明: {op_id}")
            surface.operations[op_id] = mirrors
            path = _op_path(cast(ast.Call, node))
            if path is not None:
                owner = paths.setdefault(path, op_id)
                if owner != op_id:
                    surface.errors.append(f"命令路径重名: {' '.join(path)} ← {owner}, {op_id}")
    # 变量名注册的 Gateway 方法须由同插件的 rpc_methods() 表给出名字，否则无法证明一对一。
    dynamic = {e for e in surface.entries if e.startswith(_DYNAMIC_RPC)}
    surface.entries -= dynamic
    for marker in sorted(dynamic):
        plugin = marker[len(_DYNAMIC_RPC):].rsplit(":", 1)[0]
        if plugin not in surface.rpc_tables:
            surface.errors.append(f"{plugin}: RpcMethod.key(变量) 注册的方法须写在 def rpc_methods() 的字典字面量里")
    return surface


def validate_parity(surface: Surface, manifest: ParityManifest) -> list[str]:
    """一对一核对：入口恰好一个归宿、operation 必有对应入口或 agent_only 理由。"""
    errors = list(surface.errors)
    exempt, agent_only = manifest["explicit_exemption"], manifest["agent_only"]

    # 1. 清单自身：无重复、无通配、理由齐全。
    for entry_id in (*exempt, *agent_only):
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

    # 2. 用户入口 → operation：恰好一个镜像，且不与豁免重叠。
    mirrored: dict[str, list[str]] = {}
    for op_id, mirrors in surface.operations.items():
        for entry_id in mirrors:
            mirrored.setdefault(entry_id, []).append(op_id)
    for entry_id, ops in sorted(mirrored.items()):
        if len(ops) > 1:
            errors.append(f"入口被多个 operation 镜像（须一对一）: {entry_id} ← {', '.join(sorted(ops))}")
        if entry_id not in surface.entries:
            errors.append(f"operation 镜像了不存在的入口: {', '.join(ops)} → {entry_id}")
        if entry_id in exempt:
            errors.append(f"已镜像入口不应再豁免: {entry_id}")
    classified = set(mirrored) | set(exempt)
    errors.extend(f"未归属的用户交互入口: {item}" for item in sorted(surface.entries - classified))
    errors.extend(
        f"已消失的陈旧清单条目: {item}" for item in sorted(set(exempt) - surface.entries)
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

    # 4. 模型工具：每个都要登记理由（为什么不经 self），清单里不能留已删掉的工具。
    tools = manifest["model_tools"]
    for tool_id in sorted(surface.tools):
        if tool_id not in tools:
            errors.append(f"模型工具未登记 model_tools（界面动作应收进 self 的 operation）: {tool_id}")
        elif not tools[tool_id]["reason"].strip():
            errors.append(f"model_tools 缺少理由: {tool_id}")
    errors.extend(f"model_tools 指向不存在的工具: {tool_id}" for tool_id in sorted(set(tools) - surface.tools))
    return errors


def _load_one(path: Path) -> ParityManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("legacy_backlog"):
        raise SystemExit(f"清单不再接受 legacy_backlog（{path}）：入口须镜像为 operation 或写明豁免")
    return ParityManifest(
        version=int(raw["version"]),
        explicit_exemption=dict(raw.get("explicit_exemption", {})),
        agent_only=dict(raw.get("agent_only", {})),
        model_tools=dict(raw.get("model_tools", {})),
    )


def load_manifest(paths: list[Path]) -> ParityManifest:
    """合并 Core 与 extra 的清单；同一条目在两份清单出现即视为冲突。"""
    merged = ParityManifest(version=3, explicit_exemption={}, agent_only={}, model_tools={})
    for path in paths:
        part = _load_one(path)
        clash = (set(merged["explicit_exemption"]) & set(part["explicit_exemption"])) | (
            set(merged["agent_only"]) & set(part["agent_only"])) | (set(merged["model_tools"]) & set(part["model_tools"]))
        if clash:
            raise SystemExit(f"清单条目重复 ({path}): {sorted(clash)}")
        merged["explicit_exemption"].update(part["explicit_exemption"])
        merged["agent_only"].update(part["agent_only"])
        merged["model_tools"].update(part["model_tools"])
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
        ("未登记模型工具", Surface(set(surface.entries), dict(surface.operations), tools=surface.tools | {"tool:probe/orphan"}), manifest, "tool:probe/orphan"),
    ]
    for label, probe_surface, probe_manifest, needle in cases:
        if not any(needle in msg for msg in validate_parity(probe_surface, probe_manifest)):
            print(f"SELF-TEST FAILED: 未检出「{label}」", file=sys.stderr)
            return 1
    print(f"SELF-TEST PASSED: {len(cases)} 类违规均被拦截")
    return 0


GUIDE = "docs/design/self-cli-plugin-authoring.md"


# 给一个未归属入口写出可直接粘贴的修法：镜像为 operation 是默认，豁免只在四类之内。
def scaffold(entry_id: str) -> str:
    kind, _, rest = entry_id.partition(":")
    plugin, method = (rest.split("/", 1) + [""])[:2] if kind in ("rpc", "ws") else (rest.split(":", 1)[0], "")
    path = " ".join(f'"{part}"' for part in (method or rest).replace("/", ".").split(".") if part)
    if kind == "rpc":
        code = (f'QueryMirror("{method}", ({path.replace(" ", ", ")},), "<一句话说明>", (OptionSpec(...),))\n'
                f"# 加进本插件的 MIRRORS，并在 apply 里软挂载：\n"
                f"# await ctx.inject((OPERATIONS_V1,), lambda child: register_query_mirrors(child, child.require(OPERATIONS_V1), query=query, mirrors=MIRRORS), name=\"operations\")")
    else:
        code = (f'OperationSpec(operation_id="{plugin}.<资源>.<动词>", path=(<命令路径>), summary="<一句话说明>",\n'
                f'              handler=ctx.entrypoint(<与界面入口共用的领域函数>), mirrors=("{entry_id}",), options=(...))')
    exemption = (f'"{entry_id}": {{"kind": "transport|view_only|user_only|auth_security", '
                 f'"reason": "<为什么助手不需要或不能做>", "decision": "<ADR 编号，可空>"}}')
    return (f"入口 {entry_id}\n"
            f"  默认：镜像为 operation（界面与助手调同一段领域代码）：\n    {code.replace(chr(10), chr(10) + '    ')}\n"
            f"  或：确属四类豁免时写进清单 explicit_exemption：\n    {exemption}\n"
            f"  详见 {GUIDE}")


# 运行期对账：实例里注册的每个 operation 都必须能被静态采集看到，否则它的镜像声明逃过了门禁。
def check_runtime(surface: Surface, workspace: Path) -> int:
    import asyncio
    from plugins.gateway.cli import request

    result = asyncio.run(request(workspace, "self/invoke", {"argv": ["describe"]}))
    data = result.get("data")
    rows = cast(list[dict[str, object]], data.get("operations", [])) if isinstance(data, dict) else []
    live = {str(row["operation_id"]) for row in rows}
    unseen = sorted(live - set(surface.operations))
    absent = sorted(set(surface.operations) - live)
    print(f"运行期 operation={len(live)}，静态声明={len(surface.operations)}，本实例未启用={len(absent)}")
    for op_id in unseen:
        print(f"  - 运行期有、静态采集看不到: {op_id}", file=sys.stderr)
    return 1 if unseen else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查用户界面入口与 Self-CLI operation 一对一")
    parser.add_argument("command", nargs="?", default="check", choices=("check", "report", "self-test", "runtime", "scaffold"))
    parser.add_argument("entries", nargs="*", help="scaffold：要生成修法的入口 ID；省略时取全部未归属入口")
    parser.add_argument("--plugins-dir", type=Path, default=ROOT / "plugins")
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--workspace", type=Path, help="runtime：运行中实例的 workspace，经 Gateway 读 self describe")
    args = parser.parse_args(argv)

    paths: list[Path] = args.manifest or [p for p in (DEFAULT_MANIFEST, EXTRA_MANIFEST) if p.is_file()]
    surface = collect_surface(args.plugins_dir)
    manifest = load_manifest(paths)
    if args.command == "self-test":
        return run_self_test(surface, manifest)
    if args.command == "runtime":
        if args.workspace is None:
            parser.error("runtime 需要 --workspace")
        return check_runtime(surface, args.workspace)

    errors = validate_parity(surface, manifest)
    mirrored = {entry for mirrors in surface.operations.values() for entry in mirrors}
    stats = (
        f"用户入口={len(surface.entries)} operation={len(surface.operations)} "
        f"(镜像入口={len(mirrored)}, exempt={len(manifest['explicit_exemption'])}, "
        f"agent_only={len(manifest['agent_only'])}, 模型工具={len(surface.tools)})"
    )
    if args.command == "report":
        for op_id, mirrors in sorted(surface.operations.items()):
            print(f"{op_id}: {', '.join(mirrors) or '(agent_only)'}")
    unowned = [err.split(": ", 1)[1] for err in errors if err.startswith("未归属的用户交互入口: ")]
    if args.command == "scaffold":
        for entry_id in args.entries or unowned:
            print(scaffold(entry_id) + "\n")
        return 0
    if errors:
        print(f"Interaction Parity 检查失败（共 {len(errors)} 项）：{stats}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        if unowned:
            print("\n新入口须在同一个 PR 里让助手也能做到。可直接参考：\n", file=sys.stderr)
            for entry_id in unowned[:5]:
                print(scaffold(entry_id) + "\n", file=sys.stderr)
            print("全部修法：python scripts/check_interaction_parity.py scaffold", file=sys.stderr)
        return 1
    print(f"Interaction Parity 检查通过：{stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
