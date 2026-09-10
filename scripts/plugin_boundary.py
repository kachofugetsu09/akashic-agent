#!/usr/bin/env python3
"""插件边界门：把「哪些 import 合法」变成可失败的检查。

本脚本用 AST 静态解析源码，不导入目标模块，因此不触发任何运行副作用。

规则
----
R1  Core 不得 import `plugins.*`（把产品功能留在 core 之外）。
R2  `plugins/**` 只能 import 受支持的公开面（`agent.plugin_composition`、
    `agent.plugin_contracts`）；其余 core 内部深路径都算违规。
R3  插件之间只能经由公开结构合同连接，不得 import 对方实现模块。
R4  每个 Core 拥有的 ServiceKey 必须在 `plugin_boundary.toml` 登记角色；
    表中登记的 key 也必须真实存在。
R5  已记录为「文档承诺、代码未实现」的名字必须保持不存在。

R1～R3 的既有欠账记在 `plugin_boundary_baseline.toml` 中，只允许减少。
R4、R5 没有基线：新增 Core 能力必须同时登记角色，否则本门失败。

用法
----
    python scripts/plugin_boundary.py check       # 校验，违规时退出码 1
    python scripts/plugin_boundary.py baseline    # 用当前状态重写债务账本
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "plugin_boundary.toml"
BASELINE_PATH = REPO_ROOT / "plugin_boundary_baseline.toml"

CORE_ROOTS = ("agent", "session", "infra", "core", "bootstrap", "bus", "utils", "mcp_servers")
CORE_FILES = ("main.py",)
PLUGIN_ROOT = "plugins"

# 插件可以依赖的公开面。顺序即优先级，前缀匹配。
PLUGIN_ALLOWED_PREFIXES = (
    "agent.plugin_composition",
    "agent.plugin_contracts",
)

# 插件不得 import 的 core 顶层包（用于 R2 的归属判定）。
CORE_TOP_LEVELS = frozenset(
    {"agent", "session", "infra", "core", "bootstrap", "bus", "utils", "mcp_servers", "types", "sdk"}
)

SCAN_SUFFIX = ".py"
SKIP_PREFIXES = (
    "akashic-plugin/",
    "plugin_packages/",
    "node_modules/",
    "vendor/",
)


@dataclass(frozen=True, order=True)
class Import:
    """一次静态 import 事实。"""

    importer: str
    module: str

    @property
    def key(self) -> str:
        return f"{self.importer}|{self.module}"


def tracked_python_files() -> list[str]:
    """返回仓库跟踪的 Python 文件，排除外部 checkout 与生成的插件包。"""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files: list[str] = []
    for raw in result.stdout.split("\0"):
        if not raw or not raw.endswith(SCAN_SUFFIX):
            continue
        if raw.startswith(SKIP_PREFIXES):
            continue
        files.append(raw)
    return sorted(files)


def _resolve_relative(importer: str, level: int, module: str | None) -> str | None:
    """把相对 import 解析成绝对模块名；越出仓库的返回 None。"""

    parts = importer.split("/")
    package = parts[:-1]  # 文件所在目录
    drop = level - 1
    if drop > len(package):
        return None
    base = package[: len(package) - drop] if drop else package
    if module:
        base = [*base, *module.split(".")]
    return ".".join(base) if base else None


def collect_imports(files: list[str]) -> list[Import]:
    """静态提取每个文件的 import 目标；语法错误直接失败而不是跳过。"""

    imports: list[Import] = []
    for rel in files:
        path = REPO_ROOT / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as error:  # pragma: no cover - 仓库内不应出现
            raise SystemExit(f"无法解析 {rel}: {error}") from error
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(Import(rel, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    resolved = _resolve_relative(rel, node.level, node.module)
                    if resolved:
                        imports.append(Import(rel, resolved))
                elif node.module:
                    imports.append(Import(rel, node.module))
    return imports


def is_core_file(rel: str) -> bool:
    return rel in CORE_FILES or rel.startswith(tuple(f"{root}/" for root in CORE_ROOTS))


def is_plugin_file(rel: str) -> bool:
    return rel.startswith(f"{PLUGIN_ROOT}/")


def plugin_package(rel: str) -> str | None:
    """`plugins/tools/plugin.py` → `plugins.tools`。"""

    parts = rel.split("/")
    if len(parts) < 2 or parts[0] != PLUGIN_ROOT:
        return None
    return f"{PLUGIN_ROOT}.{parts[1]}"


def module_plugin_package(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != PLUGIN_ROOT:
        return None
    return f"{PLUGIN_ROOT}.{parts[1]}"


def check_core_imports_plugin(imports: list[Import]) -> list[Import]:
    """R1：core 直接 import 插件。"""

    return [
        item
        for item in imports
        if is_core_file(item.importer) and item.module.split(".")[0] == PLUGIN_ROOT
    ]


def check_plugin_deep_core(imports: list[Import]) -> list[Import]:
    """R2：插件 import core 内部深路径。"""

    violations: list[Import] = []
    for item in imports:
        if not is_plugin_file(item.importer):
            continue
        top = item.module.split(".")[0]
        if top not in CORE_TOP_LEVELS:
            continue
        if any(
            item.module == prefix or item.module.startswith(f"{prefix}.")
            for prefix in PLUGIN_ALLOWED_PREFIXES
        ):
            continue
        violations.append(item)
    return violations


def check_cross_plugin(imports: list[Import]) -> list[Import]:
    """R3：插件 import 兄弟插件的实现模块。"""

    violations: list[Import] = []
    for item in imports:
        if not is_plugin_file(item.importer):
            continue
        own = plugin_package(item.importer)
        target = module_plugin_package(item.module)
        if own is None or target is None or own == target:
            continue
        violations.append(item)
    return violations


def _is_service_key_call(node: ast.AST) -> ast.Call | None:
    """识别 `ServiceKey[T]("name")` 形态，返回该调用节点。"""

    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Subscript):
        base = func.value
    else:
        base = func
    if isinstance(base, ast.Name) and base.id == "ServiceKey":
        return node
    return None


def discover_service_keys() -> dict[str, str]:
    """扫描 `X = ServiceKey[...]("<name>")`，返回 name → 定义文件。"""

    found: dict[str, str] = {}
    for rel in tracked_python_files():
        if not is_core_file(rel):
            continue
        path = REPO_ROOT / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
                value: ast.expr | None = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            else:
                continue
            if len(targets) != 1 or value is None:
                continue
            target = targets[0]
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            call = _is_service_key_call(value)
            if call is None or not call.args:
                continue
            arg = call.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found[arg.value] = rel
    return found


def load_policy() -> dict[str, object]:
    with POLICY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def check_capability_table(policy: dict[str, object]) -> list[str]:
    """R4：角色表与代码中的 Core ServiceKey 双向一致。"""

    declared = policy.get("capabilities", {})
    if not isinstance(declared, dict):
        raise SystemExit("plugin_boundary.toml: [capabilities] 必须是表")
    actual = discover_service_keys()
    errors: list[str] = []
    for name in sorted(actual):
        if name not in declared:
            errors.append(
                f"未登记角色的 Core ServiceKey: {name}（{actual[name]}）；"
                "请在 plugin_boundary.toml 的 [capabilities] 声明 role"
            )
    for name in sorted(declared):
        if name not in actual:
            errors.append(
                f"plugin_boundary.toml 登记的 ServiceKey 在代码中不存在: {name}"
            )
    for name, entry in sorted(declared.items()):
        if not isinstance(entry, dict) or not entry.get("role"):
            errors.append(f"[capabilities.\"{name}\"] 缺少 role")
    return errors


def check_phantom_names(policy: dict[str, object]) -> list[str]:
    """R5：文档承诺但未实现的名字必须保持不存在。"""

    phantoms = policy.get("phantom", {})
    if not isinstance(phantoms, dict):
        raise SystemExit("plugin_boundary.toml: [phantom] 必须是表")
    if not phantoms:
        return []
    text = "\n".join(
        (REPO_ROOT / rel).read_text(encoding="utf-8") for rel in tracked_python_files()
    )
    errors: list[str] = []
    for name, entry in sorted(phantoms.items()):
        # 词边界匹配：避免把长标识符里的子串当成已实现。
        if re.search(rf"\b{re.escape(name)}\b", text):
            note = entry.get("note", "") if isinstance(entry, dict) else ""
            errors.append(
                f"{name} 已在代码中出现，但 plugin_boundary.toml 仍把它记为未实现"
                f"（{note}）；实现后请更新文档并把该条从 [phantom] 移除"
            )
    return errors


def load_baseline() -> dict[str, list[str]]:
    if not BASELINE_PATH.exists():
        return {}
    with BASELINE_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    result: dict[str, list[str]] = {}
    for rule, entries in data.get("baseline", {}).items():
        if isinstance(entries, list):
            result[rule] = [str(item) for item in entries]
    return result


def _format_violation(rule: str, item: Import) -> str:
    return f"{rule}: {item.importer} imports {item.module}"


def run_check() -> int:
    policy = load_policy()
    baseline = load_baseline()
    imports = collect_imports(tracked_python_files())

    findings: dict[str, list[Import]] = {
        "R1": check_core_imports_plugin(imports),
        "R2": check_plugin_deep_core(imports),
        "R3": check_cross_plugin(imports),
    }

    errors: list[str] = []
    for rule, items in findings.items():
        known = set(baseline.get(rule, []))
        for item in items:
            if item.key in known:
                continue
            errors.append(_format_violation(rule, item))
        # 已还清的债务必须从账本移除，避免账本漂移。
        live = {item.key for item in items}
        for stale in sorted(known - live):
            errors.append(
                f"baseline-{rule}: 账本条目已不再是违规，请从 plugin_boundary_baseline.toml 删除: {stale}"
            )

    errors.extend(check_capability_table(policy))
    errors.extend(check_phantom_names(policy))

    if errors:
        print("插件边界门未通过：", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    counts = {rule: len({item.key for item in items}) for rule, items in findings.items()}
    print(
        "插件边界门通过："
        f"R1={counts['R1']}/{len(baseline.get('R1', []))} "
        f"R2={counts['R2']}/{len(baseline.get('R2', []))} "
        f"R3={counts['R3']}/{len(baseline.get('R3', []))} "
        "（当前/债务基线）"
    )
    return 0


def write_baseline() -> int:
    imports = collect_imports(tracked_python_files())
    findings = {
        "R1": check_core_imports_plugin(imports),
        "R2": check_plugin_deep_core(imports),
        "R3": check_cross_plugin(imports),
    }
    lines = [
        "# 插件边界门债务账本：既有违规的精确清单。",
        "# 由 `python scripts/plugin_boundary.py baseline` 生成。",
        "# 只允许减少：新增违规会使门失败，条目还清后必须删除。",
        "",
        "[baseline]",
    ]
    for rule in ("R1", "R2", "R3"):
        keys = sorted({item.key for item in findings[rule]})
        lines.append(f"{rule} = [")
        lines.extend(f'    "{key}",' for key in keys)
        lines.append("]")
    BASELINE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {BASELINE_PATH.relative_to(REPO_ROOT)}："
          + ", ".join(f"{rule}={len(findings[rule])}" for rule in findings))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Akashic 插件边界门")
    parser.add_argument("command", choices=("check", "baseline"))
    args = parser.parse_args()
    return run_check() if args.command == "check" else write_baseline()


if __name__ == "__main__":
    raise SystemExit(main())
