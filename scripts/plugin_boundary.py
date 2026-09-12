#!/usr/bin/env python3
"""插件边界门：把「哪些 import 合法」变成可失败的检查。

本脚本用 AST 静态解析源码，不导入目标模块，因此不触发任何运行副作用。

规则
----
R1  Core 不得 import `plugins.*`，历史迁移也计入精确欠账。
R2  插件导入 Core 只能使用冻结的公开模块清单；目录不自动授予公开资格。
R3  不得导入兄弟插件实现或经自身绝对路径绕过 generation。
R4  Core 文件中的字面 ServiceKey 必须在 `plugin_boundary.toml` 登记角色；
    表中登记的 key 也必须真实存在。
R5  已记录为「文档承诺、代码未实现」的名字必须保持不存在。

R1～R3 的既有欠账记在 `plugin_boundary_baseline.toml` 中，只允许减少。
R4、R5 没有基线；本门不证明角色归属、原子性或运行时隔离。

用法
----
    python scripts/plugin_boundary.py check       # 校验，违规时退出码 1
    python scripts/plugin_boundary.py check --base origin/main  # 禁止新增依赖
    python scripts/plugin_boundary.py baseline    # 只输出待评审账本，不写文件
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "plugin_boundary.toml"
BASELINE_PATH = REPO_ROOT / "plugin_boundary_baseline.toml"

CORE_ROOTS = (
    "agent", "session", "infra", "core", "bootstrap", "bus", "utils",
    "mcp_servers", "host_bridge", "migrations", "memory2", "prompts",
)
CORE_FILES = ("main.py",)
PLUGIN_ROOT = "plugins"

# 冻结既有公开模块，不给目录内未来新增的实现自动授予公开资格。
# 这是兼容清单，不证明其中每个对象已经原子化；扩张须单独评审 owner。
PLUGIN_ALLOWED_MODULES = frozenset({
    "agent.control.context",
    "agent.host_bridge.filesystem",
    "agent.host_bridge.factory",
    "agent.media",
    # Migration callbacks receive this source-neutral execution context from
    # the host; it is not a business implementation or plugin-name exception.
    "agent.plugin_composition",
    "agent.plugin_composition.access",
    "agent.plugin_composition.artifacts",
    "agent.plugin_composition.assets",
    "agent.plugin_composition.bindings",
    "agent.plugin_composition.channels",
    "agent.plugin_composition.claims",
    "agent.plugin_composition.commands",
    "agent.plugin_composition.context",
    "agent.plugin_composition.control_frames",
    "agent.plugin_composition.credentials",
    "agent.plugin_composition.dashboard",
    "agent.plugin_composition.deliveries",
    "agent.plugin_composition.diagnostics",
    "agent.plugin_composition.durable_deliveries",
    "agent.plugin_composition.durable_delivery_store",
    "agent.plugin_composition.effect",
    "agent.plugin_composition.events",
    "agent.plugin_composition.executor",
    "agent.plugin_composition.interaction_undo",
    "agent.plugin_composition.mcp_slots",
    "agent.plugin_composition.messages",
    "agent.plugin_composition.model",
    "agent.plugin_composition.model_settings_http",
    "agent.plugin_composition.models",
    "agent.plugin_composition.overlay",
    "agent.plugin_composition.plugin_updates",
    "agent.plugin_composition.process_slots",
    "agent.plugin_composition.processes",
    "agent.plugin_composition.runtime_lifecycle",
    "agent.plugin_composition.rpc",
    "agent.plugin_composition.archive",
    "agent.plugin_composition.process_runtime",
    "agent.plugin_composition.shell_runtime",
    "agent.plugin_composition.tasks",
    "agent.plugin_composition.timers",
    "agent.tool_catalog",
    "agent.plugin_composition.ui_slots",
    "agent.plugin_composition.workload_slots",
    "agent.plugin_contracts",
    "agent.plugin_contracts.json_store",
    "agent.plugin_contracts.message",
    "agent.plugin_composition.message_view",
    "core.common.diagnostic_log",
    "core.error_context",
    "core.net.http",
    "agent.plugin_contracts.timekit",
    "agent.plugin_contracts.turn_effects",
})

# 插件不得 import 的 core 顶层包（用于 R2 的归属判定）。
CORE_TOP_LEVELS = frozenset(
    {*CORE_ROOTS, "sdk", "main"}
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
    absolute: bool = True

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


def collect_imports(files: list[str], sources: dict[str, str] | None = None) -> list[Import]:
    """静态提取每个文件的 import 目标；语法错误直接失败而不是跳过。"""

    imports: list[Import] = []
    known_files = set(files)
    for rel in files:
        path = REPO_ROOT / rel
        try:
            source = path.read_text(encoding="utf-8") if sources is None else sources[rel]
            tree = ast.parse(source, filename=rel)
        except SyntaxError as error:  # pragma: no cover - 仓库内不应出现
            raise SystemExit(f"无法解析 {rel}: {error}") from error
        imports.extend(_literal_dynamic_imports(rel, tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(Import(rel, alias.name))
            elif isinstance(node, ast.ImportFrom):
                resolved = (
                    _resolve_relative(rel, node.level, node.module)
                    if node.level else node.module
                )
                if resolved:
                    roots = {PLUGIN_ROOT, *CORE_TOP_LEVELS}
                    if resolved not in roots or any(a.name == "*" for a in node.names):
                        imports.append(Import(rel, resolved, not node.level))
                    # 包入口可以把子模块藏在 names 中，不能只检查 node.module。
                    for alias in node.names:
                        target = f"{resolved}.{alias.name}"
                        if alias.name != "*" and (
                            resolved in roots
                            or (target.replace(".", "/") + ".py") in known_files
                            or (target.replace(".", "/") + "/__init__.py") in known_files
                        ):
                            imports.append(Import(rel, target, not node.level))
    return imports


def _literal_dynamic_imports(rel: str, tree: ast.Module) -> list[Import]:
    """检查字面动态导入及导入别名，不执行或推断计算式模块名。"""

    # 1. 只解析明确导入的入口；不做任意赋值、反射或数据流推断。
    loaders = {"__import__"}
    module_loaders: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    module_loaders.add(f"{alias.asname or alias.name}.import_module")
                elif alias.name.startswith("importlib.") and alias.asname is None:
                    module_loaders.add("importlib.import_module")
                elif alias.name == "builtins":
                    loaders.add(f"{alias.asname or alias.name}.__import__")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if node.module == "importlib" and alias.name == "import_module":
                    module_loaders.add(alias.asname or alias.name)
                elif node.module == "builtins" and alias.name == "__import__":
                    loaders.add(alias.asname or alias.name)

    # 2. 字面绝对路径与普通相对 import_module 都进入同一条依赖规则。
    imports: list[Import] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        loader = ast.unparse(node.func)
        if loader not in loaders | module_loaders:
            continue
        keywords = {item.arg: item.value for item in node.keywords}
        name = node.args[0] if node.args else keywords.get("name")
        if not isinstance(name, ast.Constant) or not isinstance(name.value, str):
            continue
        module = name.value
        if loader in module_loaders and module.startswith("."):
            package = node.args[1] if len(node.args) > 1 else keywords.get("package")
            level = len(module) - len(module.lstrip("."))
            if isinstance(package, ast.Name) and package.id == "__package__":
                resolved = _resolve_relative(rel, level, module.lstrip("."))
            elif isinstance(package, ast.Constant) and isinstance(package.value, str):
                resolved = _resolve_relative(package.value.replace(".", "/") + "/_.py", level, module.lstrip("."))
            else:
                continue
            if resolved:
                # 固定 package 字符串仍绕过 generation；只有 __package__ 绑定本代。
                imports.append(Import(rel, resolved, absolute=isinstance(package, ast.Constant)))
        else:
            imports.append(Import(rel, module))
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
        if item.module in PLUGIN_ALLOWED_MODULES:
            continue
        violations.append(item)
    return violations


def check_cross_plugin(imports: list[Import]) -> list[Import]:
    """R3：跨插件实现依赖，以及绕过 generation 的自身绝对导入。"""

    violations: list[Import] = []
    for item in imports:
        if not is_plugin_file(item.importer):
            continue
        own = plugin_package(item.importer)
        target = module_plugin_package(item.module)
        if own is None or item.module.split(".")[0] != PLUGIN_ROOT:
            continue
        if own == target and not item.absolute:
            continue
        violations.append(item)
    return violations


def _is_service_key_call(node: ast.AST, names: set[str], modules: set[str]) -> ast.Call | None:
    """识别 `ServiceKey[T]("name")` 形态，返回该调用节点。"""

    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Subscript):
        base = func.value
    else:
        base = func
    if isinstance(base, ast.Name) and base.id in names:
        return node
    if isinstance(base, ast.Attribute) and base.attr == "ServiceKey" and ast.unparse(base.value) in modules:
        return node
    return None


def discover_service_keys() -> dict[str, str]:
    """扫描字面 ServiceKey 声明（含别名和小写变量），不推断动态 key。"""

    found: dict[str, str] = {}
    for rel in tracked_python_files():
        if not is_core_file(rel):
            continue
        path = REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        names = {"ServiceKey"}
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in node.names if alias.name == "ServiceKey")
            elif isinstance(node, ast.Import):
                modules.update(alias.asname or alias.name for alias in node.names)
        for node in ast.walk(tree):
            call = _is_service_key_call(node, names, modules)
            if call is None:
                continue
            arg = call.args[0] if call.args else next(
                (item.value for item in call.keywords if item.arg == "name"), None,
            )
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
        elif entry["role"] not in {"core", "seam", "bundle", "claim"}:
            errors.append(f"[capabilities.\"{name}\"] 无效 role: {entry['role']}")
    return errors


def implementation_python_files() -> list[str]:
    """只返回实现代码，排除测试。

    R5 断言的是「这些名字没有在实现里出现」。测试为了守护策略表会合法地写出
    这些名字，把 tests/ 计入会把守护测试本身判成违规。
    """

    return [
        rel
        for rel in tracked_python_files()
        if not rel.startswith(("tests/", "tests_scenarios/"))
    ]


def check_phantom_names(policy: dict[str, object]) -> list[str]:
    """R5：文档承诺但未实现的名字必须保持不存在于实现代码。"""

    phantoms = policy.get("phantom", {})
    if not isinstance(phantoms, dict):
        raise SystemExit("plugin_boundary.toml: [phantom] 必须是表")
    if not phantoms:
        return []
    text = "\n".join(
        (REPO_ROOT / rel).read_text(encoding="utf-8")
        for rel in implementation_python_files()
    )
    errors: list[str] = []
    for name, entry in sorted(phantoms.items()):
        # 词边界匹配：避免把长标识符里的子串当成已实现。
        if re.search(rf"\b{re.escape(name)}\b", text):
            note = entry.get("note", "") if isinstance(entry, dict) else ""
            errors.append(
                f"{name} 已在实现代码中出现，但 plugin_boundary.toml 仍把它记为未实现"
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


def import_findings(imports: list[Import]) -> dict[str, list[Import]]:
    """用同一规则扫描当前源码和比较基线，规则修正不伪装成新增依赖。"""

    return {
        "R1": check_core_imports_plugin(imports),
        "R2": check_plugin_deep_core(imports),
        "R3": check_cross_plugin(imports),
    }


def base_findings(base: str) -> dict[str, list[Import]]:
    """只读 Git 快照；不 checkout、不执行基线源码、不接触运行数据。"""

    commit = subprocess.run(
        ["git", "rev-parse", "--verify", f"{base}^{{commit}}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "archive", commit], cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    sources: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
        for member in snapshot:
            if not member.isfile() or not member.name.endswith(SCAN_SUFFIX):
                continue
            if member.name.startswith(SKIP_PREFIXES):
                continue
            handle = snapshot.extractfile(member)
            assert handle is not None
            sources[member.name] = handle.read().decode("utf-8")
    return import_findings(collect_imports(sorted(sources), sources))


def run_check(base: str | None = None) -> int:
    policy = load_policy()
    baseline = load_baseline()
    imports = collect_imports(tracked_python_files())

    findings = import_findings(imports)
    previous = base_findings(base) if base else None

    errors: list[str] = []
    for rule, items in findings.items():
        known = set(baseline.get(rule, []))
        for item in items:
            if item.key in known:
                continue
            errors.append(_format_violation(rule, item))
        # 已还清的债务必须从账本移除，避免账本漂移。
        live = {item.key for item in items}
        if previous is not None:
            for added in sorted(live - {item.key for item in previous[rule]}):
                errors.append(f"{rule}: 相对 {base} 新增依赖（写入账本也不能放行）: {added}")
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
        "（当前/债务基线；不代表插件可独立安装或替换）"
    )
    return 0


def print_baseline() -> int:
    """输出待评审账本，不覆盖文件，也不自动批准新增债务。"""

    findings = import_findings(collect_imports(tracked_python_files()))
    lines = [
        "# 插件边界门债务账本：既有违规的精确清单。",
        "# 由 `python scripts/plugin_boundary.py baseline` 输出，人工评审后更新。",
        "# 只允许减少：新增违规会使门失败，条目还清后必须删除。",
        "",
        "[baseline]",
    ]
    for rule in ("R1", "R2", "R3"):
        keys = sorted({item.key for item in findings[rule]})
        lines.append(f"{rule} = [")
        lines.extend(f"    {json.dumps(key, ensure_ascii=False)}," for key in keys)
        lines.append("]")
    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Akashic 插件边界门")
    parser.add_argument("command", choices=("check", "baseline"))
    parser.add_argument("--base", help="按当前规则比较 Git 基线源码，禁止账本接纳新增依赖")
    args = parser.parse_args()
    return run_check(args.base) if args.command == "check" else print_baseline()


if __name__ == "__main__":
    raise SystemExit(main())
