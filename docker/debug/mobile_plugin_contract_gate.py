from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "docker" / "debug" / "mobile-plugin-release.lock.json"
DEFAULT_REPORT = ROOT / "docker" / "debug" / "reports" / "mobile-plugin-contract" / "gate.json"
UI_CONTRACT_RUNNER = ROOT / "docker" / "debug" / "mobile_plugin_ui_contract.mjs"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
ALLOWED_SLOTS = {
    "turn.before_reasoning",
    "turn.before_tool",
    "turn.after_answer",
    "drawer.panel",
}


@dataclass(frozen=True)
class PluginContract:
    id: str
    repository: str
    commit: str
    plugin_class: str
    module: str
    stylesheet: str
    navigation: bool
    slots: tuple[str, ...]
    query_methods: tuple[str, ...]
    node_test: str
    node_setup: str


@dataclass(frozen=True)
class PluginEvidence:
    id: str
    repository: str
    commit: str
    module_sha256: str
    module_bytes: int
    stylesheet_sha256: str
    stylesheet_bytes: int
    query_methods: tuple[str, ...]
    node_test: str


def main() -> None:
    """在干净临时目录验证核心与固定插件提交的移动 UI 发布合同。"""

    # 1. 读取发布锁并审计当前核心合同
    args = _parse_args()
    lock_path = args.lock.resolve()
    contracts = _load_lock(lock_path)
    core_status = _git_output(ROOT, "status", "--porcelain").splitlines()
    if args.require_clean_core and core_status:
        raise RuntimeError(f"核心工作树不干净: {core_status}")
    _verify_core_contract()

    # 2. 从公开仓库拉取精确提交并运行跨仓库合同
    evidence: list[PluginEvidence] = []
    with tempfile.TemporaryDirectory(prefix="akashic-mobile-plugin-contract-") as raw_temp:
        temp_root = Path(raw_temp)
        for contract in contracts:
            checkout = temp_root / contract.id
            _checkout_locked_commit(contract, checkout)
            evidence.append(_verify_plugin(contract, checkout))

    # 3. 只把可复核摘要写到忽略版本控制的报告目录
    report = {
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "core_head": _git_output(ROOT, "rev-parse", "HEAD"),
        "core_dirty_status": core_status,
        "lock": str(lock_path.relative_to(ROOT)),
        "plugins": [asdict(item) for item in evidence],
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"mobile plugin contract gate passed: {report_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证固定插件提交的移动 UI 合同")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--require-clean-core", action="store_true")
    return parser.parse_args()


def _load_lock(path: Path) -> tuple[PluginContract, ...]:
    """严格解析发布锁，不接受缺字段、额外字段或模糊版本。"""

    # 1. 校验锁文件根结构
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "plugins"}:
        raise ValueError("移动插件发布锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的移动插件发布锁版本: {raw['schema_version']}")
    plugins = raw["plugins"]
    if not isinstance(plugins, list) or not plugins:
        raise ValueError("移动插件发布锁必须包含插件")

    # 2. 把每个外部依赖收敛成完整、不可变合同
    contracts = tuple(_parse_contract(item) for item in plugins)
    ids = [item.id for item in contracts]
    if len(ids) != len(set(ids)):
        raise ValueError("移动插件发布锁包含重复 id")
    return contracts


def _parse_contract(raw: object) -> PluginContract:
    expected = {
        "id",
        "repository",
        "commit",
        "plugin_class",
        "module",
        "stylesheet",
        "navigation",
        "slots",
        "query_methods",
        "node_test",
        "node_setup",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"移动插件合同字段无效: {raw}")
    item = cast(dict[str, object], raw)
    strings = {
        name: _required_string(item, name)
        for name in (
            "id",
            "repository",
            "commit",
            "plugin_class",
            "module",
            "stylesheet",
            "node_test",
            "node_setup",
        )
    }
    if not strings["repository"].startswith("https://github.com/"):
        raise ValueError(f"插件仓库必须是公开 GitHub HTTPS 地址: {strings['repository']}")
    if COMMIT_PATTERN.fullmatch(strings["commit"]) is None:
        raise ValueError(f"插件 commit 必须是完整 SHA: {strings['commit']}")
    navigation = item["navigation"]
    if not isinstance(navigation, bool):
        raise ValueError(f"插件 navigation 必须是布尔值: {strings['id']}")
    slots = _string_tuple(item, "slots")
    if len(slots) != len(set(slots)) or any(slot not in ALLOWED_SLOTS for slot in slots):
        raise ValueError(f"插件 slots 无效: {strings['id']}")
    methods = _string_tuple(item, "query_methods")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError(f"插件 query_methods 无效: {strings['id']}")
    if strings["node_setup"] not in {"none", "npm-ci"}:
        raise ValueError(f"插件 node_setup 无效: {strings['id']}")
    for name in ("module", "stylesheet", "node_test"):
        _require_relative_path(strings[name])
    return PluginContract(
        id=strings["id"],
        repository=strings["repository"],
        commit=strings["commit"],
        plugin_class=strings["plugin_class"],
        module=strings["module"],
        stylesheet=strings["stylesheet"],
        navigation=navigation,
        slots=slots,
        query_methods=methods,
        node_test=strings["node_test"],
        node_setup=strings["node_setup"],
    )


def _verify_core_contract() -> None:
    """确认核心仍是异步 provider 调度同步插件 handler。"""

    # 1. 基类只声明同步插件边界
    base_tree = ast.parse((ROOT / "agent" / "plugins" / "base.py").read_text(encoding="utf-8"))
    plugin_class = _class_node(base_tree, "Plugin")
    base_query = _method_node(plugin_class, "mobile_ui_query")
    if not isinstance(base_query, ast.FunctionDef):
        raise RuntimeError("Plugin.mobile_ui_query 必须保持同步")
    _verify_query_signature(base_query, owner="Plugin")

    # 2. provider 对外异步，内部必须经 executor 调用同步 handler
    provider_tree = ast.parse(
        (ROOT / "agent" / "plugins" / "mobile_ui.py").read_text(encoding="utf-8")
    )
    provider_class = _class_node(provider_tree, "PluginMobileUiProvider")
    provider_query = _method_node(provider_class, "query")
    run_query = _method_node(provider_class, "_run_query")
    if not isinstance(provider_query, ast.AsyncFunctionDef) or not isinstance(
        run_query, ast.AsyncFunctionDef
    ):
        raise RuntimeError("PluginMobileUiProvider.query 必须保持异步")
    calls = [node for node in ast.walk(run_query) if isinstance(node, ast.Call)]
    if not any(_attribute_name(call.func) == "run_in_executor" for call in calls):
        raise RuntimeError("核心 mobile UI query 未经线程池隔离同步插件")
    if not any(_attribute_name(call.func) == "mobile_ui_query" for call in calls):
        raise RuntimeError("核心 mobile UI provider 未调用插件 mobile_ui_query")


def _checkout_locked_commit(contract: PluginContract, checkout: Path) -> None:
    """只获取发布锁声明的公开 Git 对象。"""

    # 1. 创建不复用宿主缓存的临时仓库
    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", contract.repository), cwd=checkout)

    # 2. 精确获取并检出完整 SHA
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", contract.commit),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    if _git_output(checkout, "rev-parse", "HEAD") != contract.commit:
        raise RuntimeError(f"插件检出提交与发布锁不一致: {contract.id}")
    if _git_output(checkout, "status", "--porcelain"):
        raise RuntimeError(f"插件检出后工作树不干净: {contract.id}")


def _verify_plugin(contract: PluginContract, checkout: Path) -> PluginEvidence:
    """验证插件 Python ABI、移动模块 ABI 与仓库自带行为测试。"""

    # 1. Python 静态合同必须与核心同步 handler 完全对齐
    plugin_source = (checkout / "plugin.py").read_text(encoding="utf-8")
    tree = ast.parse(plugin_source)
    plugin_class = _class_node(tree, contract.plugin_class)
    query = _method_node(plugin_class, "mobile_ui_query")
    if not isinstance(query, ast.FunctionDef):
        raise RuntimeError(f"插件 mobile_ui_query 必须保持同步: {contract.id}")
    _verify_query_signature(query, owner=contract.id)
    literals = {
        node.value
        for node in ast.walk(query)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    missing_methods = set(contract.query_methods) - literals
    if missing_methods:
        raise RuntimeError(f"插件查询方法与发布锁不一致: {contract.id} {sorted(missing_methods)}")
    _verify_mobile_ui_declaration(plugin_class, contract)

    # 2. 核心拥有的 JS ABI runner 与插件自有行为测试都要通过
    module_path = _inside(checkout, contract.module)
    stylesheet_path = _inside(checkout, contract.stylesheet)
    node_test = _inside(checkout, contract.node_test)
    module = module_path.read_bytes()
    stylesheet = stylesheet_path.read_bytes()
    if not module or not stylesheet:
        raise RuntimeError(f"插件移动资源不能为空: {contract.id}")
    if len(module) + len(stylesheet) > 240 * 1024:
        raise RuntimeError(f"插件移动资源超过核心预算: {contract.id}")
    _run(
        (
            "node",
            str(UI_CONTRACT_RUNNER),
            str(module_path),
            str(contract.navigation).lower(),
            json.dumps(contract.slots),
        ),
        cwd=checkout,
    )
    if contract.node_setup == "npm-ci":
        _run(("npm", "ci", "--ignore-scripts"), cwd=checkout)
    _run(("node", "--test", str(node_test)), cwd=checkout)

    # 3. 报告只记录不可变来源与内容摘要
    return PluginEvidence(
        id=contract.id,
        repository=contract.repository,
        commit=contract.commit,
        module_sha256=hashlib.sha256(module).hexdigest(),
        module_bytes=len(module),
        stylesheet_sha256=hashlib.sha256(stylesheet).hexdigest(),
        stylesheet_bytes=len(stylesheet),
        query_methods=contract.query_methods,
        node_test=contract.node_test,
    )


def _verify_query_signature(method: ast.FunctionDef, *, owner: str) -> None:
    positional = [argument.arg for argument in method.args.posonlyargs + method.args.args]
    keywords = [argument.arg for argument in method.args.kwonlyargs]
    if (
        positional != ["self", "method", "payload"]
        or keywords != ["session_id", "turn_id"]
        or method.args.vararg is not None
        or method.args.kwarg is not None
    ):
        raise RuntimeError(f"mobile_ui_query 签名漂移: {owner}")


def _verify_mobile_ui_declaration(
    plugin_class: ast.ClassDef,
    contract: PluginContract,
) -> None:
    method = _method_node(plugin_class, "mobile_ui")
    if not isinstance(method, ast.FunctionDef):
        raise RuntimeError(f"插件 mobile_ui 声明无效: {contract.id}")
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    contribution = next(
        (call for call in calls if _attribute_name(call.func) == "MobileUiContribution"),
        None,
    )
    if contribution is None:
        raise RuntimeError(f"插件缺少 MobileUiContribution: {contract.id}")
    keywords = {item.arg: item.value for item in contribution.keywords if item.arg is not None}
    module = _literal_string(keywords.get("module"))
    stylesheet = _literal_string(keywords.get("stylesheet"))
    slots = _literal_strings(keywords.get("slots")) if "slots" in keywords else ()
    navigation = "navigation" in keywords
    if (
        module != contract.module
        or stylesheet != contract.stylesheet
        or slots != contract.slots
        or navigation != contract.navigation
    ):
        raise RuntimeError(f"插件 mobile_ui 声明与发布锁不一致: {contract.id}")


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"类定义数量无效: {name}")
    return matches[0]


def _method_node(
    class_node: ast.ClassDef,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"方法定义数量无效: {class_node.name}.{name}")
    return matches[0]


def _attribute_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _literal_string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_strings(node: ast.expr | None) -> tuple[str, ...]:
    if not isinstance(node, ast.Tuple):
        raise RuntimeError("插件 mobile_ui slots 必须是字面量 tuple")
    values = tuple(_literal_string(item) for item in node.elts)
    if any(value is None for value in values):
        raise RuntimeError("插件 mobile_ui slots 必须是字符串字面量")
    return cast(tuple[str, ...], values)


def _required_string(item: dict[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"移动插件合同 {name} 必须是非空字符串")
    return value


def _string_tuple(item: dict[str, object], name: str) -> tuple[str, ...]:
    value = item[name]
    if not isinstance(value, list) or any(not isinstance(entry, str) or not entry for entry in value):
        raise ValueError(f"移动插件合同 {name} 必须是非空字符串数组")
    return tuple(cast(list[str], value))


def _require_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"移动插件合同路径必须留在插件仓库内: {value}")


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"插件合同路径逃逸临时仓库: {relative}")
    return path


def _run(command: tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _git_output(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
