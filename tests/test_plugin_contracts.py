"""公开结构合同模块：纯度与导出身份。

`agent/plugin_contracts` 是插件可以依赖的层，因此它必须自己先不依赖实现。
本测试用 AST 静态断言这一点，并在运行时断言它与旧导入路径导出同一对象。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "agent" / "plugin_contracts"

# 词汇表只允许依赖标准库与本层自身。
# 合同层可以依赖 `agent.plugin_composition.model`：它只定义 ServiceKey / CompositionError
# 等纯值身份原语，且自身零仓库内 import（实测）。合同层需要 ServiceKey 才能声明公开 key，
# 而 ServiceKey 只按 name 相等，不引入任何实现依赖。其它 composition 子模块仍不允许。
ALLOWED_TOP_LEVEL = {"agent.plugin_contracts", "agent.plugin_composition.model"}

VOCABULARY_NAMES = (
    "Body",
    "CallRef",
    "ContentPart",
    "ContentReferences",
    "Control",
    "Input",
    "Message",
    "Output",
    "Part",
    "ToolCall",
    "ToolResult",
    "freeze_json",
    "freeze_metadata",
)


def _imported_modules(path: Path) -> set[str]:
    """返回运行时 import 的模块；`if TYPE_CHECKING:` 下的只作类型标注，不算依赖。"""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_type_checking = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if is_type_checking:
            guarded.update(id(child) for child in ast.walk(node))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_contracts_module_has_no_implementation_dependency() -> None:
    """词汇表不得 import 任何项目内实现层。"""

    offenders: dict[str, set[str]] = {}
    for path in sorted(CONTRACTS_DIR.glob("*.py")):
        for module in _imported_modules(path):
            top = module.split(".")[0]
            if top in {"agent", "session", "core", "infra", "bootstrap", "bus", "plugins"}:
                if not any(
                    module == allowed or module.startswith(f"{allowed}.")
                    for allowed in ALLOWED_TOP_LEVEL
                ):
                    offenders.setdefault(path.name, set()).add(module)
    assert offenders == {}, f"结构合同模块出现实现依赖: {offenders}"


def test_session_message_reexports_same_objects() -> None:
    """旧导入路径必须导出同一对象，搬迁不产生第二套类型身份。"""

    import agent.plugin_contracts as contracts
    import session.message as legacy

    for name in VOCABULARY_NAMES:
        assert getattr(legacy, name) is getattr(contracts, name), f"{name} 出现两套身份"


def test_legacy_path_declares_its_public_surface() -> None:
    import session.message as legacy

    for name in VOCABULARY_NAMES:
        assert name in legacy.__all__, f"{name} 未出现在 session.message.__all__"


def test_message_identity_survives_both_paths() -> None:
    """经两条路径构造的对象必须能互相 isinstance 通过。"""

    from datetime import UTC, datetime

    import agent.plugin_contracts as contracts
    import session.message as legacy

    message = legacy.Message(
        message_id="m1",
        session_id="s1",
        seq=0,
        recorded_at=datetime(2026, 9, 10, tzinfo=UTC),
        author="user",
        source="test",
        body=contracts.Input(parts=(legacy.ContentPart(kind="text", value="hi"),)),
    )
    assert isinstance(message, contracts.Message)
    assert isinstance(message.body, contracts.Input)
    assert isinstance(message.body.parts[0], contracts.ContentPart)


def test_legacy_message_codec_shim_exports_every_consumed_name() -> None:
    """再导出必须覆盖所有被消费的名字，包括 yoyo 迁移用的私有 helper。"""

    import session.message_codec as shim

    for name in ("json_value", "body_to_dict", "encode_body", "decode_body", "_unique_fields"):
        assert hasattr(shim, name), f"session.message_codec 缺少 {name}"


def test_legacy_artifacts_shim_exports_every_consumed_name() -> None:
    """再导出必须覆盖所有被消费的名字，包括 Core 内部再导出用的名字。"""

    import session.artifacts as shim

    for name in (
        "AttachmentKind",
        "AttachmentReadLease",
        "AttachmentReadPort",
        "AttachmentRef",
        "check_artifact_id",
    ):
        assert hasattr(shim, name), f"session.artifacts 缺少 {name}"


def test_every_contract_import_resolves() -> None:
    """全库 `from agent.plugin_contracts* import X` 的名字与模块都必须真实存在。

    这是 move & re-export 的兜底：历史上漏过 `_unique_fields`（yoyo 迁移依赖）
    与 `AttachmentReadLease`/`AttachmentReadPort`（composition 依赖），两次都是
    「再导出没按全库被 import 的名字集合来写」。本测试把它变成一次静态全量校验。
    """

    import importlib
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    missing: list[str] = []
    for relative in files:
        try:
            tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("agent.plugin_contracts"):
                continue
            try:
                module = importlib.import_module(node.module)
            except ImportError as error:  # pragma: no cover
                missing.append(f"{relative}: 无法 import {node.module}: {error}")
                continue
            for alias in node.names:
                if not hasattr(module, alias.name):
                    missing.append(f"{relative}: {node.module} 缺少 {alias.name}")
    assert missing == []
