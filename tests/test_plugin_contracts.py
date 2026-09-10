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
ALLOWED_TOP_LEVEL = {"agent.plugin_contracts"}

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
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
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
