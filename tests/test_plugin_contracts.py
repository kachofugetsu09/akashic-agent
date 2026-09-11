"""公开结构合同模块：纯度与导出身份。

`agent/plugin_contracts` 是插件可以依赖的层，因此它必须自己先不依赖实现。
本测试用 AST 静态断言这一点，并在运行时断言它与旧导入路径导出同一对象。
"""

from __future__ import annotations

from pathlib import Path

import scripts.plugin_boundary as boundary

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "agent" / "plugin_contracts"

# 词汇表只允许依赖标准库与本层自身。
VALUE_LIBRARIES = {"__future__", "collections", "dataclasses", "datetime", "json", "math", "types", "typing"}

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


def test_contracts_module_has_no_implementation_dependency() -> None:
    """检查嵌套模块与相对导入；I/O 库和第三方实现不能自动进入值合同。"""

    offenders: dict[str, set[str]] = {}
    files = [str(path.relative_to(REPO_ROOT)) for path in CONTRACTS_DIR.rglob("*.py")]
    for item in boundary.collect_imports(files):
        module = item.module
        if module == "agent.plugin_contracts" or module.startswith("agent.plugin_contracts."):
            continue
        if module.split(".")[0] not in VALUE_LIBRARIES:
            offenders.setdefault(item.importer, set()).add(module)
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
