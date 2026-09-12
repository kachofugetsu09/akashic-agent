from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest

from plugins.akasha._boundaries import CONTENT as AKASHA_CONTENT
from plugins.akasha._boundaries import Result as AkashaResult
from plugins.akasha._boundaries import TOOLS as AKASHA_TOOLS
from plugins.akasha.recall_tool import RecallTool
from plugins.akasha.tools import FeedbackTool
from plugins.compaction._boundaries import CONTEXT, MATERIALS
from plugins.markdown_memory._boundaries import (
    COMPACTION_READER, COMPACTION_SUMMARIES, CONTENT as MARKDOWN_CONTENT,
)
from agent.plugin_contracts import ContentPart


ROOT = Path(__file__).parents[1]
OWNED = (
    ROOT / "plugins/akasha",
    ROOT / "plugins/markdown_memory",
    ROOT / "plugins/prompt",
    ROOT / "plugins/compaction",
)


def test_memory_plugins_have_no_cross_plugin_imports() -> None:
    imports: list[str] = []
    for directory in OWNED:
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names if alias.name.startswith("plugins."))
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("plugins."):
                    imports.append(node.module)
    assert imports == []


def test_owned_service_keys_use_current_abi() -> None:
    assert AKASHA_CONTENT.name == "content.v2"
    assert MARKDOWN_CONTENT.name == "content.v2"
    assert AKASHA_TOOLS.name == "tools.v1"
    assert CONTEXT.name == "context.v2"
    assert MATERIALS.name == "context.materials.v3"
    assert COMPACTION_SUMMARIES.name == "compaction.summaries.v1"
    assert COMPACTION_READER.name == "compaction.reader.v1"


@pytest.mark.asyncio
async def test_tool_argument_rejections_are_strings() -> None:
    feedback = FeedbackTool("remember", cast(object, None), cast(object, None), lambda: {})
    assert isinstance(await feedback.prepare({}), str)

    recall = RecallTool(
        memory=Path("/tmp/akasha-test-memory"), legacy_index=None, config=cast(object, None),
        catalog=cast(object, None), embeddings=cast(object, None), bindings=cast(object, None),
        select_learning=lambda: ("learning", "embedding"), records=cast(object, None),
        open_embedding=cast(object, None), max_chars=1,
    )
    assert isinstance(await recall.prepare({"query": "  "}), str)


def test_local_result_is_structural() -> None:
    result = AkashaResult("success", (ContentPart("text", "ok"),))
    assert result.outcome == "success"
    assert result.parts[0].value == "ok"
