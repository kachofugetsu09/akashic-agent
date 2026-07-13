"""
Memory2 去重能力基线测试（保留不依赖死逻辑的测试）
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from memory2.memorizer import Memorizer
from memory2.rule_schema import ProcedureRuleSchema, procedure_rules_conflict
from memory2.store import MemoryStore2


class _FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        return list(self._mapping.get(text, [0.0, 0.0, 0.0]))


def test_baseline_exact_hash_prevents_double_write(tmp_path):
    """[PASS] content_hash 去重：完全相同的 summary 写两次，DB 只有一条，reinforcement=2。"""
    store = MemoryStore2(tmp_path / "m.db")
    embedder = _FakeEmbedder({"查 Steam 必须用 steam MCP": [1.0, 0.0]})
    memorizer = Memorizer(store, cast(Any, embedder))

    async def _run():
        await memorizer.save_item(
            summary="查 Steam 必须用 steam MCP",
            memory_type="procedure",
            extra={},
            source_ref="turn1",
        )
        await memorizer.save_item(
            summary="查 Steam 必须用 steam MCP",
            memory_type="procedure",
            extra={},
            source_ref="turn2",
        )

    asyncio.run(_run())

    items = store.list_by_type("procedure")
    assert len(items) == 1, "完全相同内容不应重复写入"
    assert items[0]["reinforcement"] == 2, "重复写入应增加 reinforcement"


def test_baseline_procedure_rules_conflict_pure_logic():
    """[PASS] procedure_rules_conflict 函数正确识别工具方向对立。"""
    # 明确对立
    new: ProcedureRuleSchema = {
        "required_tools": ["steam_mcp"],
        "forbidden_tools": ["web_search"],
        "mentioned_tools": ["steam_mcp", "web_search"],
    }
    old: ProcedureRuleSchema = {
        "required_tools": ["web_search"],
        "forbidden_tools": ["steam_mcp"],
        "mentioned_tools": ["steam_mcp", "web_search"],
    }
    assert procedure_rules_conflict(new, old) is True

    # 同方向（都要求 steam_mcp）
    new2: ProcedureRuleSchema = {
        "required_tools": ["steam_mcp"],
        "forbidden_tools": [],
        "mentioned_tools": ["steam_mcp"],
    }
    old2: ProcedureRuleSchema = {
        "required_tools": ["steam_mcp"],
        "forbidden_tools": [],
        "mentioned_tools": ["steam_mcp"],
    }
    assert procedure_rules_conflict(new2, old2) is False

    # 无工具交集
    new3: ProcedureRuleSchema = {
        "required_tools": ["weather_skill"],
        "forbidden_tools": [],
        "mentioned_tools": ["weather_skill"],
    }
    old3: ProcedureRuleSchema = {
        "required_tools": ["steam_mcp"],
        "forbidden_tools": [],
        "mentioned_tools": ["steam_mcp"],
    }
    assert procedure_rules_conflict(new3, old3) is False
