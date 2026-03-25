"""
Memory2 去重能力基线测试

目的：在实施 dedup_decider / batch-internal-dedup 优化前，记录现有系统的真实能力边界。
对照组：运行这些测试，观察哪些通过、哪些体现了当前局限。

标注约定：
  # [PASS]      现有能力，优化后仍应通过
  # [BOUNDARY]  当前边界问题，注释说明优化后断言应如何变化

运行方式：
  pytest tests/test_memory2_dedup_baseline.py -v
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from memory2.memorizer import Memorizer
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.retriever import Retriever
from memory2.rule_schema import procedure_rules_conflict, resolve_procedure_rule_schema
from memory2.store import MemoryStore2


# ─── 测试基础设施 ──────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """受控向量空间：通过精确向量控制 cosine 相似度，不依赖真实 embedding API。"""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping

    async def embed(self, text: str) -> list[float]:
        # 未预设文本返回全零向量（不命中任何条目）
        return list(self._mapping.get(text, [0.0, 0.0, 0.0]))


class _SilentProvider:
    """LLM 占位：总返回空数组，让 worker 走保守路径（不 supersede）。"""

    async def chat(self, **kwargs: Any):
        class _R:
            content = "[]"

        return _R()


def _make_worker(
    store: MemoryStore2,
    embedder: _FakeEmbedder,
    *,
    provider=None,
    score_threshold: float = 0.45,
) -> PostResponseMemoryWorker:
    retriever = Retriever(store, cast(Any, embedder), score_threshold=score_threshold)
    memorizer = Memorizer(store, cast(Any, embedder))
    return PostResponseMemoryWorker(
        memorizer=cast(Any, memorizer),
        retriever=cast(Any, retriever),
        light_provider=cast(Any, provider or _SilentProvider()),
        light_model="test",
    )


# ─── A. 现有能力验证 ───────────────────────────────────────────────────────────


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


def test_baseline_tool_conflict_procedure_superseded(tmp_path):
    """[PASS] 工具方向对立的 procedure 触发 supersede。

    旧：查 Steam 必须用 web_search，禁止 steam_mcp
    新：查 Steam 必须用 steam_mcp，禁止 web_search
    → 明确对立，旧条目应被 supersede。
    """
    store = MemoryStore2(tmp_path / "m.db")
    # 旧条目向量 [1.0, 0.0]；新候选向量 [0.98, 0.1]，相似度足够触发 supersede 检查
    embedder = _FakeEmbedder(
        {
            "查 Steam 信息必须直接使用 web_search，不能先用 steam_mcp": [1.0, 0.0],
            "查 Steam 必须先用 steam_mcp，不能用 web_search": [0.98, 0.1],
        }
    )
    worker = _make_worker(store, embedder)

    # 预先写入旧条目
    store.upsert_item(
        memory_type="procedure",
        summary="查 Steam 信息必须直接使用 web_search，不能先用 steam_mcp",
        embedding=[1.0, 0.0],
        extra={
            "rule_schema": {
                "required_tools": ["web_search"],
                "forbidden_tools": ["steam_mcp"],
                "mentioned_tools": ["steam_mcp", "web_search"],
            }
        },
    )
    old_id = store.list_by_type("procedure")[0]["id"]

    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": "查 Steam 必须先用 steam_mcp，不能用 web_search",
                "memory_type": "procedure",
                "tool_requirement": "steam_mcp",
                "steps": [],
                "rule_schema": {
                    "required_tools": ["steam_mcp"],
                    "forbidden_tools": ["web_search"],
                    "mentioned_tools": ["steam_mcp", "web_search"],
                },
            },
            "test@ref",
            token_budget=256,
        )
    )

    rows = store._db.execute(
        "SELECT id, status FROM memory_items WHERE id=?", (old_id,)
    ).fetchone()
    assert rows[1] == "superseded", "方向对立的旧 procedure 应被 supersede"

    # list_by_type 不过滤 status，需用直接查询验证 active 数量
    active = store._db.execute(
        "SELECT id FROM memory_items WHERE memory_type='procedure' AND status='active'"
    ).fetchall()
    assert len(active) == 1, "supersede 后只有新条目是 active"


def test_baseline_unrelated_procedure_not_superseded(tmp_path):
    """[PASS] 与新候选无工具交集的旧条目不会被 supersede（避免误杀）。

    旧：查天气用 weather_skill
    新：查 Steam 用 steam_mcp
    → 主题不同，旧条目保留。
    """
    store = MemoryStore2(tmp_path / "m.db")
    embedder = _FakeEmbedder(
        {
            "查天气时必须使用 weather_skill": [1.0, 0.0, 0.0],
            "查 Steam 信息必须先用 steam_mcp": [0.0, 1.0, 0.0],  # 完全不同语义
        }
    )
    worker = _make_worker(store, embedder)

    store.upsert_item(
        memory_type="procedure",
        summary="查天气时必须使用 weather_skill",
        embedding=[1.0, 0.0, 0.0],
        extra={
            "rule_schema": {
                "required_tools": ["weather_skill"],
                "forbidden_tools": [],
                "mentioned_tools": ["weather_skill"],
            }
        },
    )
    weather_id = store.list_by_type("procedure")[0]["id"]

    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": "查 Steam 信息必须先用 steam_mcp",
                "memory_type": "procedure",
                "tool_requirement": "steam_mcp",
                "steps": [],
                "rule_schema": {
                    "required_tools": ["steam_mcp"],
                    "forbidden_tools": [],
                    "mentioned_tools": ["steam_mcp"],
                },
            },
            "test@ref",
            token_budget=256,
        )
    )

    row = store._db.execute(
        "SELECT status FROM memory_items WHERE id=?", (weather_id,)
    ).fetchone()
    assert row[0] == "active", "无关旧条目不应被 supersede"


def test_baseline_procedure_rules_conflict_pure_logic():
    """[PASS] procedure_rules_conflict 函数正确识别工具方向对立。"""
    # 明确对立
    new = {"required_tools": ["steam_mcp"], "forbidden_tools": ["web_search"], "mentioned_tools": ["steam_mcp", "web_search"]}
    old = {"required_tools": ["web_search"], "forbidden_tools": ["steam_mcp"], "mentioned_tools": ["steam_mcp", "web_search"]}
    assert procedure_rules_conflict(new, old) is True

    # 同方向（都要求 steam_mcp）
    new2 = {"required_tools": ["steam_mcp"], "forbidden_tools": [], "mentioned_tools": ["steam_mcp"]}
    old2 = {"required_tools": ["steam_mcp"], "forbidden_tools": [], "mentioned_tools": ["steam_mcp"]}
    assert procedure_rules_conflict(new2, old2) is False

    # 无工具交集
    new3 = {"required_tools": ["weather_skill"], "forbidden_tools": [], "mentioned_tools": ["weather_skill"]}
    old3 = {"required_tools": ["steam_mcp"], "forbidden_tools": [], "mentioned_tools": ["steam_mcp"]}
    assert procedure_rules_conflict(new3, old3) is False


# ─── B. 当前边界问题记录 ───────────────────────────────────────────────────────


def test_boundary_same_turn_synonym_procedures_both_written(tmp_path):
    """[BOUNDARY] 同轮两个同义 procedure 候选当前都会写入（批内无去重）。

    场景：用户说"查 Steam 要用 MCP"，light model 提取了两条措辞不同但含义相同的 procedure。
    两条的向量高度相似（cos ≈ 0.9999），rule_schema 同方向（都是 required=steam_mcp）。

    当前行为：两条都写入 DB（因为 rule_schema 不冲突，LLM 判断为不需要 supersede）。

    优化后预期（batch-internal dedup, threshold=0.90）：
      仅写入第一条，第二条因批内向量相似度 > 0.90 被跳过。
      → 断言应改为 assert len(items) == 1
    """
    store = MemoryStore2(tmp_path / "m.db")
    # 两条候选的向量高度相似
    syn_1 = "查 Steam 时必须先用 steam MCP 工具获取数据"
    syn_2 = "访问 Steam 信息时需要优先调用 steam MCP，不能直接搜索"
    embedder = _FakeEmbedder(
        {
            syn_1: [1.0, 0.01, 0.0],
            syn_2: [0.999, 0.02, 0.0],  # cosine ≈ 0.9999
        }
    )
    worker = _make_worker(store, embedder)
    # LLM 保守路径：不 supersede
    worker._check_supersede = AsyncMock(return_value=([], 600))

    # 模拟 run() 内部循环（跳过 LLM 提取，直接给两个同义候选）
    async def _simulate_two_synonym_saves():
        budget = 1000
        for summary in (syn_1, syn_2):
            budget = await worker._save_with_supersede(
                {
                    "summary": summary,
                    "memory_type": "procedure",
                    "tool_requirement": "steam_mcp",
                    "steps": [],
                    "rule_schema": {
                        "required_tools": ["steam_mcp"],
                        "forbidden_tools": [],
                        "mentioned_tools": ["steam_mcp"],
                    },
                },
                "test@ref",
                token_budget=budget,
            )

    asyncio.run(_simulate_two_synonym_saves())

    items = store.list_by_type("procedure")

    # ── CURRENT BEHAVIOR ──
    assert len(items) == 2, (
        "CURRENT: 同轮同义候选均写入（批内无向量去重）。"
        " 优化后（batch-internal dedup）此断言应改为 assert len(items) == 1"
    )


def test_boundary_partial_conflict_triggers_full_supersede(tmp_path):
    """[BOUNDARY] 部分工具冲突会导致整条旧 procedure 被 supersede（而非 merge）。

    场景：
      旧条目是"查 Steam 的完整 3 步流程"，包含 tool_a 和 tool_b（两者都 required）。
      新候选只是"改用 tool_c 替代 tool_a"，tool_b 的步骤仍然有效。

      procedure_rules_conflict：new.forbidden=[tool_a] ∩ old.required=[tool_a, tool_b] → 冲突（True）
      → 整条旧记忆被 supersede，tool_b 的有效信息随之丢失。

    当前行为：旧条目被整条 supersede（tool_b 信息丢失）。

    优化后预期（dedup_decider + merge）：
      LLM 识别为 partial conflict → NONE+merge → 旧条目更新（保留 tool_b）。
      → 断言应改为 row[0] == 'active'（旧条目被更新，而非 superseded）
    """
    store = MemoryStore2(tmp_path / "m.db")
    old_summary = "查询 Steam 完整流程：第一步用 tool_a 获取 App ID；第二步用 tool_b 获取评价；第三步汇总"
    new_summary = "用户纠正：查 Steam 评价不要用 tool_a，改用 tool_c"

    embedder = _FakeEmbedder(
        {
            old_summary: [1.0, 0.0, 0.0],
            new_summary: [0.95, 0.15, 0.0],  # 相似但不完全相同
        }
    )
    worker = _make_worker(store, embedder)

    # 写入旧条目（有 tool_a 和 tool_b 两个 required）
    store.upsert_item(
        memory_type="procedure",
        summary=old_summary,
        embedding=[1.0, 0.0, 0.0],
        extra={
            "rule_schema": {
                "required_tools": ["tool_a", "tool_b"],
                "forbidden_tools": [],
                "mentioned_tools": ["tool_a", "tool_b", "tool_c"],
            }
        },
    )
    old_id = store.list_by_type("procedure")[0]["id"]

    # 新候选：只更新 tool_a → tool_c，tool_b 不涉及
    asyncio.run(
        worker._save_with_supersede(
            {
                "summary": new_summary,
                "memory_type": "procedure",
                "tool_requirement": "tool_c",
                "steps": [],
                "rule_schema": {
                    "required_tools": ["tool_c"],
                    "forbidden_tools": ["tool_a"],   # ← 和旧条目 required=[tool_a] 冲突
                    "mentioned_tools": ["tool_a", "tool_c"],
                },
            },
            "test@ref",
            token_budget=256,
        )
    )

    row = store._db.execute(
        "SELECT status FROM memory_items WHERE id=?", (old_id,)
    ).fetchone()

    # ── CURRENT BEHAVIOR ──
    assert row[0] == "superseded", (
        "CURRENT: 部分工具冲突导致整条旧记忆被 supersede（tool_b 信息丢失）。"
        " 优化后（dedup_decider merge）此断言应改为 assert row[0] == 'active'"
        "（旧条目被 merge 更新而非整条退休）。"
    )


def test_boundary_same_preference_different_wording_both_survive(tmp_path):
    """[BOUNDARY] 同义 preference（措辞不同）当前都会写入。

    场景：两轮对话都触发了对"回复风格简洁"的 preference 提取，但措辞稍有不同。
    内容 hash 不同（措辞不同），rule_schema 为空（preference），向量相似度高。

    当前行为：
      - 第一轮写入"回复要简洁，不要长篇大论"
      - 第二轮写入"回复请保持简短，避免冗长"
      - 两条都在 DB 中，redundant 信息积累

    优化后预期（dedup_decider SKIP 或 NONE+merge）：
      第二轮候选被识别为同义 → skip 或 merge 进第一条。
      → 断言应改为 assert len(items) == 1
    """
    store = MemoryStore2(tmp_path / "m.db")
    pref_1 = "回复要简洁，不要长篇大论"
    pref_2 = "回复请保持简短，避免冗长"
    embedder = _FakeEmbedder(
        {
            pref_1: [1.0, 0.0, 0.0],
            pref_2: [0.97, 0.05, 0.0],  # cosine ≈ 0.988
        }
    )
    worker = _make_worker(store, embedder)
    # LLM 保守路径（不 supersede）
    worker._check_supersede = AsyncMock(return_value=([], 600))

    async def _two_turns():
        budget = 1000
        for pref in (pref_1, pref_2):
            budget = await worker._save_with_supersede(
                {"summary": pref, "memory_type": "preference", "tool_requirement": None, "steps": []},
                "test@ref",
                token_budget=budget,
            )

    asyncio.run(_two_turns())

    items = store.list_by_type("preference")

    # ── CURRENT BEHAVIOR ──
    assert len(items) == 2, (
        "CURRENT: 同义 preference（措辞不同）两条都写入（无跨轮语义去重）。"
        " 优化后（dedup_decider SKIP/merge）此断言应改为 assert len(items) == 1"
    )


def test_boundary_conflicting_preference_not_auto_superseded(tmp_path):
    """[BOUNDARY] 方向对立的 preference 不会自动 supersede（需要 LLM 判断）。

    场景：
      旧：用户喜欢回复加 emoji
      新：用户说不要在回复里加 emoji

    procedure_rules_conflict 对 preference 类型不检查（只对 procedure），
    所以对立的 preference 需要 LLM _check_supersede 判断。
    若 token_budget 耗尽，LLM 调用被跳过，旧条目就不会被 supersede。

    当前行为（token_budget 充足时）：进入 LLM 判断；budget 耗尽时：保守不 supersede。
    这里测 budget=0 的退化路径：旧条目保留（可能留下矛盾记忆）。

    优化后（dedup_decider 有独立判断路径）：
      即使 budget 为 0，也应通过 dedup_decider 的结构化决策处理。
    """
    store = MemoryStore2(tmp_path / "m.db")
    old_pref = "用户喜欢回复里加 emoji，比较生动"
    new_pref = "用户明确说不要在回复里加 emoji"
    embedder = _FakeEmbedder(
        {
            old_pref: [1.0, 0.0],
            new_pref: [0.96, 0.1],
        }
    )
    worker = _make_worker(store, embedder)

    store.upsert_item(
        memory_type="preference",
        summary=old_pref,
        embedding=[1.0, 0.0],
        extra={},
    )
    old_id = store.list_by_type("preference")[0]["id"]

    # token_budget=0 → _check_supersede 被跳过
    asyncio.run(
        worker._save_with_supersede(
            {"summary": new_pref, "memory_type": "preference", "tool_requirement": None, "steps": []},
            "test@ref",
            token_budget=0,  # 预算耗尽
        )
    )

    row = store._db.execute(
        "SELECT status FROM memory_items WHERE id=?", (old_id,)
    ).fetchone()

    # ── CURRENT BEHAVIOR (budget=0 退化路径) ──
    # preference 类型无 rule_schema 冲突检测，budget=0 时 LLM 被跳过，旧条目保留
    assert row[0] == "active", (
        "CURRENT: token_budget=0 时对立 preference 不被 supersede（LLM 调用被跳过）。"
        " 优化后（dedup_decider 独立 budget）：即使主 budget 为 0 也应正确处理。"
    )
