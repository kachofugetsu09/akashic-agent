"""
Memory2 检索排序能力基线测试

目的：记录 vector_search 当前的排序行为（纯 cosine），以及 hotness 优化后的预期变化。

标注约定：
  # [PASS]      现有能力，优化后仍应通过
  # [BOUNDARY]  当前不足，注释说明优化后断言应如何变化

运行方式：
  pytest tests/test_memory2_retrieval_baseline.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from memory2.store import MemoryStore2


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _days_ago(n: int) -> str:
    return (_now() - timedelta(days=n)).isoformat()


# ─── A. 现有能力验证 ───────────────────────────────────────────────────────────


def test_baseline_cosine_ranking_basic(tmp_path):
    """[PASS] vector_search 按 cosine 相似度降序返回结果。"""
    store = MemoryStore2(tmp_path / "m.db")

    store.upsert_item("preference", "高相似条目 A", embedding=[1.0, 0.0, 0.0], extra={})
    store.upsert_item("procedure", "中等相似条目 B", embedding=[0.8, 0.6, 0.0], extra={})
    store.upsert_item("event", "低相似条目 C", embedding=[0.0, 1.0, 0.0], extra={})

    # 查询向量与 A 最相似，与 C 正交
    results = store.vector_search(query_vec=[1.0, 0.0, 0.0], top_k=3, score_threshold=0.0)

    assert len(results) == 3
    assert results[0]["summary"] == "高相似条目 A"
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]


def test_baseline_superseded_excluded_from_retrieval(tmp_path):
    """[PASS] superseded 条目不出现在检索结果中（include_superseded=False）。"""
    store = MemoryStore2(tmp_path / "m.db")

    store.upsert_item("procedure", "旧规则（已退休）", embedding=[1.0, 0.0], extra={})
    old_id = store.list_by_type("procedure")[0]["id"]
    store.mark_superseded(old_id)

    store.upsert_item("procedure", "新规则（active）", embedding=[0.9, 0.0], extra={})

    results = store.vector_search(query_vec=[1.0, 0.0], top_k=5, score_threshold=0.0)
    summaries = [r["summary"] for r in results]

    assert "旧规则（已退休）" not in summaries, "superseded 条目不应出现在检索结果中"
    assert "新规则（active）" in summaries


def test_baseline_score_threshold_filters_low_matches(tmp_path):
    """[PASS] score_threshold 过滤低相似度条目。"""
    store = MemoryStore2(tmp_path / "m.db")

    # 条目 A：与查询高度相似
    store.upsert_item("preference", "相关条目", embedding=[1.0, 0.0, 0.0], extra={})
    # 条目 B：与查询几乎正交（cosine ≈ 0）
    store.upsert_item("preference", "不相关条目", embedding=[0.0, 1.0, 0.0], extra={})

    results = store.vector_search(
        query_vec=[1.0, 0.0, 0.0],
        top_k=5,
        score_threshold=0.5,  # 过滤低分
    )

    summaries = [r["summary"] for r in results]
    assert "相关条目" in summaries
    assert "不相关条目" not in summaries


def test_baseline_top_k_limits_result_count(tmp_path):
    """[PASS] top_k 限制返回数量。"""
    store = MemoryStore2(tmp_path / "m.db")

    for i in range(10):
        store.upsert_item("event", f"条目 {i}", embedding=[1.0 - i * 0.05, 0.0], extra={})

    results = store.vector_search(query_vec=[1.0, 0.0], top_k=3, score_threshold=0.0)
    assert len(results) == 3


def test_baseline_type_filter_works(tmp_path):
    """[PASS] memory_types 过滤只返回指定类型。"""
    store = MemoryStore2(tmp_path / "m.db")

    store.upsert_item("procedure", "procedure 条目", embedding=[1.0, 0.0], extra={})
    store.upsert_item("preference", "preference 条目", embedding=[0.99, 0.01], extra={})

    results = store.vector_search(
        query_vec=[1.0, 0.0],
        top_k=5,
        memory_types=["procedure"],
        score_threshold=0.0,
    )

    assert all(r["memory_type"] == "procedure" for r in results)
    assert len(results) == 1


# ─── B. 当前边界：纯 cosine 不感知热度 ────────────────────────────────────────


def test_boundary_high_reinforcement_not_boosted(tmp_path):
    """[BOUNDARY] 高 reinforcement 条目不会因频繁访问而排名靠前。

    场景：
      条目 A：semantic=0.90，被用户反复确认（reinforcement=10），昨天刚用过
      条目 B：semantic=0.95，只创建过一次（reinforcement=1），30 天前

    直觉上 A 更"有用"（被持续验证，且新鲜），但当前纯 cosine 排序中 B 靠前。

    当前行为：B 排在 A 前面（score_B=0.95 > score_A=0.90）。

    优化后预期（hotness_alpha=0.18, half_life=14天）：
      hotness_A = sigmoid(log1p(10)) * exp(-ln2/14*1) ≈ 0.855 * 0.951 ≈ 0.813
      hotness_B = sigmoid(log1p(1))  * exp(-ln2/14*30) ≈ 0.574 * 0.228 ≈ 0.131
      final_A = 0.82*0.90 + 0.18*0.813 ≈ 0.738 + 0.146 = 0.884
      final_B = 0.82*0.95 + 0.18*0.131 ≈ 0.779 + 0.024 = 0.803
      → A(0.884) > B(0.803)，A 应排前。
      断言应改为 assert results[0]["summary"] == "条目 A（常用且新鲜）"
    """
    store = MemoryStore2(tmp_path / "m.db")

    # 条目 A：与查询相似度 0.90，reinforcement=10，1天前更新
    store.upsert_item(
        "procedure",
        "条目 A（常用且新鲜）",
        embedding=[0.90, 0.436, 0.0],  # cosine([1,0,0], [0.9,0.436,0]) = 0.9/sqrt(0.81+0.19) = 0.9
        extra={},
    )
    item_a_id = store.list_by_type("procedure")[0]["id"]
    # 手动设置 reinforcement=10，updated_at=昨天
    store._db.execute(
        "UPDATE memory_items SET reinforcement=10, updated_at=? WHERE id=?",
        (_days_ago(1), item_a_id),
    )
    store._db.commit()

    # 条目 B：与查询相似度 0.95，reinforcement=1，30天前创建
    store.upsert_item(
        "procedure",
        "条目 B（一次性，陈旧）",
        embedding=[0.95, 0.312, 0.0],  # cosine([1,0,0], [0.95,0.312,0]) = 0.95/sqrt(0.9025+0.097) ≈ 0.95
        extra={},
    )
    item_b_id = [r["id"] for r in store.list_by_type("procedure") if r["id"] != item_a_id][0]
    store._db.execute(
        "UPDATE memory_items SET reinforcement=1, created_at=?, updated_at=? WHERE id=?",
        (_days_ago(30), _days_ago(30), item_b_id),
    )
    store._db.commit()

    results = store.vector_search(
        query_vec=[1.0, 0.0, 0.0],
        top_k=5,
        score_threshold=0.0,
        # hotness_alpha 未传入（当前接口不支持），纯 cosine 模式
    )

    assert len(results) == 2

    # ── CURRENT BEHAVIOR ──
    # 纯 cosine：B(0.95) > A(0.90)，B 排前
    assert results[0]["summary"] == "条目 B（一次性，陈旧）", (
        "CURRENT: 纯 cosine 排序，高语义分的陈旧条目 B 排在常用新鲜条目 A 前。"
        " 优化后（hotness_alpha=0.18）：final_score_A=0.884 > final_score_B=0.803，"
        "A 应排前，断言改为 results[0]['summary'] == '条目 A（常用且新鲜）'"
    )
    assert results[1]["summary"] == "条目 A（常用且新鲜）"


def test_boundary_recent_update_not_boosted(tmp_path):
    """[BOUNDARY] 最近被更新（验证）的条目不因 recency 而排名靠前。

    场景：
      条目 A：semantic=0.88，2天前刚被 reinforced
      条目 B：semantic=0.92，60天前创建后再未被使用

    当前行为：B 排前（纯 cosine 0.92 > 0.88）。

    优化后预期（hotness_alpha=0.18, half_life=14天）：
      hotness_A = sigmoid(log1p(3)) * exp(-ln2/14*2) ≈ 0.757 * 0.906 ≈ 0.686
      hotness_B = sigmoid(log1p(1)) * exp(-ln2/14*60) ≈ 0.574 * 0.054 ≈ 0.031
      final_A = 0.82*0.88 + 0.18*0.686 ≈ 0.722 + 0.123 = 0.845
      final_B = 0.82*0.92 + 0.18*0.031 ≈ 0.754 + 0.006 = 0.760
      → A 排前，断言应改为 results[0]['summary'] == '条目 A（近期使用）'
    """
    store = MemoryStore2(tmp_path / "m.db")

    # 条目 A：semantic≈0.88，reinforcement=3，2天前
    store.upsert_item(
        "preference",
        "条目 A（近期使用）",
        embedding=[0.88, 0.475, 0.0],
        extra={},
    )
    item_a_id = store.list_by_type("preference")[0]["id"]
    store._db.execute(
        "UPDATE memory_items SET reinforcement=3, updated_at=? WHERE id=?",
        (_days_ago(2), item_a_id),
    )
    store._db.commit()

    # 条目 B：semantic≈0.92，reinforcement=1，60天前
    store.upsert_item(
        "preference",
        "条目 B（陈旧未用）",
        embedding=[0.92, 0.392, 0.0],
        extra={},
    )
    item_b_id = [r["id"] for r in store.list_by_type("preference") if r["id"] != item_a_id][0]
    store._db.execute(
        "UPDATE memory_items SET reinforcement=1, created_at=?, updated_at=? WHERE id=?",
        (_days_ago(60), _days_ago(60), item_b_id),
    )
    store._db.commit()

    results = store.vector_search(
        query_vec=[1.0, 0.0, 0.0],
        top_k=5,
        score_threshold=0.0,
    )

    # ── CURRENT BEHAVIOR ──
    assert results[0]["summary"] == "条目 B（陈旧未用）", (
        "CURRENT: 纯 cosine，B(0.92)排前。"
        " 优化后：final_A(0.845) > final_B(0.760)，A 应排前。"
    )


def test_boundary_score_does_not_include_hotness_fields(tmp_path):
    """[OPTIMIZED] vector_search 现在始终返回 _score_debug 字段，
    包含 semantic/hotness/final 三个拆分分数，便于观测和调试。
    """
    store = MemoryStore2(tmp_path / "m.db")
    store.upsert_item("event", "测试条目", embedding=[1.0, 0.0], extra={})

    results = store.vector_search(query_vec=[1.0, 0.0], top_k=1, score_threshold=0.0)
    assert len(results) == 1

    debug = results[0].get("_score_debug")
    assert debug is not None, "优化后：_score_debug 字段应始终存在"
    assert "semantic" in debug
    assert "hotness" in debug
    assert "final" in debug
    # hotness_alpha=0（默认）时，hotness 分量为 0，final == semantic
    assert debug["hotness"] == 0.0
    assert abs(debug["final"] - debug["semantic"]) < 1e-6


def test_emotional_weight_extends_hotness_half_life_in_ranking(tmp_path):
    store = MemoryStore2(tmp_path / "m.db")
    age = _days_ago(20)

    low = store.upsert_item(
        "event",
        "普通事件",
        embedding=[0.88, 0.475, 0.0],
        extra={},
        emotional_weight=0,
    )
    high = store.upsert_item(
        "event",
        "情绪事件",
        embedding=[0.88, 0.475, 0.0],
        extra={},
        emotional_weight=8,
    )
    low_id = low.split(":", 1)[1]
    high_id = high.split(":", 1)[1]
    store._db.execute(
        "UPDATE memory_items SET reinforcement=3, updated_at=? WHERE id IN (?, ?)",
        (age, low_id, high_id),
    )
    store._db.commit()

    results = store.vector_search(
        query_vec=[1.0, 0.0, 0.0],
        top_k=2,
        score_threshold=0.0,
        hotness_alpha=0.20,
    )

    assert results[0]["summary"] == "情绪事件"
    assert results[0]["_score_debug"]["hotness"] > results[1]["_score_debug"]["hotness"]


# ─── C. 热度公式规格预验证（与实现无关的数学验证）──────────────────────────────


def _hotness_formula(reinforcement: int, age_days: float, half_life: float = 14.0) -> float:
    """热度公式：sigmoid(log1p(reinforcement)) * exp_decay(age_days)
    直接按设计文档内联，用于验证公式数学正确性，不依赖 store.py 实现。
    """
    freq = 1.0 / (1.0 + math.exp(-math.log1p(reinforcement)))
    recency = math.exp(-math.log(2) / half_life * age_days)
    return freq * recency


def test_hotness_formula_fresh_frequently_accessed_is_hot():
    """[SPEC] 新鲜且频繁访问的条目热度最高（接近 1.0）。"""
    score = _hotness_formula(reinforcement=20, age_days=0)
    assert score > 0.9, f"新鲜且高访问条目热度应接近 1，实际 {score:.4f}"


def test_hotness_formula_old_unaccessed_is_cold():
    """[SPEC] 陈旧且从未被二次访问的条目热度最低（接近 0）。"""
    score = _hotness_formula(reinforcement=1, age_days=90, half_life=14.0)
    assert score < 0.05, f"90天前创建、未被引用的条目热度应接近 0，实际 {score:.4f}"


def test_hotness_formula_half_life_decay():
    """[SPEC] 在 half_life 天后热度衰减到初始值的一半（时间衰减正确性）。"""
    half_life = 14.0
    score_fresh = _hotness_formula(reinforcement=5, age_days=0, half_life=half_life)
    score_at_half_life = _hotness_formula(reinforcement=5, age_days=half_life, half_life=half_life)
    ratio = score_at_half_life / score_fresh
    assert abs(ratio - 0.5) < 0.01, f"half_life 处应衰减到 50%，实际 {ratio:.4f}"


def test_hotness_formula_blended_score_changes_ranking():
    """[SPEC] 热度加权后，常用新鲜条目排名优于高语义分陈旧条目。

    这是 hotness 优化的核心目标：
      条目 A: semantic=0.90, reinforcement=10, 1天前
      条目 B: semantic=0.95, reinforcement=1,  30天前
      alpha=0.18, half_life=14 → A 的 final_score 应超过 B
    """
    alpha = 0.18
    half_life = 14.0

    sem_a, reinforcement_a, age_a = 0.90, 10, 1
    sem_b, reinforcement_b, age_b = 0.95,  1, 30

    hot_a = _hotness_formula(reinforcement_a, age_a, half_life)
    hot_b = _hotness_formula(reinforcement_b, age_b, half_life)

    final_a = (1 - alpha) * sem_a + alpha * hot_a
    final_b = (1 - alpha) * sem_b + alpha * hot_b

    assert final_a > final_b, (
        f"热度加权后 A({final_a:.4f}) 应超过 B({final_b:.4f})。"
        f"  A: semantic={sem_a}, hotness={hot_a:.4f} → final={final_a:.4f}"
        f"  B: semantic={sem_b}, hotness={hot_b:.4f} → final={final_b:.4f}"
    )


def test_hotness_alpha_zero_equals_pure_cosine():
    """[SPEC] alpha=0 时 final_score == semantic_score（向后兼容保证）。"""
    alpha = 0.0
    semantic = 0.85
    hotness = 0.9  # 随意一个热度值

    final = (1 - alpha) * semantic + alpha * hotness
    assert final == semantic, "alpha=0 时热度不参与排序，确保向后兼容"
