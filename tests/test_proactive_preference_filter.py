"""
TDD: 用户偏好应影响 proactive loop 是否发送消息。

修订说明（v2）：
  - 补充端到端用例（真实 DefaultMemoryRetrievalPort + 候选 item + 向量库负偏好命中 →
    preference_block 非空 → engine 收到 interest_match 低 → 不发送）
  - 补充 compose_message/reflect 约束用例（preference_block 必须传入生成阶段）
  - 补充 config 加载用例（config.json 中 preference 字段能被正确解析）

场景：向量数据库中存有用户偏好（如"只关注 Falcons 和 NiKo，不关心其他战队"），
当主动推送候选内容来自 NAVI 等用户明确不关注的来源时，
proactive loop 不应该发出消息（哪怕信息本身"值得一发"）。

测试边界：
  1. DefaultMemoryRetrievalPort 的偏好 RAG 查询包含 item 来源名称
  2. ProactiveRetrievedMemory 有独立 preference_block 字段
  3. score_features 接收 preference_block 参数（与 retrieved_memory_block 分开）
  4. interest_match 低于偏好否决阈值 → engine 不发送
  5. interest_match 高于阈值 → 正常走 compose_message 逻辑
  6. preference_veto_enabled=False 时，低 interest_match 不硬否决
  7. preference_block 为空时不影响原有流程
"""

from __future__ import annotations

import pytest

from agent.memory import MemoryStore
from core.memory.port import DefaultMemoryPort
from feeds.base import FeedItem
from memory2.memorizer import Memorizer
from memory2.retriever import Retriever
from memory2.store import MemoryStore2
from proactive.config import ProactiveConfig
from proactive.engine import ProactiveEngine, _STOP_NONE
from proactive.ports import DefaultMemoryRetrievalPort, ProactiveRetrievedMemory
from proactive.state import ProactiveStateStore
from proactive.components import build_proactive_preference_query

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _navi_item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="NAVI beat 3DMAX to march to EPL playoffs",
        content="NAVI defeated 3DMAX in a convincing match.",
        url="https://www.hltv.org/news/44042/navi-beat-3dmax",
        author=None,
        published_at=None,
    )


def _falcons_item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="Falcons win ESL Pro League",
        content="Falcons secured the trophy with NiKo at the helm.",
        url="https://www.hltv.org/news/00001/falcons-win-epl",
        author=None,
        published_at=None,
    )


def _hltv_major_race_item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="科隆 Major 名额冲刺分析：B8 势头很猛，Legacy 基本稳了",
        content=(
            "刚看到 HLTV 的科隆 Major 名额冲刺分析，B8 势头很猛，Legacy 基本稳了。"
            "虽然没直接提到 NiKo 和 Falcons 的战况，但大赛前的格局变动总是值得留意。"
        ),
        url=(
            "https://www.hltv.org/news/44056/"
            "cologne-major-race-update-b8-surge-after-pcc-sign-up-legacy-all-but-confirm-spot-after-epl-run"
        ),
        author=None,
        published_at=None,
    )


class _FakePreferenceEmbedder:
    _KEYWORDS = (
        ("hltv", "cs", "counter-strike"),
        ("falcons", "niko"),
        ("major", "名额", "冲刺", "科隆", "b8", "legacy", "race", "格局"),
        ("只想看", "只关注", "不想看", "不关心", "偏好"),
    )

    async def embed(self, text: str) -> list[float]:
        raw = (text or "").lower()
        vector = [
            float(sum(1 for kw in group if kw in raw)) for group in self._KEYWORDS
        ]
        norm = sum(v * v for v in vector) ** 0.5
        if norm <= 0:
            return [0.0 for _ in vector]
        return [v / norm for v in vector]


def _build_real_preference_memory_port(tmp_path):
    store = MemoryStore2(tmp_path / "memory2.db")
    embedder = _FakePreferenceEmbedder()
    memorizer = Memorizer(store, embedder)
    retriever = Retriever(
        store=store,
        embedder=embedder,
        score_threshold=0.2,
        score_thresholds={
            "procedure": 0.2,
            "preference": 0.2,
            "event": 0.2,
            "profile": 0.2,
        },
        relative_delta=0.2,
    )
    return DefaultMemoryPort(
        MemoryStore(tmp_path), memorizer=memorizer, retriever=retriever
    )


def _sense_with_item(item: FeedItem):
    class _Sense:
        def compute_energy(self):
            return 0.5

        def collect_recent(self):
            return []

        def collect_recent_proactive(self, n=5):
            return []

        def compute_interruptibility(self, **kw):
            return 1.0, {
                "f_reply": 1.0,
                "f_activity": 1.0,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            }

        async def fetch_items(self, n):
            return [item]

        def filter_new_items(self, items):
            return items, [("rss:hltv", "item1")], []

        def read_memory_text(self):
            return ""

        def has_global_memory(self):
            return False

        def last_user_at(self):
            return None

        def refresh_sleep_context(self):
            return False

        def target_session_key(self):
            return "telegram:123"

    return _Sense()


# ---------------------------------------------------------------------------
# Test 1: build_proactive_preference_query 包含 item 来源信息
# ---------------------------------------------------------------------------


def test_preference_query_includes_item_source_name():
    """偏好查询字符串中应包含 item 的 source_name 和标题关键词。"""
    items = [_navi_item()]
    query = build_proactive_preference_query(items=items, max_items=3)
    assert (
        "HLTV" in query or "hltv" in query.lower()
    ), f"偏好查询未包含来源名称 HLTV: {query!r}"
    assert (
        "NAVI" in query or "navi" in query.lower()
    ), f"偏好查询未包含内容关键词 NAVI: {query!r}"


def test_preference_query_includes_multiple_sources():
    """多条 item 时，偏好查询应覆盖所有来源。"""
    items = [_navi_item(), _falcons_item()]
    query = build_proactive_preference_query(items=items, max_items=3)
    query_lower = query.lower()
    assert (
        "hltv" in query_lower or "HLTV" in query
    ), f"偏好查询未包含来源 HLTV: {query!r}"
    assert (
        "falcons" in query_lower or "navi" in query_lower
    ), f"偏好查询未包含任何 item 关键词: {query!r}"


def test_preference_query_contains_preference_signal_words():
    """偏好查询应包含"偏好/关注/兴趣/不喜欢"等检索信号词。"""
    query = build_proactive_preference_query(items=[_navi_item()], max_items=3)
    preference_words = [
        "偏好",
        "兴趣",
        "关注",
        "喜欢",
        "不关心",
        "preference",
        "interest",
    ]
    assert any(
        w in query for w in preference_words
    ), f"偏好查询中缺少偏好信号词: {query!r}"


# ---------------------------------------------------------------------------
# Test 2: ProactiveRetrievedMemory 有独立 preference_block 字段
# ---------------------------------------------------------------------------


def test_proactive_retrieved_memory_has_preference_block():
    """ProactiveRetrievedMemory 应有独立的 preference_block 字段，默认为空字符串。"""
    result = ProactiveRetrievedMemory()
    assert hasattr(
        result, "preference_block"
    ), "ProactiveRetrievedMemory 缺少 preference_block 字段"
    assert (
        result.preference_block == ""
    ), f"preference_block 默认值应为空字符串，实际: {result.preference_block!r}"


def test_proactive_retrieved_memory_empty_has_preference_block():
    """empty() 工厂方法创建的实例也应有 preference_block 字段。"""
    result = ProactiveRetrievedMemory.empty("test")
    assert hasattr(result, "preference_block")
    assert result.preference_block == ""


# ---------------------------------------------------------------------------
# Test 3: DefaultMemoryRetrievalPort 发起偏好专项 RAG 查询
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_port_sends_preference_specific_query():
    """DefaultMemoryRetrievalPort 应针对 item 来源发起独立的 preference 类型查询。"""
    pref_calls: list[dict] = []

    class _Memory:
        async def retrieve_related(self, query: str, **kwargs):
            if kwargs.get("memory_types") == ["preference"]:
                pref_calls.append({"query": query, **kwargs})
                return [
                    {
                        "id": "p1",
                        "memory_type": "preference",
                        "summary": "只关注 Falcons 和 NiKo，不关心其他战队",
                    }
                ]
            if kwargs.get("memory_types") == ["procedure", "preference"]:
                return []
            if kwargs.get("memory_types") == ["event"]:
                return []
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            if not items:
                return "", []
            return "## block\n- 偏好", [str(i.get("id")) for i in items if i.get("id")]

    cfg = ProactiveConfig()
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_navi_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert pref_calls, "未发起 preference 专项查询"
    # 查询应包含 item 来源相关信息
    pref_query = pref_calls[0]["query"].lower()
    assert (
        "hltv" in pref_query or "navi" in pref_query
    ), f"偏好查询未包含 item 来源信息: {pref_query!r}"


@pytest.mark.asyncio
async def test_memory_port_populates_preference_block():
    """当偏好 RAG 返回内容时，preference_block 应被填充。"""

    class _Memory:
        async def retrieve_related(self, query: str, **kwargs):
            if kwargs.get("memory_types") == ["preference"]:
                return [
                    {
                        "id": "p1",
                        "memory_type": "preference",
                        "summary": "只关注 Falcons 和 NiKo",
                    }
                ]
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            if not items:
                return "", []
            return "## 用户偏好\n- 只关注 Falcons 和 NiKo", ["p1"]

    cfg = ProactiveConfig()
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_navi_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert result.preference_block, "偏好 RAG 有返回但 preference_block 为空"
    assert (
        "Falcons" in result.preference_block or "偏好" in result.preference_block
    ), f"preference_block 内容不符预期: {result.preference_block!r}"


@pytest.mark.asyncio
async def test_memory_port_preference_block_empty_when_no_preference_hits():
    """无偏好 RAG 结果时，preference_block 应为空字符串。"""

    class _Memory:
        async def retrieve_related(self, query: str, **kwargs):
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            return "", []

    cfg = ProactiveConfig()
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_Memory(),
        item_id_fn=lambda _: "item1",
    )
    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_navi_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert (
        result.preference_block == ""
    ), f"无偏好命中时 preference_block 应为空: {result.preference_block!r}"


# ---------------------------------------------------------------------------
# Test 4 & 5: engine 的偏好否决门
# ---------------------------------------------------------------------------


def _base_sense():
    class _Sense:
        def compute_energy(self):
            return 0.5

        def collect_recent(self):
            return []

        def collect_recent_proactive(self, n=5):
            return []

        def compute_interruptibility(self, **kw):
            return 1.0, {
                "f_reply": 1.0,
                "f_activity": 1.0,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            }

        async def fetch_items(self, n):
            return [_navi_item()]

        def filter_new_items(self, items):
            return items, [("rss:hltv", "item1")], []

        def read_memory_text(self):
            return ""

        def has_global_memory(self):
            return False

        def last_user_at(self):
            return None

        def refresh_sleep_context(self):
            return False

        def target_session_key(self):
            return "telegram:123"

    return _Sense()


def _decide_with_interest(interest_match: float, *, send_called_out: list):
    """DecidePort mock，捕获 preference_block 参数并返回指定 interest_match。"""

    class _Decide:
        async def score_features(self, **kw):
            send_called_out.append({"preference_block": kw.get("preference_block", "")})
            return {
                "topic_continuity": 0.8,
                "interest_match": interest_match,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            return "NAVI beat 3DMAX!"

        async def reflect(self, *a, **kw):
            raise AssertionError("feature_scoring_enabled=true 时不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return "k"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return "item1"

    return _Decide()


def _retrieval_with_preference(preference_block: str):
    class _Retrieval:
        async def retrieve_proactive_context(self, **kwargs):
            return ProactiveRetrievedMemory(
                query="q",
                block="## 操作规范\n- 规范1",
                item_ids=["p1"],
                preference_block=preference_block,
            )

    return _Retrieval()


@pytest.mark.asyncio
async def test_engine_does_not_send_when_interest_match_below_veto_threshold(tmp_path):
    """interest_match 低于偏好否决阈值时，engine 不应发送消息。"""
    send_calls: list[dict] = []
    score_calls: list[dict] = []

    class _Act:
        async def send(self, message, meta=None):
            send_calls.append({"message": message})
            return True

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,  # 总分门槛放行
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
        # 偏好否决：interest_match < 0.15 时硬拒绝
        preference_veto_enabled=True,
        preference_interest_veto_threshold=0.15,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_base_sense(),
        decide=_decide_with_interest(0.05, send_called_out=score_calls),  # 极低
        act=_Act(),
        memory_retrieval=_retrieval_with_preference("用户只关注 Falcons 和 NiKo"),
    )

    await engine.tick()

    assert (
        not send_calls
    ), f"interest_match=0.05 低于否决阈值 0.15，不应发送，但实际发送了: {send_calls}"


@pytest.mark.asyncio
async def test_engine_sends_when_interest_match_above_veto_threshold(tmp_path):
    """interest_match 高于偏好否决阈值时，engine 应正常发送。"""
    send_calls: list[dict] = []
    score_calls: list[dict] = []

    class _Act:
        async def send(self, message, meta=None):
            send_calls.append({"message": message})
            return True

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
        preference_veto_enabled=True,
        preference_interest_veto_threshold=0.15,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_base_sense(),
        decide=_decide_with_interest(0.90, send_called_out=score_calls),  # 高偏好
        act=_Act(),
        memory_retrieval=_retrieval_with_preference("用户热爱 CS:GO 所有战队"),
    )

    await engine.tick()

    assert send_calls, "interest_match=0.90 高于否决阈值，应发送消息，但实际未发送"


@pytest.mark.asyncio
async def test_engine_sends_when_preference_veto_disabled(tmp_path):
    """preference_veto_enabled=False 时，即使 interest_match 极低也不硬否决。"""
    send_calls: list[dict] = []
    score_calls: list[dict] = []

    class _Act:
        async def send(self, message, meta=None):
            send_calls.append({"message": message})
            return True

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
        preference_veto_enabled=False,  # 关闭偏好否决
        preference_interest_veto_threshold=0.15,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_base_sense(),
        decide=_decide_with_interest(0.05, send_called_out=score_calls),
        act=_Act(),
        memory_retrieval=_retrieval_with_preference("用户只关注 Falcons 和 NiKo"),
    )

    await engine.tick()

    assert (
        send_calls
    ), "preference_veto_enabled=False 时不应硬否决，但 interest_match=0.05 时未发送"


# ---------------------------------------------------------------------------
# Test 6: score_features 接收 preference_block 参数
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_passes_preference_block_to_score_features(tmp_path):
    """engine 应将 preference_block 作为独立参数传给 score_features。"""
    captured: dict[str, str] = {}

    class _Decide:
        async def score_features(self, **kw):
            captured["preference_block"] = kw.get("preference_block", "MISSING")
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.8,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            return "hello"

        async def reflect(self, *a, **kw):
            raise AssertionError("不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return "k"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return "item1"

    class _Act:
        async def send(self, message, meta=None):
            return True

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
        preference_veto_enabled=True,
        preference_interest_veto_threshold=0.15,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_base_sense(),
        decide=_Decide(),
        act=_Act(),
        memory_retrieval=_retrieval_with_preference("## 偏好\n- 只关注 Falcons"),
    )

    await engine.tick()

    assert (
        captured.get("preference_block") != "MISSING"
    ), "score_features 未收到 preference_block 参数"
    assert "Falcons" in captured.get(
        "preference_block", ""
    ), f"preference_block 内容不正确: {captured.get('preference_block')!r}"


# ---------------------------------------------------------------------------
# Test 7: preference_block 为空时不影响现有流程
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_normal_flow_when_preference_block_empty(tmp_path):
    """preference_block 为空时，引擎应走正常判断逻辑（无硬否决）。"""
    send_calls: list[dict] = []

    class _Act:
        async def send(self, message, meta=None):
            send_calls.append({"message": message})
            return True

    class _Decide:
        async def score_features(self, **kw):
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.8,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            return "Falcons win!"

        async def reflect(self, *a, **kw):
            raise AssertionError("不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return "k"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return "item1"

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
        preference_veto_enabled=True,
        preference_interest_veto_threshold=0.15,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_base_sense(),
        decide=_Decide(),
        act=_Act(),
        memory_retrieval=_retrieval_with_preference(""),  # 无偏好
    )

    await engine.tick()

    assert send_calls, "无偏好限制 + high interest_match，应正常发送，但实际未发送"


# ---------------------------------------------------------------------------
# Test 8: 端到端——真实 DefaultMemoryRetrievalPort + 负偏好命中 → preference_block 非空
# （验证 RSS候选 → 偏好检索 → preference_block 填充 链路，不 mock DecidePort 打分）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_preference_rag_populates_block_for_disliked_source():
    """
    端到端链路：DefaultMemoryRetrievalPort 对 NAVI 相关 item 发起偏好查询，
    向量库返回"只关注 Falcons/NiKo"偏好记忆 → preference_block 非空。
    这覆盖了"RSS候选 → 偏好检索 → preference_block"的真实链路，
    而不是 mock DecidePort 直接返回数值。
    """
    # 记录每次 retrieve_related 的 memory_types 参数
    preference_retrieval_queries: list[str] = []

    class _RealMemory:
        async def retrieve_related(self, query: str, **kwargs):
            mt = kwargs.get("memory_types", [])
            if mt == ["preference"]:
                preference_retrieval_queries.append(query)
                # 模拟向量库命中"不关心 NAVI"偏好记忆
                return [
                    {
                        "id": "pref-001",
                        "memory_type": "preference",
                        "summary": "用户只关注 Falcons 和 NiKo，不关心 NAVI 等其他战队",
                    }
                ]
            if mt == ["procedure", "preference"]:
                return []
            if mt == ["event"]:
                return []
            return []

        def select_for_injection(self, items):
            return items

        def format_injection_with_ids(self, items):
            if not items:
                return "", []
            summaries = "\n".join(f"- {i['summary']}" for i in items if "summary" in i)
            ids = [str(i.get("id")) for i in items if i.get("id")]
            return f"## 用户偏好\n{summaries}", ids

    cfg = ProactiveConfig(preference_retrieval_enabled=True, preference_top_k=4)
    port = DefaultMemoryRetrievalPort(
        cfg=cfg,
        memory=_RealMemory(),
        item_id_fn=lambda item: item.title or "unknown",
    )

    result = await port.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_navi_item()],
        recent=[{"role": "user", "content": "只看 Falcons"}],
        decision_signals={},
        is_crisis=False,
    )

    # 偏好专项查询必须发生
    assert (
        preference_retrieval_queries
    ), "未发起 preference 类型 RAG 查询，preference_block 无法被填充"
    # 查询应包含 item 来源相关词
    assert any(
        "hltv" in q.lower() or "navi" in q.lower() for q in preference_retrieval_queries
    ), f"偏好查询未包含 item 来源信息: {preference_retrieval_queries}"

    # preference_block 必须被填充
    assert (
        result.preference_block
    ), "向量库返回了负偏好命中，但 preference_block 未被填充"
    assert (
        "Falcons" in result.preference_block or "NiKo" in result.preference_block
    ), f"preference_block 未包含偏好内容: {result.preference_block!r}"


# ---------------------------------------------------------------------------
# Test 9: compose_message 和 reflect 都收到 preference_block（生成约束用例）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_message_receives_preference_block(tmp_path):
    """compose_message 调用时必须收到 preference_block，不能只在打分阶段传入。"""
    compose_kwargs_captured: list[dict] = {}

    class _Decide:
        async def score_features(self, **kw):
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.8,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            compose_kwargs_captured.update(kw)
            return "hello"

        async def reflect(self, *a, **kw):
            raise AssertionError("不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return "k"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return "item1"

    class _Act:
        async def send(self, message, meta=None):
            return True

    cfg = ProactiveConfig(
        enabled=True,
        feature_scoring_enabled=True,
        feature_send_threshold=0.0,
        score_llm_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
        preference_veto_enabled=True,
        preference_interest_veto_threshold=0.15,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_base_sense(),
        decide=_Decide(),
        act=_Act(),
        memory_retrieval=_retrieval_with_preference("## 偏好\n- 只关注 Falcons"),
    )

    await engine.tick()

    assert (
        "preference_block" in compose_kwargs_captured
    ), "compose_message 未收到 preference_block 参数"
    assert "Falcons" in compose_kwargs_captured.get("preference_block", ""), (
        f"compose_message 收到的 preference_block 内容不正确: "
        f"{compose_kwargs_captured.get('preference_block')!r}"
    )


# ---------------------------------------------------------------------------
# Test 10: config.json 中 preference 字段能被正确加载
# ---------------------------------------------------------------------------


def test_config_loader_parses_preference_fields(tmp_path):
    """agent/config.py 的 config loader 必须解析 preference 相关字段。"""
    import json
    from agent.config import load_config

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude-test",
                "api_key": "test",
                "proactive": {
                    "enabled": False,
                    "default_chat_id": "123",
                    "preference_veto_enabled": False,
                    "preference_interest_veto_threshold": 0.25,
                    "preference_retrieval_enabled": False,
                    "preference_top_k": 8,
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))
    p = cfg.proactive

    assert (
        p.preference_veto_enabled is False
    ), f"preference_veto_enabled 未被正确加载: {p.preference_veto_enabled!r}"
    assert (
        p.preference_interest_veto_threshold == 0.25
    ), f"preference_interest_veto_threshold 未被正确加载: {p.preference_interest_veto_threshold!r}"
    assert (
        p.preference_retrieval_enabled is False
    ), f"preference_retrieval_enabled 未被正确加载: {p.preference_retrieval_enabled!r}"
    assert (
        p.preference_top_k == 8
    ), f"preference_top_k 未被正确加载: {p.preference_top_k!r}"


@pytest.mark.asyncio
async def test_real_vector_retrieval_hits_hltv_major_race_preference(tmp_path):
    port = _build_real_preference_memory_port(tmp_path)
    await port.save_item(
        summary=(
            "HLTV 的 CS 资讯里，我只想看 NiKo、Falcons 相关消息；"
            "不想看 B8、Legacy 这种 Major 名额分析。"
        ),
        memory_type="preference",
        extra={},
        source_ref="pref-major-race",
    )
    await port.save_item(
        summary="更关心 CS2 枪皮和 V 社更新。",
        memory_type="preference",
        extra={},
        source_ref="pref-skins",
    )
    retrieval = DefaultMemoryRetrievalPort(
        cfg=ProactiveConfig(preference_retrieval_enabled=True, preference_top_k=4),
        memory=port,
        item_id_fn=lambda _: "item1",
    )

    # 1. 走真实 query builder + retriever 检索偏好记忆。
    # 2. 用 HLTV Major 名额分析候选触发 preference RAG。
    # 3. 断言命中的正是这条场景相关偏好。
    result = await retrieval.retrieve_proactive_context(
        session_key="telegram:123",
        channel="telegram",
        chat_id="123",
        items=[_hltv_major_race_item()],
        recent=[],
        decision_signals={},
        is_crisis=False,
    )

    assert "NiKo" in result.preference_block
    assert "Falcons" in result.preference_block
    assert "B8" in result.preference_block
    assert "Legacy" in result.preference_block


@pytest.mark.asyncio
async def test_engine_vetoes_hltv_major_race_after_real_vector_retrieval(tmp_path):
    send_calls: list[str] = []
    score_inputs: list[dict] = []
    port = _build_real_preference_memory_port(tmp_path)
    await port.save_item(
        summary=(
            "HLTV 的 CS 资讯里，我只想看 NiKo、Falcons 相关消息；"
            "不想看 B8、Legacy 这种 Major 名额分析。"
        ),
        memory_type="preference",
        extra={},
        source_ref="pref-major-race",
    )
    retrieval = DefaultMemoryRetrievalPort(
        cfg=ProactiveConfig(preference_retrieval_enabled=True, preference_top_k=4),
        memory=port,
        item_id_fn=lambda _: "item1",
    )

    class _Decide:
        async def score_features(self, **kw):
            score_inputs.append(kw)
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.05,
                "content_novelty": 0.7,
                "reconnect_value": 0.6,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            return "这条本来会发送，但应该先被 veto。"

        async def reflect(self, *a, **kw):
            raise AssertionError("不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return "k"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return "item1"

    class _Act:
        async def send(self, message, meta=None):
            send_calls.append(message)
            return True

    engine = ProactiveEngine(
        cfg=ProactiveConfig(
            enabled=True,
            feature_scoring_enabled=True,
            feature_send_threshold=0.0,
            score_llm_threshold=0.0,
            default_channel="telegram",
            default_chat_id="123",
            preference_veto_enabled=True,
            preference_interest_veto_threshold=0.15,
        ),
        state=ProactiveStateStore(tmp_path / "state.json"),
        presence=None,
        rng=None,
        sense=_sense_with_item(_hltv_major_race_item()),
        decide=_Decide(),
        act=_Act(),
        memory_retrieval=retrieval,
    )

    # 1. 先用真实向量检索拿到偏好 block。
    # 2. 再让 feature score 给出低 interest_match。
    # 3. 最后验证 engine 在发送前被 preference_veto 拦下。
    await engine.tick()

    assert score_inputs
    assert "NiKo" in score_inputs[0].get("preference_block", "")
    assert not send_calls
