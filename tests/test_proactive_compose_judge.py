from __future__ import annotations

from datetime import datetime, timezone

import pytest

from feeds.base import FeedItem
from proactive.components import ProactiveJudgeResult
from proactive.components import ProactiveJudge
from proactive.config import ProactiveConfig
from proactive.engine import ProactiveEngine
from proactive.ports import ProactiveRetrievedMemory
from proactive.state import ProactiveStateStore


def _item() -> FeedItem:
    return FeedItem(
        source_name="HLTV",
        source_type="rss",
        title="TeamAtlas win ESL Pro League",
        content="TeamAtlas secured the trophy with PlayerNova at the helm.",
        url="https://example.com/a",
        published_at=datetime.now(timezone.utc),
    )


def _portal_item() -> FeedItem:
    content = (
        "PlayStation Portal 即将上线 1080p 高画质模式，通过更高码率流传输，让远程游戏体验更清晰流畅。"
        " 这次更新强调画面清晰度与串流质量提升，文章核心就是 1080p 模式、高码率，以及对远程游玩体验的改善。"
        " 作为测试数据，这里额外重复一次相同信息，避免 compose 阶段触发补抓正文。"
        " PlayStation Portal 即将上线 1080p 高画质模式，通过更高码率流传输，让远程游戏体验更清晰流畅。"
    )
    return FeedItem(
        source_name="VGC News",
        source_type="rss",
        title="PlayStation Portal is getting a new 1080p High Quality mode this week",
        content=content,
        url="https://www.videogameschronicle.com/news/playstation-portal-is-getting-a-new-1080p-high-quality-mode-this-week/",
        author=None,
        published_at=datetime.now(timezone.utc),
    )


class _Sense:
    def compute_energy(self):
        return 0.6

    def collect_recent(self):
        return [{"role": "user", "content": "最近有啥 CS 新闻"}]

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
        return [_item()]

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


class _MemoryRetrieval:
    async def retrieve_proactive_context(self, **kwargs):
        return ProactiveRetrievedMemory.empty()


class _Act:
    def __init__(self):
        self.calls: list[str] = []

    async def send(self, message, meta=None):
        self.calls.append(message)
        return True


@pytest.mark.asyncio
async def test_compose_judge_no_content_skip_send(tmp_path):
    class _Decide:
        async def score_features(self, **kw):
            raise AssertionError("compose_judge 模式不应走 feature score")

        async def compose_message(self, **kw):
            raise AssertionError("compose_judge 模式不应走旧 compose_message")

        async def compose_for_judge(self, **kw):
            return "<no_content/>"

        async def judge_message(self, **kw):
            raise AssertionError("compose 输出 no_content 时不应调用 judge")

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
        compose_judge_enabled=True,
        score_llm_threshold=0.0,
        score_pre_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
    )
    act = _Act()
    engine = ProactiveEngine(
        cfg=cfg,
        state=ProactiveStateStore(tmp_path / "state.json"),
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=act,
        memory_retrieval=_MemoryRetrieval(),
    )

    await engine.tick()

    assert act.calls == []


@pytest.mark.asyncio
async def test_compose_judge_veto_skip_send(tmp_path):
    class _Decide:
        async def score_features(self, **kw):
            raise AssertionError("compose_judge 模式不应走 feature score")

        async def compose_message(self, **kw):
            raise AssertionError("compose_judge 模式不应走旧 compose_message")

        async def compose_for_judge(self, **kw):
            return "有条新消息值得你看。"

        async def judge_message(self, **kw):
            return ProactiveJudgeResult(
                final_score=0.45,
                should_send=False,
                vetoed_by="balance",
                dims_deterministic={"urgency": 0.9, "balance": 0.0, "dynamics": 1.0},
                dims_llm={
                    "information_gap": 0.75,
                    "relevance": 0.75,
                    "expected_impact": 0.75,
                },
                dims_llm_raw={"information_gap": 4, "relevance": 4, "expected_impact": 4},
            )

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
        compose_judge_enabled=True,
        score_llm_threshold=0.0,
        score_pre_threshold=0.0,
        default_channel="telegram",
        default_chat_id="123",
    )
    act = _Act()
    engine = ProactiveEngine(
        cfg=cfg,
        state=ProactiveStateStore(tmp_path / "state.json"),
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=act,
        memory_retrieval=_MemoryRetrieval(),
    )

    await engine.tick()

    assert act.calls == []


@pytest.mark.asyncio
async def test_compose_for_judge_prompt_includes_portal_negative_preference(monkeypatch):
    preference_block = (
        "## 【流程规范】用户偏好与规则\n"
        "- 用户明确要求：禁止推送任何主机（PS5、Xbox、Switch）相关的硬件新闻、独占游戏、系统更新等内容。"
        "用户是纯 PC 党，此类信息视为无效噪音直接过滤。\n"
        "- 用户明确对 VGC News 的游戏硬件新闻推送保持高度警惕，尤其反感其对 PlayStation 设备功能更新的过度宣传，"
        "认为此类内容常夸大体验提升且缺乏实际使用反馈，属于典型的平台生态营销信息。\n"
    )

    class _Provider:
        async def chat(self, **kwargs):
            user_msg = kwargs["messages"][1]["content"]
            assert "PlayStation Portal is getting a new 1080p High Quality mode this week" in user_msg
            assert "禁止推送任何主机" in user_msg
            assert "<no_content/>" in user_msg

            class _Resp:
                content = "<no_content/>"

            return _Resp()

    judge = ProactiveJudge(
        provider=_Provider(),
        model="m",
        max_tokens=128,
        format_items=lambda items: "\n".join(
            f"- {item.title}\n原文链接: {item.url}" for item in items
        ),
        format_recent=lambda _: "",
        cfg=ProactiveConfig(),
    )
    monkeypatch.setattr(
        judge,
        "_enrich_items_for_compose",
        lambda items: __import__("asyncio").sleep(0, result=items),
    )
    result = await judge.compose_for_judge(
        items=[_portal_item()],
        recent=[],
        preference_block=preference_block,
    )

    assert result == "<no_content/>"


@pytest.mark.asyncio
async def test_judge_prompt_includes_portal_negative_preference():
    preference_block = (
        "## 【流程规范】用户偏好与规则\n"
        "- 用户明确要求：禁止推送任何主机（PS5、Xbox、Switch）相关的硬件新闻、独占游戏、系统更新等内容。"
        "用户是纯 PC 党，此类信息视为无效噪音直接过滤。\n"
        "- 用户明确对 VGC News 的游戏硬件新闻推送保持高度警惕，尤其反感其对 PlayStation 设备功能更新的过度宣传，"
        "认为此类内容常夸大体验提升且缺乏实际使用反馈，属于典型的平台生态营销信息。\n"
    )

    class _Provider:
        async def chat(self, **kwargs):
            user_msg = kwargs["messages"][1]["content"]
            assert "用户偏好与禁推规则" in user_msg
            assert "禁止推送任何主机" in user_msg
            assert "PlayStation Portal 即将上线 1080p 高画质模式" in user_msg

            class _Resp:
                content = '{"information_gap":4,"relevance":1,"expected_impact":1}'

            return _Resp()

    judge = ProactiveJudge(
        provider=_Provider(),
        model="m",
        max_tokens=128,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        cfg=ProactiveConfig(judge_veto_llm_dim_min=2),
    )
    result = await judge.judge_message(
        message="PlayStation Portal 即将上线 1080p 高画质模式，通过更高码率流传输，让远程游戏体验更清晰流畅。",
        recent=[],
        recent_proactive_text="",
        preference_block=preference_block,
        age_hours=1.0,
        sent_24h=0,
        interrupt_factor=1.0,
    )

    assert result.vetoed_by == "llm_dim"


@pytest.mark.asyncio
async def test_compose_judge_empty_string_skip_send(tmp_path):
    class _Decide:
        async def score_features(self, **kw):
            raise AssertionError("compose_judge 模式不应走 feature score")

        async def compose_message(self, **kw):
            raise AssertionError("compose_judge 模式不应走旧 compose_message")

        async def compose_for_judge(self, **kw):
            return ""

        async def judge_message(self, **kw):
            raise AssertionError("compose 输出空字符串时不应调用 judge")

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

    engine = ProactiveEngine(
        cfg=ProactiveConfig(
            compose_judge_enabled=True,
            score_llm_threshold=0.0,
            score_pre_threshold=0.0,
            default_channel="telegram",
            default_chat_id="123",
        ),
        state=ProactiveStateStore(tmp_path / "state.json"),
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=_Act(),
        memory_retrieval=_MemoryRetrieval(),
    )

    await engine.tick()


@pytest.mark.asyncio
async def test_compose_judge_none_fallback_allows_send(tmp_path):
    sent: list[str] = []

    class _Decide:
        async def score_features(self, **kw):
            raise AssertionError("compose_judge 模式不应走 feature score")

        async def compose_message(self, **kw):
            raise AssertionError("compose_judge 模式不应走旧 compose_message")

        async def compose_for_judge(self, **kw):
            return "这条消息应该发出。"

        async def judge_message(self, **kw):
            return None

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

    class _SendAct:
        async def send(self, message, meta=None):
            sent.append(message)
            return True

    engine = ProactiveEngine(
        cfg=ProactiveConfig(
            compose_judge_enabled=True,
            score_llm_threshold=0.0,
            score_pre_threshold=0.0,
            default_channel="telegram",
            default_chat_id="123",
        ),
        state=ProactiveStateStore(tmp_path / "state.json"),
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=_SendAct(),
        memory_retrieval=_MemoryRetrieval(),
    )

    await engine.tick()
    assert sent


class _StubProvider:
    async def chat(self, **kwargs):
        class _Resp:
            content = '{"information_gap":4,"relevance":4,"expected_impact":4}'

        return _Resp()


@pytest.mark.asyncio
async def test_judge_does_not_veto_by_urgency():
    judge = ProactiveJudge(
        provider=_StubProvider(),
        model="m",
        max_tokens=128,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        cfg=ProactiveConfig(
            judge_urgency_horizon_hours=12.0,
            judge_veto_urgency_min=0.2,
            judge_send_threshold=0.0,
        ),
    )
    result = await judge.judge_message(
        message="msg",
        recent=[],
        recent_proactive_text="",
        age_hours=20.0,
        sent_24h=0,
        interrupt_factor=1.0,
    )
    assert result.vetoed_by is None
    assert result.should_send is True


@pytest.mark.asyncio
async def test_judge_veto_by_balance():
    judge = ProactiveJudge(
        provider=_StubProvider(),
        model="m",
        max_tokens=128,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        cfg=ProactiveConfig(
            judge_balance_daily_max=4,
            judge_veto_balance_min=0.2,
        ),
    )
    result = await judge.judge_message(
        message="msg",
        recent=[],
        recent_proactive_text="",
        age_hours=1.0,
        sent_24h=4,
        interrupt_factor=1.0,
    )
    assert result.vetoed_by == "balance"


class _LowDimProvider:
    async def chat(self, **kwargs):
        class _Resp:
            content = '{"information_gap":1,"relevance":4,"expected_impact":4}'

        return _Resp()


@pytest.mark.asyncio
async def test_judge_veto_by_llm_dim():
    judge = ProactiveJudge(
        provider=_LowDimProvider(),
        model="m",
        max_tokens=128,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        cfg=ProactiveConfig(judge_veto_llm_dim_min=2),
    )
    result = await judge.judge_message(
        message="msg",
        recent=[],
        recent_proactive_text="",
        age_hours=1.0,
        sent_24h=0,
        interrupt_factor=1.0,
    )
    assert result.vetoed_by == "llm_dim"


@pytest.mark.asyncio
async def test_judge_score_is_zero_when_all_weights_zero():
    judge = ProactiveJudge(
        provider=_StubProvider(),
        model="m",
        max_tokens=128,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        cfg=ProactiveConfig(
            judge_weight_urgency=0.0,
            judge_weight_balance=0.0,
            judge_weight_dynamics=0.0,
            judge_weight_information_gap=0.0,
            judge_weight_relevance=0.0,
            judge_weight_expected_impact=0.0,
            judge_send_threshold=0.5,
        ),
    )
    result = await judge.judge_message(
        message="msg",
        recent=[],
        recent_proactive_text="",
        age_hours=1.0,
        sent_24h=0,
        interrupt_factor=1.0,
    )
    assert result.final_score == 0.0
    assert result.should_send is False
