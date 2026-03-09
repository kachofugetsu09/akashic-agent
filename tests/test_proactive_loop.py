from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.memory import MemoryStore
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from feeds.base import FeedItem
from proactive.config import ProactiveConfig
from proactive.engine import DecisionContext, GateResult, ProactiveEngine, _STOP_NONE
from proactive.loop import ProactiveLoop, _parse_decision
from proactive.ports import ProactiveSendMeta, ProactiveSourceRef
from proactive.presence import PresenceStore
from session.manager import SessionManager


def _utc(**kwargs) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**kwargs)


class _DummyFeedRegistry:
    async def fetch_all(
        self,
        limit_per_source: int = 3,
        per_source_limits: dict[str, int] | None = None,
    ):
        return []


class _DummyProvider:
    async def chat(self, **kwargs):
        raise RuntimeError("not used in this test")


def _build_loop(tmp_path, push_tool, chat_id: str = "7674283004", default_channel: str = "telegram"):
    session_manager = SessionManager(tmp_path)
    return ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=session_manager,
        provider=_DummyProvider(),
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel=default_channel,
            default_chat_id=chat_id,
        ),
        model="test-model",
        max_tokens=128,
        state_path=tmp_path / "proactive_state.json",
    ), session_manager


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture(autouse=True)
async def _shared_http_resources():
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    try:
        yield
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()


def test_parse_decision_string_false_is_false():
    d = _parse_decision(
        '{"score": 0.9, "should_send": "false", "message": "hello", "reasoning": "r"}'
    )
    assert d.should_send is False


def test_decision_context_groups_fields_into_snapshots():
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    decide = ctx.ensure_decide()
    act = ctx.ensure_act()

    sense.energy = 0.6
    fetch.new_items = []
    score.base_score = 0.4
    decide.decision_message = "hi"
    act.state_summary_tag = "none"
    ctx.state.session_key = "telegram:1"

    assert ctx.sense is not None and ctx.sense.energy == 0.6
    assert ctx.fetch is not None and ctx.fetch.new_items == []
    assert ctx.score is not None and ctx.score.base_score == 0.4
    assert ctx.decide is not None and ctx.decide.decision_message == "hi"
    assert ctx.act is not None and ctx.act.state_summary_tag == "none"
    assert ctx.state.session_key == "telegram:1"


@pytest.mark.asyncio
async def test_engine_classify_state_summary_tag_supports_fenced_json():
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp(
            '```json\n{"state_summary_tag":"general_encouragement"}\n```'
        )
    )
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._light_provider = provider
    engine._light_model = "test-model"

    tag = await engine._classify_state_summary_tag("先别太逼自己，今天先缓一缓。")

    assert tag == "general_encouragement"


@pytest.mark.asyncio
async def test_engine_request_light_text_uses_expected_chat_kwargs():
    provider = _DummyProvider()
    provider.chat = AsyncMock(return_value=_Resp("rewritten"))
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._light_provider = provider
    engine._light_model = "test-model"

    text = await engine._request_light_text(
        system_content="system",
        user_content="user",
        max_tokens=77,
    )

    assert text == "rewritten"
    kwargs = provider.chat.await_args.kwargs
    assert kwargs["tools"] == []
    assert kwargs["model"] == "test-model"
    assert kwargs["max_tokens"] == 77


@pytest.mark.asyncio
async def test_engine_stage_gate_returns_structured_result_for_scheduler_reject():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        dedupe_seen_ttl_hours=24,
        delivery_dedupe_hours=24,
        semantic_dedupe_window_hours=24,
        anyaction_enabled=True,
    )
    engine._state = SimpleNamespace(cleanup=MagicMock())
    engine._sense = SimpleNamespace(last_user_at=lambda: None)
    engine._anyaction = SimpleNamespace(
        should_act=lambda **kwargs: (False, {"reason": "min_interval"})
    )

    result = await engine._stage_gate(DecisionContext())

    assert result.proceed is False
    assert result.stop_result is _STOP_NONE
    assert result.reason_code == "scheduler_reject"


@pytest.mark.asyncio
async def test_engine_stage_pre_score_returns_structured_below_threshold_result():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        score_weight_energy=1.0,
        score_weight_recent=1.0,
        score_pre_threshold=0.5,
    )
    engine._try_skill_action = AsyncMock()
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    score = ctx.ensure_score()
    sense.de = 0.1
    sense.dr = 0.1
    sense.interrupt_factor = 1.0
    sense.interruptibility = 1.0
    sense.interrupt_detail = {
        "f_reply": 1.0,
        "f_activity": 1.0,
        "f_fatigue": 1.0,
        "random_delta": 0.0,
    }
    sense.sleep_mod = 1.0
    sense.energy = 0.2
    sense.recent = []
    sense.health_events = []

    result = await engine._stage_pre_score(ctx)

    assert result.proceed is False
    assert result.return_score == score.pre_score
    assert result.reason_code == "below_threshold"
    engine._try_skill_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_stage_sense_returns_structured_snapshot():
    sleep_ctx = SimpleNamespace(
        health_events=[{"severity": "high"}],
        sleep_modifier=0.5,
        state="sleeping",
        available=True,
        prob=0.8,
        data_lag_min=5,
    )
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(score_recent_scale=10)
    engine._sense = SimpleNamespace(
        refresh_sleep_context=lambda: True,
        sleep_context=lambda: sleep_ctx,
        compute_energy=lambda: 0.2,
        collect_recent=lambda: [{"role": "user", "content": "hi"}],
        compute_interruptibility=lambda **kwargs: (
            0.75,
            {
                "f_reply": 0.9,
                "f_activity": 0.8,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            },
        ),
    )
    ctx = DecisionContext()

    result = engine._stage_sense(ctx)

    assert result.sleep_state == "sleeping"
    assert result.sleep_available is True
    assert result.health_event_count == 1
    assert result.energy == 0.2
    assert result.recent_count == 1
    assert result.interruptibility == 0.75
    assert result.interrupt_factor == ctx.ensure_sense().interrupt_factor


@pytest.mark.asyncio
async def test_engine_stage_score_returns_snapshot_fields_for_no_candidates():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        score_content_halfsat=3.0,
        score_weight_energy=1.0,
        score_weight_content=1.0,
        score_weight_recent=1.0,
        score_llm_threshold=0.6,
    )
    engine._rng = None
    engine._sense = SimpleNamespace(target_session_key=lambda: "")
    engine._presence = None
    engine._state = SimpleNamespace()
    engine._try_skill_action = AsyncMock()
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    sense.de = 0.2
    sense.dr = 0.1
    sense.interrupt_factor = 1.0
    sense.interruptibility = 1.0
    fetch.new_items = []
    sense.health_events = []
    score.force_reflect = False
    sense.energy = 0.3
    fetch.has_memory = False

    result = await engine._stage_score(ctx)

    assert result.proceed is False
    assert result.reason_code == "no_candidates"
    assert result.return_score == score.base_score
    assert result.base_score == score.base_score
    assert result.draw_score == score.draw_score
    assert result.force_reflect is False


@pytest.mark.asyncio
async def test_engine_stage_score_draw_threshold_still_triggers_skill_action():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        score_content_halfsat=3.0,
        score_weight_energy=1.0,
        score_weight_content=1.0,
        score_weight_recent=1.0,
        score_llm_threshold=0.95,
    )
    engine._rng = None
    engine._sense = SimpleNamespace(target_session_key=lambda: "")
    engine._presence = None
    engine._state = SimpleNamespace()
    engine._try_skill_action = AsyncMock()
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    ctx.state.now_utc = datetime.now(timezone.utc)
    sense.de = 0.2
    sense.dr = 0.1
    sense.interrupt_factor = 1.0
    sense.interruptibility = 1.0
    fetch.new_items = [
        FeedItem(
            source_name="Test",
            source_type="rss",
            title="A",
            content="body",
            url="https://example.com/a",
            author=None,
            published_at=None,
        )
    ]
    sense.health_events = []
    score.force_reflect = False
    sense.energy = 0.3
    fetch.has_memory = False

    result = await engine._stage_score(ctx)

    assert result.proceed is False
    assert result.reason_code == "draw_score_below_threshold"
    engine._try_skill_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_stage_fetch_filter_returns_structured_snapshot():
    items = [
        FeedItem(
            source_name="Test",
            source_type="rss",
            title="A",
            content="body",
            url="https://example.com/a",
            author=None,
            published_at=None,
        )
    ]
    entries = [("rss:test", "item-1")]
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        interest_filter=SimpleNamespace(enabled=False),
        pending_queue_enabled=False,
        items_per_source=3,
    )
    engine._sense = SimpleNamespace(
        fetch_items=AsyncMock(return_value=items),
        filter_new_items=lambda raw: (raw, entries, []),
        has_global_memory=lambda: True,
    )
    ctx = DecisionContext()

    result = await engine._stage_fetch_filter(ctx)

    assert result.total_items == 1
    assert result.discovered_count == 1
    assert result.selected_count == 1
    assert result.semantic_duplicate_count == 0
    assert result.pending_enabled is False
    assert result.has_memory is True
    assert ctx.ensure_fetch().new_items == items
    assert ctx.ensure_fetch().new_entries == entries


@pytest.mark.asyncio
async def test_engine_stage_decide_returns_structured_feature_reject_result():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        feature_scoring_enabled=True,
        feature_send_threshold=0.8,
        threshold=0.7,
        score_llm_threshold=0.6,
        llm_reject_cooldown_hours=0,
        feature_weight_topic_continuity=1.0,
        feature_weight_interest_match=1.0,
        feature_weight_content_novelty=1.0,
        feature_weight_reconnect_value=1.0,
        feature_weight_message_readiness=1.0,
        feature_weight_disturb_risk=1.0,
        feature_weight_interrupt_penalty=1.0,
        feature_weight_d_recent_bonus=0.0,
        feature_weight_d_content_bonus=0.0,
        feature_weight_d_energy_bonus=0.0,
    )
    engine._decide = SimpleNamespace(
        score_features=AsyncMock(
            return_value={
                "topic_continuity": 0.1,
                "interest_match": 0.1,
                "content_novelty": 0.1,
                "reconnect_value": 0.1,
                "disturb_risk": 0.9,
                "message_readiness": 0.1,
                "confidence": 0.1,
            }
        ),
        compose_message=AsyncMock(return_value=""),
    )
    engine._memory_retrieval = None
    engine._state = SimpleNamespace(mark_rejection_cooldown=MagicMock())
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    act = ctx.ensure_act()
    ctx.state.session_key = "telegram:1"
    ctx.state.now_utc = datetime.now(timezone.utc)
    fetch.new_items = []
    fetch.new_entries = []
    sense.recent = []
    sense.health_events = []
    act.high_events = []
    sense.interruptibility = 1.0
    sense.interrupt_detail = {"f_reply": 1.0, "f_activity": 1.0, "f_fatigue": 1.0}
    score.pre_score = 0.2
    score.base_score = 0.3
    score.draw_score = 0.3
    score.dc = 0.1
    sense.de = 0.1
    sense.dr = 0.1
    score.is_crisis = False
    score.sent_24h = 0
    score.fresh_items_24h = 0

    result = await engine._stage_decide(ctx)

    assert result.proceed is False
    assert result.reason_code == "feature_score_reject"
    assert result.should_send is False
    assert result.decision_message == ""
    assert result.decision_mode == "feature"
    assert result.feature_final_score == ctx.ensure_decide().feature_final_score
    assert result.history_gate_reason == "disabled"
    assert result.history_scope_mode == "disabled"


@pytest.mark.asyncio
async def test_engine_stage_decide_feature_mode_still_composes_and_sets_candidates():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        feature_scoring_enabled=True,
        feature_send_threshold=0.4,
        threshold=0.7,
        score_llm_threshold=0.6,
        llm_reject_cooldown_hours=0,
        feature_weight_topic_continuity=1.0,
        feature_weight_interest_match=1.0,
        feature_weight_content_novelty=1.0,
        feature_weight_reconnect_value=1.0,
        feature_weight_message_readiness=1.0,
        feature_weight_disturb_risk=0.0,
        feature_weight_interrupt_penalty=0.0,
        feature_weight_d_recent_bonus=0.0,
        feature_weight_d_content_bonus=0.0,
        feature_weight_d_energy_bonus=0.0,
    )
    item = FeedItem(
        source_name="Test",
        source_type="rss",
        title="A",
        content="body",
        url="https://example.com/a",
        author=None,
        published_at=None,
    )
    engine._decide = SimpleNamespace(
        score_features=AsyncMock(
            return_value={
                "topic_continuity": 1.0,
                "interest_match": 1.0,
                "content_novelty": 1.0,
                "reconnect_value": 1.0,
                "disturb_risk": 0.0,
                "message_readiness": 1.0,
                "confidence": 1.0,
                "topic_continuity_reason": "r1",
                "interest_match_reason": "r2",
                "content_novelty_reason": "r3",
                "reconnect_value_reason": "r4",
                "disturb_risk_reason": "r5",
                "message_readiness_reason": "r6",
                "confidence_reason": "r7",
            }
        ),
        compose_message=AsyncMock(return_value="hello proactive"),
        item_id_for=lambda _: "item-1",
    )
    engine._memory_retrieval = None
    engine._state = SimpleNamespace(mark_rejection_cooldown=MagicMock())
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    act = ctx.ensure_act()
    ctx.state.session_key = "telegram:1"
    ctx.state.now_utc = datetime.now(timezone.utc)
    fetch.new_items = [item]
    fetch.new_entries = [("rss:test", "item-1")]
    sense.recent = []
    sense.health_events = []
    act.high_events = []
    sense.interruptibility = 1.0
    sense.interrupt_detail = {"f_reply": 1.0, "f_activity": 1.0, "f_fatigue": 1.0}
    score.pre_score = 0.2
    score.base_score = 0.3
    score.draw_score = 0.3
    score.dc = 0.1
    sense.de = 0.1
    sense.dr = 0.1
    score.is_crisis = False
    score.sent_24h = 0
    score.fresh_items_24h = 0

    result = await engine._stage_decide(ctx)

    assert result.proceed is True
    assert result.decision_mode == "feature"
    assert result.should_send is True
    assert result.decision_message == "hello proactive"
    assert ctx.ensure_act().compose_items == [item]
    assert ctx.ensure_act().compose_entries == [("rss:test", "item-1")]
    engine._decide.compose_message.assert_awaited_once()


def test_engine_stage_trace_writer_emits_strategy_envelope():
    emitted: list[dict] = []
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._stage_trace_writer = emitted.append

    engine._trace_stage_result(
        DecisionContext(),
        stage="gate",
        result=GateResult(proceed=True, stop_result=None, reason_code="pass"),
    )

    assert emitted[0]["trace_type"] == "proactive_stage"
    assert emitted[0]["subject"]["kind"] == "global"
    assert emitted[0]["payload"]["stage"] == "gate"
    assert emitted[0]["payload"]["result"]["reason_code"] == "pass"


def test_engine_stage_trace_writer_serializes_stop_none_sentinel():
    emitted: list[dict] = []
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._stage_trace_writer = emitted.append

    engine._trace_stage_result(
        DecisionContext(),
        stage="gate",
        result=GateResult(proceed=False, stop_result=_STOP_NONE, reason_code="scheduler_reject"),
    )

    assert isinstance(emitted[0]["payload"]["result"]["stop_result"], str)


@pytest.mark.asyncio
async def test_send_uses_configured_channel(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop, _ = _build_loop(tmp_path, push_tool, chat_id="7674283004", default_channel="qq")

    await loop._send("主动消息")

    push_tool.execute.assert_called_once_with(
        channel="qq",
        chat_id="7674283004",
        message="主动消息",
    )


@pytest.mark.asyncio
async def test_send_writes_proactive_message_into_target_session(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop, session_manager = _build_loop(tmp_path, push_tool)

    await loop._send("你好，这是一次主动触达")

    session = session_manager.get_or_create("telegram:7674283004")
    assert session.messages
    last = session.messages[-1]
    assert last["role"] == "assistant"
    assert last["content"] == "你好，这是一次主动触达"
    assert last.get("proactive") is True


@pytest.mark.asyncio
async def test_send_persists_source_refs_and_state_summary_tag(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop, session_manager = _build_loop(tmp_path, push_tool)

    await loop._send(
        "Rare Atom 这条看起来像是临时补强。",
        ProactiveSendMeta(
            evidence_item_ids=["item-1"],
            source_refs=[
                ProactiveSourceRef(
                    item_id="item-1",
                    source_type="rss",
                    source_name="HLTV",
                    title="Rare Atom signs x9",
                    url="https://example.com/ra",
                    published_at="2026-03-09T00:00:00+00:00",
                )
            ],
            state_summary_tag="none",
        ),
    )

    session = session_manager.get_or_create("telegram:7674283004")
    last = session.messages[-1]
    assert last["content"] == "Rare Atom 这条看起来像是临时补强。"
    assert last["evidence_item_ids"] == ["item-1"]
    assert last["state_summary_tag"] == "none"
    assert last["source_refs"][0]["source_name"] == "HLTV"
    history = session.get_history()
    assert "[proactive_meta]" in history[-1]["content"]
    assert "HLTV | Rare Atom signs x9 | https://example.com/ra" in history[-1]["content"]


@pytest.mark.asyncio
async def test_tick_delivery_dedupe_blocks_repeat_send_for_seen_item(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    feed = _DummyFeedRegistry()
    item = FeedItem(
        source_name="TestFeed",
        source_type="rss",
        title="Same News",
        content="content",
        url="https://example.com/a",
        author=None,
        published_at=None,
    )
    feed.fetch_all = AsyncMock(side_effect=[[item], [item]])

    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp(
            '{"reasoning":"ok","score":0.9,"should_send":true,"message":"ping"}'
        )
    )
    session_manager = SessionManager(tmp_path)
    loop = ProactiveLoop(
        feed_registry=feed,
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="7674283004",
            delivery_dedupe_hours=24,
            message_dedupe_enabled=False,
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
    )

    await loop._tick()
    await loop._tick()

    assert provider.chat.await_count == 2
    assert push_tool.execute.await_count == 1


@pytest.mark.asyncio
async def test_reflect_includes_global_memory(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户偏好：关注单机游戏发售与DLC，不爱电竞资讯。")

    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.2,"should_send":false,"message":""}')
    )
    session_manager = SessionManager(tmp_path)
    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            use_global_memory=True,
            global_memory_max_chars=3000,
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        memory_store=memory,
    )

    await loop._reflect(items=[], recent=[])

    kwargs = provider.chat.await_args.kwargs
    user_prompt = kwargs["messages"][1]["content"]
    assert "用户偏好：关注单机游戏发售与DLC，不爱电竞资讯。" in user_prompt


# ── Dynamic energy / presence 集成测试 ────────────────────────────


def _make_presence(tmp_path, session_key: str, last_user_minutes_ago: float | None):
    p = PresenceStore(tmp_path / "presence.json")
    if last_user_minutes_ago is not None:
        p.record_user_message(session_key, now=_utc(minutes=last_user_minutes_ago))
    return p


def _build_loop_with_presence(tmp_path, provider, presence, feed=None):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    session_manager = SessionManager(tmp_path)
    loop = ProactiveLoop(
        feed_registry=feed or _DummyFeedRegistry(),
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="123",
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        presence=presence,
    )
    return loop, push_tool


@pytest.mark.asyncio
async def test_tick_skips_llm_when_energy_above_cool_threshold(tmp_path):
    """用户刚发消息（5 分钟前），电量高，不应调用 LLM。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock()

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=5)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._tick()

    provider.chat.assert_not_called()


@pytest.mark.asyncio
async def test_tick_calls_llm_in_crisis_mode_no_content(tmp_path):
    """72h 未发消息（危机模式），即使没有 feed 和记忆，W_content 托底 → 应调用 LLM。"""
    import random
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 72)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=SessionManager(tmp_path),
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="123",
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        presence=presence,
        rng=random.Random(1),   # 固定种子，结果确定
    )

    await loop._tick()

    provider.chat.assert_called_once()


@pytest.mark.asyncio
async def test_tick_calls_llm_when_no_presence_data(tmp_path):
    """无心跳记录（从未收到消息），视作电量为 0，应进入 LLM 反思。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=None)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._tick()

    provider.chat.assert_called_once()


@pytest.mark.asyncio
async def test_reflect_contains_energy_context(tmp_path):
    """LLM prompt 里应包含电量和冲动信息。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 48)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._reflect(items=[], recent=[], energy=0.06, urge=0.82)

    user_prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "电量" in user_prompt
    assert "冲动" in user_prompt


@pytest.mark.asyncio
async def test_reflect_prompt_requires_direct_opinion_not_counter_question(tmp_path):
    """proactive 文案应强调直接表达观点，避免“你怎么看”式收尾。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 48)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._reflect(items=[], recent=[], energy=0.06, urge=0.82)

    user_prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "直接表达你的判断/观点" in user_prompt
    assert "你怎么看/你觉得呢/你怎么想/要不要我继续" in user_prompt


@pytest.mark.asyncio
async def test_reflect_prompt_allows_interest_based_new_topic(tmp_path):
    """即使与近期对话无关，只要符合长期兴趣，也允许自然开启新话题。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 48)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._reflect(items=[], recent=[], energy=0.06, urge=0.82)

    user_prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "只是加分项，不是前置条件" in user_prompt
    assert "长期兴趣高度匹配" in user_prompt


@pytest.mark.asyncio
async def test_reflect_prompt_discourages_repeating_user_state_summary(tmp_path):
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 48)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._reflect(items=[], recent=[], energy=0.06, urge=0.82)

    user_prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "用户此后还没有回复，则新消息禁止重复这一层" in user_prompt
    assert "若本次只是新资讯，直接进入新内容" in user_prompt


@pytest.mark.asyncio
async def test_reflect_prompt_requires_evidence_and_exact_source(tmp_path):
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 48)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._reflect(items=[], recent=[], energy=0.06, urge=0.82)

    user_prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "只有在你能指出消息依据的确切证据时" in user_prompt
    assert "若你找不到确切证据或来源不清，应降低 score，并把 should_send 设为 false" in user_prompt
    assert "优先自然带上“来源名 + 可点击原文链接”" in user_prompt
    assert "系统不会在发送前替你自动补来源" in user_prompt


@pytest.mark.asyncio
async def test_reflect_contains_crisis_hint_when_energy_very_low(tmp_path):
    """电量极低（危机模式）时，prompt 里应包含危机提示。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 72)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._reflect(items=[], recent=[], energy=0.02, urge=0.99)

    user_prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "危机" in user_prompt


@pytest.mark.asyncio
async def test_tick_skips_llm_when_no_content_and_no_crisis(tmp_path):
    """有 presence（1h前）但无内容（无 feed、无记忆），且非危机 → D_content=0 → 跳过 LLM。

    使用固定 rng seed + 1h 场景保证确定性：
    1h 时 energy≈0.49 → d_energy≈0.51 → base_score_max≈0.23 < score_llm_threshold=0.40，
    random_weight∈[0.5, 1.5] 的最大扰动也无法突破阈值，不依赖随机采样结果。
    """
    import random
    provider = _DummyProvider()
    provider.chat = AsyncMock()

    # 1h 前有互动（非危机，非冷启动），无 feed，无记忆
    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=SessionManager(tmp_path),
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="123",
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        presence=presence,
        rng=random.Random(42),  # 固定种子，防止 random_weight 偶发超阈值
    )

    await loop._tick()

    provider.chat.assert_not_called()


@pytest.mark.asyncio
async def test_tick_calls_llm_when_low_energy_with_memory(tmp_path):
    """电量低 + 有记忆 → W_content > 0 → 进入 LLM。"""
    import random
    from agent.memory import MemoryStore
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 24)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    session_manager = SessionManager(tmp_path)
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户喜欢魂类游戏，最近在玩 Elden Ring。")

    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="123",
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        presence=presence,
        memory_store=memory,
        rng=random.Random(1),
    )

    await loop._tick()

    provider.chat.assert_called_once()


@pytest.mark.asyncio
async def test_tick_without_new_items_still_runs_when_low_energy_and_memory(tmp_path):
    """no new feed items + low energy + memory 时仍可进入 LLM。"""
    import random
    from agent.memory import MemoryStore
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 24)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    session_manager = SessionManager(tmp_path)
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户喜欢 Python。")

    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),   # 始终返回空 feed
        session_manager=session_manager,
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="123",
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        presence=presence,
        memory_store=memory,
        rng=random.Random(1),
    )

    await loop._tick()

    provider.chat.assert_called_once()


@pytest.mark.asyncio
async def test_reflect_always_contains_full_memory(tmp_path):
    """_reflect 应始终注入全量记忆，不论是否危机模式。"""
    from agent.memory import MemoryStore
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    memory = MemoryStore(tmp_path)
    memory.write_long_term("## 偏好\n\n- 喜欢魂类游戏\n- 不喜欢电竞\n\n## 工作\n\n- 用 Python\n")
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=SessionManager(tmp_path),
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(enabled=True),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        memory_store=memory,
    )

    # 正常模式：全量记忆
    await loop._reflect(items=[], recent=[], energy=0.15, urge=0.3, is_crisis=False)
    prompt_normal = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "偏好" in prompt_normal
    assert "工作" in prompt_normal
    assert "Python" in prompt_normal
    assert "魂类" in prompt_normal


@pytest.mark.asyncio
async def test_reflect_crisis_adds_topic_hint(tmp_path):
    """危机模式应在全量记忆基础上额外注入一条随机话题作为开场提示。"""
    import random
    from agent.memory import MemoryStore
    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.3,"should_send":false,"message":""}')
    )

    memory = MemoryStore(tmp_path)
    memory.write_long_term("## 偏好\n\n- 喜欢魂类游戏\n- 不喜欢电竞\n\n## 工作\n\n- 用 Python\n")
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=SessionManager(tmp_path),
        provider=provider,
        push_tool=push_tool,
        config=ProactiveConfig(enabled=True),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        memory_store=memory,
        rng=random.Random(1),
    )

    await loop._reflect(items=[], recent=[], energy=0.02, urge=0.99, is_crisis=True)
    prompt_crisis = provider.chat.await_args.kwargs["messages"][1]["content"]
    # 全量记忆仍在
    assert "偏好" in prompt_crisis or "工作" in prompt_crisis
    # 额外有话题提示
    assert "话题" in prompt_crisis or "开场" in prompt_crisis or "开始聊" in prompt_crisis


@pytest.mark.asyncio
async def test_send_records_proactive_sent_in_presence(tmp_path):
    """_send 成功后，presence 应记录 last_proactive_at。"""
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    presence = PresenceStore(tmp_path / "presence.json")
    session_manager = SessionManager(tmp_path)
    loop = ProactiveLoop(
        feed_registry=_DummyFeedRegistry(),
        session_manager=session_manager,
        provider=_DummyProvider(),
        push_tool=push_tool,
        config=ProactiveConfig(
            enabled=True,
            default_channel="telegram",
            default_chat_id="456",
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        presence=presence,
    )

    assert presence.get_last_proactive_at("telegram:456") is None
    await loop._send("你好")
    assert presence.get_last_proactive_at("telegram:456") is not None


# ── P1-1: record_action/mark_items_seen 与 session_key 解耦 ─────────


@pytest.mark.asyncio
async def test_sent_without_session_key_still_marks_items_seen(tmp_path):
    """sent=True 且 session_key 为空时：mark_items_seen 执行，mark_delivery 跳过。"""
    from unittest.mock import MagicMock, patch
    from proactive.engine import ProactiveEngine
    from proactive.state import ProactiveStateStore
    from proactive.anyaction import AnyActionGate, QuotaStore

    state = ProactiveStateStore(tmp_path / "state.json")
    seen_calls: list = []
    delivery_calls: list = []

    original_mark_seen = state.mark_items_seen
    original_mark_delivery = state.mark_delivery

    def _track_seen(entries, **kw):
        seen_calls.append(entries)
        return original_mark_seen(entries, **kw)

    def _track_delivery(session_key, delivery_key, **kw):
        delivery_calls.append((session_key, delivery_key))
        return original_mark_delivery(session_key, delivery_key, **kw)

    state.mark_items_seen = _track_seen
    state.mark_delivery = _track_delivery

    item = FeedItem(
        source_name="S", source_type="rss", title="T",
        content="c", url="https://x.com/1", author=None, published_at=None,
    )

    class _Sense:
        def compute_energy(self): return 0.0
        def collect_recent(self): return []
        def collect_recent_proactive(self, n=5): return []
        def compute_interruptibility(self, **kw): return 1.0, {"f_time":1,"f_reply":1,"f_activity":1,"f_fatigue":1,"random_delta":0}
        async def fetch_items(self, n): return [item]
        def filter_new_items(self, items): return items, [("s:s","id1")], []
        def read_memory_text(self): return ""
        def has_global_memory(self): return False
        def last_user_at(self): return None
        def target_session_key(self): return ""  # 空 session_key
        def quiet_hours(self): return 23, 10, 0.0

    class _Decide:
        async def score_features(self, **kw): return None
        async def compose_message(self, **kw): return ""
        async def reflect(self, items, recent, **kw):
            class D:
                score = 0.9; should_send = True; message = "hi"; reasoning = "ok"; evidence_item_ids = []
            return D()
        def randomize_decision(self, d): return d, 0.0
        def resolve_evidence_item_ids(self, d, items): return ["id1"]
        def build_delivery_key(self, ids, msg): return "key1"
        def semantic_entries(self, items): return [{"source_key":"s:s","item_id":"id1","text":"t"}]
        def item_id_for(self, item): return "id1"

    class _Act:
        async def send(self, msg, meta=None): return True  # 发送成功

    cfg = ProactiveConfig(
        enabled=True, default_channel="", default_chat_id="",
        anyaction_enabled=False,
        feature_scoring_enabled=False,
        score_pre_threshold=0.0,
        score_llm_threshold=0.0,
    )
    engine = ProactiveEngine(
        cfg=cfg, state=state, presence=None, rng=None,
        sense=_Sense(), decide=_Decide(), act=_Act(),
    )
    await engine.tick()

    assert len(seen_calls) > 0, "mark_items_seen 应被调用"
    assert len(delivery_calls) == 0, "session_key 为空时 mark_delivery 不应调用"


# ── P2-4: gate 拒绝返回值语义 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_quota_exhausted_returns_zero(tmp_path):
    """gate 拒绝原因为 quota_exhausted → tick 返回 0.0（调度最长间隔）。"""
    from proactive.engine import ProactiveEngine
    from proactive.state import ProactiveStateStore
    from proactive.anyaction import AnyActionGate, QuotaStore
    from datetime import timezone

    state = ProactiveStateStore(tmp_path / "state.json")

    class _Gate:
        def should_act(self, *, now_utc, last_user_at):
            return False, {"reason": "quota_exhausted", "used_today": 24, "remaining_today": 0}
        def record_action(self, *, now_utc): pass

    class _Sense:
        def compute_energy(self): return 0.5
        def collect_recent(self): return []
        def collect_recent_proactive(self, n=5): return []
        def compute_interruptibility(self, **kw): return 1.0, {"f_time":1,"f_reply":1,"f_activity":1,"f_fatigue":1,"random_delta":0}
        async def fetch_items(self, n): return []
        def filter_new_items(self, items): return [], [], []
        def read_memory_text(self): return ""
        def has_global_memory(self): return False
        def last_user_at(self): return None
        def target_session_key(self): return "telegram:123"
        def quiet_hours(self): return 23, 10, 0.0

    class _Decide:
        async def score_features(self, **kw): return None
        async def compose_message(self, **kw): return ""
        async def reflect(self, *a, **kw): raise AssertionError("不应调用")
        def randomize_decision(self, d): return d, 0.0
        def resolve_evidence_item_ids(self, d, items): return []
        def build_delivery_key(self, ids, msg): return ""
        def semantic_entries(self, items): return []
        def item_id_for(self, item): return ""

    class _Act:
        async def send(self, msg, meta=None): return False

    cfg = ProactiveConfig(enabled=True, anyaction_enabled=True, default_channel="telegram", default_chat_id="123")
    engine = ProactiveEngine(
        cfg=cfg, state=state, presence=None, rng=None,
        sense=_Sense(), decide=_Decide(), act=_Act(),
        anyaction=_Gate(),
    )
    result = await engine.tick()
    assert result == 0.0


@pytest.mark.asyncio
async def test_gate_min_interval_returns_none(tmp_path):
    """gate 拒绝原因为 min_interval → tick 返回 None（调度器按能量自算）。"""
    from proactive.engine import ProactiveEngine
    from proactive.state import ProactiveStateStore

    state = ProactiveStateStore(tmp_path / "state.json")

    class _Gate:
        def should_act(self, *, now_utc, last_user_at):
            return False, {"reason": "min_interval", "used_today": 1, "remaining_today": 23, "seconds_since_last_action": 60}
        def record_action(self, *, now_utc): pass

    class _Sense:
        def compute_energy(self): return 0.5
        def collect_recent(self): return []
        def collect_recent_proactive(self, n=5): return []
        def compute_interruptibility(self, **kw): return 1.0, {"f_time":1,"f_reply":1,"f_activity":1,"f_fatigue":1,"random_delta":0}
        async def fetch_items(self, n): return []
        def filter_new_items(self, items): return [], [], []
        def read_memory_text(self): return ""
        def has_global_memory(self): return False
        def last_user_at(self): return None
        def target_session_key(self): return "telegram:123"
        def quiet_hours(self): return 23, 10, 0.0

    class _Decide:
        async def score_features(self, **kw): return None
        async def compose_message(self, **kw): return ""
        async def reflect(self, *a, **kw): raise AssertionError("不应调用")
        def randomize_decision(self, d): return d, 0.0
        def resolve_evidence_item_ids(self, d, items): return []
        def build_delivery_key(self, ids, msg): return ""
        def semantic_entries(self, items): return []
        def item_id_for(self, item): return ""

    class _Act:
        async def send(self, msg, meta=None): return False

    cfg = ProactiveConfig(enabled=True, anyaction_enabled=True, default_channel="telegram", default_chat_id="123")
    engine = ProactiveEngine(
        cfg=cfg, state=state, presence=None, rng=None,
        sense=_Sense(), decide=_Decide(), act=_Act(),
        anyaction=_Gate(),
    )
    result = await engine.tick()
    assert result is None


# ── P2-3: 时区配置校验 ────────────────────────────────────────────


def test_invalid_timezone_raises_when_anyaction_enabled():
    """anyaction_enabled=True 且时区无效 → 配置加载 fail-fast。"""
    from agent.config import _validated_timezone
    with pytest.raises(ValueError, match="anyaction_timezone"):
        _validated_timezone("Invalid/Timezone_XYZ", enabled=True)


def test_invalid_timezone_ignored_when_anyaction_disabled():
    """anyaction_enabled=False 时，无效时区不报错（功能未启用，副作用不扩大）。"""
    from agent.config import _validated_timezone
    result = _validated_timezone("Invalid/Timezone_XYZ", enabled=False)
    assert result == "Invalid/Timezone_XYZ"


def test_safe_zone_fallback_on_invalid_timezone():
    """运行时 _safe_zone 遇到无效时区应回退 UTC，不抛异常。"""
    from proactive.anyaction import _safe_zone
    from zoneinfo import ZoneInfo
    tz = _safe_zone("Not/A/Real_Zone")
    assert tz == ZoneInfo("UTC")


# ── Fix A: delivery_key 不含 message 文本 ──────────────────────────

def test_delivery_key_with_evidence_ignores_message():
    """有证据时，delivery_key 只取决于 item_ids，与 message 文本无关。"""
    from proactive.ports import DefaultDecidePort
    from unittest.mock import MagicMock
    decide = DefaultDecidePort(
        reflector=MagicMock(),
        randomize_fn=lambda d: (d, 0.0),
        source_key_fn=lambda i: "rss:test",
        item_id_fn=lambda i: "id1",
        semantic_text_fn=lambda i, n: "",
        semantic_text_max_chars=240,
    )
    ids = ["u_abc123"]
    key1 = decide.build_delivery_key(ids, "版本A的消息措辞")
    key2 = decide.build_delivery_key(ids, "版本B完全不同的说法")
    assert key1 == key2, "同 evidence 不同措辞应产生相同 delivery_key"


def test_delivery_key_empty_ids_uses_time_bucket_and_prefix():
    """空 evidence 时，不同内容不退化到同一 hash。"""
    from proactive.ports import DefaultDecidePort
    from unittest.mock import MagicMock
    decide = DefaultDecidePort(
        reflector=MagicMock(),
        randomize_fn=lambda d: (d, 0.0),
        source_key_fn=lambda i: "rss:test",
        item_id_fn=lambda i: "id1",
        semantic_text_fn=lambda i, n: "",
        semantic_text_max_chars=240,
    )
    key1 = decide.build_delivery_key([], "消息A内容")
    key2 = decide.build_delivery_key([], "消息B完全不同的内容")
    assert key1 != key2, "空 evidence 不同消息前缀应产生不同 delivery_key"


# ── Fix C: rejection_cooldown ────────────────────────────────────


def test_rejection_cooldown_mark_and_check(tmp_path):
    """mark_rejection_cooldown → is_rejection_cooled 返回 True；超时后返回 False。"""
    from datetime import timezone
    from proactive.state import ProactiveStateStore

    state = ProactiveStateStore(tmp_path / "state.json")
    entries = [("rss:test", "u_abc123")]

    # 未标记时不冷却
    assert not state.is_rejection_cooled("rss:test", "u_abc123", ttl_hours=12)

    state.mark_rejection_cooldown(entries, hours=12)
    assert state.is_rejection_cooled("rss:test", "u_abc123", ttl_hours=12)

    # 模拟过期
    state._state["rejection_cooldown"]["rss:test"]["u_abc123"] = (
        (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat()
    )
    assert not state.is_rejection_cooled("rss:test", "u_abc123", ttl_hours=12)


def test_rejection_cooldown_disabled_when_hours_zero(tmp_path):
    """hours=0 时，mark/check 均为 no-op。"""
    from proactive.state import ProactiveStateStore
    state = ProactiveStateStore(tmp_path / "state.json")
    entries = [("rss:test", "u_x")]
    state.mark_rejection_cooldown(entries, hours=0)  # should be no-op
    assert not state.is_rejection_cooled("rss:test", "u_x", ttl_hours=0)


def test_rejection_cooldown_filters_in_next_tick(tmp_path):
    """LLM 拒绝后条目进入 rejection_cooldown，下轮 filter_new_items 跳过该条目。"""
    from proactive.state import ProactiveStateStore
    from proactive.components import ProactiveItemFilter
    from proactive.item_id import compute_item_id, compute_source_key
    from unittest.mock import MagicMock

    item = FeedItem(
        source_name="test", source_type="rss",
        title="被拒绝的内容", content="", url="https://example.com/rejected",
        author=None, published_at=None,
    )
    source_key = compute_source_key(item)
    item_id = compute_item_id(item)

    state = ProactiveStateStore(tmp_path / "state.json")
    # 模拟 LLM 已拒绝，写入 cooldown
    state.mark_rejection_cooldown([(source_key, item_id)], hours=12)

    cfg = MagicMock()
    cfg.dedupe_seen_ttl_hours = 336
    cfg.semantic_dedupe_enabled = False
    cfg.llm_reject_cooldown_hours = 12

    item_filter = ProactiveItemFilter(
        cfg=cfg,
        state=state,
        source_key_fn=compute_source_key,
        item_id_fn=compute_item_id,
        semantic_text_fn=lambda i, n: "",
        build_tfidf_vectors_fn=lambda texts, n: [],
        cosine_fn=lambda a, b: 0.0,
    )

    new_items, new_entries, _ = item_filter.filter_new_items([item])
    assert len(new_items) == 0, "rejection_cooldown 中的条目应被过滤掉"


@pytest.mark.asyncio
async def test_llm_rejection_writes_rejection_cooldown(tmp_path):
    """LLM 返回 should_send=False → new_entries 写入 rejection_cooldown，不写 seen_items。"""
    from proactive.state import ProactiveStateStore
    from proactive.engine import ProactiveEngine
    from proactive.item_id import compute_item_id, compute_source_key
    from unittest.mock import AsyncMock, MagicMock

    item = FeedItem(
        source_name="src", source_type="rss",
        title="LLM 拒绝的条目", content="", url="https://example.com/llm-rejected",
        author=None, published_at=None,
    )
    source_key = compute_source_key(item)
    item_id = compute_item_id(item)

    state = ProactiveStateStore(tmp_path / "state.json")

    sense = MagicMock()
    sense.compute_energy.return_value = 0.5
    sense.collect_recent.return_value = []
    sense.compute_interruptibility.return_value = (
        0.5,
        {"f_time": 0.5, "f_reply": 0.5, "f_activity": 0.5, "f_fatigue": 0.5, "random_delta": 0.0},
    )
    sense.fetch_items = AsyncMock(return_value=[item])
    sense.filter_new_items.return_value = (
        [item],
        [(source_key, item_id)],
        [],
    )
    sense.read_memory_text.return_value = ""
    sense.has_global_memory.return_value = False
    sense.last_user_at.return_value = None
    sense.target_session_key.return_value = "telegram:123"
    sense.quiet_hours.return_value = (23, 8, 0.0)
    sense.refresh_sleep_context.return_value = False
    sense.sleep_context.return_value = None

    # decide: reflect 返回 should_send=False
    from proactive.loop import _Decision
    decide = MagicMock()
    decide.reflect = AsyncMock(return_value=_Decision(
        score=0.2, should_send=False, message="", reasoning="not interesting"
    ))
    decide.randomize_decision.side_effect = lambda d: (d, 0.0)
    decide.semantic_entries.return_value = []

    cfg = MagicMock()
    cfg.anyaction_enabled = False
    cfg.score_weight_energy = 0.40
    cfg.score_weight_content = 0.30
    cfg.score_weight_recent = 0.20
    cfg.score_recent_scale = 8.0
    cfg.score_content_halfsat = 2.5
    cfg.score_pre_threshold = 0.01
    cfg.score_llm_threshold = 0.01   # 很低，确保进 LLM
    cfg.items_per_source = 5
    cfg.interest_filter.enabled = False
    cfg.feature_scoring_enabled = False
    cfg.threshold = 0.9   # 高 threshold → should_send 不触发
    cfg.dedupe_seen_ttl_hours = 336
    cfg.delivery_dedupe_hours = 10
    cfg.semantic_dedupe_window_hours = 72
    cfg.llm_reject_cooldown_hours = 12

    engine = ProactiveEngine(
        cfg=cfg, state=state, presence=None,
        rng=None, sense=sense, decide=decide, act=MagicMock(),
    )

    await engine.tick()

    # rejection_cooldown 应写入
    assert state.is_rejection_cooled(source_key, item_id, ttl_hours=12), \
        "LLM 拒绝后条目应进入 rejection_cooldown"

    # seen_items 不应写入
    assert not state.is_item_seen(source_key=source_key, item_id=item_id, ttl_hours=336), \
        "LLM 拒绝后条目不应写入 seen_items（仅软冷却）"
