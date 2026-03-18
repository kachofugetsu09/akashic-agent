import asyncio
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import unittest.mock as mock

import pytest

from agent.memory import MemoryStore
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from proactive.event import GenericContentEvent
from proactive.config import ProactiveConfig
from proactive.tick import (
    DecisionContext,
    EvaluateResult,
    GateSenseResult,
    ProactiveEngine,
    _build_recent_proactive_context_signal,
)
from proactive.item_id import compute_item_id
from proactive.loop import ProactiveLoop
from proactive.loop_helpers import _parse_decision
from proactive.ports import ProactiveSendMeta, ProactiveSourceRef, RecentProactiveMessage
from proactive.presence import PresenceStore
from session.manager import SessionManager


def _utc(**kwargs) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**kwargs)


class _DummyProvider:
    async def chat(self, **kwargs):
        raise RuntimeError("not used in this test")


def _build_loop(
    tmp_path, push_tool, chat_id: str = "7674283004", default_channel: str = "telegram"
):
    session_manager = SessionManager(tmp_path)
    return (
        ProactiveLoop(
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
        ),
        session_manager,
    )


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


@pytest.fixture(autouse=True)
def _isolate_mcp_alert_sources():
    with mock.patch("proactive.mcp_sources.fetch_alert_events", return_value=[]), mock.patch(
        "proactive.mcp_sources.acknowledge_events", return_value=None
    ), mock.patch(
        "proactive.mcp_sources.fetch_content_events", return_value=[]
    ), mock.patch(
        "proactive.mcp_sources.acknowledge_content_entries", return_value=None
    ):
        yield


def test_parse_decision_string_false_is_false():
    d = _parse_decision(
        '{"score": 0.9, "should_send": "false", "message": "hello", "reasoning": "r"}'
    )
    assert d.should_send is False


def test_build_recent_proactive_context_signal_marks_followup_fatigue():
    last_user_at = datetime(2026, 3, 15, 1, 0, tzinfo=timezone.utc)
    msgs = [
        RecentProactiveMessage(
            content="第一条补充",
            timestamp=datetime(2026, 3, 15, 1, 5, tzinfo=timezone.utc),
        ),
        RecentProactiveMessage(
            content="第二条补充",
            timestamp=datetime(2026, 3, 15, 1, 10, tzinfo=timezone.utc),
        ),
    ]

    signal = _build_recent_proactive_context_signal(
        msgs,
        last_user_at=last_user_at,
        candidate_items_count=0,
    )

    assert signal["exists"] is True
    assert signal["count_since_last_user"] == 2
    assert signal["already_followed_up"] is True
    assert signal["followup_fatigue"] == "high"
    assert signal["has_new_feed"] is False
    assert signal["latest_excerpt"] == "第二条补充"


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
async def test_proactive_loop_run_waits_before_first_tick():
    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._cfg = SimpleNamespace(
        threshold=0.36,
        default_channel="telegram",
        default_chat_id="7674283004",
    )
    loop._manual_trigger_event = asyncio.Event()
    ticked = {"done": False}

    async def _tick():
        ticked["done"] = True
        loop._running = False
        return 0.5

    interval_calls = {"count": 0}

    def _next_interval(base_score=None):
        interval_calls["count"] += 1
        assert ticked["done"] is False, "run() 首次启动不应立即执行 tick"
        loop._running = False
        loop._manual_trigger_event.set()
        return 60

    loop._tick = _tick
    loop._next_interval = _next_interval

    await loop.run()

    assert ticked["done"] is False
    assert interval_calls["count"] == 1


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
async def test_engine_gate_and_sense_returns_structured_result_for_scheduler_reject():
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

    result = await engine._gate_and_sense(DecisionContext())

    assert result.proceed is False
    assert result.return_score is None
    assert result.reason_code == "scheduler_reject"


@pytest.mark.asyncio
async def test_engine_gate_and_sense_bypasses_quota_when_alert_exists():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        dedupe_seen_ttl_hours=24,
        delivery_dedupe_hours=24,
        semantic_dedupe_window_hours=24,
        anyaction_enabled=True,
        score_recent_scale=10,
        score_weight_energy=1.0,
        score_weight_recent=1.0,
        score_pre_threshold=0.0,
    )
    engine._state = SimpleNamespace(cleanup=MagicMock())
    engine._sense = SimpleNamespace(
        last_user_at=lambda: None,
        refresh_sleep_context=lambda: False,
        sleep_context=lambda: None,
        compute_energy=lambda: 0.4,
        collect_recent=lambda: [],
        compute_interruptibility=lambda **kwargs: (
            1.0,
            {
                "f_reply": 1.0,
                "f_activity": 1.0,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            },
        ),
    )
    engine._anyaction = SimpleNamespace(
        should_act=lambda **kwargs: (False, {"reason": "quota_exhausted"})
    )

    fake_alert_payload = [
        {
            "event_id": "alert-001",
            "kind": "alert",
            "source_type": "health_event",
            "source_name": "fitbit",
            "content": "心率偏高",
            "severity": "high",
        }
    ]
    with mock.patch(
        "proactive.mcp_sources.fetch_alert_events", return_value=fake_alert_payload
    ):
        result = await engine._gate_and_sense(DecisionContext())

    assert result.proceed is True
    assert result.return_score is None
    assert result.reason_code == "continue"


@pytest.mark.asyncio
async def test_engine_gate_and_sense_returns_structured_below_threshold_result():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        dedupe_seen_ttl_hours=24,
        delivery_dedupe_hours=24,
        semantic_dedupe_window_hours=24,
        anyaction_enabled=False,
        score_recent_scale=10,
        score_weight_energy=1.0,
        score_weight_recent=1.0,
        score_pre_threshold=0.5,
    )
    engine._state = SimpleNamespace(cleanup=MagicMock())
    engine._sense = SimpleNamespace(
        refresh_sleep_context=lambda: False,
        sleep_context=lambda: None,
        compute_energy=lambda: 0.2,
        collect_recent=lambda: [],
        compute_interruptibility=lambda **kwargs: (
            1.0,
            {
                "f_reply": 1.0,
                "f_activity": 1.0,
                "f_fatigue": 1.0,
                "random_delta": 0.0,
            },
        ),
    )
    engine._try_skill_action = AsyncMock()
    ctx = DecisionContext()
    score = ctx.ensure_score()

    result = await engine._gate_and_sense(ctx)

    assert result.proceed is False
    assert result.return_score == score.pre_score
    assert result.reason_code == "below_threshold"
    engine._try_skill_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_gate_and_sense_returns_structured_snapshot():
    sleep_ctx = SimpleNamespace(
        sleep_modifier=0.5,
        state="sleeping",
        available=True,
        prob=0.8,
        data_lag_min=5,
    )
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        dedupe_seen_ttl_hours=24,
        delivery_dedupe_hours=24,
        semantic_dedupe_window_hours=24,
        anyaction_enabled=False,
        score_recent_scale=10,
        score_weight_energy=1.0,
        score_weight_recent=1.0,
        score_pre_threshold=0.0,
    )
    engine._state = SimpleNamespace(cleanup=MagicMock())
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

    fake_mcp_payload = [{
        "event_id": "test-001", "kind": "alert", "source_type": "health_event",
        "source_name": "fitbit", "title": "hr_elevated_rest",
        "content": "心率偏高", "severity": "high", "published_at": None,
    }]
    import unittest.mock as mock
    with mock.patch("proactive.mcp_sources.fetch_alert_events", return_value=fake_mcp_payload):
        result = await engine._gate_and_sense(ctx)

    assert result.sleep_state == "sleeping"
    assert result.sleep_available is True
    assert result.health_event_count == 1
    assert result.energy == 0.2
    assert result.recent_count == 1
    assert result.interruptibility == 0.75
    assert result.interrupt_factor == ctx.ensure_sense().interrupt_factor


@pytest.mark.asyncio
async def test_engine_evaluate_no_candidates_falls_through_to_draw_threshold():
    # 无候选内容时不再硬退出，继续走 draw_score 门槛判断。
    # de=0.2 dr=0.1 D_content=0 → base_score 远低于 0.6 阈值，应走 draw_score_below_threshold。
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        score_content_halfsat=3.0,
        score_weight_energy=1.0,
        score_weight_content=1.0,
        score_weight_recent=1.0,
        score_llm_threshold=0.6,
    )
    engine._rng = None
    engine._sense = SimpleNamespace(
        target_session_key=lambda: "",
        has_global_memory=lambda: False,
    )
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

    engine._load_content_snapshot = AsyncMock(return_value=([], [], []))

    result = await engine._evaluate(ctx)

    assert result.proceed is False
    assert result.reason_code == "no_valid_source"
    assert result.base_score == 0.0
    assert result.draw_score == 0.0
    assert result.force_reflect is False


@pytest.mark.asyncio
async def test_engine_evaluate_draw_threshold_still_triggers_skill_action():
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        score_content_halfsat=3.0,
        score_weight_energy=1.0,
        score_weight_content=1.0,
        score_weight_recent=1.0,
        score_llm_threshold=0.95,
    )
    engine._rng = None
    engine._sense = SimpleNamespace(
        target_session_key=lambda: "",
        has_global_memory=lambda: False,
    )
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
        GenericContentEvent(
            event_id="test-a",
            source_name="Test",
            source_type="rss",
            title="A",
            content="body",
            url="https://example.com/a",
            published_at=None,
        )
    ]
    sense.health_events = []
    score.force_reflect = False
    sense.energy = 0.3
    fetch.has_memory = False

    engine._load_content_snapshot = AsyncMock(
        return_value=(list(fetch.new_items), [("rss:test", "test-a")], [])
    )

    result = await engine._evaluate(ctx)

    assert result.proceed is False
    assert result.reason_code == "draw_score_below_threshold"
    engine._try_skill_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_evaluate_returns_structured_snapshot():
    payloads = [
        {
            "event_id": "evt-1",
            "kind": "content",
            "source_type": "rss",
            "source_name": "Test",
            "title": "A",
            "content": "body",
            "url": "https://example.com/a",
            "published_at": None,
            "ack_server": "feed",
        }
    ]
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        interest_filter=SimpleNamespace(enabled=False),
        items_per_source=3,
        score_content_halfsat=3.0,
        score_weight_energy=1.0,
        score_weight_content=1.0,
        score_weight_recent=1.0,
        score_llm_threshold=0.6,
    )
    engine._sense = SimpleNamespace(
        has_global_memory=lambda: True,
        target_session_key=lambda: "",
    )
    engine._presence = None
    engine._rng = None
    engine._state = SimpleNamespace()
    engine._try_skill_action = AsyncMock()
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    sense.de = 0.2
    sense.dr = 0.1
    sense.interrupt_factor = 1.0
    sense.interruptibility = 1.0
    sense.health_events = []
    with mock.patch("proactive.mcp_sources.fetch_content_events", return_value=payloads):
        result = await engine._evaluate(ctx)

    assert result.total_items == 1
    assert result.discovered_count == 1
    assert result.selected_count == 1
    assert result.semantic_duplicate_count == 0
    assert result.has_memory is True
    feed_events = ctx.ensure_fetch().new_items
    assert len(feed_events) == 1
    assert isinstance(feed_events[0], GenericContentEvent)
    assert feed_events[0].event_id == "evt-1"
    assert ctx.ensure_fetch().new_entries == [
        ("mcp:feed:evt-1", compute_item_id(feed_events[0]))
    ]


@pytest.mark.asyncio
async def test_engine_evaluate_skips_rejection_cooled_events(tmp_path):
    from proactive.state import ProactiveStateStore

    payloads = [
        {
            "event_id": "evt-1",
            "kind": "content",
            "source_type": "rss",
            "source_name": "Test",
            "title": "A",
            "content": "body",
            "url": "https://example.com/a",
            "published_at": None,
            "ack_server": "feed",
        }
    ]
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        llm_reject_cooldown_hours=12,
        score_content_halfsat=3.0,
        score_weight_energy=1.0,
        score_weight_content=1.0,
        score_weight_recent=1.0,
        score_llm_threshold=0.6,
    )
    state = ProactiveStateStore(tmp_path / "state.json")
    engine._state = state
    engine._sense = SimpleNamespace(
        has_global_memory=lambda: False,
        target_session_key=lambda: "",
    )
    engine._presence = None
    engine._rng = None
    engine._try_skill_action = AsyncMock()
    engine._decide = SimpleNamespace()
    ctx = DecisionContext()
    ctx.state.now_utc = datetime.now(timezone.utc)
    sense = ctx.ensure_sense()
    sense.de = 0.2
    sense.dr = 0.1
    sense.interrupt_factor = 1.0
    sense.interruptibility = 1.0
    sense.health_events = []

    item_id = compute_item_id(GenericContentEvent.from_mcp_payload(payloads[0]))
    state.mark_rejection_cooldown([("mcp:feed:evt-1", item_id)], hours=12)

    with mock.patch("proactive.mcp_sources.fetch_content_events", return_value=payloads):
        result = await engine._evaluate(ctx)

    assert result.total_items == 0
    assert result.selected_count == 0
    assert ctx.ensure_fetch().new_items == []
    assert ctx.ensure_fetch().new_entries == []


def test_populate_decision_signals_keeps_health_subset_by_source_type():
    from proactive.event import GenericAlertEvent

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(score_llm_threshold=0.6, threshold=0.7)
    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    fetch = ctx.ensure_fetch()
    score = ctx.ensure_score()
    decide = ctx.ensure_decide()
    act = ctx.ensure_act()

    # 1. 构造一条健康告警和一条普通告警，验证 health_events 只保留健康来源子集。
    sense.health_events = [
        GenericAlertEvent.from_mcp_payload(
            {
                "event_id": "h1",
                "kind": "alert",
                "source_type": "health_event",
                "source_name": "fitbit",
                "content": "心率偏高",
                "severity": "high",
            }
        ),
        GenericAlertEvent.from_mcp_payload(
            {
                "event_id": "c1",
                "kind": "alert",
                "source_type": "calendar_alert",
                "source_name": "calendar",
                "content": "10分钟后开会",
                "severity": "high",
            }
        ),
    ]
    fetch.new_items = []
    score.sent_24h = 0
    score.fresh_items_24h = 0
    score.pre_score = 0.0
    score.base_score = 0.0
    score.draw_score = 0.0
    sense.interrupt_detail = {"f_reply": 1.0, "f_activity": 1.0, "f_fatigue": 1.0}
    sense.interruptibility = 1.0
    sense.sleep_ctx = None

    # 2. 填充 decision_signals，并校验 alert/health 两层信号是否符合当前 MCP 语义。
    engine._populate_decision_signals(ctx)

    assert len(decide.decision_signals["alert_events"]) == 2
    assert len(decide.decision_signals["health_events"]) == 1
    assert decide.decision_signals["health_events"][0]["source_type"] == "health_event"
    assert len(act.high_events) == 2


def test_engine_trace_writer_emits_strategy_envelope():
    emitted: list[dict] = []
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._trace_writer = emitted.append

    engine._trace(
        DecisionContext(),
        stage="gate_and_sense",
        result=GateSenseResult(
            proceed=True,
            return_score=None,
            reason_code="continue",
            sleep_state="awake",
            sleep_available=True,
            health_event_count=0,
            energy=0.5,
            recent_count=1,
            interruptibility=0.8,
            interrupt_factor=0.92,
            sleep_mod=1.0,
        ),
    )

    assert emitted[0]["trace_type"] == "proactive_stage"
    assert emitted[0]["subject"]["kind"] == "global"
    assert emitted[0]["payload"]["stage"] == "gate_and_sense"
    assert emitted[0]["payload"]["result"]["reason_code"] == "continue"


def test_engine_trace_writer_serializes_none_return_score():
    emitted: list[dict] = []
    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._trace_writer = emitted.append

    engine._trace(
        DecisionContext(),
        stage="evaluate",
        result=EvaluateResult(
            proceed=False,
            return_score=None,
            reason_code="draw_score_force_reflect",
            base_score=0.2,
            draw_score=0.1,
            force_reflect=True,
            total_items=0,
            discovered_count=0,
            selected_count=0,
            semantic_duplicate_count=0,
            has_memory=False,
        ),
    )

    assert emitted[0]["payload"]["result"]["return_score"] is None


def test_emit_observe_decision_includes_sent_message_for_send_stage():
    emitted = []

    class _Writer:
        def emit(self, trace):
            emitted.append(trace)

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._observe_writer = _Writer()
    engine._cfg = SimpleNamespace(threshold=0.7)
    ctx = DecisionContext()
    ctx.state.tick_id = "tick-1"
    ctx.state.session_key = "telegram:1"
    decide = ctx.ensure_decide()
    decide.decision_message = "实际发出的正文"
    decide.should_send = True

    engine._emit_observe_decision(
        ctx,
        stage="send",
        reason_code="sent",
        should_send=True,
        action="chat",
        delivery_attempted=True,
        delivery_result="sent",
    )

    assert emitted[0].stage == "send"
    assert emitted[0].sent_message == "实际发出的正文"


@pytest.mark.asyncio
async def test_send_uses_configured_channel(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop, _ = _build_loop(
        tmp_path, push_tool, chat_id="7674283004", default_channel="qq"
    )

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
    assert (
        "HLTV | Rare Atom signs x9 | https://example.com/ra" in history[-1]["content"]
    )


# ── Dynamic energy / presence 集成测试 ────────────────────────────


def _make_presence(tmp_path, session_key: str, last_user_minutes_ago: float | None):
    p = PresenceStore(tmp_path / "presence.json")
    if last_user_minutes_ago is not None:
        p.record_user_message(session_key, now=_utc(minutes=last_user_minutes_ago))
    return p


def _build_loop_with_presence(tmp_path, provider, presence):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    session_manager = SessionManager(tmp_path)
    loop = ProactiveLoop(
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
    provider.chat = AsyncMock(return_value=_Resp("<no_content/>"))

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 72)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    loop = ProactiveLoop(
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
        rng=random.Random(1),  # 固定种子，结果确定
    )

    await loop._tick()

    assert provider.chat.call_count >= 1


@pytest.mark.asyncio
async def test_tick_calls_llm_when_no_presence_data(tmp_path):
    """无心跳记录（从未收到消息），视作电量为 0，应进入 LLM 反思。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock(return_value=_Resp("<no_content/>"))

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=None)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)

    await loop._tick()

    assert provider.chat.call_count >= 1


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
    provider.chat = AsyncMock(return_value=_Resp("<no_content/>"))

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 24)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    session_manager = SessionManager(tmp_path)
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户喜欢魂类游戏，最近在玩 Elden Ring。")

    loop = ProactiveLoop(
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

    assert provider.chat.call_count >= 1


@pytest.mark.asyncio
async def test_tick_without_new_items_still_runs_when_low_energy_and_memory(tmp_path):
    """no new feed items + low energy + memory 时仍可进入 LLM。"""
    import random
    from agent.memory import MemoryStore

    provider = _DummyProvider()
    provider.chat = AsyncMock(return_value=_Resp("<no_content/>"))

    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 24)
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")
    session_manager = SessionManager(tmp_path)
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户喜欢 Python。")

    loop = ProactiveLoop(
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

    assert provider.chat.call_count >= 1


@pytest.mark.asyncio
async def test_send_records_proactive_sent_in_presence(tmp_path):
    """_send 成功后，presence 应记录 last_proactive_at。"""
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    presence = PresenceStore(tmp_path / "presence.json")
    session_manager = SessionManager(tmp_path)
    loop = ProactiveLoop(
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


# ── P2-4: gate 拒绝返回值语义 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_quota_exhausted_returns_zero(tmp_path):
    """gate 拒绝原因为 quota_exhausted → tick 返回 0.0（调度最长间隔）。"""
    from proactive.tick import ProactiveEngine
    from proactive.state import ProactiveStateStore
    from proactive.anyaction import AnyActionGate, QuotaStore
    from datetime import timezone

    state = ProactiveStateStore(tmp_path / "state.json")

    class _Gate:
        def should_act(self, *, now_utc, last_user_at):
            return False, {
                "reason": "quota_exhausted",
                "used_today": 24,
                "remaining_today": 0,
            }

        def record_action(self, *, now_utc):
            pass

    class _Sense:
        def compute_energy(self):
            return 0.5

        def collect_recent(self):
            return []

        def collect_recent_proactive(self, n=5):
            return []

        def compute_interruptibility(self, **kw):
            return 1.0, {
                "f_time": 1,
                "f_reply": 1,
                "f_activity": 1,
                "f_fatigue": 1,
                "random_delta": 0,
            }

        async def fetch_items(self, n):
            return []

        def filter_new_items(self, items):
            return [], [], []

        def read_memory_text(self):
            return ""

        def has_global_memory(self):
            return False

        def last_user_at(self):
            return None

        def target_session_key(self):
            return "telegram:123"

        def quiet_hours(self):
            return 23, 10, 0.0

    class _Decide:
        async def score_features(self, **kw):
            return None

        async def compose_message(self, **kw):
            return ""

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return ""

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return ""

    class _Act:
        async def send(self, msg, meta=None):
            return False

    cfg = ProactiveConfig(
        enabled=True,
        anyaction_enabled=True,
        default_channel="telegram",
        default_chat_id="123",
    )
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=_Act(),
        anyaction=_Gate(),
    )
    result = await engine.tick()
    assert result == 0.0


@pytest.mark.asyncio
async def test_gate_min_interval_returns_none(tmp_path):
    """gate 拒绝原因为 min_interval → tick 返回 None（调度器按能量自算）。"""
    from proactive.tick import ProactiveEngine
    from proactive.state import ProactiveStateStore

    state = ProactiveStateStore(tmp_path / "state.json")

    class _Gate:
        def should_act(self, *, now_utc, last_user_at):
            return False, {
                "reason": "min_interval",
                "used_today": 1,
                "remaining_today": 23,
                "seconds_since_last_action": 60,
            }

        def record_action(self, *, now_utc):
            pass

    class _Sense:
        def compute_energy(self):
            return 0.5

        def collect_recent(self):
            return []

        def collect_recent_proactive(self, n=5):
            return []

        def compute_interruptibility(self, **kw):
            return 1.0, {
                "f_time": 1,
                "f_reply": 1,
                "f_activity": 1,
                "f_fatigue": 1,
                "random_delta": 0,
            }

        async def fetch_items(self, n):
            return []

        def filter_new_items(self, items):
            return [], [], []

        def read_memory_text(self):
            return ""

        def has_global_memory(self):
            return False

        def last_user_at(self):
            return None

        def target_session_key(self):
            return "telegram:123"

        def quiet_hours(self):
            return 23, 10, 0.0

    class _Decide:
        async def score_features(self, **kw):
            return None

        async def compose_message(self, **kw):
            return ""

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return []

        def build_delivery_key(self, ids, msg):
            return ""

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return ""

    class _Act:
        async def send(self, msg, meta=None):
            return False

    cfg = ProactiveConfig(
        enabled=True,
        anyaction_enabled=True,
        default_channel="telegram",
        default_chat_id="123",
    )
    engine = ProactiveEngine(
        cfg=cfg,
        state=state,
        presence=None,
        rng=None,
        sense=_Sense(),
        decide=_Decide(),
        act=_Act(),
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
        datetime.now(timezone.utc) - timedelta(hours=13)
    ).isoformat()
    assert not state.is_rejection_cooled("rss:test", "u_abc123", ttl_hours=12)


def test_rejection_cooldown_disabled_when_hours_zero(tmp_path):
    """hours=0 时，mark/check 均为 no-op。"""
    from proactive.state import ProactiveStateStore

    state = ProactiveStateStore(tmp_path / "state.json")
    entries = [("rss:test", "u_x")]
    state.mark_rejection_cooldown(entries, hours=0)  # should be no-op
    assert not state.is_rejection_cooled("rss:test", "u_x", ttl_hours=0)


def test_rejection_cooldown_filters_in_next_tick(tmp_path):
    """LLM 拒绝后条目进入 rejection_cooldown，状态层能识别冷却命中。"""
    from proactive.state import ProactiveStateStore
    from proactive.item_id import compute_item_id, compute_source_key

    item = GenericContentEvent(
        event_id="rejected",
        source_name="test",
        source_type="rss",
        title="被拒绝的内容",
        content="",
        url="https://example.com/rejected",
        published_at=None,
    )
    source_key = compute_source_key(item)
    item_id = compute_item_id(item)

    state = ProactiveStateStore(tmp_path / "state.json")
    state.mark_rejection_cooldown([(source_key, item_id)], hours=12)
    assert state.is_rejection_cooled(source_key, item_id, ttl_hours=12) is True


def _build_event(*, event_id: str, source_name: str, title: str, published_at=None):
    from proactive.event import GenericContentEvent

    return GenericContentEvent(
        event_id=event_id,
        source_type="rss",
        source_name=source_name,
        content=title,
        title=title,
        url="https://example.com/" + event_id,
        published_at=published_at,
    )


def test_prepare_compose_candidates_prefers_interest_ranked_items():
    from types import SimpleNamespace
    from proactive.tick import ProactiveEngine, DecisionContext

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        interest_filter=SimpleNamespace(
            enabled=True,
            memory_max_chars=4000,
            keyword_max_count=80,
            min_token_len=2,
            min_score=0.0,
            top_k=10,
            exploration_ratio=0.0,
        )
    )
    engine._decide = SimpleNamespace(item_id_for=lambda item: item.title)

    ctx = DecisionContext()
    fetch = ctx.ensure_fetch()
    decide = ctx.ensure_decide()
    act = ctx.ensure_act()
    decide.preference_block = "只关注 Niko 和 Major"

    # 1. 构造“无关大组 + 关注小组”的候选，顺序先无关后关注。
    fetch.new_items = [
        _build_event(event_id="o1", source_name="Other", title="Other match A"),
        _build_event(event_id="o2", source_name="Other", title="Other match B"),
        _build_event(event_id="o3", source_name="Other", title="Other match C"),
        _build_event(event_id="n1", source_name="HLTV", title="Niko semifinal"),
        _build_event(event_id="n2", source_name="HLTV", title="Niko final"),
    ]
    fetch.new_entries = [
        ("rss:other", "Other match A"),
        ("rss:other", "Other match B"),
        ("rss:other", "Other match C"),
        ("rss:hltv", "Niko semifinal"),
        ("rss:hltv", "Niko final"),
    ]

    # 2. 触发 compose 候选选择。
    engine._prepare_feature_compose_candidates(ctx)

    # 3. 应优先选择高兴趣组（Niko）。
    assert act.compose_items
    assert "Niko" in (act.compose_items[0].title or "")
    assert act.compose_entries == [
        ("rss:hltv", "Niko semifinal"),
        ("rss:hltv", "Niko final"),
    ]


def test_select_compose_items_sorts_group_by_published_at_asc():
    from types import SimpleNamespace
    from datetime import datetime, timezone, timedelta
    from proactive.tick import ProactiveEngine

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._decide = SimpleNamespace(item_id_for=lambda item: item.title)

    now = datetime.now(timezone.utc)
    newer = GenericContentEvent(
        event_id="niko-final",
        source_name="HLTV",
        source_type="rss",
        title="Niko final",
        content="",
        url="https://example.com/f",
        published_at=now,
    )
    older = GenericContentEvent(
        event_id="niko-semifinal",
        source_name="HLTV",
        source_type="rss",
        title="Niko semifinal",
        content="",
        url="https://example.com/s",
        published_at=now - timedelta(hours=3),
    )

    # 1. 逆序输入同组内容。
    items = [newer, older]
    entries: list[tuple[str, str]] = []

    # 2. 选择 compose 候选。
    compose_items, _ = engine._select_compose_items(items, entries)

    # 3. 组内应按发布时间正序。
    assert compose_items[0].title == "Niko semifinal"


def test_select_compose_items_keeps_top_interest_single_item_over_larger_lower_interest_group():
    from datetime import datetime, timezone, timedelta
    from proactive.tick import ProactiveEngine

    engine = ProactiveEngine.__new__(ProactiveEngine)

    now = datetime.now(timezone.utc)
    niko = GenericContentEvent(
        event_id="niko",
        source_name="NiKo Twitter",
        source_type="rss",
        title="NiKo Twitter clip",
        content="",
        url="https://example.com/niko",
        published_at=now,
    )
    hooxi = GenericContentEvent(
        event_id="hooxi",
        source_name="HLTV News",
        source_type="rss",
        title="HooXi on G2 changes",
        content="",
        url="https://example.com/hooxi",
        published_at=now - timedelta(minutes=10),
    )
    navi = GenericContentEvent(
        event_id="navi",
        source_name="HLTV News",
        source_type="rss",
        title="NAVI win EPL playoffs",
        content="",
        url="https://example.com/navi",
        published_at=now - timedelta(minutes=5),
    )

    # 1. 输入顺序已经代表 interest rank 结果：NiKo 最高，其余较低。
    items = [niko, hooxi, navi]
    entries = [
        ("rss:niko", "NiKo Twitter clip"),
        ("rss:hltv", "HooXi on G2 changes"),
        ("rss:hltv", "NAVI win EPL playoffs"),
    ]

    # 2. 选择 compose 候选时，应只围绕最高兴趣 seed 补上下文。
    compose_items, compose_entries = engine._select_compose_items(items, entries)

    # 3. 高兴趣单条不能再被更大但低兴趣的组覆盖。
    assert [item.title for item in compose_items] == ["NiKo Twitter clip"]
    assert len(compose_entries) == 1


def test_select_compose_items_aggregates_same_source_short_news():
    from types import SimpleNamespace
    from datetime import datetime, timezone, timedelta
    from proactive.tick import ProactiveEngine

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._decide = SimpleNamespace(item_id_for=lambda item: item.title)
    now = datetime.now(timezone.utc)
    items = [
        GenericContentEvent("week-10", "rss", "HLTV News", "", "Short news: Week 10", "https://e/10", now - timedelta(hours=2)),
        GenericContentEvent("roster-tracker", "rss", "HLTV News", "", "Roster tracker: March 2026", "https://e/11", now - timedelta(hours=1)),
        GenericContentEvent("headtr1ck", "rss", "HLTV News", "", "Inner Circle sign headtr1ck", "https://e/12", now),
    ]
    entries = [
        ("rss:hltv", "Short news: Week 10"),
        ("rss:hltv", "Roster tracker: March 2026"),
        ("rss:hltv", "Inner Circle sign headtr1ck"),
    ]

    # 1. 输入顺序代表兴趣排序，三条都是低价值同源资讯流。
    # 2. MVP 期望：即便 token 不重合，只要同来源且接近，也能聚成一组。
    compose_items, compose_entries = engine._select_compose_items(items, entries)

    # 3. 当前实现还做不到，这条测试先作为聚合实现目标。
    assert [item.title for item in compose_items] == [
        "Short news: Week 10",
        "Roster tracker: March 2026",
        "Inner Circle sign headtr1ck",
    ]
    assert compose_entries == entries


def test_select_compose_items_aggregates_same_topic_across_sources():
    from types import SimpleNamespace
    from datetime import datetime, timezone, timedelta
    from proactive.tick import ProactiveEngine

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._decide = SimpleNamespace(item_id_for=lambda item: item.title)
    now = datetime.now(timezone.utc)
    items = [
        GenericContentEvent("niko-pressure", "rss", "NiKo Twitter", "", "NiKo talks about pressure in playoffs", "https://e/niko", now),
        GenericContentEvent("hooxi-niko", "rss", "HLTV News", "", "HooXi reflects on NiKo leadership in G2", "https://e/hooxi", now - timedelta(minutes=20)),
    ]
    entries = [
        ("rss:niko", "NiKo talks about pressure in playoffs"),
        ("rss:hltv", "HooXi reflects on NiKo leadership in G2"),
    ]

    # 1. 两条来自不同 source，但同一核心话题是 NiKo。
    # 2. MVP 期望：跨来源同话题也可以聚成一个 compose group。
    compose_items, compose_entries = engine._select_compose_items(items, entries)

    # 3. 当前实现只按同 source 补上下文，这条测试先锁定目标行为。
    assert [item.title for item in compose_items] == [
        "HooXi reflects on NiKo leadership in G2",
        "NiKo talks about pressure in playoffs",
    ]
    assert compose_entries == [
        ("rss:hltv", "HooXi reflects on NiKo leadership in G2"),
        ("rss:niko", "NiKo talks about pressure in playoffs"),
    ]


def test_prepare_compose_candidates_prefers_hot_single_over_same_source_news_bundle():
    from types import SimpleNamespace
    from proactive.tick import ProactiveEngine, DecisionContext

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        interest_filter=SimpleNamespace(
            enabled=True,
            memory_max_chars=4000,
            keyword_max_count=80,
            min_token_len=2,
            min_score=0.0,
            top_k=10,
            exploration_ratio=0.0,
        )
    )
    engine._decide = SimpleNamespace(item_id_for=lambda item: item.title)
    ctx = DecisionContext()
    fetch = ctx.ensure_fetch()
    decide = ctx.ensure_decide()
    act = ctx.ensure_act()
    decide.preference_block = "只关注 NiKo，普通资讯流可以聚合但不要压过 NiKo"
    fetch.new_items = [
        _build_event(event_id="n1", source_name="NiKo Twitter", title="NiKo playoff interview"),
        _build_event(event_id="h1", source_name="HLTV News", title="Short news: Week 10"),
        _build_event(event_id="h2", source_name="HLTV News", title="Roster tracker: March 2026"),
        _build_event(event_id="h3", source_name="HLTV News", title="Inner Circle sign headtr1ck"),
    ]
    fetch.new_entries = [
        ("rss:niko", "NiKo playoff interview"),
        ("rss:hltv", "Short news: Week 10"),
        ("rss:hltv", "Roster tracker: March 2026"),
        ("rss:hltv", "Inner Circle sign headtr1ck"),
    ]

    # 1. 先按兴趣排序，再进入聚合选择。
    # 2. MVP 期望：同源新闻 bundle 可以存在，但不能压过高兴趣单条。
    engine._prepare_feature_compose_candidates(ctx)

    # 3. 最高兴趣单条仍应优先。
    assert [item.title for item in act.compose_items] == ["NiKo playoff interview"]
    assert act.compose_entries == [("rss:niko", "NiKo playoff interview")]


@pytest.mark.asyncio
async def test_compose_judge_reject_marks_rejection_cooldown():
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from proactive.tick import ProactiveEngine, DecisionContext
    from proactive.judge import ProactiveJudgeResult

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        compose_no_content_token="<no_content/>",
        llm_reject_cooldown_hours=12,
        score_llm_threshold=0.0,
        threshold=0.7,
    )
    engine._state = SimpleNamespace(mark_rejection_cooldown=MagicMock())
    async def _compose_for_judge(**kw):
        return "content"

    async def _judge_message(**kw):
        return ProactiveJudgeResult(
            final_score=0.1,
            should_send=False,
            vetoed_by="llm_dim",
            dims_deterministic={"urgency": 0.5, "balance": 1.0, "dynamics": 1.0},
            dims_llm={"information_gap": 0.0, "relevance": 0.0, "expected_impact": 0.0},
            dims_llm_raw={"information_gap": 1, "relevance": 1, "expected_impact": 1},
        )

    engine._decide = SimpleNamespace(
        item_id_for=lambda item: item.title,
        pre_compose_veto=lambda **kw: None,
        compose_for_judge=_compose_for_judge,
        judge_message=_judge_message,
        resolve_evidence_item_ids=lambda decision, items: [item.title for item in items],
    )

    ctx = DecisionContext()
    fetch = ctx.ensure_fetch()
    sense = ctx.ensure_sense()
    score = ctx.ensure_score()
    ctx.state.now_utc = datetime.now(timezone.utc)
    sense.recent = []
    sense.interruptibility = 1.0
    sense.interrupt_detail = {
        "f_reply": 1.0,
        "f_activity": 1.0,
        "f_fatigue": 1.0,
    }
    score.sent_24h = 0
    fetch.new_items = [
        _build_event(event_id="n1", source_name="HLTV", title="Niko semifinal")
    ]
    fetch.new_entries = [("rss:hltv", "Niko semifinal")]
    # 设置主源为 content，确保能进入 compose
    fetch.selected_primary_source = "content"
    fetch.context_mode = "none"
    fetch.content_items = fetch.new_items

    # 1. 先走 compose，再走 judge，验证现行主链路会写 cooldown。
    compose = await engine._compose(ctx)
    assert compose.proceed is True
    result = await engine._judge_and_send(ctx)
    assert result == score.base_score
    assert ctx.ensure_decide().should_send is False
    assert ctx.ensure_decide().judge_vetoed_by == "llm_dim"

    # 2. LLM 拒绝应写入本轮真正进入 compose 的条目，而不是原始候选首条。
    engine._state.mark_rejection_cooldown.assert_called_once_with(
        [("rss:hltv", "Niko semifinal")],
        hours=12,
    )


@pytest.mark.asyncio
async def test_compose_judge_without_candidates_uses_user_recent_only():
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from proactive.tick import ProactiveEngine, DecisionContext

    compose_calls: list[dict] = []

    async def _compose_for_judge(**kw):
        compose_calls.append(kw)
        return "<no_content/>"

    engine = ProactiveEngine.__new__(ProactiveEngine)
    engine._cfg = SimpleNamespace(
        compose_no_content_token="<no_content/>",
        llm_reject_cooldown_hours=12,
        score_llm_threshold=0.0,
        threshold=0.7,
        context_as_assist_enabled=True,
        bg_context_main_topic_min_interval_hours=6,
    )
    engine._state = SimpleNamespace(
        mark_rejection_cooldown=lambda *a, **kw: None,
        get_bg_context_last_main_at=lambda: None,
    )
    engine._sense = SimpleNamespace(
        collect_recent_proactive=lambda n: [],
    )
    engine._decide = SimpleNamespace(
        item_id_for=lambda item: item.title,
        pre_compose_veto=lambda **kw: None,
        compose_for_judge=_compose_for_judge,
        judge_message=None,
    )

    ctx = DecisionContext()
    sense = ctx.ensure_sense()
    score = ctx.ensure_score()
    fetch = ctx.ensure_fetch()
    ctx.state.now_utc = datetime.now(timezone.utc)
    sense.recent = [
        {"role": "assistant", "content": "旧的主动资讯"},
        {"role": "user", "content": "我最近有点累"},
        {"role": "assistant", "content": "再一条旧资讯"},
    ]
    sense.interruptibility = 1.0
    sense.interrupt_detail = {
        "f_reply": 1.0,
        "f_activity": 1.0,
        "f_fatigue": 1.0,
    }
    score.sent_24h = 0
    # 设置主源为 context-only，确保能进入 compose
    fetch.selected_primary_source = "context"
    fetch.context_mode = "context_only"
    fetch.background_context = [{"topic": "test", "summary": "test context"}]

    compose = await engine._compose(ctx)

    assert compose.proceed is False
    assert compose.reason_code == "compose_no_content"

    assert len(compose_calls) == 1
    assert compose_calls[0]["items"] == []
    assert compose_calls[0]["recent"] == [
        {"role": "user", "content": "我最近有点累"}
    ]
