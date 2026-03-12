from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.provider import LLMResponse
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from feeds.base import FeedItem
from proactive.components import ProactiveMessageComposer
from proactive.config import ProactiveConfig
from proactive.engine import ProactiveEngine
from proactive.item_id import compute_item_id, compute_source_key
from proactive.ports import RecentProactiveMessage
from proactive.state import ProactiveStateStore
from proactive.loop_helpers import _format_items, _format_recent


@pytest.fixture(autouse=True)
def _shared_http_resources():
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    try:
        yield
    finally:
        clear_default_shared_http_resources(resources)
        asyncio.run(resources.aclose())


def _item(title: str, url: str, minutes_ago: int = 0) -> FeedItem:
    return FeedItem(
        source_name="Feed",
        source_type="rss",
        title=title,
        content=f"{title} content",
        url=url,
        author=None,
        published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


def _interruptibility():
    return (
        0.8,
        {
            "f_reply": 0.8,
            "f_activity": 0.8,
            "f_fatigue": 0.8,
            "random_delta": 0.0,
        },
    )


def _sense(items, entries):
    sense = MagicMock()
    sense.compute_energy.return_value = 0.5
    sense.collect_recent.return_value = []
    sense.collect_recent_proactive.return_value = []
    sense.compute_interruptibility.return_value = _interruptibility()
    sense.fetch_items = AsyncMock(return_value=items)
    sense.filter_new_items.return_value = (items, entries, [])
    sense.read_memory_text.return_value = ""
    sense.has_global_memory.return_value = False
    sense.last_user_at.return_value = None
    sense.target_session_key.return_value = "telegram:123"
    sense.quiet_hours.return_value = (23, 8, 0.0)
    sense.refresh_sleep_context.return_value = False
    sense.sleep_context.return_value = None
    return sense


def _cfg(**overrides) -> ProactiveConfig:
    cfg = ProactiveConfig(
        enabled=True,
        default_channel="telegram",
        default_chat_id="123",
        threshold=0.5,
        score_pre_threshold=0.0,
        score_llm_threshold=0.0,
        pending_queue_enabled=True,
        pending_candidate_limit=3,
        llm_reject_cooldown_hours=12,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _make_feature_decide(captured: dict[str, object]):
    class _FeatureDecide:
        async def score_features(self, **kw):
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.9,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            # 1. 记录最终进入生成阶段的聚合条目。
            # 2. 用标题拼接消息，方便断言选中了哪一组。
            # 3. 不引入额外格式逻辑，保持测试直观。
            feed_items = kw["items"]
            captured["items"] = feed_items
            return " | ".join((item.title or "") for item in feed_items)

        async def reflect(self, *a, **kw):
            raise AssertionError("feature mode 不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return [compute_item_id(item) for item in items]

        def build_delivery_key(self, ids, msg):
            return "|".join(ids) or "no-evidence"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return compute_item_id(item)

    return _FeatureDecide()


class _Decide:
    def __init__(self, evidence_ids: list[str], *, should_send: bool, score: float = 0.9):
        self._evidence_ids = evidence_ids
        self.reflect = AsyncMock(return_value=self._decision(should_send, score))
        self.score_features = AsyncMock(return_value=None)
        self.compose_message = AsyncMock(return_value="")

    def _decision(self, should_send: bool, score: float):
        class _D:
            pass

        d = _D()
        d.score = score
        d.should_send = should_send
        d.message = "ping" if should_send else ""
        d.reasoning = "ok"
        d.evidence_item_ids = list(self._evidence_ids)
        return d

    def randomize_decision(self, d):
        return d, 0.0

    def resolve_evidence_item_ids(self, d, items):
        valid = [compute_item_id(item) for item in items]
        for evidence_id in self._evidence_ids:
            if evidence_id in valid:
                return [evidence_id]
        return valid[:1]

    def build_delivery_key(self, ids, msg):
        return "|".join(ids) or "no-evidence"

    def semantic_entries(self, items):
        return [
            {
                "source_key": compute_source_key(item),
                "item_id": compute_item_id(item),
                "text": item.title or "",
            }
            for item in items
        ]

    def item_id_for(self, item):
        return compute_item_id(item)


@pytest.mark.asyncio
async def test_send_success_consumes_only_evidence_item(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        _item("A", "https://example.com/a", minutes_ago=3),
        _item("B", "https://example.com/b", minutes_ago=2),
        _item("C", "https://example.com/c", minutes_ago=1),
    ]
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]
    evidence_id = compute_item_id(items[1])

    engine = ProactiveEngine(
        cfg=_cfg(),
        state=state,
        presence=None,
        rng=None,
        sense=_sense(items, entries),
        decide=_Decide([evidence_id], should_send=True),
        act=MagicMock(send=AsyncMock(return_value=True)),
    )

    await engine.tick()

    assert state.is_item_seen(entries[1][0], entries[1][1], ttl_hours=336)
    assert not state.is_item_seen(entries[0][0], entries[0][1], ttl_hours=336)
    assert not state.is_item_seen(entries[2][0], entries[2][1], ttl_hours=336)
    assert state.pending_stats()[entries[0][0]] == 2
    send_meta = engine._act.send.await_args.args[1]
    assert send_meta.evidence_item_ids == [evidence_id]
    assert send_meta.source_refs[0].source_name == "Feed"
    assert send_meta.source_refs[0].url == "https://example.com/b"


@pytest.mark.asyncio
async def test_feature_mode_uses_same_item_for_message_and_source_ref(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        FeedItem(
            source_name="A9VG (Bilibili)",
            source_type="rss",
            title="失物招领有限公司",
            content="A9VG live",
            url="https://t.bilibili.com/1",
            author=None,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ),
        FeedItem(
            source_name="PC Gamer UK - Games",
            source_type="rss",
            title="Banquet for Fools",
            content="Banquet for Fools is bursting with strange ideas.",
            url="https://www.pcgamer.com/banquet-for-fools",
            author=None,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        ),
    ]
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]
    captured: dict[str, object] = {}

    class _FeatureDecide:
        async def score_features(self, **kw):
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.9,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            feed_items = kw["items"]
            captured["items"] = feed_items
            assert len(feed_items) == 1
            item = feed_items[0]
            return f"刚看到 {item.source_name} 提到 {item.title}"

        async def reflect(self, *a, **kw):
            raise AssertionError("feature mode 不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            valid = [compute_item_id(item) for item in items]
            return valid[:1]

        def build_delivery_key(self, ids, msg):
            return "|".join(ids) or "no-evidence"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return compute_item_id(item)

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(
            feature_scoring_enabled=True,
            feature_send_threshold=0.0,
            pending_queue_enabled=False,
        ),
        state=state,
        presence=None,
        rng=None,
        sense=_sense(items, entries),
        decide=_FeatureDecide(),
        act=act,
    )

    await engine.tick()

    send_message, send_meta = act.send.await_args.args
    composed_items = captured["items"]
    assert len(composed_items) == 1
    assert composed_items[0].source_name == "A9VG (Bilibili)"
    assert "A9VG (Bilibili)" in send_message
    assert send_meta.evidence_item_ids == [compute_item_id(items[0])]
    assert send_meta.source_refs[0].source_name == "A9VG (Bilibili)"
    assert send_meta.source_refs[0].title == "失物招领有限公司"


@pytest.mark.asyncio
async def test_feature_mode_groups_same_topic_items_into_one_message(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        FeedItem(
            source_name="PC Gamer UK - Games",
            source_type="rss",
            title="Patch notes published",
            content="another topic",
            url="https://www.pcgamer.com/demo",
            author=None,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=3),
        ),
        FeedItem(
            source_name="PC Gamer UK - Games",
            source_type="rss",
            title="Banquet for Fools release date confirmed",
            content="Banquet for Fools release window",
            url="https://www.pcgamer.com/banquet-release",
            author=None,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ),
        FeedItem(
            source_name="PC Gamer UK - Games",
            source_type="rss",
            title="Banquet for Fools demo is out now",
            content="Banquet for Fools playable demo",
            url="https://www.pcgamer.com/banquet-demo",
            author=None,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        ),
    ]
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]
    captured: dict[str, object] = {}

    class _FeatureDecide:
        async def score_features(self, **kw):
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.9,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            feed_items = kw["items"]
            captured["items"] = feed_items
            titles = [item.title for item in feed_items]
            return " | ".join(titles)

        async def reflect(self, *a, **kw):
            raise AssertionError("feature mode 不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return [compute_item_id(item) for item in items]

        def build_delivery_key(self, ids, msg):
            return "|".join(ids) or "no-evidence"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return compute_item_id(item)

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(
            feature_scoring_enabled=True,
            feature_send_threshold=0.0,
            pending_queue_enabled=False,
        ),
        state=state,
        presence=None,
        rng=None,
        sense=_sense(items, entries),
        decide=_FeatureDecide(),
        act=act,
    )

    await engine.tick()

    send_message, send_meta = act.send.await_args.args
    composed_items = captured["items"]
    assert len(composed_items) == 2
    assert all("Banquet for Fools" in (item.title or "") for item in composed_items)
    assert "Banquet for Fools release date confirmed" in send_message
    assert "Banquet for Fools demo is out now" in send_message
    assert len(send_meta.evidence_item_ids) == 2
    assert len(send_meta.source_refs) == 2
    assert all("Banquet for Fools" in (ref.title or "") for ref in send_meta.source_refs)


@pytest.mark.asyncio
async def test_feature_mode_keeps_first_item_when_topics_are_all_different(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        _item("Falcons roster update", "https://example.com/falcons", minutes_ago=3),
        _item("Steam sale starts", "https://example.com/steam", minutes_ago=2),
        _item("Fitbit battery tips", "https://example.com/fitbit", minutes_ago=1),
    ]
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]
    captured: dict[str, object] = {}

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(
            feature_scoring_enabled=True,
            feature_send_threshold=0.0,
            pending_queue_enabled=False,
        ),
        state=state,
        presence=None,
        rng=None,
        sense=_sense(items, entries),
        decide=_make_feature_decide(captured),
        act=act,
    )

    await engine.tick()

    send_message, send_meta = act.send.await_args.args
    composed_items = captured["items"]
    assert len(composed_items) == 1
    assert composed_items[0].title == "Falcons roster update"
    assert send_message == "Falcons roster update"
    assert len(send_meta.evidence_item_ids) == 1
    assert send_meta.source_refs[0].title == "Falcons roster update"


@pytest.mark.asyncio
async def test_feature_mode_prefers_larger_same_topic_cluster(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        _item("Falcons roster update", "https://example.com/falcons-1", minutes_ago=5),
        _item("Falcons map pool changes", "https://example.com/falcons-2", minutes_ago=4),
        _item("Steam sale starts", "https://example.com/steam-1", minutes_ago=3),
        _item("Steam sale best RPG picks", "https://example.com/steam-2", minutes_ago=2),
        _item("Steam sale hidden gems", "https://example.com/steam-3", minutes_ago=1),
    ]
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]
    captured: dict[str, object] = {}

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(
            feature_scoring_enabled=True,
            feature_send_threshold=0.0,
            pending_queue_enabled=False,
        ),
        state=state,
        presence=None,
        rng=None,
        sense=_sense(items, entries),
        decide=_make_feature_decide(captured),
        act=act,
    )

    await engine.tick()

    send_message, send_meta = act.send.await_args.args
    composed_items = captured["items"]
    titles = [item.title for item in composed_items]
    assert len(composed_items) == 3
    assert titles == [
        "Steam sale starts",
        "Steam sale best RPG picks",
        "Steam sale hidden gems",
    ]
    assert "Falcons roster update" not in send_message
    assert len(send_meta.evidence_item_ids) == 3
    assert [ref.title for ref in send_meta.source_refs] == titles


@pytest.mark.asyncio
async def test_real_composer_prompt_contains_only_selected_topic_cluster(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        _item("Falcons roster update", "https://example.com/falcons-1", minutes_ago=5),
        _item("Falcons map pool changes", "https://example.com/falcons-2", minutes_ago=4),
        _item("Steam sale starts", "https://example.com/steam-1", minutes_ago=3),
        _item("Steam sale best RPG picks", "https://example.com/steam-2", minutes_ago=2),
        _item("Steam sale hidden gems", "https://example.com/steam-3", minutes_ago=1),
    ]
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]

    class _Provider:
        def __init__(self):
            self.calls: list[dict] = []

        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            return LLMResponse(content="给用户的最终消息", tool_calls=[])

    provider = _Provider()
    composer = ProactiveMessageComposer(
        provider=provider,
        model="test-model",
        max_tokens=256,
        format_items=_format_items,
        format_recent=_format_recent,
        collect_global_memory=lambda: "",
        max_tool_iterations=2,
    )

    class _RealComposerDecide:
        async def score_features(self, **kw):
            return {
                "topic_continuity": 0.8,
                "interest_match": 0.9,
                "content_novelty": 0.7,
                "reconnect_value": 0.7,
                "disturb_risk": 0.1,
                "message_readiness": 0.8,
                "confidence": 0.9,
            }

        async def compose_message(self, **kw):
            # 1. 使用真实 ProactiveMessageComposer 生成 prompt。
            # 2. 让 provider 记录最终注入的 messages。
            # 3. 返回 provider 的最终文本，保持链路完整。
            return await composer.compose_message(**kw)

        async def reflect(self, *a, **kw):
            raise AssertionError("feature mode 不应走 reflect")

        def randomize_decision(self, d):
            return d, 0.0

        def resolve_evidence_item_ids(self, d, items):
            return [compute_item_id(item) for item in items]

        def build_delivery_key(self, ids, msg):
            return "|".join(ids) or "no-evidence"

        def semantic_entries(self, items):
            return []

        def item_id_for(self, item):
            return compute_item_id(item)

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(
            feature_scoring_enabled=True,
            feature_send_threshold=0.0,
            pending_queue_enabled=False,
        ),
        state=state,
        presence=None,
        rng=None,
        sense=_sense(items, entries),
        decide=_RealComposerDecide(),
        act=act,
    )

    await engine.tick()

    assert provider.calls, "真实 composer 未调用 provider"
    user_prompt = provider.calls[0]["messages"][1]["content"]
    assert "Steam sale starts" in user_prompt
    assert "Steam sale best RPG picks" in user_prompt
    assert "Steam sale hidden gems" in user_prompt
    assert "Falcons roster update" not in user_prompt
    assert "Falcons map pool changes" not in user_prompt


@pytest.mark.asyncio
async def test_same_state_summary_is_blocked_within_current_silence(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    item = _item("A", "https://example.com/a", minutes_ago=1)
    entry = (compute_source_key(item), compute_item_id(item))
    sense = _sense([item], [entry])
    sense.collect_recent_proactive.return_value = [
        RecentProactiveMessage(
            content="月底面试确实烦，但你的底子在那儿。",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
            state_summary_tag="interview_anxiety_reassurance",
            source_refs=[],
        )
    ]

    class _RepeatDecide(_Decide):
        def _decision(self, should_send: bool, score: float):
            d = super()._decision(should_send, score)
            d.message = "月底面试确实烦，但你的底子在那儿。刚好又有个新消息。"
            return d

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(),
        state=state,
        presence=None,
        rng=None,
        sense=sense,
        decide=_RepeatDecide([entry[1]], should_send=True),
        act=act,
        light_provider=None,
        light_model="",
    )

    await engine.tick()

    act.send.assert_not_called()
    assert state.is_rejection_cooled(entry[0], entry[1], ttl_hours=12)


@pytest.mark.asyncio
async def test_pending_item_survives_when_feed_no_longer_returns_it(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    pending_item = _item("Backlog", "https://example.com/backlog", minutes_ago=10)
    pending_entry = (compute_source_key(pending_item), compute_item_id(pending_item))
    state.upsert_pending_items([pending_item], now=datetime.now(timezone.utc))

    decide = _Decide([pending_entry[1]], should_send=False, score=0.2)
    engine = ProactiveEngine(
        cfg=_cfg(llm_reject_cooldown_hours=0),
        state=state,
        presence=None,
        rng=None,
        sense=_sense([], []),
        decide=decide,
        act=MagicMock(send=AsyncMock(return_value=False)),
    )

    await engine.tick()

    decide.reflect.assert_awaited()
    reflect_items = decide.reflect.await_args.args[0]
    assert [compute_item_id(item) for item in reflect_items] == [pending_entry[1]]
    assert state.pending_stats()[pending_entry[0]] == 1
    assert not state.is_item_seen(*pending_entry, ttl_hours=336)


@pytest.mark.asyncio
async def test_delivery_dedupe_consumes_only_evidence_item(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    items = [
        _item("A", "https://example.com/a", minutes_ago=2),
        _item("B", "https://example.com/b", minutes_ago=1),
    ]
    state.upsert_pending_items(items)
    evidence_id = compute_item_id(items[0])
    state.mark_delivery("telegram:123", evidence_id)
    entries = [(compute_source_key(item), compute_item_id(item)) for item in items]

    act = MagicMock(send=AsyncMock(return_value=True))
    engine = ProactiveEngine(
        cfg=_cfg(),
        state=state,
        presence=None,
        rng=None,
        sense=_sense([], []),
        decide=_Decide([evidence_id], should_send=True),
        act=act,
    )

    await engine.tick()

    act.send.assert_not_called()
    assert state.is_item_seen(entries[0][0], entries[0][1], ttl_hours=336)
    assert not state.is_item_seen(entries[1][0], entries[1][1], ttl_hours=336)
    assert state.pending_stats()[entries[1][0]] == 1


@pytest.mark.asyncio
async def test_message_dedupe_keeps_pending_and_writes_cooldown(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    item = _item("A", "https://example.com/a", minutes_ago=1)
    entry = (compute_source_key(item), compute_item_id(item))

    message_deduper = MagicMock()
    message_deduper.is_duplicate = AsyncMock(return_value=(True, "dup"))
    engine = ProactiveEngine(
        cfg=_cfg(),
        state=state,
        presence=None,
        rng=None,
        sense=_sense([item], [entry]),
        decide=_Decide([entry[1]], should_send=True),
        act=MagicMock(send=AsyncMock(return_value=True)),
        message_deduper=message_deduper,
    )

    await engine.tick()

    assert state.pending_stats()[entry[0]] == 1
    assert state.is_rejection_cooled(entry[0], entry[1], ttl_hours=12)
    assert not state.is_item_seen(entry[0], entry[1], ttl_hours=336)


@pytest.mark.asyncio
async def test_should_send_false_keeps_pending_and_writes_cooldown(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    item = _item("A", "https://example.com/a", minutes_ago=1)
    entry = (compute_source_key(item), compute_item_id(item))

    engine = ProactiveEngine(
        cfg=_cfg(),
        state=state,
        presence=None,
        rng=None,
        sense=_sense([item], [entry]),
        decide=_Decide([entry[1]], should_send=False, score=0.2),
        act=MagicMock(send=AsyncMock(return_value=False)),
    )

    await engine.tick()

    assert state.pending_stats()[entry[0]] == 1
    assert state.is_rejection_cooled(entry[0], entry[1], ttl_hours=12)
    assert not state.is_item_seen(entry[0], entry[1], ttl_hours=336)


def test_pending_cleanup_expires_old_items(tmp_path):
    state = ProactiveStateStore(tmp_path / "state.json")
    old_item = _item("Old", "https://example.com/old", minutes_ago=60 * 25)
    entry = (compute_source_key(old_item), compute_item_id(old_item))
    state.upsert_pending_items([old_item])

    state.cleanup(
        seen_ttl_hours=336,
        delivery_ttl_hours=24,
        semantic_ttl_hours=72,
        pending_ttl_hours=24,
    )

    assert state.pending_stats().get(entry[0], 0) == 0


def test_state_loads_old_version_without_pending_items(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "seen_items": {},
                "deliveries": {},
                "semantic_items": [],
                "rejection_cooldown": {},
            }
        ),
        encoding="utf-8",
    )

    state = ProactiveStateStore(path)

    assert state._state["version"] == 4
    assert state._state["pending_items"] == {}
