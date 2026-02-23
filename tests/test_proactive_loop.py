from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest

from agent.memory import MemoryStore
from feeds.base import FeedItem
from proactive.loop import ProactiveConfig, ProactiveLoop, _parse_decision
from proactive.presence import PresenceStore
from session.manager import SessionManager


def _utc(**kwargs) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**kwargs)


class _DummyFeedRegistry:
    async def fetch_all(self, limit_per_source: int = 3):
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


def test_parse_decision_string_false_is_false():
    d = _parse_decision(
        '{"score": 0.9, "should_send": "false", "message": "hello", "reasoning": "r"}'
    )
    assert d.should_send is False


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
async def test_tick_dedupes_seen_items_and_skips_second_reflect(tmp_path):
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
        return_value=_Resp('{"reasoning":"ok","score":0.9,"should_send":true,"message":"ping"}')
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
            only_new_items_trigger=True,
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
    )

    await loop._tick()
    await loop._tick()

    assert provider.chat.await_count == 1
    assert push_tool.execute.await_count == 1


@pytest.mark.asyncio
async def test_tick_delivery_dedupe_blocks_duplicate_send(tmp_path):
    """delivery dedup 阻止重复推送：tick1 有新 item 触发发送，tick2 同 item 已 seen 但有 memory
    支撑 content_weight > 0 使 LLM 被调到，delivery key 复用 fallback item id → 去重拦截第二次发送。"""
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    feed = _DummyFeedRegistry()
    item = FeedItem(
        source_name="TestFeed",
        source_type="rss",
        title="A",
        content="content",
        url="https://example.com/a",
        author=None,
        published_at=None,
    )
    feed.fetch_all = AsyncMock(side_effect=[[item], [item]])

    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp(
            '{"reasoning":"ok","score":0.9,"should_send":true,"message":"same msg","evidence_item_ids":[]}'
        )
    )
    session_manager = SessionManager(tmp_path)
    memory = MemoryStore(tmp_path)
    memory.write_long_term("用户偏好：关注游戏资讯。")  # has_memory=True → tick2 content_weight > 0
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
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
        memory_store=memory,
    )

    await loop._tick()
    await loop._tick()

    assert provider.chat.await_count == 2   # LLM 被调两次
    assert push_tool.execute.await_count == 1  # 第二次被 delivery dedup 拦截


@pytest.mark.asyncio
async def test_tick_semantic_dedupe_blocks_cross_source_duplicates(tmp_path):
    push_tool = AsyncMock()
    push_tool.execute = AsyncMock(return_value="文本已发送")

    feed = _DummyFeedRegistry()
    item_a = FeedItem(
        source_name="SourceA",
        source_type="rss",
        title="Elden Ring DLC announced",
        content="New trailer and release date window.",
        url="https://a.example.com/post-1",
        author=None,
        published_at=None,
    )
    item_b = FeedItem(
        source_name="SourceB",
        source_type="rss",
        title="Elden Ring DLC announced",
        content="New trailer and release date window.",
        url="https://b.example.com/news-77",
        author=None,
        published_at=None,
    )
    feed.fetch_all = AsyncMock(side_effect=[[item_a], [item_b]])

    provider = _DummyProvider()
    provider.chat = AsyncMock(
        return_value=_Resp('{"reasoning":"ok","score":0.95,"should_send":true,"message":"ping"}')
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
            only_new_items_trigger=True,
            semantic_dedupe_enabled=True,
            semantic_dedupe_threshold=0.90,
            semantic_dedupe_window_hours=72,
            semantic_dedupe_ngram=3,
        ),
        model="test-model",
        state_path=tmp_path / "proactive_state.json",
    )

    await loop._tick()
    await loop._tick()

    assert provider.chat.await_count == 1
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
            only_new_items_trigger=False,
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
            only_new_items_trigger=False,
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
    """电量低但没有内容（无 feed，无记忆），且非危机 → W_content=0 → 跳过 LLM。"""
    provider = _DummyProvider()
    provider.chat = AsyncMock()

    # 电量低（24h前），但无记忆无 feed
    presence = _make_presence(tmp_path, "telegram:123", last_user_minutes_ago=60 * 24)
    loop, _ = _build_loop_with_presence(tmp_path, provider, presence)
    # 无记忆注入，无 feed（_DummyFeedRegistry 返回 []）

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
            only_new_items_trigger=False,
            energy_min_urge=0.01,   # 低阈值，确保有记忆时必然触发
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
    """no new feed items + low energy + memory → 不被旧 only_new_items_trigger 拦截。"""
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
            only_new_items_trigger=True,   # 有 presence 时此开关不阻止 LLM 调用
            energy_min_urge=0.01,
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
