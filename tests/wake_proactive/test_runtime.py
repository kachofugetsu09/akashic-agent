from __future__ import annotations

import json
import random
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock

import pytest

from agent.plugins.specs import ProactiveSourceSpec, RegisteredProactiveSource
from agent.persona import reset_veda
from agent.provider import LLMResponse, ToolCall
from core.clock import ReplayClock
from plugins.wake_proactive.plugin import WakeProactivePlugin, WakeRuntimeFactory
from plugins.wake_proactive.context import WakeContext
from plugins.wake_proactive.prompt import build_messages
from plugins.wake_proactive.runtime import WakeRuntime, select_content_page
from plugins.wake_proactive.state import WakeStateStore
from proactive_v2.frame import new_proactive_frame
from proactive_v2.lifecycle import ProactiveLifecycleBuilder, ProactiveLifecycleSpec
from proactive_v2.runtime_scope import ProactiveRuntimeScope
from session.embedding_store import MessageEmbeddingStore


class FakeGateway:
    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.acks: list[dict] = []

    async def call(self, server, tool_name, args, *, timeout=None):
        if tool_name == "fetch":
            return list(self.events)
        if tool_name == "ack":
            self.acks.append(dict(args))
            return {"ok": True}
        raise AssertionError(tool_name)


class FakeContextGateway:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot

    async def call(self, server, tool_name, args, *, timeout=None):
        if tool_name == "fetch":
            return dict(self.snapshot)
        raise AssertionError(tool_name)


class FailingGateway:
    async def call(self, server, tool_name, args, *, timeout=None):
        raise RuntimeError("source unavailable")


class FlakyAckGateway(FakeGateway):
    def __init__(self, events: list[dict]) -> None:
        super().__init__(events)
        self.ack_attempts = 0

    async def call(self, server, tool_name, args, *, timeout=None):
        if tool_name == "ack":
            self.ack_attempts += 1
            if self.ack_attempts == 1:
                raise RuntimeError("temporary ack failure")
        return await super().call(server, tool_name, args, timeout=timeout)


class FakeWebFetch:
    name = "web_fetch"
    description = "读取网页"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    async def execute(self, **kwargs):
        return json.dumps({"url": kwargs["url"], "text": f"正文 {kwargs['url']}"})

    def to_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FakeTools:
    def get_tool(self, name):
        return FakeWebFetch() if name == "web_fetch" else None

    def get_mcp_server_names(self):
        return set()

    def get_tool_names_by_source(self, source_type, source_name):
        _ = (source_type, source_name)
        return set()


class FakeOrchestrator:
    def __init__(self) -> None:
        self.results = []

    async def handle_proactive_turn(self, *, result, **kwargs):
        self.results.append(result)
        effects = result.side_effects if result.decision == "skip" else result.success_side_effects
        for effect in effects:
            await effect.run()
        return result.decision == "reply"


class FailingOrchestrator(FakeOrchestrator):
    async def handle_proactive_turn(self, *, result, **kwargs):
        self.results.append(result)
        return False


class FixedRng:
    def gammavariate(self, shape, scale):
        return 0.000001

    def random(self):
        return 0.000001


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


def test_consume_rejects_unknown_item_without_partial_update(tmp_path):
    now = datetime(2026, 7, 18, tzinfo=UTC)
    store = WakeStateStore(tmp_path / "wake.db")
    store.ingest_with_ids(
        "content",
        [
            {
                "event_id": "known",
                "source_id": "source-a",
                "ack_server": "feed_plugin:main",
                "published_at": now.isoformat(),
            }
        ],
        now,
    )

    with pytest.raises(RuntimeError, match="every canonical item_id"):
        store.consume(["feed_plugin:main:known", "feed_plugin:main:missing"], now)

    assert store.unread("content")[0]["id"] == "feed_plugin:main:known"
    store.close()


def _source(channel: Literal["alert", "content", "context"]) -> RegisteredProactiveSource:
    return RegisteredProactiveSource(
        plugin_id="feed_plugin",
        spec=ProactiveSourceSpec(
            id="main",
            channels=(channel,),
            server="feed",
            fetch_tool="fetch",
            ack_tool="ack",
        ),
    )


def _scope(tmp_path: Path, gateway, provider, orchestrator, source) -> ProactiveRuntimeScope:
    embedding_api = SimpleNamespace(
        model_id="test-embedding",
        embed_batch=AsyncMock(return_value=[[0.1, 0.2], [0.2, 0.3]]),
    )
    memory = SimpleNamespace(
        read_long_term=lambda: "用户关心 agent 架构",
        embedding_api=embedding_api,
        query=AsyncMock(
            return_value=SimpleNamespace(
                records=[SimpleNamespace(summary="用户持续研究主动唤醒")]
            )
        ),
    )
    return ProactiveRuntimeScope(
        cfg=SimpleNamespace(
            agent_tick_model="",
            agent_tick_web_fetch_max_chars=8000,
            default_channel="telegram",
            default_chat_id="1",
            drift_enabled=True,
            drift_max_steps=20,
        ),
        provider=provider,
        model="fake-model",
        max_tokens=1000,
        memory=memory,
        state_store=SimpleNamespace(workspace_dir=tmp_path),
        sense=SimpleNamespace(collect_recent=lambda: []),
        any_action_gate=None,
        passive_busy_fn=None,
        deduper=None,
        rng=FixedRng(),
        workspace_context_fn=lambda: "只分享真正有增量的内容",
        mcp_gateway=cast(Any, gateway),
        proactive_sources=[source],
        shared_tools=cast(Any, FakeTools()),
        turn_orchestrator=cast(Any, orchestrator),
        presence=None,
    )


@pytest.mark.asyncio
async def test_content_vertical_slice_filters_investigates_and_shares(
    tmp_path,
    request,
):
    events = [
        {
            "kind": "content",
            "event_id": "old",
            "title": "旧标题",
            "source_name": "Research",
            "published_at": "2026-07-10T00:00:00+00:00",
            "url": "https://example.com/old",
            "preprocess_score": 0.2,
        },
        {
            "kind": "content",
            "event_id": "new",
            "title": "新标题",
            "source_name": "Research",
            "published_at": "2026-07-11T00:00:00+00:00",
            "url": "https://example.com/new",
            "preprocess_score": 0.9,
        },
    ]
    ids = ["feed_plugin:main:new", "feed_plugin:main:old"]
    provider = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "screen",
                            "scratchpad",
                            {
                                "items": [
                                    {
                                        "item_id": "candidate_1",
                                        "initial_interest": "likely_interesting",
                                    }
                                ]
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "final",
                            "share_content",
                            {
                                "opening": "这个变化值得留意。",
                                "items": [
                                    {
                                        "item_id": "candidate_1",
                                        "summary": "新内容已经发布。",
                                        "why_it_matters": "符合你最近关注的方向",
                                    }
                                ],
                            },
                        )
                    ],
                ),
            ]
        )
    )
    gateway = FakeGateway(events)
    orchestrator = FakeOrchestrator()
    scope = _scope(tmp_path, gateway, provider, orchestrator, _source("content"))
    runtime = WakeRuntime(
        scope,
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)),
    )
    request.addfinalizer(runtime.close)
    frame = new_proactive_frame("telegram:1")
    frame.input = frame.input.__class__(
        session_key="telegram:1",
        started_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    lifecycle = ProactiveLifecycleBuilder().build(
        cast(ProactiveLifecycleSpec, WakeProactivePlugin().proactive_lifecycles()[0]),
        runtime.build_modules(),
    )
    result = await lifecycle.run(frame)

    assert result.output is not None
    assert result.output.next_interval_seconds == 300
    assert result.slots["wake:run_state"].ctx.now_utc == datetime(2026, 7, 12, tzinfo=UTC)
    assert provider.chat.await_count == 2
    assert len(orchestrator.results) == 1
    assert orchestrator.results[0].decision == "reply"
    assert orchestrator.results[0].evidence == [ids[0]]
    assert "符合你最近关注的方向" in orchestrator.results[0].outbound.content
    observations = runtime._state.observations("content")
    assert len(observations) == 1
    candidates = json.loads(observations[0]["candidates_json"])
    llm_input = json.loads(observations[0]["llm_input_json"])
    assert [candidate["id"] for candidate in candidates] == ids
    first_prompt = llm_input[1]["content"]
    system_prompt = llm_input[0]["content"]
    assert first_prompt.index("新标题") < first_prompt.index("旧标题")
    assert "来源：Research" in first_prompt
    assert "preprocess_score" not in first_prompt
    assert "scratchpad" not in system_prompt
    assert "scratchpad" in llm_input[2]["content"]
    assert "自行决定调查范围" in llm_input[2]["content"]
    assert "当前 ContextEvent" in first_prompt
    assert "没有有效 ContextEvent" in first_prompt
    final_prompt = provider.chat.await_args_list[1].kwargs["messages"][0]["content"]
    assert "unknown 时保持中性" in final_prompt
    assert "敏感经历" in final_prompt
    assert "scratchpad" not in final_prompt
    first_messages = provider.chat.await_args_list[0].kwargs["messages"]
    final_messages = provider.chat.await_args_list[1].kwargs["messages"]
    assert first_messages[:2] == final_messages[:2]
    assert "scratchpad" in first_messages[-1]["content"]
    assert "scratchpad" not in final_messages[-1]["content"]
    assert "share_content" in final_messages[-1]["content"]
    assert "skip_content" in final_messages[-1]["content"]
    assert gateway.acks == [{"event_ids": ["new", "old"]}]
    assert [event["id"] for event in runtime._state.unread("content")] == [ids[1]]


@pytest.mark.asyncio
async def test_content_skip_keeps_full_unread_window_when_title_page_is_capped(
    tmp_path, request
):
    now = datetime(2026, 7, 12, tzinfo=UTC)
    events = [
        {
            "kind": "content",
            "event_id": f"event-{index}",
            "title": f"标题 {index}",
            "published_at": now.isoformat(),
            "preprocess_score": 0.9,
        }
        for index in range(121)
    ]
    gateway = FakeGateway(events)
    scope = _scope(
        tmp_path,
        gateway,
        SimpleNamespace(
            chat=AsyncMock(
                side_effect=[
                    LLMResponse(
                        content=None,
                        tool_calls=[ToolCall("screen", "scratchpad", {"items": []})],
                    ),
                    LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                "final",
                                "skip_content",
                                {"reason": "没有值得打扰用户的内容"},
                            )
                        ],
                    ),
                ]
            )
        ),
        FakeOrchestrator(),
        _source("content"),
    )
    scope.memory.embedding_api = None
    runtime = WakeRuntime(
        scope,
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(now),
    )
    request.addfinalizer(runtime.close)
    state = runtime.begin(new_proactive_frame("telegram:1"))

    await runtime.ingest(state)
    await runtime.decide(state)

    observation = runtime._state.observations("content")[0]
    assert len(json.loads(observation["candidates_json"])) == 120
    assert state.ctx.content_backlog_count == 1
    assert len(runtime._state.unread("content")) == 121
    assert len(gateway.acks[0]["event_ids"]) == 121


@pytest.mark.asyncio
async def test_decayed_content_is_acknowledged_without_wake(tmp_path, request):
    now = datetime(2026, 7, 12, tzinfo=UTC)
    gateway = FakeGateway(
        [
            {
                "kind": "content",
                "event_id": "stale",
                "title": "三十天前的内容",
                "published_at": (now - timedelta(days=30)).isoformat(),
                "preprocess_score": 0.9,
            }
        ]
    )
    scope = _scope(
        tmp_path,
        gateway,
        SimpleNamespace(chat=AsyncMock()),
        FakeOrchestrator(),
        _source("content"),
    )
    scope.memory.embedding_api = None
    runtime = WakeRuntime(
        scope,
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(now),
    )
    request.addfinalizer(runtime.close)
    state = runtime.begin(new_proactive_frame("telegram:1"))

    await runtime.ingest(state)
    await runtime.decide(state)

    assert runtime._state.unread("content") == []
    assert runtime._state.observations("content") == []
    assert gateway.acks == [{"event_ids": ["stale"]}]


@pytest.mark.asyncio
async def test_shared_ack_route_keeps_original_source_grouping_and_order(
    tmp_path, request
) -> None:
    now = datetime(2026, 7, 12, tzinfo=UTC)
    events = [
        {
            "kind": "content", "event_id": "a-old", "source_id": "source-a",
            "title": "A旧", "published_at": "2026-07-10T00:00:00+00:00",
        },
        {
            "kind": "content", "event_id": "b-new", "source_id": "source-b",
            "title": "B新", "published_at": "2026-07-12T00:00:00+00:00",
        },
        {
            "kind": "content", "event_id": "a-new", "source_id": "source-a",
            "title": "A新", "published_at": "2026-07-11T00:00:00+00:00",
        },
    ]
    gateway = FakeGateway(events)
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            gateway,
            SimpleNamespace(chat=AsyncMock()),
            FakeOrchestrator(),
            _source("content"),
        ),
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(now),
    )
    request.addfinalizer(runtime.close)
    enriched = [dict(event, ack_server="feed_plugin:main") for event in events]
    assert runtime._state.ingest("content", enriched, now) == 3
    unread = runtime._state.unread("content")

    prompt = build_messages(
        ctx=WakeContext(content_events=unread),
        memory_text="",
        proactive_context="",
        recent_passive_conversation="",
        recent_proactive_messages="",
    )[1]["content"]

    assert prompt.index("来源：source-a") < prompt.index("来源：source-b")
    assert prompt.index("A新") < prompt.index("A旧")
    assert "item_id=candidate_1" in prompt
    assert "feed_plugin:main:" not in prompt
    await runtime._ack_and_consume(unread, now)
    assert gateway.acks == [{"event_ids": ["a-new", "a-old", "b-new"]}]


@pytest.mark.parametrize("mode", ["content", "alert", "context"])
def test_prompt_exposes_current_context_and_only_selected_mode(mode) -> None:
    ctx = WakeContext(
        content_events=[
            {
                "id": "feed:item",
                "title": "已有 content 标题",
                "published_at": "2026-07-12T00:00:00+00:00",
            }
        ]
    )
    event = (
        None
        if mode == "content"
        else {"event_id": "event-1", "summary": "本轮单条事件"}
    )

    messages = build_messages(
        ctx=ctx,
        memory_text="memory",
        proactive_context="proactive context",
        recent_passive_conversation="recent passive conversation",
        recent_proactive_messages=(
            "2026-07-12T00:00:00+00:00 | session=mobile:owner | "
            "assistant(proactive): recent proactive message"
        ),
        current_context="presence=active | confidence=0.90",
        mode=cast(Any, mode),
        event=event,
    )

    assert "【当前 ContextEvent】" in messages[1]["content"]
    assert "【截至当前时间的最近被动对话】" in messages[1]["content"]
    assert "recent passive conversation" in messages[1]["content"]
    assert "【截至当前时间已经发送的主动消息】" in messages[1]["content"]
    assert "理解你最近主动和用户聊过什么" in messages[1]["content"]
    assert "不是内容价值的扣分表" in messages[1]["content"]
    assert "话题、结论或事件相近都不自动禁止再次分享" in messages[1]["content"]
    assert "recent proactive message" in messages[1]["content"]
    assert "presence=active | confidence=0.90" in messages[1]["content"]
    assert f"mode={mode}" in messages[1]["content"]
    assert "mode=content" not in messages[0]["content"]
    assert "mode=alert" not in messages[0]["content"]
    assert "mode=context" not in messages[0]["content"]
    assert f"mode={mode}" in messages[2]["content"]
    assert all(
        f"mode={other_mode}" not in messages[2]["content"]
        for other_mode in {"content", "alert", "context"} - {mode}
    )
    if mode == "content":
        assert "已有 content 标题" in messages[1]["content"]
        assert "scratchpad" in messages[2]["content"]
        assert "<example>" in messages[2]["content"]
        assert "主题 X" in messages[2]["content"]
        assert "内容形态 Y" in messages[2]["content"]
        assert "自行决定调查范围" in messages[2]["content"]
    else:
        assert "本轮单条事件" in messages[1]["content"]


def test_content_final_prompt_has_no_default_share_or_skip_tendency() -> None:
    messages = build_messages(
        ctx=WakeContext(content_events=[]),
        memory_text="",
        proactive_context="",
        recent_passive_conversation="",
        recent_proactive_messages="",
        content_phase="final",
    )

    assert "没有默认的 share 或 skip 倾向" in messages[2]["content"]
    assert "每次都重新判断这件事此刻对用户意味着什么" in messages[2]["content"]
    assert "发送次数本身不是用户的态度" in messages[2]["content"]
    assert "始终对用户本人和他在意的一切保持真诚好奇" in messages[0]["content"]
    assert "这种好奇不会因为一个话题已经聊过" in messages[0]["content"]


def test_legacy_reservoir_migrates_ack_and_original_sources(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE reservoir_events(
            item_id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT NOT NULL,
            source_event_id TEXT NOT NULL, published_at TEXT NOT NULL,
            first_seen_at TEXT NOT NULL, preprocess_score REAL NOT NULL,
            payload_json TEXT NOT NULL, embedding_json TEXT,
            status TEXT NOT NULL DEFAULT 'unread', consumed_at TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO reservoir_events VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "feed_plugin:main:event", "content", "feed_plugin:main", "event",
            "2026-07-11T00:00:00+00:00", "2026-07-11T00:00:00+00:00", 0.0,
            json.dumps({"source_id": "source-a"}), None, "unread", None,
        ),
    )
    connection.commit()
    connection.close()

    store = WakeStateStore(path)
    try:
        event = store.unread("content")[0]
        assert event["_reservoir_original_source_id"] == "source-a"
        assert event["_reservoir_ack_source_id"] == "feed_plugin:main"
        assert store.ingest(
            "content",
            [
                {
                    "ack_server": "feed_plugin:main",
                    "event_id": "new-event",
                    "source_id": "source-b",
                    "published_at": "2026-07-12T00:00:00+00:00",
                }
            ],
            datetime(2026, 7, 12, tzinfo=UTC),
        ) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_alerts_are_naturalized_by_one_llm_call_per_tick(
    tmp_path, request, caplog
):
    events = [
        {"kind": "alert", "event_id": "a1", "title": "提醒一", "body": "内容一"},
        {"kind": "alert", "event_id": "a2", "title": "提醒二", "body": "内容二"},
    ]
    gateway = FakeGateway(events)
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        "alert",
                        "send_event",
                        {"message": "这几天恢复状态有些往下走，今晚尽量早点休息。"},
                    )
                ],
            )
        )
    )
    orchestrator = FakeOrchestrator()
    scope = _scope(tmp_path, gateway, provider, orchestrator, _source("alert"))
    scope.memory.embedding_api.embed_batch.side_effect = RuntimeError("embedding down")
    runtime = WakeRuntime(
        scope,
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)),
    )
    request.addfinalizer(runtime.close)
    frame = new_proactive_frame("telegram:1")
    state = runtime.begin(frame)

    with caplog.at_level("INFO", logger="plugins.wake_proactive.runtime"):
        await runtime.ingest(state)
        await runtime.decide(state)

    assert len(orchestrator.results) == 1
    assert orchestrator.results[0].outbound.content == "这几天恢复状态有些往下走，今晚尽量早点休息。"
    assert provider.chat.await_count == 1
    request_messages = provider.chat.await_args.kwargs["messages"]
    assert "mode=alert" in request_messages[1]["content"]
    assert any(title in request_messages[1]["content"] for title in ("提醒一", "提醒二"))
    assert state.next_interval_seconds == 1
    assert len(gateway.acks) == 1
    assert set(gateway.acks[0]) == {"event_ids"}
    assert len(runtime._state.unread("alert")) == 1
    assert scope.memory.embedding_api.embed_batch.await_count == 0
    assert "[wake.source] poll ok" in caplog.text
    assert "new=alerts:2,content:0" in caplog.text
    assert "[wake.event] llm done kind=alert" in caplog.text
    assert len(runtime._state.observations("alert")) == 1


@pytest.mark.asyncio
async def test_ack_failure_does_not_repeat_delivered_alert_and_retries_next_tick(
    tmp_path,
    request,
):
    now = datetime(2026, 7, 12, tzinfo=UTC)
    gateway = FlakyAckGateway(
        [{"kind": "alert", "event_id": "a1", "title": "只发送一次"}]
    )
    orchestrator = FakeOrchestrator()
    store = WakeStateStore(tmp_path / "wake.db")
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            gateway,
            SimpleNamespace(
                chat=AsyncMock(
                    return_value=LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                "alert",
                                "send_event",
                                {"message": "这是一条只发送一次的自然提醒。"},
                            )
                        ],
                    )
                )
            ),
            orchestrator,
            _source("alert"),
        ),
        state_store=store,
        clock=FixedClock(now),
    )
    request.addfinalizer(runtime.close)

    first = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(first)
    await runtime.decide(first)

    assert len(orchestrator.results) == 1
    assert store.unread("alert") == []
    assert store.pending_acknowledgements() == {"feed_plugin:main": ["a1"]}

    second = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(second)
    await runtime.decide(second)

    assert gateway.ack_attempts == 2
    assert store.pending_acknowledgements() == {}
    assert store.unread("alert") == []
    assert len(orchestrator.results) == 1


@pytest.mark.asyncio
async def test_alert_llm_failure_keeps_event_unread_for_retry(tmp_path, request):
    gateway = FakeGateway(
        [{"kind": "alert", "event_id": "a1", "title": "需要自然处理"}]
    )
    orchestrator = FakeOrchestrator()
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            gateway,
            SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("llm down"))),
            orchestrator,
            _source("alert"),
        ),
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)),
    )
    request.addfinalizer(runtime.close)
    state = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(state)

    with pytest.raises(RuntimeError, match="llm down"):
        await runtime.decide(state)

    assert len(runtime._state.unread("alert")) == 1
    assert orchestrator.results == []
    assert len(runtime._state.observations("alert")) == 1


@pytest.mark.asyncio
async def test_source_poll_failure_is_visible_in_main_log(tmp_path, request, caplog):
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            FailingGateway(),
            SimpleNamespace(chat=AsyncMock()),
            FakeOrchestrator(),
            _source("alert"),
        ),
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)),
    )
    request.addfinalizer(runtime.close)

    with caplog.at_level("INFO", logger="plugins.wake_proactive.runtime"):
        with pytest.raises(RuntimeError, match="所有 proactive sources 拉取失败"):
            await runtime.ingest(runtime.begin(new_proactive_frame("telegram:1")))

    assert "[wake.source] poll failed sources=1" in caplog.text


@pytest.mark.asyncio
async def test_replay_content_only_draws_again_when_a_new_event_arrives(
    tmp_path, request
):
    start = datetime(2026, 7, 12, 12, tzinfo=UTC)
    clock = ReplayClock(tmp_path / "replay" / "clock.json", start)
    events = [
        {
            "kind": "content",
            "event_id": "replay-1",
            "title": "回放时间推进",
            "content": "一条用于验证 hazard 跨模拟时间累积的回放事件。",
            "published_at": start.isoformat(),
            "preprocess_score": 0.5,
        }
    ]
    item_id = "feed_plugin:main:replay-1"
    provider = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "c1",
                            "scratchpad",
                            {
                                "items": [
                                    {
                                        "item_id": "candidate_2",
                                        "initial_interest": "likely_interesting",
                                    }
                                ]
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "c2",
                            "share_content",
                            {
                                "opening": "时间推进后，这条达到了唤醒阈值。",
                                "items": [
                                    {
                                        "item_id": "candidate_2",
                                        "summary": "回放 hazard 已跨 tick 累积。",
                                        "why_it_matters": "验证模拟时间有效",
                                    }
                                ],
                            },
                        )
                    ],
                ),
            ]
        )
    )
    gateway = FakeGateway(events)
    orchestrator = FakeOrchestrator()
    store = WakeStateStore(tmp_path / "wake.db")
    runtime = WakeRuntime(
        _scope(tmp_path, gateway, provider, orchestrator, _source("content")),
        state_store=store,
        clock=clock,
    )
    runtime._content_draw = lambda _session_key, _now: 1.0
    request.addfinalizer(runtime.close)

    first = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(first)
    await runtime.decide(first)

    assert first.hazard_result is not None
    assert first.hazard_result.should_wake is False
    first_meter = store.load_hazard_monitor("telegram:1")
    assert first_meter is not None
    assert first_meter["should_wake"] == 0
    assert first_meter["candidate_count"] == 1
    assert provider.chat.await_count == 0
    assert runtime.next_interval(first) == 1

    _ = clock.advance(timedelta(hours=1))
    second = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(second)
    await runtime.decide(second)

    assert second.hazard_result is None
    assert provider.chat.await_count == 0

    gateway.events.append(
        {
            "kind": "content",
            "event_id": "replay-2",
            "title": "第二条真实事件",
            "published_at": clock.now().isoformat(),
            "preprocess_score": 0.99,
        }
    )
    runtime._content_draw = lambda _session_key, _now: 0.0
    third = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(third)
    await runtime.decide(third)

    assert third.hazard_result is not None
    assert third.hazard_result.should_wake is True
    assert provider.chat.await_count == 2
    assert len(orchestrator.results) == 1
    assert orchestrator.results[0].decision == "reply"
    assert len(store.observations("content")) == 1


@pytest.mark.asyncio
async def test_runtime_owns_rng_when_live_scope_does_not_supply_one(tmp_path, request):
    now = datetime(2026, 7, 12, tzinfo=UTC)
    gateway = FakeGateway(
        [
            {
                "kind": "content",
                "event_id": "live-assembly",
                "title": "真实组装不注入随机源",
                "published_at": now.isoformat(),
                "preprocess_score": 0.2,
            }
        ]
    )
    provider = SimpleNamespace(chat=AsyncMock())
    scope = _scope(tmp_path, gateway, provider, FakeOrchestrator(), _source("content"))
    scope.rng = None
    runtime = WakeRuntime(scope, clock=FixedClock(now))
    request.addfinalizer(runtime.close)
    assert isinstance(runtime._rng, random.Random)
    runtime._rng.seed(0)
    state = runtime.begin(new_proactive_frame("telegram:1"))

    await runtime.ingest(state)
    await runtime.decide(state)

    assert state.hazard_result is not None
    assert provider.chat.await_count == 0


def test_replay_content_draw_is_stable_across_runtime_restart(tmp_path, request):
    now = datetime(2026, 7, 12, 12, tzinfo=UTC)
    clock = ReplayClock(tmp_path / "replay" / "clock.json", now)
    scope = _scope(
        tmp_path,
        FakeGateway([]),
        SimpleNamespace(chat=AsyncMock()),
        FakeOrchestrator(),
        _source("content"),
    )
    first = WakeRuntime(scope, clock=clock)
    second = WakeRuntime(scope, clock=clock)
    request.addfinalizer(first.close)
    request.addfinalizer(second.close)

    assert first._content_draw("telegram:1", now) == second._content_draw(
        "telegram:1", now
    )
    assert first._content_draw(
        "telegram:1", now
    ) != first._content_draw("telegram:1", now + timedelta(hours=1))


def test_live_content_draw_uses_runtime_rng(tmp_path, request):
    scope = _scope(
        tmp_path,
        FakeGateway([]),
        SimpleNamespace(chat=AsyncMock()),
        FakeOrchestrator(),
        _source("content"),
    )
    scope.rng = SimpleNamespace(random=lambda: 0.75)
    runtime = WakeRuntime(scope, clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)))
    request.addfinalizer(runtime.close)

    assert runtime._content_draw(
        "telegram:1", datetime(2026, 7, 12, tzinfo=UTC)
    ) == 0.75


def test_replay_last_user_time_cannot_see_future_message(tmp_path, request):
    session_db = tmp_path / "sessions.db"
    with closing(sqlite3.connect(session_db)) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY, session_key TEXT, seq INTEGER,
                role TEXT, content TEXT, extra TEXT, ts TEXT
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, 'telegram:1', ?, 'user', '', '{}', ?)",
            [
                ("past", 1, "2026-07-11T00:00:00+00:00"),
                ("future", 2, "2026-07-13T00:00:00+00:00"),
            ],
        )
        db.commit()
    now = datetime(2026, 7, 12, tzinfo=UTC)
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            FakeGateway([]),
            SimpleNamespace(chat=AsyncMock()),
            FakeOrchestrator(),
            _source("content"),
        ),
        clock=ReplayClock(tmp_path / "replay" / "clock.json", now),
    )
    request.addfinalizer(runtime.close)

    assert runtime._last_user_at("telegram:1", now) == datetime(
        2026, 7, 11, tzinfo=UTC
    )


def test_plugin_exposes_complete_wake_lifecycle():
    plugin = WakeProactivePlugin()
    assert plugin.proactive_lifecycles()[0].id == "wake"
    assert isinstance(plugin.proactive_runtime_factories()[0], WakeRuntimeFactory)
    assert plugin.proactive_module_factories()[0].lifecycle_id == "wake"


def test_future_message_embeddings_do_not_affect_current_interest(tmp_path, request):
    session_db = tmp_path / "sessions.db"
    with closing(sqlite3.connect(session_db)) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                extra TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages(id, session_key, seq, role, content, extra, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("u0", "s", -1, "user", "主动前用户", "{}", "2026-07-09T00:00:00+00:00"),
                ("ap0", "s", 0, "assistant", "主动推送", "{\"proactive\": true}", "2026-07-09T00:01:00+00:00"),
                ("u1", "s", 1, "user", "过去用户", "{}", "2026-07-10T00:00:00+00:00"),
                ("a1", "s", 2, "assistant", "过去助手", "{}", "2026-07-10T00:01:00+00:00"),
                ("u2", "s", 3, "user", "未来用户", "{}", "2026-07-13T00:00:00+00:00"),
                ("a2", "s", 4, "assistant", "未来助手", "{}", "2026-07-13T00:01:00+00:00"),
            ],
        )
        db.commit()
    embedding_store = MessageEmbeddingStore(session_db)
    for message_id, content, vector in (
        ("u0", "主动前用户", [0.0, 1.0]),
        ("ap0", "主动推送", [0.0, 1.0]),
        ("u1", "过去用户", [1.0, 0.0]),
        ("a1", "过去助手", [1.0, 0.0]),
        ("u2", "未来用户", [0.0, 1.0]),
        ("a2", "未来助手", [0.0, 1.0]),
    ):
        embedding_store.upsert(
            message_id=message_id,
            content=content,
            model="test-embedding",
            embedding=vector,
        )
    embedding_store.close()

    scope = _scope(
        tmp_path,
        FakeGateway([]),
        SimpleNamespace(chat=AsyncMock()),
        FakeOrchestrator(),
        _source("content"),
    )
    runtime = WakeRuntime(
        scope,
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)),
    )
    request.addfinalizer(runtime.close)
    current_event = {
        "preprocess_score": 0.9,
        "preprocess_features": {"interest": 0.1},
        "_event_embedding": [0.0, 1.0],
    }
    future_event = dict(current_event)

    runtime._apply_semantic_interest(
        [current_event], datetime(2026, 7, 12, tzinfo=UTC)
    )
    runtime._apply_semantic_interest(
        [future_event], datetime(2026, 7, 14, tzinfo=UTC)
    )

    assert current_event["_wake_interest_score"] == pytest.approx(0.1)
    assert future_event["_wake_interest_score"] > 0.99
    now = datetime(2026, 7, 12, tzinfo=UTC)
    recent = runtime._read_recent_passive_conversation("s", now)
    assert "主动推送" not in recent
    assert "过去用户" in recent
    assert "未来用户" not in recent
    proactive = runtime._read_recent_proactive_messages(now)
    assert "2026-07-09T00:01:00+00:00" in proactive
    assert "session=s" in proactive
    assert "主动推送" in proactive
    assert "未来助手" not in proactive


def test_recent_proactive_messages_are_workspace_wide_and_bounded(tmp_path, request):
    session_db = tmp_path / "sessions.db"
    now = datetime(2026, 7, 12, tzinfo=UTC)
    recent_rows = [
        (
            f"p{index}",
            f"mobile:{index % 2}",
            index,
            "assistant",
            f"最近主动消息 {index}",
            '{"proactive": true}',
            (now - timedelta(hours=32 - index)).isoformat(),
        )
        for index in range(32)
    ]
    boundary_rows = [
        (
            "old",
            "telegram:old",
            1,
            "assistant",
            "窗口外主动消息",
            '{"proactive": true}',
            (now - timedelta(days=8)).isoformat(),
        ),
        (
            "future",
            "telegram:future",
            1,
            "assistant",
            "未来主动消息",
            '{"proactive": true}',
            (now + timedelta(minutes=1)).isoformat(),
        ),
    ]
    with closing(sqlite3.connect(session_db)) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                extra TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [*recent_rows, *boundary_rows],
        )
        db.commit()

    runtime = WakeRuntime(
        _scope(
            tmp_path,
            FakeGateway([]),
            SimpleNamespace(chat=AsyncMock()),
            FakeOrchestrator(),
            _source("content"),
        ),
        clock=FixedClock(now),
    )
    request.addfinalizer(runtime.close)

    proactive = runtime._read_recent_proactive_messages(now)
    proactive_lines = proactive.splitlines()

    assert len(proactive_lines) == 30
    assert not any(line.endswith("最近主动消息 0") for line in proactive_lines)
    assert not any(line.endswith("最近主动消息 1") for line in proactive_lines)
    assert any(line.endswith("最近主动消息 2") for line in proactive_lines)
    assert any(line.endswith("最近主动消息 31") for line in proactive_lines)
    assert "session=mobile:0" in proactive
    assert "session=mobile:1" in proactive
    assert "窗口外主动消息" not in proactive
    assert "未来主动消息" not in proactive


def test_semantic_interest_keeps_moderate_match_meaningful(tmp_path, request):
    session_db = tmp_path / "sessions.db"
    with closing(sqlite3.connect(session_db)) as db:
        db.execute(
            """
            CREATE TABLE messages(
                id TEXT, session_key TEXT, seq INTEGER, role TEXT,
                content TEXT, extra TEXT, ts TEXT
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("u", "s", 1, "user", "用户", "{}", "2026-07-11T00:00:00+00:00"),
                ("a", "s", 2, "assistant", "助手", "{}", "2026-07-11T00:01:00+00:00"),
            ],
        )
        db.commit()
    embedding_store = MessageEmbeddingStore(session_db)
    for message_id, content in (("u", "用户"), ("a", "助手")):
        embedding_store.upsert(
            message_id=message_id,
            content=content,
            model="test-embedding",
            embedding=[1.0, 0.0],
        )
    embedding_store.close()
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            FakeGateway([]),
            SimpleNamespace(chat=AsyncMock()),
            FakeOrchestrator(),
            _source("content"),
        ),
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)),
    )
    request.addfinalizer(runtime.close)
    event = {
        "preprocess_features": {"interest": 0.1},
        "_event_embedding": [0.8, 0.6],
    }

    runtime._apply_semantic_interest([event], datetime(2026, 7, 12, tzinfo=UTC))

    assert event["_wake_semantic_interest"] == pytest.approx(0.8**4)
    assert event["_wake_interest_score"] == pytest.approx(0.46864)


def test_turn_prototype_limit_uses_latest_time_across_sessions(tmp_path, request):
    session_db = tmp_path / "sessions.db"
    rows = []
    base = datetime(2026, 7, 1, tzinfo=UTC)
    for index in range(256):
        session_key = f"zzz-old-{index:03d}"
        user_id = f"old-u-{index}"
        assistant_id = f"old-a-{index}"
        at = base + timedelta(minutes=index)
        rows.extend(
            [
                (user_id, session_key, 1, "user", user_id, "{}", at.isoformat()),
                (
                    assistant_id,
                    session_key,
                    2,
                    "assistant",
                    assistant_id,
                    "{}",
                    (at + timedelta(seconds=1)).isoformat(),
                ),
            ]
        )
    rows.extend(
        [
            ("latest-u", "aaa-latest", 1, "user", "latest-u", "{}", "2026-07-11T00:00:00+00:00"),
            ("latest-a", "aaa-latest", 2, "assistant", "latest-a", "{}", "2026-07-11T00:00:01+00:00"),
        ]
    )
    with closing(sqlite3.connect(session_db)) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                extra TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages(id, session_key, seq, role, content, extra, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()
    embeddings = MessageEmbeddingStore(session_db)
    for message_id, _session_key, _seq, _role, content, _extra, _ts in rows:
        embeddings.upsert(
            message_id=message_id,
            content=content,
            model="test-embedding",
            embedding=[1.0, 0.0] if message_id.startswith("latest") else [0.0, 1.0],
        )
    embeddings.close()
    scope = _scope(
        tmp_path,
        FakeGateway([]),
        SimpleNamespace(chat=AsyncMock()),
        FakeOrchestrator(),
        _source("content"),
    )
    runtime = WakeRuntime(scope, clock=FixedClock(datetime(2026, 7, 12, tzinfo=UTC)))
    request.addfinalizer(runtime.close)

    prototypes = runtime._load_turn_prototypes(datetime(2026, 7, 12, tzinfo=UTC))

    assert len(prototypes) == 256
    assert any(vector[0] == pytest.approx(1.0) for vector in prototypes)


@pytest.mark.asyncio
async def test_context_snapshot_without_event_id_only_reevaluates_queued_content(
    tmp_path,
    request,
):
    now = datetime(2026, 7, 12, 12, tzinfo=UTC)
    gateway = FakeContextGateway(
        {
            "presence": "sleeping",
            "confidence": 0.9,
            "observed_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=15)).isoformat(),
        }
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        "context",
                        "skip_event",
                        {"reason": "用户醒来本身不值得打扰"},
                    )
                ],
            )
        )
    )
    orchestrator = FakeOrchestrator()
    store = WakeStateStore(tmp_path / "wake.db")
    scope = _scope(tmp_path, gateway, provider, orchestrator, _source("context"))
    scope.cfg.drift_enabled = False
    clock = FixedClock(now)
    runtime = WakeRuntime(scope, state_store=store, clock=clock)
    request.addfinalizer(runtime.close)

    first = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(first)
    await runtime.decide(first)

    assert first.context_reevaluate is False
    assert store.load_context("feed_plugin:main").presence == "sleeping"
    assert '"presence": "sleeping"' in runtime._current_context_text(now)
    assert provider.chat.await_count == 0
    assert orchestrator.results == []

    _ = store.ingest(
        "content",
        [
            {
                "ack_server": "queued",
                "event_id": "c1",
                "title": "等待重评的标题",
                "published_at": now.isoformat(),
                    "preprocess_score": 0.5,
            }
        ],
        now,
    )
    store.save_hazard(
        session_key="telegram:1",
        hazard=0.0,
        threshold=99.0,
        updated_at=now - timedelta(hours=1),
        last_wake_at=None,
    )
    gateway.snapshot = {
        "presence": "active",
        "confidence": 0.9,
        "observed_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=15)).isoformat(),
    }
    second = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(second)
    await runtime.decide(second)

    assert second.context_reevaluate is True
    assert second.hazard_result is None
    assert store.load_context("feed_plugin:main").presence == "active"
    assert '"presence": "active"' in runtime._current_context_text(now)
    assert len(store.unread("content")) == 1
    assert provider.chat.await_count == 1
    assert "mode=context" in provider.chat.await_args.kwargs["messages"][1]["content"]
    assert len(orchestrator.results) == 1
    assert orchestrator.results[0].decision == "skip"
    assert len(store.observations("context")) == 1

    clock.advance(timedelta(hours=1))
    third = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(third)
    await runtime.decide(third)

    assert third.context_reevaluate is False
    assert runtime._current_context_text(clock.now()) == "没有有效 ContextEvent"
    assert third.hazard_result is None
    assert len(store.unread("content")) == 1
    assert provider.chat.await_count == 1
    assert len(orchestrator.results) == 1


@pytest.mark.asyncio
async def test_context_transition_can_be_shared_by_single_llm_call(tmp_path, request):
    now = datetime(2026, 7, 12, 12, tzinfo=UTC)
    gateway = FakeContextGateway(
        {
            "presence": "sleeping",
            "confidence": 0.9,
            "summary": "用户当前可能已经睡着",
            "observed_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=15)).isoformat(),
        }
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        "context",
                        "send_event",
                        {"message": "醒啦？如果刚起来，先慢慢缓一会儿。"},
                    )
                ],
            )
        )
    )
    orchestrator = FakeOrchestrator()
    runtime = WakeRuntime(
        _scope(tmp_path, gateway, provider, orchestrator, _source("context")),
        state_store=WakeStateStore(tmp_path / "wake.db"),
        clock=FixedClock(now),
    )
    request.addfinalizer(runtime.close)

    await runtime.ingest(runtime.begin(new_proactive_frame("telegram:1")))
    gateway.snapshot = {
        "presence": "active",
        "confidence": 0.9,
        "summary": "用户当前更可能醒着",
        "transition": "sleeping->active",
        "observed_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=15)).isoformat(),
    }
    state = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(state)
    await runtime.decide(state)

    assert provider.chat.await_count == 1
    prompt = provider.chat.await_args.kwargs["messages"][1]["content"]
    assert "mode=context" in prompt
    assert "用户当前更可能醒着" in prompt
    assert orchestrator.results[0].outbound.content == "醒啦？如果刚起来，先慢慢缓一会儿。"


@pytest.mark.asyncio
async def test_context_transitions_are_globally_throttled_to_once_per_three_hours(
    tmp_path, request
):
    start = datetime(2026, 7, 12, 0, tzinfo=UTC)

    def snapshot(presence: str, now: datetime) -> dict:
        return {
            "presence": presence,
            "confidence": 0.9,
            "observed_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=10)).isoformat(),
        }

    clock = FixedClock(start)
    gateway = FakeContextGateway(snapshot("sleeping", start))
    store = WakeStateStore(tmp_path / "wake.db")
    runtime = WakeRuntime(
        _scope(
            tmp_path,
            gateway,
            SimpleNamespace(chat=AsyncMock()),
            FakeOrchestrator(),
            _source("context"),
        ),
        state_store=store,
        clock=clock,
    )
    request.addfinalizer(runtime.close)

    initial = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(initial)
    assert initial.context_reevaluate is False

    gateway.snapshot = snapshot("active", start)
    first = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(first)
    assert first.context_reevaluate is True

    clock.advance(timedelta(hours=1))
    gateway.snapshot = snapshot("sleeping", clock.now())
    throttled = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(throttled)
    assert throttled.context_reevaluate is False

    clock.advance(timedelta(hours=2))
    gateway.snapshot = snapshot("active", clock.now())
    boundary = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(boundary)
    assert boundary.context_reevaluate is True

    audit = store.context_reevaluation_state()
    assert audit is not None
    assert audit["last_signaled_at"] == clock.now().isoformat()
    assert audit["last_candidate_at"] == clock.now().isoformat()
    assert audit["suppressed_count"] == 1


def _drift_runtime(tmp_path, provider, orchestrator, now):
    _ = reset_veda(tmp_path)
    store = WakeStateStore(tmp_path / "wake.db")
    scope = _scope(
        tmp_path,
        FakeGateway([]),
        provider,
        orchestrator,
        _source("content"),
    )
    scope.presence = SimpleNamespace(
        get_last_user_at=lambda _session_key: now - timedelta(hours=12)
    )
    runtime = WakeRuntime(scope, state_store=store, clock=FixedClock(now))
    store.save_drift_progress(
        session_key="telegram:1",
        hazard=0.9,
        threshold=0.8,
        updated_at=now - timedelta(hours=1),
    )
    return runtime, store


@pytest.mark.asyncio
async def test_drift_due_event_calls_model_and_sends_reply(
    tmp_path,
    request,
):
    now = datetime(2026, 7, 12, 12, tzinfo=UTC)
    skill_dir = tmp_path / "drift" / "skills" / "explore-curiosity"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: explore-curiosity\ndescription: 探索一个轻量想法\n---\n",
        encoding="utf-8",
    )
    provider = SimpleNamespace(
        chat=AsyncMock(
            side_effect=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "select",
                            "select_skill",
                            {
                                "skill_name": "explore-curiosity",
                                "decision": "explore",
                                "intention": "想想事件钟",
                                "reason": "当前适合做一个轻量探索",
                            },
                        )
                    ],
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "push",
                            "message_push",
                            {"message": "突然想到，时间回放也许可以做成事件钟。"},
                        )
                    ],
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            "finish",
                            "finish_drift",
                            {
                                "skill_used": "explore-curiosity",
                                "status": "completed",
                                "briefing": "分享了事件钟想法",
                                "self_update": {
                                    "next_tendency": "下次按当时状态自由选择",
                                    "reflection": "本轮普通闭环",
                                    "pattern": "ordinary",
                                },
                            },
                        )
                    ],
                ),
            ]
        )
    )
    orchestrator = FakeOrchestrator()
    runtime, store = _drift_runtime(tmp_path, provider, orchestrator, now)
    request.addfinalizer(runtime.close)
    _ = store.ingest_context(
        [
            {
                "_source": "activity:main",
                "presence": "driving",
                "road_kind": "highway",
                "confidence": 0.91,
                "observed_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=1)).isoformat(),
            }
        ],
        now,
    )

    state = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(state)
    await runtime.decide(state)

    cast(FixedClock, runtime._clock).advance(timedelta(minutes=1))
    due = runtime.begin(new_proactive_frame("telegram:1"))
    await runtime.ingest(due)
    await runtime.decide(due)

    assert provider.chat.await_count == 3
    assert len(orchestrator.results) == 1
    assert orchestrator.results[0].outbound.content == "突然想到，时间回放也许可以做成事件钟。"
    observations = store.observations("drift")
    assert len(observations) == 1
    assert json.loads(observations[0]["llm_input_json"]) == []
    prompt = provider.chat.await_args_list[0].kwargs["messages"][1]["content"]
    assert "drift_skills" in prompt
    assert "explore-curiosity" in prompt
    assert "current_context_events" in prompt
    assert '"presence": "driving"' in prompt
    assert '"road_kind": "highway"' in prompt
    first_tools = {
        schema["function"]["name"]
        for schema in provider.chat.await_args_list[0].kwargs["tools"]
    }
    second_tools = {
        schema["function"]["name"]
        for schema in provider.chat.await_args_list[1].kwargs["tools"]
    }
    third_tools = {
        schema["function"]["name"]
        for schema in provider.chat.await_args_list[2].kwargs["tools"]
    }
    assert first_tools == {"select_skill", "idle_drift"}
    assert {"read_file", "write_file", "message_push", "finish_drift"} <= second_tools
    assert third_tools == {"finish_drift"}
    assert state.drift_ctx is None
    assert due.drift_ctx is not None
    assert due.drift_ctx.drift_finished is True
    assert due.drift_ctx.drift_message_sent is True
    drift = store.load_drift("telegram:1")
    assert drift["last_drift_at"] == (now + timedelta(minutes=1)).isoformat()
    assert drift["last_fingerprint"] == "突然想到，时间回放也许可以做成事件钟。"
    assert drift["hazard"] == 0


def test_content_page_uses_score_with_source_diversity_decay() -> None:
    published_at = "2026-07-12T00:00:00+00:00"
    events = [
        {
            "id": f"a:{index}",
            "_reservoir_original_source_id": "a",
            "published_at": published_at,
            "preprocess_score": 0.9 - index * 0.01,
        }
        for index in range(5)
    ] + [
        {
            "id": f"b:{index}",
            "_reservoir_original_source_id": "b",
            "published_at": published_at,
            "preprocess_score": 0.7 - index * 0.01,
        }
        for index in range(2)
    ]

    page = select_content_page(
        events,
        now=datetime(2026, 7, 12, tzinfo=UTC),
        limit=4,
    )

    assert [event["id"] for event in page] == ["a:0", "b:0", "a:1", "b:1"]


def test_ineligible_backfill_stays_unread_for_continuous_downweighting(tmp_path) -> None:
    store = WakeStateStore(tmp_path / "wake.db")
    now = datetime(2026, 7, 12, tzinfo=UTC)

    inserted = store.ingest(
        "content",
        [
            {
                "event_id": "old",
                "ack_server": "feed:main",
                "source_id": "source",
                "published_at": "2025-01-01T00:00:00+00:00",
                "wake_eligible": False,
            }
        ],
        now,
    )

    assert inserted == 1
    assert len(store.unread("content")) == 1
    status = store._conn.execute(
        "SELECT status FROM reservoir_events"
    ).fetchone()[0]
    assert status == "unread"
    store.close()
