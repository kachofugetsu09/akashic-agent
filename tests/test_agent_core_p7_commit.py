from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.passive_turn import AgentCore, AgentCoreDeps
from agent.core.response_parser import parse_response
from agent.core.runtime_support import TurnRunResult
from agent.core.types import ContextBundle
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.types import AfterReasoningCtx
from agent.looping.lifecycle_consumers import (
    register_turn_committed_consumers,
)
from bootstrap.wiring import wire_turn_lifecycle
from bus.event_bus import EventBus
from bus.events import InboundMessage
from bus.events_lifecycle import TurnCommitted


def _load_meme_plugin_class() -> Any:
    path = Path(__file__).parents[1] / "plugins" / "02_meme" / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        "test_p7_02_meme_plugin",
        path,
        submodule_search_locations=[str(path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MemePlugin


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict[str, object]] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict[str, object]]:
        return self.messages[-max_messages:]

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg: dict[str, object] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if media:
            msg["media"] = list(media)
        msg.update(kwargs)
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_commit_fanout_enqueues_recent_before_return():
    session = _DummySession("telegram:trace")
    session_manager = SimpleNamespace(get_or_create=MagicMock(return_value=session))
    event_bus = EventBus()
    release_recent = asyncio.Event()

    async def _refresh_recent_turns(*, session) -> None:
        await release_recent.wait()

    async def _engine_refresh_recent_turns(request) -> None:
        await _refresh_recent_turns(session=request.session)

    register_turn_committed_consumers(
        event_bus=event_bus,
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, SimpleNamespace(schedule_consolidation=MagicMock())),
        memory_engine=cast(
            Any,
            SimpleNamespace(
                refresh_recent_turns=AsyncMock(
                    side_effect=_engine_refresh_recent_turns
                )
            ),
        ),
    )

    await event_bus.emit(
        TurnCommitted(
            session_key="telegram:trace",
            channel="telegram",
            chat_id="trace",
            input_message="你好",
            persisted_user_message="你好",
            assistant_response="收到",
            tools_used=[],
            thinking=None,
            raw_reply="收到",
            meme_tag=None,
            meme_media_count=0,
            tool_chain_raw=[],
            tool_call_groups=[],
            post_reply_budget={
                "history_window": 500,
                "history_messages": 1,
                "history_chars": 12,
                "history_tokens": 4,
                "prompt_tokens": 3,
                "next_turn_baseline_tokens": 1,
            },
            react_stats={},
        )
    )

    recent_tasks = [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "recent_context:telegram:trace"
    ]
    assert recent_tasks
    release_recent.set()
    await asyncio.gather(*recent_tasks)
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_context_store_commit_persists_commits_and_dispatches():
    order: list[str] = []
    session = _DummySession("telegram:123")
    presence = SimpleNamespace(record_user_message=MagicMock(side_effect=lambda _key: None))
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("persist")),
    )
    outbound = SimpleNamespace(dispatch=AsyncMock(side_effect=lambda *_args, **_kwargs: order.append("dispatch") or True))
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []

    event_bus.on(
        TurnCommitted,
        lambda event: order.append("committed") or committed_events.append(event),
    )
    register_turn_committed_consumers(
        event_bus=event_bus,
        session_manager=cast(Any, session_manager),
        scheduler=cast(Any, SimpleNamespace(schedule_consolidation=MagicMock())),
        memory_engine=cast(
            Any,
            SimpleNamespace(refresh_recent_turns=AsyncMock()),
        ),
    )

    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=["refactor"],
                retrieved_memory_block="remembered",
            )
        )
    )
    context = SimpleNamespace(
        render=MagicMock(
            return_value=SimpleNamespace(system_prompt="p", messages=[]),
        )
    )
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="整理好了",
                tools_used=["noop"],
                tool_chain=[{"text": "", "calls": []}],
                thinking="思考",
                streamed=True,
                context_retry={
                    "selected_plan": "full",
                    "react_stats": {
                        "iteration_count": 3,
                        "turn_input_sum_tokens": 42100,
                        "turn_input_peak_tokens": 18800,
                        "final_call_input_tokens": 17500,
                    },
                },
            )
        )
    )
    tools = SimpleNamespace(
        set_context=MagicMock()
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=cast(
                Any,
                SimpleNamespace(
                    session_manager=session_manager,
                    presence=presence,
                ),
            ),
            context_store=cast(Any, context_store),
            context=cast(Any, context),
            tools=cast(Any, tools),
            reasoner=cast(Any, reasoner),
            event_bus=event_bus,
            outbound_port=cast(Any, outbound),
            history_window=500,
        )
    )

    out = await agent_core.process(
        InboundMessage(
            channel="telegram",
            sender="hua",
            chat_id="123",
            content="你好",
            metadata={"req_id": "r1"},
        ),
        "telegram:123",
        dispatch_outbound=True,
    )
    await event_bus.drain()

    assert out.content == "整理好了"
    assert out.media == []
    assert out.metadata["req_id"] == "r1"
    assert out.metadata["tools_used"] == ["noop"]
    assert out.metadata["streamed_reply"] is True
    assert order == ["persist", "committed", "dispatch"]
    presence.record_user_message.assert_called_once_with("telegram:123")
    session_manager.append_messages.assert_awaited_once()
    assert session.messages[-1]["content"] == "整理好了"
    assert session.messages[-1]["reasoning_content"] == "思考"
    assert session.messages[-1].get("cited_memory_ids", []) == []
    assert len(committed_events) == 1
    tc = committed_events[0]
    assert tc.persisted_user_message == "你好"
    assert tc.assistant_response == "整理好了"
    assert tc.meme_media_count == 0
    assert tc.raw_reply == "整理好了"
    assert tc.post_reply_budget["history_window"] == 500
    assert tc.post_reply_budget["history_messages"] == 2
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_turn_committed_omits_user_message_when_user_turn_not_persisted():
    session = _DummySession("cli:direct")
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        append_messages=AsyncMock(),
    )
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []
    event_bus.on(TurnCommitted, lambda event: committed_events.append(event))

    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=[],
                retrieved_memory_block="",
            )
        )
    )
    context = SimpleNamespace(
        render=MagicMock(
            return_value=SimpleNamespace(system_prompt="p", messages=[]),
        )
    )
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="完成",
                tools_used=[],
                tool_chain=[],
                thinking=None,
                streamed=False,
                context_retry={},
            )
        )
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=cast(
                Any,
                SimpleNamespace(
                    session_manager=session_manager,
                    presence=None,
                ),
            ),
            context_store=cast(Any, context_store),
            context=cast(Any, context),
            tools=cast(
                Any,
                SimpleNamespace(set_context=MagicMock()),
            ),
            reasoner=cast(Any, reasoner),
            event_bus=event_bus,
            outbound_port=cast(
                Any,
                SimpleNamespace(dispatch=AsyncMock(return_value=True)),
            ),
            history_window=500,
        )
    )

    await agent_core.process(
        InboundMessage(
            channel="cli",
            sender="hua",
            chat_id="direct",
            content="内部提示词",
            metadata={"omit_user_turn": True},
        ),
        "cli:direct",
        dispatch_outbound=False,
    )
    await event_bus.drain()

    assert committed_events[0].persisted_user_message is None
    assert committed_events[0].assistant_response == "完成"
    assert [msg["role"] for msg in session.messages] == ["assistant"]
    session_manager.append_messages.assert_awaited_once()
    await event_bus.aclose()


def test_response_parser_keeps_reply_protocols_for_plugins():
    text = "答复正文\n§cited:[mem_1]§ <meme:shy>"

    parsed = parse_response(text, tool_chain=[])

    assert parsed.clean_text == text
    assert parsed.metadata.raw_text == text


# ── 新链 (AfterReasoning + AfterTurn) 端到端测试 ──


class _CitationPersistModule:
    async def run(self, frame):
        ctx = frame.slots["reasoning:ctx"]
        ctx.reply = "原始回复 <meme:shy>"
        frame.slots["persist:assistant:cited_memory_ids"] = ["mem_1"]
        return frame


@pytest.mark.asyncio
async def test_new_chain_after_reasoning_persists_meme_and_fires_turn_committed(tmp_path: Path):
    from agent.core.passive_turn import ContextStore
    from agent.plugins.context import PluginContext, PluginKVStore

    MemePlugin = _load_meme_plugin_class()

    order: list[str] = []
    memes = tmp_path / "memes"
    (memes / "shy").mkdir(parents=True)
    image = memes / "shy" / "001.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    (memes / "manifest.json").write_text(
        '{"categories":{"shy":{"desc":"害羞","enabled":true}}}',
        encoding="utf-8",
    )
    session = _DummySession("telegram:456")
    presence = SimpleNamespace(
        record_user_message=MagicMock(side_effect=lambda _key: order.append("presence")),
    )
    session_manager = SimpleNamespace(
        get_or_create=MagicMock(return_value=session),
        peek_next_message_id=MagicMock(return_value="telegram:456:0"),
        append_messages=AsyncMock(side_effect=lambda *a, **kw: order.append("persist")),
    )
    event_bus = EventBus()
    committed_events: list[TurnCommitted] = []
    event_bus.on(
        TurnCommitted,
        lambda event: order.append("committed") or committed_events.append(event),
    )
    dispatch_port = SimpleNamespace(
        dispatch=AsyncMock(side_effect=lambda *a, **kw: order.append("dispatch")),
    )
    plugin_dir = tmp_path / "plugins" / "meme"
    plugin_dir.mkdir(parents=True)
    meme_plugin = MemePlugin()
    meme_plugin.context = PluginContext(
        event_bus=event_bus,
        tool_registry=None,
        plugin_id="meme",
        plugin_dir=plugin_dir,
        kv_store=PluginKVStore(plugin_dir / ".kv.json"),
        workspace=tmp_path,
    )
    await meme_plugin.initialize()
    event_bus.on(AfterReasoningCtx, meme_plugin.decorate_meme)
    context_store = SimpleNamespace(
        prepare=AsyncMock(
            return_value=ContextBundle(
                skill_mentions=[],
                retrieved_memory_block="",
            )
        ),
    )
    context = SimpleNamespace(
        render=MagicMock(return_value=SimpleNamespace(system_prompt="p", messages=[])),
    )
    tools = SimpleNamespace(set_context=MagicMock())
    reasoner = SimpleNamespace(
        run_turn=AsyncMock(
            return_value=TurnRunResult(
                reply="原始回复 <meme:shy>\n§cited:[mem_1]§",
                tools_used=["noop"],
                tool_chain=[{"text": "done", "calls": []}],
                thinking="思考",
                streamed=True,
                context_retry={
                    "selected_plan": "full",
                    "react_stats": {
                        "iteration_count": 2,
                        "turn_input_sum_tokens": 5000,
                    },
                },
            )
        ),
    )
    agent_core = AgentCore(
        AgentCoreDeps(
            session=cast(
                Any,
                SimpleNamespace(
                    session_manager=session_manager,
                    presence=presence,
                ),
            ),
            context_store=cast(ContextStore, context_store),
            context=cast(Any, context),
            tools=cast(Any, tools),
            reasoner=cast(Any, reasoner),
            event_bus=event_bus,
            outbound_port=cast(Any, dispatch_port),
            history_window=100,
            after_reasoning_plugin_modules_before_emit=[_CitationPersistModule()],
        )
    )
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(event_bus),
        active_turn_states={},
    )
    msg = InboundMessage(
        channel="telegram",
        sender="hua",
        chat_id="456",
        content="你好",
        metadata={"req_id": "r2"},
    )

    out = await agent_core.process(msg, "telegram:456")
    await event_bus.drain()

    # 1. outbound 内容来自 meme 插件装饰后
    assert out.content == "原始回复"
    assert out.media == [str(image)]
    assert out.metadata["req_id"] == "r2"
    assert out.metadata["streamed_reply"] is True

    # 3. persist 写入 session
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["content"] == "原始回复"
    assert session.messages[1]["reasoning_content"] == "思考"
    assert session.messages[1]["cited_memory_ids"] == ["mem_1"]
    presence.record_user_message.assert_called_once_with("telegram:456")
    session_manager.append_messages.assert_awaited_once()

    # 4. TurnCommitted 字段正确
    assert len(committed_events) == 1
    tc = committed_events[0]
    assert tc.session_key == "telegram:456"
    assert tc.input_message == "你好"
    assert tc.persisted_user_message == "你好"
    assert tc.assistant_response == "原始回复"
    assert tc.tools_used == ["noop"]
    assert tc.thinking == "思考"
    assert tc.raw_reply == "原始回复 <meme:shy>\n§cited:[mem_1]§"
    assert tc.meme_tag == "shy"
    assert tc.meme_media_count == 1
    assert tc.post_reply_budget["history_window"] == 100
    assert tc.post_reply_budget["history_messages"] == 2
    assert tc.react_stats["iteration_count"] == 2
    assert tc.react_stats["turn_input_sum_tokens"] == 5000

    # 5. 执行顺序: presence → persist → committed → dispatch
    assert order == ["presence", "persist", "committed", "dispatch"]

    # 6. dispatch 实际发送
    dispatch_port.dispatch.assert_awaited_once()

    await event_bus.aclose()
