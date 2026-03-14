import asyncio
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from agent.looping.core import AgentLoop
from agent.looping.handlers import ConversationTurnHandler
from agent.memory import MemoryStore
from agent.policies.history_route import DecisionMeta, RouteDecision
from agent.provider import LLMResponse
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage
from core.memory.port import DefaultMemoryPort
from memory2.query_rewriter import RewriteDecision
from memory2.sufficiency_checker import SufficiencyResult


class _NoopTool(Tool):
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "ok"


class _Provider:
    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts or [])

    async def chat(self, **kwargs):
        if self._texts:
            return LLMResponse(content=self._texts.pop(0), tool_calls=[])
        return LLMResponse(
            content='{"decision":"RETRIEVE","confidence":"high"}', tool_calls=[]
        )


class _DummySession:
    def __init__(self, key: str) -> None:
        self.key = key
        self.messages: list[dict] = []
        self.metadata: dict[str, object] = {}
        self.last_consolidated = 0

    def get_history(self, max_messages: int = 500) -> list[dict]:
        return self.messages[-max_messages:]

    def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        msg.update(kwargs)
        if media:
            msg["media"] = list(media)
        self.messages.append(msg)


def _make_loop(provider: _Provider, **kwargs: Any) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(_NoopTool())
    workspace = kwargs.pop("workspace", Path(tempfile.mkdtemp(prefix="loop-test-")))
    return AgentLoop(
        bus=MagicMock(),
        provider=cast(Any, provider),
        light_provider=cast(Any, provider),
        tools=tools,
        session_manager=MagicMock(),
        workspace=workspace,
        memory_port=kwargs.pop(
            "memory_port", DefaultMemoryPort(MemoryStore(workspace))
        ),
        **kwargs,
    )


def test_route_gate_no_retrieve_when_high_confidence_no_retrieve():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"q","confidence":"high"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(loop._decide_history_route(user_msg="你好", metadata={}))
    assert decision.needs_history is False
    assert decision.rewritten_query == "q"
    assert loop._trace_route_reason(decision) == "ok"


def test_route_gate_fail_open_on_low_confidence():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"q","confidence":"low"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(loop._decide_history_route(user_msg="你好", metadata={}))
    assert decision.needs_history is True
    assert loop._trace_route_reason(decision) == "ok"


def test_route_gate_supports_fenced_json_payload():
    loop = _make_loop(
        _Provider(
            [
                '```json\n{"decision":"NO_RETRIEVE","rewritten_query":"偏好","confidence":"high"}\n```'
            ]
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(
        loop._decide_history_route(user_msg="我之前喜欢什么游戏", metadata={})
    )
    assert decision.needs_history is False
    assert decision.rewritten_query == "偏好"
    assert loop._trace_route_reason(decision) == "ok"


def test_route_decision_exposes_structured_meta():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"偏好","confidence":"high"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(
        loop._decide_history_route(user_msg="我之前喜欢什么游戏", metadata={})
    )
    assert decision.needs_history is False
    assert decision.rewritten_query == "偏好"
    assert decision.fail_open is False
    assert decision.meta.source == "llm"
    assert decision.meta.confidence == "high"
    assert decision.meta.reason_code == "llm_no_retrieve"


def test_route_decision_marks_low_confidence_fail_open():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"q","confidence":"weird"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(loop._decide_history_route(user_msg="你好", metadata={}))
    assert decision.needs_history is True
    assert decision.rewritten_query == "q"
    assert decision.fail_open is True
    assert decision.meta.source == "llm"
    assert decision.meta.confidence == "low"
    assert decision.meta.reason_code == "llm_low_confidence_fail_open"


def test_flow_execution_state_not_triggered_by_single_char_xian_zai():
    loop = _make_loop(_Provider(), memory_route_intention_enabled=True)
    assert loop._is_flow_execution_state("我先问个问题", {}) is False
    assert loop._is_flow_execution_state("我们再看看", {}) is False
    assert loop._is_flow_execution_state("先查再说", {}) is True


def test_flow_execution_state_uses_task_tool_flag_not_any_tool_count():
    loop = _make_loop(_Provider(), memory_route_intention_enabled=True)
    assert (
        loop._is_flow_execution_state(
            "普通问题",
            {"last_turn_tool_calls_count": 3, "last_turn_had_task_tool": False},
        )
        is False
    )
    assert (
        loop._is_flow_execution_state(
            "普通问题",
            {"last_turn_tool_calls_count": 0, "last_turn_had_task_tool": True},
        )
        is True
    )


def test_process_inner_parallelizes_procedure_retrieve_and_route_gate():
    loop = _make_loop(_Provider(), memory_route_intention_enabled=True)
    session = _DummySession("cli:1")
    loop.session_manager.get_or_create.return_value = session
    loop.session_manager.append_messages = AsyncMock(return_value=None)
    loop._run_with_safety_retry = AsyncMock(return_value=("ok", [], []))

    async def _slow_retrieve(*args, **kwargs):
        await asyncio.sleep(0.12)
        return []

    loop._memory_port = MagicMock()
    loop._memory_port.retrieve_related = AsyncMock(side_effect=_slow_retrieve)
    loop._memory_port.select_for_injection = MagicMock(return_value=[])
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))

    async def _slow_route_decision(*args, **kwargs):
        await asyncio.sleep(0.12)
        return RouteDecision(
            needs_history=False,
            rewritten_query="q",
            fail_open=False,
            latency_ms=120,
            meta=DecisionMeta(
                source="llm",
                confidence="high",
                reason_code="llm_no_retrieve",
            ),
        )

    loop._decide_history_route = _slow_route_decision  # type: ignore[assignment]
    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")
    start = time.perf_counter()
    asyncio.run(loop._process_inner(msg, msg.session_key))
    elapsed = time.perf_counter() - start

    # 若串行应接近 0.24s；并行时应接近单个分支耗时。
    assert elapsed < 0.22


def test_build_procedure_context_hint_handles_none_media():
    loop = _make_loop(_Provider())
    handler = ConversationTurnHandler(loop)
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="hello",
        media=None,  # type: ignore[arg-type]
        metadata={},
    )

    assert handler._build_procedure_context_hint(msg) == ""


def test_retrieve_memory_block_prefers_query_rewriter_primary_path():
    loop = _make_loop(_Provider())
    session = _DummySession("cli:1")
    handler = ConversationTurnHandler(loop)
    loop._trace_memory_retrieve = MagicMock()
    loop._query_rewriter = MagicMock()
    loop._query_rewriter.decide = AsyncMock(
        return_value=RewriteDecision(
            needs_retrieval=False,
            procedure_query="下载",
            history_query="下载历史",
            memory_types_hint=["procedure"],
            latency_ms=12,
        )
    )
    loop._decide_history_route = AsyncMock(
        side_effect=AssertionError("should not call history route fallback")
    )
    loop._memory_port = MagicMock()
    loop._memory_port.select_for_injection = MagicMock(return_value=[])
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")

    with patch(
        "agent.looping.handlers.build_procedure_queries",
        side_effect=AssertionError("should not call fallback query builder"),
    ):
        asyncio.run(
            handler._retrieve_memory_block(
                msg=msg,
                key=msg.session_key,
                session=session,
                main_history=[],
            )
        )

    loop._query_rewriter.decide.assert_awaited_once()
    assert loop._trace_memory_retrieve.call_args.kwargs["gate_type"] == "query_rewriter"


def test_retrieve_memory_block_passes_procedure_query_and_user_msg_to_multi_query():
    loop = _make_loop(_Provider())
    session = _DummySession("cli:1")
    handler = ConversationTurnHandler(loop)
    loop._trace_memory_retrieve = MagicMock()
    loop._query_rewriter = MagicMock()
    loop._query_rewriter.decide = AsyncMock(
        return_value=RewriteDecision(
            needs_retrieval=True,
            procedure_query="B站视频下载流程",
            history_query="用户的B站下载偏好历史",
            memory_types_hint=["procedure", "preference"],
            latency_ms=8,
        )
    )
    loop._memory_port = MagicMock()
    loop._memory_port.select_for_injection = MagicMock(return_value=[])
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="把这个B站视频下载下来",
    )

    with (
        patch(
            "agent.looping.handlers.retrieve_procedure_items",
            new=AsyncMock(return_value=[]),
        ) as proc_mock,
        patch(
            "agent.looping.handlers.retrieve_history_items",
            new=AsyncMock(return_value=([], "disabled")),
        ),
    ):
        asyncio.run(
            handler._retrieve_memory_block(
                msg=msg,
                key=msg.session_key,
                session=session,
                main_history=[],
            )
        )

    proc_mock.assert_awaited_once()
    assert proc_mock.call_args.kwargs["queries"] == [
        "B站视频下载流程",
        "把这个B站视频下载下来",
    ]


def test_process_inner_schedules_consolidation_only_after_append_messages():
    loop = _make_loop(_Provider())
    session = _DummySession("cli:1")
    session.messages = [{"role": "user", "content": "x"} for _ in range(41)]
    loop.session_manager.get_or_create.return_value = session
    loop._run_with_safety_retry = AsyncMock(return_value=("ok", [], []))

    append_done = False

    async def _append_messages(*args, **kwargs):
        nonlocal append_done
        append_done = True
        return None

    loop.session_manager.append_messages = AsyncMock(side_effect=_append_messages)
    loop._memory_port = MagicMock()
    loop._memory_port.retrieve_related = AsyncMock(return_value=[])
    loop._memory_port.select_for_injection = MagicMock(return_value=[])
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    loop._decide_history_route = AsyncMock(
        return_value=RouteDecision(
            needs_history=False,
            rewritten_query="q",
            fail_open=False,
            latency_ms=0,
            meta=DecisionMeta(
                source="llm",
                confidence="high",
                reason_code="llm_no_retrieve",
            ),
        )
    )
    scheduled_after_append: list[bool] = []
    real_create_task = asyncio.create_task

    def _fake_create_task(coro, *args, **kwargs):
        name = getattr(getattr(coro, "cr_code", None), "co_name", "")
        if name == "_consolidate_memory_bg":
            scheduled_after_append.append(append_done)
            # 避免未调度协程告警
            try:
                coro.close()
            except Exception:
                pass
            return MagicMock()
        return real_create_task(coro, *args, **kwargs)

    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")
    with patch("agent.looping.core.asyncio.create_task", side_effect=_fake_create_task):
        asyncio.run(loop._process_inner(msg, msg.session_key))

    assert scheduled_after_append


def test_retrieve_memory_block_triggers_sufficiency_check_on_low_score_items():
    """当检索结果最高分低于阈值时，sufficiency checker 被调用。"""
    loop = _make_loop(_Provider())
    loop._query_rewriter = MagicMock()
    loop._query_rewriter.decide = AsyncMock(
        return_value=RewriteDecision(
            needs_retrieval=True,
            procedure_query="仁王游戏讨论",
            history_query="用户关于仁王的历史",
            memory_types_hint=["event"],
            latency_ms=10,
        )
    )
    low_score_items = [
        {
            "id": "x1",
            "memory_type": "procedure",
            "score": 0.479,
            "summary": "西历2236读书进度规则",
            "extra_json": {},
        },
    ]
    checker_mock = MagicMock()
    checker_mock.check = AsyncMock(
        return_value=SufficiencyResult(
            is_sufficient=False,
            reason="irrelevant",
            refined_query="用户与仁王游戏相关的讨论历史",
            latency_ms=40,
        )
    )
    loop._sufficiency_checker = checker_mock
    loop._memory_port = MagicMock()
    loop._memory_port.retrieve_related = AsyncMock(return_value=low_score_items)
    loop._memory_port.select_for_injection = MagicMock(return_value=low_score_items)
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    loop._trace_memory_retrieve = MagicMock()

    session = _DummySession("cli:1")
    handler = ConversationTurnHandler(loop)
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="我之前和你聊过什么有关仁王的内容吗",
    )

    asyncio.run(
        handler._retrieve_memory_block(
            msg=msg,
            key=msg.session_key,
            session=session,
            main_history=[],
        )
    )

    checker_mock.check.assert_awaited_once()


def test_retrieve_memory_block_uses_refined_query_on_insufficient():
    """sufficiency checker 返回 insufficient 时，用 refined_query 重查 history。"""
    loop = _make_loop(_Provider())
    loop._query_rewriter = MagicMock()
    loop._query_rewriter.decide = AsyncMock(
        return_value=RewriteDecision(
            needs_retrieval=True,
            procedure_query="仁王游戏讨论",
            history_query="用户关于仁王的历史",
            memory_types_hint=["event"],
            latency_ms=10,
        )
    )
    checker_mock = MagicMock()
    checker_mock.check = AsyncMock(
        return_value=SufficiencyResult(
            is_sufficient=False,
            reason="irrelevant",
            refined_query="用户与仁王游戏相关的讨论历史",
            latency_ms=40,
        )
    )
    loop._sufficiency_checker = checker_mock
    loop._memory_port = MagicMock()
    loop._memory_port.select_for_injection = MagicMock(
        return_value=[
            {
                "id": "x1",
                "memory_type": "procedure",
                "score": 0.48,
                "summary": "无关规则",
                "extra_json": {},
            },
        ]
    )
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    loop._trace_memory_retrieve = MagicMock()
    loop._memory_port.retrieve_related = AsyncMock(return_value=[])
    history_calls: list[str] = []

    session = _DummySession("cli:1")
    handler = ConversationTurnHandler(loop)
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="我之前和你聊过什么有关仁王的内容吗",
    )

    async def _fake_history_items(memory, query, **kwargs):
        history_calls.append(query)
        return [], "disabled"

    with patch(
        "agent.looping.handlers.retrieve_history_items",
        side_effect=_fake_history_items,
    ):
        asyncio.run(
            handler._retrieve_memory_block(
                msg=msg,
                key=msg.session_key,
                session=session,
                main_history=[],
            )
        )

    assert any("仁王" in q for q in history_calls)


def test_retrieve_memory_block_skips_sufficiency_check_when_checker_is_none():
    """loop._sufficiency_checker 为 None 时，不触发 check，主路径正常跑完。"""
    loop = _make_loop(_Provider())
    loop._query_rewriter = None
    loop._sufficiency_checker = None
    loop._memory_port = MagicMock()
    loop._memory_port.retrieve_related = AsyncMock(return_value=[])
    loop._memory_port.select_for_injection = MagicMock(return_value=[])
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    loop._trace_memory_retrieve = MagicMock()

    session = _DummySession("cli:1")
    handler = ConversationTurnHandler(loop)
    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")

    asyncio.run(
        handler._retrieve_memory_block(
            msg=msg,
            key=msg.session_key,
            session=session,
            main_history=[],
        )
    )


def test_retrieve_memory_block_no_second_retrieval_when_sufficient():
    """sufficiency checker 返回 sufficient 时，不做第二次检索。"""
    loop = _make_loop(_Provider())
    loop._query_rewriter = MagicMock()
    loop._query_rewriter.decide = AsyncMock(
        return_value=RewriteDecision(
            needs_retrieval=True,
            procedure_query="天气查询",
            history_query="用户天气偏好",
            memory_types_hint=["procedure"],
            latency_ms=10,
        )
    )
    checker_mock = MagicMock()
    checker_mock.check = AsyncMock(
        return_value=SufficiencyResult(
            is_sufficient=True,
            reason="sufficient",
            refined_query=None,
            latency_ms=20,
        )
    )
    loop._sufficiency_checker = checker_mock
    retrieve_call_count = 0

    async def _count_retrieve(query, **kwargs):
        nonlocal retrieve_call_count
        retrieve_call_count += 1
        return [
            {
                "id": "w1",
                "memory_type": "procedure",
                "score": 0.538,
                "summary": "天气查询强制走 weather 技能",
                "extra_json": {},
            }
        ]

    loop._memory_port = MagicMock()
    loop._memory_port.retrieve_related = AsyncMock(side_effect=_count_retrieve)
    loop._memory_port.select_for_injection = MagicMock(return_value=[])
    loop._memory_port.format_injection_with_ids = MagicMock(return_value=("", []))
    loop._trace_memory_retrieve = MagicMock()

    session = _DummySession("cli:1")
    handler = ConversationTurnHandler(loop)
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="北京今天天气怎么样",
    )

    asyncio.run(
        handler._retrieve_memory_block(
            msg=msg,
            key=msg.session_key,
            session=session,
            main_history=[],
        )
    )

    assert retrieve_call_count <= 2


def test_consolidate_memory_calls_profile_extractor_when_set():
    loop = _make_loop(_Provider(['{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}']))
    loop._memory_port = MagicMock()
    loop._memory_port.read_long_term = MagicMock(return_value="MEMORY")
    loop._memory_port.append_history_once = MagicMock(return_value=True)
    loop._memory_port.append_pending_once = MagicMock(return_value=True)
    loop._memory_port.save_from_consolidation = AsyncMock()
    loop._memory_port.save_item = AsyncMock(return_value="new:profile-1")
    loop._profile_extractor = MagicMock()
    loop._profile_extractor.extract = AsyncMock(return_value=[])
    session = _DummySession("cli:1")
    session.messages = [
        {"role": "user", "content": "我买了 Zigbee 网关和加湿器", "timestamp": "2026-03-15T10:00:00"},
        {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
    ]
    session._channel = "cli"
    session._chat_id = "1"

    asyncio.run(loop._consolidate_memory(session, archive_all=True, await_vector_store=True))

    loop._profile_extractor.extract.assert_awaited_once()
    extract_call = loop._profile_extractor.extract.await_args
    conversation_arg = (
        extract_call.args[0]
        if extract_call.args
        else str(extract_call.kwargs.get("conversation", ""))
    )
    assert "Zigbee" in conversation_arg


def test_consolidate_memory_works_without_profile_extractor():
    loop = _make_loop(_Provider(['{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}']))
    loop._memory_port = MagicMock()
    loop._memory_port.read_long_term = MagicMock(return_value="MEMORY")
    loop._memory_port.append_history_once = MagicMock(return_value=True)
    loop._memory_port.append_pending_once = MagicMock(return_value=True)
    loop._memory_port.save_from_consolidation = AsyncMock()
    loop._memory_port.save_item = AsyncMock(return_value="new:profile-1")
    loop._profile_extractor = None
    session = _DummySession("cli:1")
    session.messages = [
        {"role": "user", "content": "我买了 Zigbee 网关和加湿器", "timestamp": "2026-03-15T10:00:00"},
        {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
    ]
    session._channel = "cli"
    session._chat_id = "1"

    asyncio.run(loop._consolidate_memory(session, archive_all=True, await_vector_store=True))
