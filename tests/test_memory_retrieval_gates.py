import asyncio
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from agent.core.types import HistoryMessage
from agent.core.runtime_support import TurnRunResult
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig, MemoryConfig
from agent.looping.memory_gate import (
    _decide_history_route,
    _is_flow_execution_state,
    _trace_route_reason,
)
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import RetrievalRequest
from agent.memory import MemoryStore
from agent.policies.history_route import DecisionMeta, RouteDecision
from agent.provider import LLMResponse
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage
from core.memory.engine import MemoryEngineRetrieveResult
from core.memory.port import DefaultMemoryPort
from memory2.query_rewriter import GateDecision
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


def _make_loop(
    provider: _Provider,
    *,
    workspace: Any = None,
    memory_port: Any = None,
    profile_extractor: Any = None,
    memory_route_intention_enabled: bool = False,
    **_unused: Any,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(_NoopTool())
    _workspace = workspace or Path(tempfile.mkdtemp(prefix="loop-test-"))
    _memory_port = memory_port or DefaultMemoryPort(MemoryStore(_workspace))
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, provider),
            light_provider=cast(Any, provider),
            tools=tools,
            session_manager=MagicMock(),
            workspace=_workspace,
            memory_port=_memory_port,
            profile_extractor=profile_extractor,
        ),
        AgentLoopConfig(
            memory=MemoryConfig(route_intention_enabled=memory_route_intention_enabled),
        ),
    )


def _make_retrieval(loop: AgentLoop) -> DefaultMemoryRetrievalPipeline:
    return cast(DefaultMemoryRetrievalPipeline, loop._retrieval_pipeline)


def _req(msg: InboundMessage, session: _DummySession) -> RetrievalRequest:
    return RetrievalRequest(
        message=msg.content,
        session_key=msg.session_key,
        channel=msg.channel,
        chat_id=msg.chat_id,
        history=[
            HistoryMessage(
                role=str(m.get("role", "")),
                content=str(m.get("content", "")),
            )
            for m in session.get_history(max_messages=500)
        ],
        session_metadata=session.metadata,
        timestamp=msg.timestamp,
    )


def test_route_gate_no_retrieve_when_high_confidence_no_retrieve():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"q","confidence":"high"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(
        _decide_history_route(
            user_msg="你好",
            metadata={},
            recent_history="",
            light_provider=cast(Any, loop.light_provider),
            light_model=loop.light_model,
            route_intention_enabled=True,
            gate_llm_timeout_ms=800,
            gate_max_tokens=96,
        )
    )
    assert decision.needs_history is False
    assert decision.rewritten_query == "q"
    assert _trace_route_reason(decision) == "ok"


def test_route_gate_fail_open_on_low_confidence():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"q","confidence":"low"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(
        _decide_history_route(
            user_msg="你好",
            metadata={},
            recent_history="",
            light_provider=cast(Any, loop.light_provider),
            light_model=loop.light_model,
            route_intention_enabled=True,
            gate_llm_timeout_ms=800,
            gate_max_tokens=96,
        )
    )
    assert decision.needs_history is True
    assert _trace_route_reason(decision) == "ok"


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
        _decide_history_route(
            user_msg="我之前喜欢什么游戏",
            metadata={},
            recent_history="",
            light_provider=cast(Any, loop.light_provider),
            light_model=loop.light_model,
            route_intention_enabled=True,
            gate_llm_timeout_ms=800,
            gate_max_tokens=96,
        )
    )
    assert decision.needs_history is False
    assert decision.rewritten_query == "偏好"
    assert _trace_route_reason(decision) == "ok"


def test_route_decision_exposes_structured_meta():
    loop = _make_loop(
        _Provider(
            ['{"decision":"NO_RETRIEVE","rewritten_query":"偏好","confidence":"high"}']
        ),
        memory_route_intention_enabled=True,
    )
    decision = asyncio.run(
        _decide_history_route(
            user_msg="我之前喜欢什么游戏",
            metadata={},
            recent_history="",
            light_provider=cast(Any, loop.light_provider),
            light_model=loop.light_model,
            route_intention_enabled=True,
            gate_llm_timeout_ms=800,
            gate_max_tokens=96,
        )
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
    decision = asyncio.run(
        _decide_history_route(
            user_msg="你好",
            metadata={},
            recent_history="",
            light_provider=cast(Any, loop.light_provider),
            light_model=loop.light_model,
            route_intention_enabled=True,
            gate_llm_timeout_ms=800,
            gate_max_tokens=96,
        )
    )
    assert decision.needs_history is True
    assert decision.rewritten_query == "q"
    assert decision.fail_open is True
    assert decision.meta.source == "llm"
    assert decision.meta.confidence == "low"
    assert decision.meta.reason_code == "llm_low_confidence_fail_open"


def test_flow_execution_state_not_triggered_by_single_char_xian_zai():
    loop = _make_loop(_Provider(), memory_route_intention_enabled=True)
    assert _is_flow_execution_state("我先问个问题", {}) is False
    assert _is_flow_execution_state("我们再看看", {}) is False
    assert _is_flow_execution_state("先查再说", {}) is True


def test_flow_execution_state_uses_task_tool_flag_not_any_tool_count():
    loop = _make_loop(_Provider(), memory_route_intention_enabled=True)
    assert (
        _is_flow_execution_state(
            "普通问题",
            {"last_turn_tool_calls_count": 3, "last_turn_had_task_tool": False},
        )
        is False
    )
    assert (
        _is_flow_execution_state(
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
    loop._reasoner.run_turn = AsyncMock(return_value=TurnRunResult(reply="ok"))

    async def _slow_retrieve(*args, **kwargs):
        await asyncio.sleep(0.12)
        return []

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

    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = None
    memory_port = MagicMock()
    memory_port.retrieve_related = AsyncMock(side_effect=_slow_retrieve)
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port
    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")
    with patch(
        "core.memory.default_runtime_facade._decide_history_route",
        new=AsyncMock(side_effect=_slow_route_decision),
    ):
        start = time.perf_counter()
        asyncio.run(loop._core_runner.process(msg, msg.session_key))
        elapsed = time.perf_counter() - start

    # 若串行应接近 0.24s；并行时应接近单个分支耗时。
    assert elapsed < 0.22

def test_retrieve_memory_block_prefers_query_rewriter_primary_path():
    loop = _make_loop(_Provider())
    session = _DummySession("cli:1")
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = MagicMock()
    retrieval._memory.query_rewriter.decide = AsyncMock(
        return_value=GateDecision(
            needs_episodic=False,
            episodic_query="下载历史",
            latency_ms=12,
        )
    )
    memory_port = MagicMock()
    memory_port.retrieve_related = AsyncMock(return_value=[])
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port
    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")

    asyncio.run(retrieval.retrieve(_req(msg, session)))
    retrieval._memory.query_rewriter.decide.assert_awaited_once()


def test_retrieve_memory_block_query_rewriter_path_uses_raw_msg_for_procedure_lane():
    loop = _make_loop(_Provider())
    session = _DummySession("cli:1")
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = MagicMock()
    retrieval._memory.query_rewriter.decide = AsyncMock(
        return_value=GateDecision(
            needs_episodic=True,
            episodic_query="用户的B站下载偏好历史",
            latency_ms=8,
        )
    )
    memory_port = MagicMock()
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="把这个B站视频下载下来",
    )
    engine_calls: list[dict[str, object]] = []

    with (
        patch(
            "core.memory.default_runtime_facade._retrieve_engine_items",
            new=AsyncMock(
                side_effect=lambda **kwargs: (
                    engine_calls.append(kwargs)
                    or MemoryEngineRetrieveResult(text_block="", hits=[], raw={"items": []})
                )
            ),
        ),
    ):
        asyncio.run(retrieval.retrieve(_req(msg, session)))

    procedure_calls = [call for call in engine_calls if call["mode"] == "procedure"]
    assert len(procedure_calls) == 1
    assert procedure_calls[0]["query"] == "把这个B站视频下载下来"


def test_process_inner_schedules_consolidation_only_after_append_messages():
    loop = _make_loop(_Provider())
    session = _DummySession("cli:1")
    session.messages = [{"role": "user", "content": "x"} for _ in range(41)]
    loop.session_manager.get_or_create.return_value = session
    loop._reasoner.run_turn = AsyncMock(return_value=TurnRunResult(reply="ok"))

    append_done = False

    async def _append_messages(*args, **kwargs):
        nonlocal append_done
        append_done = True
        return None

    loop.session_manager.append_messages = AsyncMock(side_effect=_append_messages)
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = MagicMock()
    retrieval._memory.query_rewriter.decide = AsyncMock(
        return_value=GateDecision(
            needs_episodic=False,
            episodic_query="q",
            latency_ms=0,
        )
    )
    memory_port = MagicMock()
    memory_port.retrieve_related = AsyncMock(return_value=[])
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port
    scheduled_after_append: list[bool] = []
    real_schedule = loop._scheduler.schedule_consolidation

    def _wrapped_schedule(session_obj, session_key):
        scheduled_after_append.append(append_done)
        return real_schedule(session_obj, session_key)

    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")
    with patch.object(loop._scheduler, "schedule_consolidation", side_effect=_wrapped_schedule):
        asyncio.run(loop._core_runner.process(msg, msg.session_key))

    assert scheduled_after_append


def test_retrieve_memory_block_triggers_sufficiency_check_on_low_score_items():
    """当注入 block 为空时，sufficiency checker 被调用。"""
    loop = _make_loop(_Provider())
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = MagicMock()
    retrieval._memory.query_rewriter.decide = AsyncMock(
        return_value=GateDecision(
            needs_episodic=True,
            episodic_query="用户关于仁王的历史",
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
    retrieval._memory.sufficiency_checker = checker_mock
    memory_port = MagicMock()
    memory_port.retrieve_related = AsyncMock(return_value=low_score_items)
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port

    session = _DummySession("cli:1")
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="我之前和你聊过什么有关仁王的内容吗",
    )

    asyncio.run(retrieval.retrieve(_req(msg, session)))

    checker_mock.check.assert_awaited_once()


def test_retrieve_memory_block_uses_refined_query_on_insufficient():
    """sufficiency checker 返回 insufficient 时，用 refined_query 重查 history。"""
    loop = _make_loop(_Provider())
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = MagicMock()
    retrieval._memory.query_rewriter.decide = AsyncMock(
        return_value=GateDecision(
            needs_episodic=True,
            episodic_query="用户关于仁王的历史",
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
    retrieval._memory.sufficiency_checker = checker_mock
    memory_port = MagicMock()
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    memory_port.retrieve_related = AsyncMock(return_value=[])
    retrieval._memory.port = memory_port
    history_calls: list[str] = []

    session = _DummySession("cli:1")
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="我之前和你聊过什么有关仁王的内容吗",
    )

    async def _fake_history_items(**kwargs):
        history_calls.append(str(kwargs.get("rewritten_query", "")))
        return [], "disabled", None, None

    with patch(
        "core.memory.default_runtime_facade._retrieve_episodic_items",
        side_effect=_fake_history_items,
    ):
        asyncio.run(retrieval.retrieve(_req(msg, session)))

    assert any("仁王" in q for q in history_calls)


def test_retrieve_memory_block_skips_sufficiency_check_when_checker_is_none():
    """loop._sufficiency_checker 为 None 时，不触发 check，主路径正常跑完。"""
    loop = _make_loop(_Provider())
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = None
    retrieval._memory.sufficiency_checker = None
    memory_port = MagicMock()
    memory_port.retrieve_related = AsyncMock(return_value=[])
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port

    session = _DummySession("cli:1")
    msg = InboundMessage(channel="cli", sender="u", chat_id="1", content="hello")

    asyncio.run(retrieval.retrieve(_req(msg, session)))


def test_retrieve_memory_block_no_second_retrieval_when_sufficient():
    """sufficiency checker 返回 sufficient 时，不做第二次检索。"""
    loop = _make_loop(_Provider())
    retrieval = _make_retrieval(loop)
    retrieval._memory.query_rewriter = MagicMock()
    retrieval._memory.query_rewriter.decide = AsyncMock(
        return_value=GateDecision(
            needs_episodic=True,
            episodic_query="用户天气偏好",
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
    retrieval._memory.sufficiency_checker = checker_mock
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

    memory_port = MagicMock()
    memory_port.retrieve_related = AsyncMock(side_effect=_count_retrieve)
    memory_port.build_injection_block = MagicMock(return_value=("", []))
    retrieval._memory.port = memory_port

    session = _DummySession("cli:1")
    msg = InboundMessage(
        channel="cli",
        sender="u",
        chat_id="1",
        content="北京今天天气怎么样",
    )

    asyncio.run(retrieval.retrieve(_req(msg, session)))

    assert retrieve_call_count <= 2


def test_consolidate_memory_runs_long_term_extraction_in_parallel():
    """consolidation 触发两次并行 LLM 调用：event 提取 + 合并的 profile/preference/procedure 提取。"""
    memory_port = MagicMock()
    memory_port.read_long_term = MagicMock(return_value="MEMORY")
    memory_port.append_history_once = MagicMock(return_value=True)
    memory_port.append_pending_once = MagicMock(return_value=True)
    memory_port.save_from_consolidation = AsyncMock()
    memory_port.save_item = AsyncMock(return_value="new:profile-1")
    loop = _make_loop(
        _Provider([
            '{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}',
            '{"profile":[],"preference":[],"procedure":[]}',
        ]),
        memory_port=memory_port,
    )
    session = _DummySession("cli:1")
    session.messages = [
        {"role": "user", "content": "我买了 Zigbee 网关和加湿器", "timestamp": "2026-03-15T10:00:00"},
        {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
    ]
    session._channel = "cli"
    session._chat_id = "1"

    asyncio.run(loop._consolidate_memory(session, archive_all=True, await_vector_store=True))

    # event 写入成功
    memory_port.append_history_once.assert_called_once()
    memory_port.save_from_consolidation.assert_awaited_once()


def test_consolidate_memory_works_without_profile_extractor():
    memory_port = MagicMock()
    memory_port.read_long_term = MagicMock(return_value="MEMORY")
    memory_port.append_history_once = MagicMock(return_value=True)
    memory_port.append_pending_once = MagicMock(return_value=True)
    memory_port.save_from_consolidation = AsyncMock()
    memory_port.save_item = AsyncMock(return_value="new:profile-1")
    loop = _make_loop(
        _Provider(['{"history_entries":["[2026-03-15 10:00] 用户聊了 Zigbee 方案"],"pending_items":[]}']),
        memory_port=memory_port,
        profile_extractor=None,
    )
    session = _DummySession("cli:1")
    session.messages = [
        {"role": "user", "content": "我买了 Zigbee 网关和加湿器", "timestamp": "2026-03-15T10:00:00"},
        {"role": "assistant", "content": "记住了", "timestamp": "2026-03-15T10:01:00"},
    ]
    session._channel = "cli"
    session._chat_id = "1"

    asyncio.run(loop._consolidate_memory(session, archive_all=True, await_vector_store=True))
