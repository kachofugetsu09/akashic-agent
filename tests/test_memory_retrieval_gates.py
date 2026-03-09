import asyncio
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from agent.looping.core import AgentLoop
from agent.memory import MemoryStore
from agent.policies.history_route import DecisionMeta, RouteDecision
from agent.provider import LLMResponse
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage
from core.memory.port import DefaultMemoryPort


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
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
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
        memory_port=kwargs.pop("memory_port", DefaultMemoryPort(MemoryStore(workspace))),
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
            ['```json\n{"decision":"NO_RETRIEVE","rewritten_query":"偏好","confidence":"high"}\n```']
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
    assert scheduled_after_append[0] is True
