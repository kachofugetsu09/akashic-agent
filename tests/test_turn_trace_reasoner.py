from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.core.types import ContextRenderResult, ContextRequest
from agent.looping.ports import LLMConfig
from agent.provider import LLMResponse, ToolCall
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from agent.tools.turn_trace import InspectTurnTraceTool
from agent.tracing.turn_trace_query import TurnTraceQueryService
from tests.test_tool_access_gateway_reasoner import _Provider, _RecordingTool
from tests.test_turn_trace_query import _insert_turn, _make_observe_db


def _msg(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime.now(timezone.utc),
    )


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        key="cli:1",
        messages=[],
        metadata={},
        get_history=lambda max_messages=40, *, start_index=None: [],
        last_consolidated=0,
    )


def _make_reasoner(provider: _Provider, observe_db) -> DefaultReasoner:
    tools = ToolRegistry()
    tools.set_context(session_key="cli:1", _session_key="cli:1")
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(_RecordingTool("search_docs"), risk="read-only")
    tools.register(_RecordingTool("fetch_doc_chunk"), risk="read-only")
    tools.register(_RecordingTool("read_file"), always_on=True, risk="read-only")
    tools.register(_RecordingTool("shell"), always_on=True, risk="read-only")
    tools.register(_RecordingTool("list_dir"), always_on=True, risk="read-only")
    tools.register(
        InspectTurnTraceTool(TurnTraceQueryService(observe_db)),
        always_on=False,
        risk="read-only",
    )

    def _render(request: ContextRequest, **_kwargs: object) -> ContextRenderResult:
        return ContextRenderResult(
            system_prompt="",
            turn_injection_context={
                "turn_injection": request.turn_injection_prompt or ""
            },
            messages=[{"role": "user", "content": request.current_message}],
            debug_breakdown=[],
        )

    return DefaultReasoner(
        llm=cast(
            Any,
            LLMServices(provider=provider, light_provider=provider),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=256),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
        memory_window=10,
        context=cast(Any, SimpleNamespace(render=_render)),
        session_manager=cast(Any, SimpleNamespace(save_async=lambda *_args, **_kw: None)),
    )


@pytest.mark.asyncio
async def test_tool_history_answer_uses_structured_turn_trace(tmp_path) -> None:
    observe_db = tmp_path / "observe.db"
    _make_observe_db(observe_db)
    _insert_turn(
        observe_db,
        session_key="cli:1",
        user_msg="第一个：根据项目文档回答",
        tool_chain=[
            {
                "text": "",
                "calls": [
                    {"name": "search_docs", "status": "success", "result": "{}"},
                    {"name": "fetch_doc_chunk", "status": "success", "result": "{}"},
                ],
            }
        ],
    )
    _insert_turn(
        observe_db,
        session_key="cli:1",
        user_msg="第二个：根据项目文档和源码回答",
        tool_chain=[
            {"text": "", "calls": [{"name": "read_file", "status": "success", "result": "a"}]},
            {"text": "", "calls": [{"name": "read_file", "status": "success", "result": "b"}]},
            {"text": "", "calls": [{"name": "read_file", "status": "success", "result": "c"}]},
        ],
        react_iteration_count=4,
    )
    _insert_turn(
        observe_db,
        session_key="cli:1",
        user_msg="第三个：其他问题",
        tool_chain=[],
    )
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "trace1",
                        "inspect_turn_trace",
                        {"selector": "nth_user_question_in_window", "n": 2},
                    )
                ],
            ),
            LLMResponse(content="第二个问题使用了 read_file x3。", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(provider, observe_db)

    result = await reasoner.run_turn(
        msg=_msg("刚才第二个问题你用了哪些工具？"),
        session=cast(Any, _session()),
    )

    assert "inspect_turn_trace" in result.tools_used
    assert "search_docs" not in result.tools_used
    assert "fetch_doc_chunk" not in result.tools_used
    assert result.reply is not None
    assert "read_file x3" in result.reply
    tool_call = result.tool_chain[0]["calls"][0]
    assert tool_call["name"] == "inspect_turn_trace"
    assert tool_call["arguments"] == {"selector": "nth_user_question_in_window", "n": 2}

    second_call_messages = provider.calls[1]["messages"]
    tool_payload_text = next(
        msg["content"]
        for msg in second_call_messages
        if msg.get("role") == "tool"
    )
    assert '"real_tools": {"read_file": 3}' in tool_payload_text
    assert '"real_tools": {"search_docs"' not in tool_payload_text
