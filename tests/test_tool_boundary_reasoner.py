from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.core.types import ContextRenderResult, ContextRequest
from agent.looping.ports import LLMConfig
from agent.provider import LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool


class _RecordingTool(Tool):
    def __init__(self, name: str, result: str | None = None) -> None:
        self._name = name
        self._result = result or json.dumps({"ok": True})
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self._result


class _Provider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


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


def _make_reasoner(
    provider: _Provider,
    *,
    search_docs: _RecordingTool | None = None,
    fetch_doc_chunk: _RecordingTool | None = None,
) -> DefaultReasoner:
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(search_docs or _RecordingTool("search_docs"), risk="read-only")
    tools.register(
        fetch_doc_chunk or _RecordingTool("fetch_doc_chunk"),
        risk="read-only",
    )
    tools.register(_RecordingTool("read_file"), always_on=True, risk="read-only")
    tools.register(_RecordingTool("shell"), always_on=True, risk="read-only")
    tools.register(_RecordingTool("list_dir"), always_on=True, risk="read-only")

    def _render(request: ContextRequest, **_kwargs: object) -> ContextRenderResult:
        return ContextRenderResult(
            system_prompt="",
            turn_injection_context={"turn_injection": request.turn_injection_prompt or ""},
            messages=[{"role": "user", "content": request.current_message}],
            debug_breakdown=[],
        )

    return DefaultReasoner(
        llm=cast(Any, LLMServices(provider=provider, light_provider=provider)),
        llm_config=LLMConfig(model="m", max_iterations=6, max_tokens=256),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
        memory_window=10,
        context=cast(Any, SimpleNamespace(render=_render)),
        session_manager=cast(Any, SimpleNamespace(save_async=lambda *_args, **_kw: None)),
    )


def test_redundant_visible_tool_search_soft_stop_does_not_execute_tool_search() -> None:
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "s1",
                        "tool_search",
                        {"query": "select:search_docs,fetch_doc_chunk"},
                    )
                ],
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(provider)

    result = asyncio.run(
        reasoner.run_turn(
            msg=_msg("根据项目文档回答agent runtime负责什么，并展开原文证据"),
            session=cast(Any, _session()),
        )
    )

    tool_messages = [m for m in provider.calls[1]["messages"] if m.get("role") == "tool"]
    payload = json.loads(tool_messages[-1]["content"])
    assert payload["error_code"] == "tool_boundary_soft_stop"
    assert result.tools_used == []
    assert result.tool_chain[0]["calls"][0]["status"] == "soft_stopped_by_tool_boundary"
    assert result.context_retry["tool_boundary"]["decisions"][0]["reason"] == (
        "redundant_visible_tool_search"
    )


def test_repeated_fetch_round_prevented_after_citation_evidence() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 1,
                "hits": [{"chunk_id": "c1", "citation": "my_md/doc.md > Agent Runtime"}],
            }
        ),
    )
    fetch_doc_chunk = _RecordingTool(
        "fetch_doc_chunk",
        json.dumps(
            {
                "ok": True,
                "chunk": {
                    "chunk_id": "c1",
                    "citation": "my_md/doc.md > Agent Runtime",
                    "text": "Agent runtime 负责管理 agent 的一次运行过程。",
                },
            }
        ),
    )
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("q1", "search_docs", {"query": "agent runtime"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall("f1", "fetch_doc_chunk", {"chunk_id": "c1"})],
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        search_docs=search_docs,
        fetch_doc_chunk=fetch_doc_chunk,
    )

    result = asyncio.run(
        reasoner.run_turn(
            msg=_msg("根据项目文档回答agent runtime负责什么，并展开原文证据"),
            session=cast(Any, _session()),
        )
    )

    assert len(search_docs.calls) == 1
    assert len(fetch_doc_chunk.calls) == 1
    assert result.tools_used == ["search_docs", "fetch_doc_chunk"]
    assert result.context_retry["turn_completion"]["action"] == "final_only"
    assert result.context_retry["turn_completion"]["metadata"]["react_boundary"] is True
    assert result.context_retry["tool_boundary"]["ledger_summary"][
        "has_citation_evidence"
    ] is True
