from __future__ import annotations

import asyncio
import json
import logging
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
    search_docs: _RecordingTool,
    fetch_doc_chunk: _RecordingTool,
    read_file: _RecordingTool | None = None,
) -> DefaultReasoner:
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(search_docs, risk="read-only")
    tools.register(fetch_doc_chunk, risk="read-only")
    tools.register(
        read_file or _RecordingTool("read_file"),
        always_on=True,
        risk="read-only",
    )
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


def test_doc_rag_evidence_complete_switches_next_call_to_final_only(caplog) -> None:
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
            LLMResponse(content="final answer with citation", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        search_docs=search_docs,
        fetch_doc_chunk=fetch_doc_chunk,
    )

    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            reasoner.run_turn(
                msg=_msg("根据项目文档回答agent runtime负责什么，并展开原文证据"),
                session=cast(Any, _session()),
            )
        )

    assert result.reply == "final answer with citation"
    assert len(search_docs.calls) == 1
    assert len(fetch_doc_chunk.calls) == 1
    assert result.tools_used == ["search_docs", "fetch_doc_chunk"]
    assert result.context_retry["turn_completion"]["action"] == "final_only"
    assert result.context_retry["turn_completion"]["reason"] == (
        "document_rag_evidence_complete"
    )
    assert result.context_retry["tool_boundary"]["ledger_summary"][
        "has_successful_retrieval"
    ] is True
    assert result.context_retry["tool_boundary"]["ledger_summary"][
        "has_citation_evidence"
    ] is True
    assert result.context_retry["turn_completion"]["metadata"]["react_boundary"] is True
    assert result.context_retry["turn_completion"]["metadata"]["batch_skip_count"] == 0
    assert provider.calls[-1]["tools"] == []
    assert "[react_boundary] final_only reason=document_rag_evidence_complete" in caplog.text
    assert (
        "[turn_completion] final_only reason=document_rag_evidence_complete"
    ) in caplog.text


def test_final_only_call_includes_evidence_contract_constraints() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 2,
                "hits": [
                    {
                        "chunk_id": "c1",
                        "citation": "my_md/doc.md > Agent Runtime",
                        "snippet": "Agent runtime 负责管理 agent 的一次运行过程。",
                    },
                    {
                        "chunk_id": "c2",
                        "citation": "my_md/doc.md > Tool Calling",
                        "snippet": "工具调用用于让 agent 访问外部能力。",
                    },
                ],
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
                    "content": "Agent runtime 负责管理 agent 的一次运行过程。",
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
            LLMResponse(
                content="",
                tool_calls=[ToolCall("f2", "fetch_doc_chunk", {"chunk_id": "c2"})],
            ),
            LLMResponse(content="final answer with constrained evidence", tool_calls=[]),
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

    final_messages = provider.calls[-1]["messages"]
    evidence_hints = [
        str(message.get("content", ""))
        for message in final_messages
        if "Evidence contract for this answer" in str(message.get("content", ""))
    ]
    assert evidence_hints
    hint = evidence_hints[-1]
    assert "Only successful fetch_doc_chunk results may be described" in hint
    assert "search_docs hits are retrieval summaries" in hint
    assert "Do not describe soft-stopped chunks as expanded original text" in hint
    assert "c2" in hint
    assert result.context_retry["evidence_contract"]["sufficiency"][
        "tool_stop_allowed"
    ] is True


def test_no_hit_retrieval_does_not_switch_to_final_only() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps({"ok": True, "hit_count": 0, "hits": []}),
    )
    fetch_doc_chunk = _RecordingTool("fetch_doc_chunk")
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("q1", "search_docs", {"query": "missing"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall("q2", "search_docs", {"query": "try again"})],
            ),
            LLMResponse(content="final without final-only", tool_calls=[]),
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

    assert result.reply == "final without final-only"
    assert len(search_docs.calls) == 1
    assert result.context_retry.get("turn_completion", {}).get("action") != "final_only"
    assert provider.calls[-1]["tools"] != []


def test_chunk_without_citation_does_not_switch_to_final_only() -> None:
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
                    "content": "text without citation",
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
            LLMResponse(
                content="",
                tool_calls=[ToolCall("f2", "fetch_doc_chunk", {"chunk_id": "c2"})],
            ),
            LLMResponse(content="final without citation final-only", tool_calls=[]),
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

    assert result.reply == "final without citation final-only"
    assert len(fetch_doc_chunk.calls) == 2
    assert result.context_retry.get("turn_completion", {}).get("action") != "final_only"
    assert provider.calls[-1]["tools"] != []


def test_proactive_boundary_does_not_final_only_without_citation() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 1,
                "hits": [{"chunk_id": "c1", "snippet": "Agent runtime text"}],
            }
        ),
    )
    fetch_doc_chunk = _RecordingTool("fetch_doc_chunk")
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
            LLMResponse(content="best effort answer", tool_calls=[]),
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

    assert result.reply == "best effort answer"
    assert len(provider.calls) == 3
    assert provider.calls[1]["tools"] != []
    assert result.context_retry.get("turn_completion", {}).get("action") != "final_only"


def test_explicit_local_source_request_does_not_switch_to_final_only() -> None:
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
    read_file = _RecordingTool("read_file", "source text")
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
            LLMResponse(
                content="",
                tool_calls=[ToolCall("f2", "fetch_doc_chunk", {"chunk_id": "c2"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "r1",
                        "read_file",
                        {"path": "agent/core/passive_turn.py", "limit": 20},
                    )
                ],
            ),
            LLMResponse(content="final with source", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        search_docs=search_docs,
        fetch_doc_chunk=fetch_doc_chunk,
        read_file=read_file,
    )

    result = asyncio.run(
        reasoner.run_turn(
            msg=_msg(
                "根据项目文档和源码回答，并展开原文证据，请读取 "
                "agent/core/passive_turn.py"
            ),
            session=cast(Any, _session()),
        )
    )

    assert result.reply == "final with source"
    assert len(fetch_doc_chunk.calls) == 1
    assert len(read_file.calls) == 1
    assert result.context_retry.get("turn_completion", {}).get("action") != "final_only"
    assert provider.calls[3]["tools"] != []


def test_final_only_ignores_tool_calls_returned_by_provider() -> None:
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
            LLMResponse(
                content="",
                tool_calls=[ToolCall("f2", "fetch_doc_chunk", {"chunk_id": "c2"})],
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall("bad", "search_docs", {"query": "again"})],
            ),
            LLMResponse(content="summary after ignored tool call", tool_calls=[]),
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
    assert result.context_retry["turn_completion"]["action"] == "final_only"
    assert result.reply is not None
    assert "final_only_tool_call" in result.reply
    assert provider.calls[2]["tools"] == []
    assert provider.calls[3]["tools"] == []


def test_same_batch_redundant_fetches_are_batch_skipped_and_final_only() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 3,
                "hits": [
                    {
                        "chunk_id": "c1",
                        "citation": "my_md/doc.md > Agent Runtime",
                        "snippet": "Agent runtime 负责管理 agent 的一次运行过程。",
                    },
                    {
                        "chunk_id": "c2",
                        "citation": "my_md/doc.md > Tool Calling",
                        "snippet": "工具调用用于让 agent 访问外部能力。",
                    },
                    {
                        "chunk_id": "c3",
                        "citation": "my_md/doc.md > System Overview",
                        "snippet": "系统全景描述 agent 运行边界。",
                    },
                ],
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
                    "content": "Agent runtime 负责管理 agent 的一次运行过程。",
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
                tool_calls=[
                    ToolCall("f1", "fetch_doc_chunk", {"chunk_id": "c1"}),
                    ToolCall("f2", "fetch_doc_chunk", {"chunk_id": "c2"}),
                    ToolCall("f3", "fetch_doc_chunk", {"chunk_id": "c3"}),
                ],
            ),
            LLMResponse(content="final answer with citation", tool_calls=[]),
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

    assert result.reply == "final answer with citation"
    assert result.tools_used == ["search_docs", "fetch_doc_chunk"]
    assert len(fetch_doc_chunk.calls) == 1
    assert provider.calls[-1]["tools"] == []
    assert result.context_retry["turn_completion"]["action"] == "final_only"
    assert result.context_retry["turn_completion"]["metadata"]["react_boundary"] is True
    statuses = [
        call.get("status")
        for group in result.tool_chain
        for call in group.get("calls", [])
        if call.get("name") == "fetch_doc_chunk"
    ]
    assert statuses == [
        "success",
        "batch_skipped_by_react_boundary",
        "batch_skipped_by_react_boundary",
    ]
    assert result.context_retry["turn_completion"]["metadata"]["batch_skip_count"] == 2
    assert result.context_retry["evidence_contract"]["metadata"]["fetched_text_count"] == 1
    assert result.context_retry["evidence_contract"]["metadata"][
        "soft_stopped_candidate_count"
    ] == 0
    assert result.context_retry["tool_boundary"]["ledger_summary"]["class_counts"][
        "evidence_expand"
    ] == 1

    final_messages = provider.calls[-1]["messages"]
    tool_results = {
        message.get("tool_call_id"): message
        for message in final_messages
        if message.get("role") == "tool"
    }
    assert {"f1", "f2", "f3"} <= set(tool_results)
    assert "react_boundary_batch_skip" in str(tool_results["f2"].get("content", ""))
    assert "react_boundary_batch_skip" in str(tool_results["f3"].get("content", ""))


def test_simple_doc_same_batch_fetch_is_skipped_after_search_evidence() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 1,
                "hits": [
                    {
                        "chunk_id": "c1",
                        "citation": "my_md/doc.md > Agent Runtime",
                        "snippet": "Agent runtime 负责管理 agent 的一次运行过程。",
                    }
                ],
            }
        ),
    )
    fetch_doc_chunk = _RecordingTool("fetch_doc_chunk")
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("q1", "search_docs", {"query": "agent runtime"}),
                    ToolCall("f1", "fetch_doc_chunk", {"chunk_id": "c1"}),
                ],
            ),
            LLMResponse(content="simple final with citation", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        search_docs=search_docs,
        fetch_doc_chunk=fetch_doc_chunk,
    )

    result = asyncio.run(
        reasoner.run_turn(
            msg=_msg("请从文档知识库中检索agent runtime负责什么？回答必须带文档引用"),
            session=cast(Any, _session()),
        )
    )

    assert result.reply == "simple final with citation"
    assert result.tools_used == ["search_docs"]
    assert len(fetch_doc_chunk.calls) == 0
    assert provider.calls[-1]["tools"] == []
    statuses = [
        call.get("status")
        for group in result.tool_chain
        for call in group.get("calls", [])
        if call.get("name") == "fetch_doc_chunk"
    ]
    assert statuses == ["batch_skipped_by_react_boundary"]


def test_simple_doc_retrieval_enters_final_only_without_fetch() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 1,
                "hits": [
                    {
                        "chunk_id": "c1",
                        "citation": "my_md/doc.md > Agent Runtime",
                        "snippet": "Agent runtime 负责管理 agent 的一次运行过程。",
                    }
                ],
            }
        ),
    )
    fetch_doc_chunk = _RecordingTool("fetch_doc_chunk")
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("q1", "search_docs", {"query": "agent runtime"})],
            ),
            LLMResponse(content="simple final with citation", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        search_docs=search_docs,
        fetch_doc_chunk=fetch_doc_chunk,
    )

    result = asyncio.run(
        reasoner.run_turn(
            msg=_msg("请从文档知识库中检索agent runtime负责什么？回答必须带文档引用"),
            session=cast(Any, _session()),
        )
    )

    assert result.reply == "simple final with citation"
    assert result.tools_used == ["search_docs"]
    assert len(fetch_doc_chunk.calls) == 0
    assert provider.calls[-1]["tools"] == []
    assert result.context_retry["turn_completion"]["reason"] == (
        "document_rag_retrieval_complete"
    )


def test_proactive_final_only_includes_evidence_contract_hint() -> None:
    search_docs = _RecordingTool(
        "search_docs",
        json.dumps(
            {
                "ok": True,
                "hit_count": 1,
                "hits": [
                    {
                        "chunk_id": "c1",
                        "citation": "my_md/doc.md > Agent Runtime",
                        "snippet": "Agent runtime 负责管理 agent 的一次运行过程。",
                    }
                ],
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
                    "content": "Agent runtime 负责管理 agent 的一次运行过程。",
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
            LLMResponse(content="final answer", tool_calls=[]),
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

    final_messages = provider.calls[-1]["messages"]
    hint_text = "\n".join(str(message.get("content", "")) for message in final_messages)

    assert "Evidence contract for this answer" in hint_text
    assert "Only successful fetch_doc_chunk results may be described" in hint_text
    assert result.context_retry["evidence_contract"]["sufficiency"][
        "tool_stop_allowed"
    ] is True
