from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any, cast

import pytest

import agent.model_runtime.query_compaction as query_compaction_module
from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.looping.ports import LLMConfig
from agent.model_runtime.execution_history import active_shell_execution_origins
from agent.model_runtime.types import LLMResponse, ModelUsage, ToolCall
from agent.provider import ContextLengthError
from agent.tool_runtime import append_tool_result
from agent.model_runtime.query_compaction import (
    COMPACTION_TOOL_NAME,
    ContextCompactionError,
    QueryCompactor,
    ReactCompaction,
    build_compaction_messages,
    parse_react_compaction,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from session.manager import Session, SessionManager


def _batch(index: int, *, model_state: bool = False) -> list[dict[str, Any]]:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": "probe",
                    "arguments": json.dumps({"index": index}),
                },
            }
        ],
    }
    if model_state:
        assistant["model_state"] = {"opaque": index}
    return [
        assistant,
        {
            "role": "tool",
            "tool_call_id": f"call-{index}",
            "content": f"result-{index}",
        },
    ]


def _record_batch(
    compactor: QueryCompactor,
    messages: list[dict],
    index: int,
    *,
    model_state: bool = False,
) -> None:
    start = len(messages)
    messages.extend(_batch(index, model_state=model_state))
    compactor.record_completed_batch(messages, batch_start=start)


def _record_execution_batch(
    compactor: QueryCompactor,
    messages: list[dict],
    index: int,
    *,
    name: str,
    arguments: dict[str, object],
    result: dict[str, object],
) -> None:
    call_id = f"exec-{index}"
    start = len(messages)
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
        ]
    )
    append_tool_result(
        messages,
        tool_call_id=call_id,
        content=json.dumps(result),
        tool_name=name,
        execution_status="success",
    )
    compactor.record_completed_batch(messages, batch_start=start)


class _ControlledProvider:
    def __init__(
        self,
        estimates: list[int],
        *,
        summary: str | BaseException = "## Goal\n继续完成任务",
        appended_tokens_per_message: int = 1,
    ) -> None:
        self.context_window = 64_000
        self.compaction_trigger_tokens = 47_360
        self.hard_input_tokens = 57_600
        self.estimates = list(estimates)
        self.summary = summary
        self.appended_tokens_per_message = appended_tokens_per_message
        self.summary_calls: list[dict[str, Any]] = []

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        if not self.estimates:
            raise AssertionError("缺少预设的完整上下文估算")
        return self.estimates.pop(0)

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        return len(messages) * self.appended_tokens_per_message

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.summary_calls.append(kwargs)
        if isinstance(self.summary, BaseException):
            raise self.summary
        return LLMResponse(content=self.summary)


class _SummarySequenceProvider(_ControlledProvider):
    def __init__(
        self,
        estimates: list[int],
        summaries: list[str | LLMResponse],
    ) -> None:
        super().__init__(estimates)
        self.summaries = list(summaries)

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.summary_calls.append(kwargs)
        if not self.summaries:
            raise AssertionError("缺少预设的摘要响应")
        response = self.summaries.pop(0)
        if isinstance(response, LLMResponse):
            return response
        content = response
        return LLMResponse(
            content=content,
            thinking="仅有推理内容" if not content else None,
            usage=ModelUsage(input_tokens=1),
        )


def _compactor(
    provider: object,
    base_messages: list[dict],
) -> QueryCompactor:
    return QueryCompactor(
        provider=cast(Any, provider),
        model="test-model",
        base_messages=base_messages,
        scope_id="turn:test",
    )


@pytest.mark.asyncio
async def test_compaction_triggers_at_exact_74_percent_after_closed_batches() -> None:
    base = [{"role": "user", "content": "完成长任务"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_359])
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2, model_state=True)

    before = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[{"type": "function"}],
    )

    assert before.compacted is False
    assert provider.summary_calls == []

    provider.estimates.extend([47_360, 2_000])
    at_boundary = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[{"type": "function"}],
    )

    assert at_boundary.compacted is True
    assert at_boundary.estimated_tokens == 2_000
    assert compactor.compaction is not None
    assert compactor.compaction.compacted_tool_groups == 1
    assert compactor.compaction.estimated_tokens_before == 47_360
    assert (
        sum(
            1
            for message in messages
            for call in cast(list[dict[str, Any]], message.get("tool_calls", []))
            if call["function"]["name"] == COMPACTION_TOOL_NAME
        )
        == 1
    )
    assert "model_state" not in messages[-2]


@pytest.mark.asyncio
async def test_compaction_summary_disables_thinking() -> None:
    base = [{"role": "user", "content": "完成长任务"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_360, 2_000])
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)

    await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert provider.summary_calls[0]["disable_thinking"] is True


@pytest.mark.asyncio
async def test_compaction_retries_thinking_only_summary_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = [{"role": "user", "content": "完成长任务"}]
    messages = deepcopy(base)
    provider = _SummarySequenceProvider(
        [47_360, 2_000],
        ["", "", "", "## Goal\n继续完成任务"],
    )
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(query_compaction_module.asyncio, "sleep", record_delay)
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)

    prepared = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert prepared.compacted is True
    assert prepared.summary_usage is not None
    assert prepared.summary_usage.input_tokens == 4
    assert prepared.summary_usage.request_count == 4
    assert len(provider.summary_calls) == 4
    assert delays == [2.0, 4.0, 8.0]
    assert all(call["disable_thinking"] is True for call in provider.summary_calls)


@pytest.mark.asyncio
async def test_compaction_retries_success_response_with_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = [{"role": "user", "content": "完成长任务"}]
    messages = deepcopy(base)
    provider = _SummarySequenceProvider(
        [47_360, 2_000],
        [
            LLMResponse(
                content="不应接受带工具调用的摘要",
                tool_calls=[ToolCall("summary-tool", "probe", {})],
            ),
            "## Goal\n继续完成任务",
        ],
    )
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(query_compaction_module.asyncio, "sleep", record_delay)
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)

    prepared = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert prepared.compacted is True
    assert len(provider.summary_calls) == 2
    assert delays == [2.0]


@pytest.mark.asyncio
async def test_compaction_cancellation_during_backoff_preserves_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = [{"role": "user", "content": "完成长任务"}]
    messages = deepcopy(base)
    provider = _SummarySequenceProvider([47_360], [""])
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)
    before = deepcopy(messages)

    async def cancel_delay(_delay: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(query_compaction_module.asyncio, "sleep", cancel_delay)

    with pytest.raises(asyncio.CancelledError):
        await compactor.prepare(
            messages,
            pending_start=len(messages),
            tools=[],
        )

    assert messages == before
    assert len(provider.summary_calls) == 1
    assert compactor.compaction is None


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_batches", [0, 1])
async def test_compaction_never_splits_current_query_or_only_recent_batch(
    completed_batches: int,
) -> None:
    base = [{"role": "user", "content": "完成不可切分的长步骤"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_360])
    compactor = _compactor(provider, base)
    for index in range(completed_batches):
        _record_batch(compactor, messages, index + 1)
    before = deepcopy(messages)

    prepared = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert prepared.compacted is False
    assert messages == before
    assert provider.summary_calls == []
    assert compactor.has_compactable_prefix is False


@pytest.mark.asyncio
async def test_compaction_pins_live_execution_until_terminal_result() -> None:
    base = [{"role": "user", "content": "完成长时间训练"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_360, 47_360, 2_000])
    compactor = _compactor(provider, base)
    _record_execution_batch(
        compactor,
        messages,
        1,
        name="shell",
        arguments={"command": "python train.py"},
        result={"process_status": "running", "execution_id": 4201},
    )
    _record_batch(compactor, messages, 2)
    _record_batch(compactor, messages, 3)
    before = deepcopy(messages)

    pinned = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert pinned.compacted is False
    assert messages == before
    assert provider.summary_calls == []

    _record_execution_batch(
        compactor,
        messages,
        4,
        name="write_stdin",
        arguments={"execution_id": 4201},
        result={"process_status": "succeeded", "exit_code": 0},
    )
    completed = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert completed.compacted is True
    assert compactor.compaction is not None
    assert compactor.compaction.compacted_tool_groups == 1


@pytest.mark.parametrize("transport_status", ["blocked", "denied", "skipped", "error"])
def test_active_execution_history_rejects_unsuccessful_transport(
    transport_status: str,
) -> None:
    messages = _batch(1)
    messages[0]["tool_calls"][0]["function"] = {
        "name": "shell",
        "arguments": json.dumps({"command": "sleep 30"}),
    }
    append_tool_result(
        messages,
        tool_call_id="call-1",
        content=json.dumps(
            {"process_status": "running", "execution_id": 4201}
        ),
        tool_name="shell",
        execution_status=transport_status,
    )

    assert active_shell_execution_origins(messages) == {}


def test_active_execution_history_rejects_malformed_transport_marker() -> None:
    messages = _batch(1)
    messages[0]["tool_calls"][0]["function"] = {
        "name": "shell",
        "arguments": json.dumps({"command": "sleep 30"}),
    }
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": (
                '<tool_execution transport_status="success" extra="x" />\n'
                '{"process_status":"running","execution_id":4201}'
            ),
        }
    )

    assert active_shell_execution_origins(messages) == {}


def test_active_execution_history_rejects_raw_json_without_transport() -> None:
    messages = _batch(1)
    messages[0]["tool_calls"][0]["function"] = {
        "name": "shell",
        "arguments": json.dumps({"command": "sleep 30"}),
    }
    messages[1]["content"] = json.dumps(
        {"process_status": "running", "execution_id": 4201}
    )

    assert active_shell_execution_origins(messages) == {}


@pytest.mark.asyncio
async def test_token_meter_uses_exact_provider_usage_plus_appended_delta() -> None:
    base = [{"role": "user", "content": "继续"}]
    messages = deepcopy(base)
    provider = _ControlledProvider(
        [10],
        appended_tokens_per_message=7,
    )
    compactor = _compactor(provider, base)
    tools = [{"type": "function", "function": {"name": "probe"}}]

    initial = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=tools,
    )
    compactor.record_response(
        message_count=len(messages),
        tools=tools,
        usage=ModelUsage(input_tokens=100),
    )
    messages.append({"role": "assistant", "content": "delta"})
    next_request = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=tools,
    )

    assert initial.estimate_quality == "approximate"
    assert next_request.estimate_quality == "exact_plus_delta"
    assert next_request.estimated_tokens == 107


@pytest.mark.asyncio
async def test_repeated_compaction_replaces_pair_and_updates_summary() -> None:
    base = [{"role": "user", "content": "完成三阶段任务"}]
    messages = deepcopy(base)
    provider = _ControlledProvider(
        [47_360, 2_000, 47_360, 2_100],
        summary="## Goal\n第一份摘要",
    )
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)

    first = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )
    provider.summary = "## Goal\n第二份摘要"
    _record_batch(compactor, messages, 3)
    second = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert first.compacted is True
    assert second.compacted is True
    assert compactor.compaction is not None
    assert compactor.compaction.generation == 2
    assert compactor.compaction.compacted_tool_groups == 2
    compact_calls = [
        call
        for message in messages
        for call in cast(list[dict[str, Any]], message.get("tool_calls", []))
        if call["function"]["name"] == COMPACTION_TOOL_NAME
    ]
    assert len(compact_calls) == 1
    assert "第二份摘要" in cast(str, messages[2]["content"])
    assert "Previous compaction summary" in cast(
        str,
        provider.summary_calls[1]["messages"][0]["content"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary", "error"),
    [
        (RuntimeError("summary unavailable"), RuntimeError),
        ("", ContextCompactionError),
    ],
)
async def test_summary_failure_does_not_mutate_active_context(
    summary: str | BaseException,
    error: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def skip_delay(_delay: float) -> None:
        return None

    monkeypatch.setattr(query_compaction_module.asyncio, "sleep", skip_delay)
    base = [{"role": "user", "content": "长任务"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_360], summary=summary)
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)
    before = deepcopy(messages)

    with pytest.raises(error):
        await compactor.prepare(
            messages,
            pending_start=len(messages),
            tools=[],
        )

    assert messages == before
    assert compactor.compaction is None
    assert len(provider.summary_calls) == (4 if summary == "" else 1)


@pytest.mark.asyncio
async def test_insufficient_compaction_fails_without_committing_projection() -> None:
    base = [{"role": "user", "content": "长任务"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_360, 57_600])
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)
    before = deepcopy(messages)

    with pytest.raises(ContextCompactionError, match="insufficient"):
        await compactor.prepare(
            messages,
            pending_start=len(messages),
            tools=[],
        )

    assert messages == before
    assert compactor.compaction is None


@pytest.mark.asyncio
async def test_compaction_may_remain_above_soft_limit_but_below_hard_limit() -> None:
    base = [{"role": "user", "content": "长任务"}]
    messages = deepcopy(base)
    provider = _ControlledProvider([47_360, 50_000])
    compactor = _compactor(provider, base)
    _record_batch(compactor, messages, 1)
    _record_batch(compactor, messages, 2)

    prepared = await compactor.prepare(
        messages,
        pending_start=len(messages),
        tools=[],
    )

    assert prepared.compacted is True
    assert prepared.estimated_tokens == 50_000


def test_compaction_pair_is_internal_protocol_not_registered_tool() -> None:
    compaction = ReactCompaction(
        summary="## Goal\n完成任务",
        compacted_tool_groups=2,
        generation=1,
        trigger="soft_limit",
        context_window=64_000,
        soft_limit_tokens=47_360,
        estimated_tokens_before=47_360,
        estimated_tokens_after=12_000,
    )

    pair = build_compaction_messages(compaction, call_id="cmp_test")

    assert pair[0]["tool_calls"][0]["id"] == pair[1]["tool_call_id"]
    assert pair[0]["tool_calls"][0]["function"]["name"] == COMPACTION_TOOL_NAME
    assert (
        parse_react_compaction(
            compaction.to_payload(),
            source="test",
        )
        == compaction
    )


class _ProbeTool(Tool):
    name = "probe"
    description = "执行一步"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class _LoopProvider:
    context_window = 1_000
    compaction_trigger_tokens = 740
    hard_input_tokens = 900

    def __init__(self, *, overflow_once: bool = False) -> None:
        first_usage = (
            None if overflow_once else ModelUsage(input_tokens=10, output_tokens=2)
        )
        second_usage = (
            None if overflow_once else ModelUsage(input_tokens=730, output_tokens=2)
        )
        final_usage = (
            None if overflow_once else ModelUsage(input_tokens=10, output_tokens=2)
        )
        self._responses = [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("call-1", "probe", {})],
                usage=first_usage,
            ),
            LLMResponse(
                content="",
                tool_calls=[ToolCall("call-2", "probe", {})],
                usage=second_usage,
            ),
            LLMResponse(
                content="完成",
                tool_calls=[],
                usage=final_usage,
            ),
        ]
        self.overflow_once = overflow_once
        self.overflow_raised = False
        self.real_calls: list[dict[str, Any]] = []
        self.summary_calls: list[dict[str, Any]] = []

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        if _contains_compaction_pair(messages):
            return 100
        completed = sum(1 for message in messages if message.get("role") == "tool")
        if self.overflow_once:
            return 100
        return 740 if completed >= 2 else 100

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        return len(messages) * 10

    async def chat(self, **kwargs: Any) -> LLMResponse:
        if not kwargs["tools"]:
            self.summary_calls.append(deepcopy(kwargs))
            return LLMResponse(
                content="## Goal\n完成 probe 流程",
                usage=ModelUsage(input_tokens=11, output_tokens=3),
            )
        self.real_calls.append(deepcopy(kwargs))
        if (
            self.overflow_once
            and not self.overflow_raised
            and sum(
                1 for message in kwargs["messages"] if message.get("role") == "tool"
            )
            >= 2
            and not _contains_compaction_pair(kwargs["messages"])
        ):
            self.overflow_raised = True
            raise ContextLengthError("context_length_exceeded")
        return self._responses.pop(0)


def _contains_compaction_pair(messages: list[dict]) -> bool:
    return any(
        call.get("function", {}).get("name") == COMPACTION_TOOL_NAME
        for message in messages
        for call in message.get("tool_calls", [])
    )


def _reasoner(provider: object) -> tuple[DefaultReasoner, ToolRegistry]:
    tools = ToolRegistry()
    tools.register(_ProbeTool(), always_on=True)
    return (
        DefaultReasoner(
            llm=cast(
                Any,
                LLMServices(
                    provider=cast(Any, provider),
                    light_provider=cast(Any, provider),
                ),
            ),
            llm_config=LLMConfig(
                model="test-model",
                max_iterations=6,
                max_tokens=0,
            ),
            tools=tools,
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            memory_window=40,
        ),
        tools,
    )


def test_reasoner_compacts_before_provider_without_tool_side_effects() -> None:
    provider = _LoopProvider()
    reasoner, tools = _reasoner(provider)

    result = asyncio.run(reasoner.run([{"role": "user", "content": "执行两步后回答"}]))

    assert result.reply == "完成"
    assert len(result.metadata["tool_chain"]) == 2
    assert result.metadata["tools_used"] == ["probe", "probe"]
    assert result.metadata["react_compaction"]["trigger"] == "soft_limit"
    assert COMPACTION_TOOL_NAME not in tools.get_registered_names()
    assert all(
        call["name"] != COMPACTION_TOOL_NAME
        for group in result.metadata["tool_chain"]
        for call in group["calls"]
    )
    assert len(provider.summary_calls) == 1
    assert provider.summary_calls[0]["tools"] == []
    assert "cache_namespace" not in provider.summary_calls[0]
    assert _contains_compaction_pair(provider.real_calls[-1]["messages"])
    assert result.metadata["react_stats"]["model_usage"]["request_count"] == 4
    assert result.metadata["react_stats"]["model_usage"]["input_tokens"] == 761
    assert result.metadata["react_stats"]["model_usage"]["output_tokens"] == 9


def test_reasoner_forces_one_compaction_after_provider_overflow() -> None:
    provider = _LoopProvider(overflow_once=True)
    reasoner, _ = _reasoner(provider)

    result = asyncio.run(reasoner.run([{"role": "user", "content": "执行两步后回答"}]))

    assert result.reply == "完成"
    assert provider.overflow_raised is True
    assert len(provider.summary_calls) == 1
    assert result.metadata["react_compaction"]["trigger"] == "context_overflow"
    assert _contains_compaction_pair(provider.real_calls[-1]["messages"])
    assert result.metadata["react_stats"]["model_usage"]["request_count"] == 4


class _ImmediateOverflowProvider(_LoopProvider):
    def __init__(self) -> None:
        super().__init__()

    async def chat(self, **kwargs: Any) -> LLMResponse:
        raise ContextLengthError("initial context too long")


def test_initial_overflow_without_closed_batch_preserves_provider_error() -> None:
    reasoner, _ = _reasoner(_ImmediateOverflowProvider())

    with pytest.raises(ContextLengthError, match="initial context too long"):
        asyncio.run(reasoner.run([{"role": "user", "content": "过长输入"}]))


def _tool_group(index: int, *, model_state: bool = False) -> dict[str, object]:
    group: dict[str, object] = {
        "text": "",
        "calls": [
            {
                "call_id": f"call-{index}",
                "name": "probe",
                "arguments": {"index": index},
                "result": f"result-{index}",
            }
        ],
    }
    if model_state:
        group["model_state"] = {
            "schema_version": 1,
            "runtime_id": "main",
            "transport": "responses",
            "model": "test-model",
            "items": [{"type": "reasoning", "id": f"rs-{index}"}],
        }
    return group


def test_sessiondb_reloads_compaction_projection_and_keeps_full_trace(
    tmp_path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:compaction")
    session.add_message("user", "旧问题")
    session.add_message("assistant", "旧回答")
    manager.save(session)
    existing_rows = manager._store._conn.execute(
        """
        SELECT id, seq, role, content, tool_chain, extra
        FROM messages
        WHERE session_key = ?
        ORDER BY seq
        """,
        (session.key,),
    ).fetchall()
    statements: list[str] = []
    manager._store._conn.set_trace_callback(statements.append)

    session.add_message("user", "完成长任务")
    session.add_message(
        "assistant",
        "最终完成",
        tool_chain=[
            _tool_group(1),
            _tool_group(2),
            _tool_group(3, model_state=True),
        ],
        react_compaction=ReactCompaction(
            summary="## Goal\n完成长任务\n## Progress\n前两步完成",
            compacted_tool_groups=2,
            generation=1,
            trigger="soft_limit",
            context_window=64_000,
            soft_limit_tokens=47_360,
            estimated_tokens_before=47_500,
            estimated_tokens_after=12_000,
        ).to_payload(),
    )
    manager.save(session)

    current_rows = manager._store._conn.execute(
        """
        SELECT id, seq, role, content, tool_chain, extra
        FROM messages
        WHERE session_key = ?
        ORDER BY seq
        """,
        (session.key,),
    ).fetchall()
    assert current_rows[: len(existing_rows)] == existing_rows
    normalized = [" ".join(statement.upper().split()) for statement in statements]
    assert not any(
        statement.startswith(("UPDATE MESSAGES ", "DELETE FROM MESSAGES "))
        for statement in normalized
    )
    manager._store._conn.set_trace_callback(None)
    manager.close()

    reloaded = SessionManager(tmp_path)
    restored = reloaded.get_existing("cli:compaction")
    assistant = restored.messages[-1]
    assert len(cast(list[object], assistant["tool_chain"])) == 3
    compaction = cast(dict[str, object], assistant["react_compaction"])
    assert compaction["compacted_tool_groups"] == 2

    history = restored.get_history()
    compact_calls = [
        call
        for message in history
        for call in cast(list[dict], message.get("tool_calls", []))
        if call["function"]["name"] == COMPACTION_TOOL_NAME
    ]
    assert len(compact_calls) == 1
    assert not any(message.get("tool_call_id") == "call-1" for message in history)
    assert not any(message.get("tool_call_id") == "call-2" for message in history)
    assert any(message.get("tool_call_id") == "call-3" for message in history)
    retained_assistant = next(
        message
        for message in history
        if any(
            call["id"] == "call-3"
            for call in cast(list[dict], message.get("tool_calls", []))
        )
    )
    assert "model_state" not in retained_assistant
    assert history[-1] == {"role": "assistant", "content": "最终完成"}
    reloaded.close()


def test_sessiondb_rejects_corrupt_compaction_metadata(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:corrupt-compaction")
    manager._store.insert_message(
        session.key,
        role="assistant",
        content="损坏数据",
        ts="2026-07-31T00:00:00+00:00",
        seq=0,
        extra={
            "react_compaction": {
                "schema_version": 99,
                "summary": "invalid",
            }
        },
    )
    manager.close()

    reloaded = SessionManager(tmp_path)
    with pytest.raises(ValueError, match="schema_version"):
        reloaded.get_existing("cli:corrupt-compaction")
    reloaded.close()


def test_replay_rejects_compaction_cut_beyond_full_tool_chain() -> None:
    session = Session("cli:invalid-cut")
    session.add_message("user", "问题")
    session.add_message(
        "assistant",
        "回答",
        tool_chain=[_tool_group(1)],
        react_compaction=ReactCompaction(
            summary="## Goal\n完成",
            compacted_tool_groups=2,
            generation=1,
            trigger="soft_limit",
            context_window=64_000,
            soft_limit_tokens=47_360,
            estimated_tokens_before=48_000,
            estimated_tokens_after=10_000,
        ).to_payload(),
    )

    with pytest.raises(ValueError, match="超过 tool_chain 长度"):
        session.get_history()
