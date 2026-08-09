import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.core.passive_turn import DefaultReasoner
from agent.control.ports import TurnUserInput
from agent.core.runtime_support import LLMServices, SessionLike, ToolDiscoveryState
from agent.lifecycle.types import AfterStepCtx
from agent.looping.ports import LLMConfig
from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextPayloadSegments,
)
from agent.provider import ContextLengthError, LLMResponse, ToolCall
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from bus.event_bus import EventBus
from bus.events_lifecycle import ToolCallCompleted, ToolCallStarted
from session.compaction_runtime import CompactionProjection
from session.manager import Session
from session.store import CompactionHead

_TEST_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS = 1


class _ProviderContextBudget:
    context_window = 1_000_000

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        return max(
            1,
            len(json.dumps([messages, tools], ensure_ascii=False)) // 3,
        )

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        if not messages:
            return 0
        return max(1, len(json.dumps(messages, ensure_ascii=False)) // 3)


class ContextPressureStopModule:
    slot = "context_pressure.stop"
    requires = ("after_step.copy_input", "step:ctx")
    produces = (
        "step:early_stop_reason",
        "step:telemetry:context_pressure_tokens",
        "step:telemetry:context_pressure_threshold",
    )

    async def run(self, frame: object) -> object:
        raw_slots = getattr(frame, "slots", None)
        if not isinstance(raw_slots, dict):
            return frame
        slots = cast(dict[str, object], raw_slots)
        ctx = slots.get("step:ctx")
        if not isinstance(ctx, AfterStepCtx) or not ctx.has_more:
            return frame
        if ctx.context_tokens_estimate <= _TEST_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS:
            return frame
        slots["step:early_stop_reason"] = "context_pressure"
        slots["step:telemetry:context_pressure_tokens"] = ctx.context_tokens_estimate
        slots["step:telemetry:context_pressure_threshold"] = (
            _TEST_CONTEXT_PRESSURE_STOP_THRESHOLD_TOKENS
        )
        return frame


class _DummyTool(Tool):
    def __init__(self, name: str = "dummy") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        properties: dict[str, Any] = {"x": {"type": "integer"}}
        if self._name == "message_push":
            properties["message"] = {"type": "string"}
        return {"type": "object", "properties": properties, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"{self._name}-ok"


class _InflateTool(Tool):
    name = "inflate_probe"
    description = "inflate_probe"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        return f"payload-{kwargs.get('value', '')}-" + ("x" * 2400)


class _Provider(_ProviderContextBudget):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


class _TimeoutProvider(_ProviderContextBudget):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        raise asyncio.TimeoutError


class _UnknownWindowOverflowProvider(_ProviderContextBudget):
    context_window = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        raise ContextLengthError("provider context overflow")


class _MandatoryCompactionRuntime:
    """Provide the narrow projection port required by reasoner test turns."""

    @staticmethod
    def _history(session: SessionLike) -> list[dict[str, Any]]:
        return [dict(message) for message in session.get_history(max_messages=500)]

    async def projection(
        self,
        session: SessionLike,
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection:
        history = self._history(session)
        units: tuple[CommittedContextUnit, ...] = ()
        if history:
            ids = tuple(f"test-message-{index}" for index in range(len(history)))
            units = (
                CommittedContextUnit(
                    source_from_seq=0,
                    consolidated_through_seq=len(history) - 1,
                    source_message_ids=ids,
                    messages=tuple(history),
                    message_refs=tuple((message_id, index) for index, message_id in enumerate(ids)),
                ),
            )
        return CompactionProjection(
            segments=ContextPayloadSegments(
                prefix=tuple(prefix),
                committed_units=units,
                current_anchor=tuple(current_anchor),
                pending=tuple(pending),
            ),
            active=None,
            head=CompactionHead(
                session_key=str(getattr(session, "key", "test-session")),
                parent_generation=0,
                next_generation=1,
            ),
        )

    async def recover_pending(self, session: object) -> None:
        return None

    async def commit_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("test compaction gate unexpectedly attempted a commit")


def _build_reasoner(**kwargs: Any) -> DefaultReasoner:
    """Construct a reasoner with the mandatory session compaction runtime."""

    return DefaultReasoner(
        compaction_runtime=_MandatoryCompactionRuntime(),
        **kwargs,
    )


async def _run_with_compaction_gate(
    reasoner: DefaultReasoner,
    initial_messages: list[dict[str, Any]],
    **kwargs: Any,
):
    """Run the legacy-shaped fixture through the required compaction gate."""

    payload = [dict(message) for message in initial_messages]
    if not payload or payload[0].get("role") != "system":
        payload.insert(0, {"role": "system", "content": "test context"})
    history = [dict(message) for message in payload[1:]]
    session = Session(
        key="test:reasoner",
        created_at=datetime(2026, 8, 8, tzinfo=UTC),
        messages=history,
        last_consolidated=0,
    )
    runtime = reasoner._compaction_runtime
    if runtime is None:
        raise AssertionError("reasoner test fixture must install compaction runtime")
    projection = await runtime.projection(
        session,
        prefix=[],
        current_anchor=[],
        pending=[],
    )
    state = reasoner._build_compaction_state(
        session=session,
        projection=projection,
        initial_messages=payload,
        history_count=len(history),
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="reasoner",
    )
    return await reasoner.run(payload, compaction_state=state, **kwargs)


def test_default_reasoner_runs_tool_loop_and_returns_reasoner_result():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "dummy", {})],
                cache_prompt_tokens=100,
                cache_hit_tokens=40,
            ),
            LLMResponse(
                content="final",
                tool_calls=[],
                cache_prompt_tokens=120,
                cache_hit_tokens=60,
            ),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "final"
    assert result.metadata["tools_used"] == ["dummy"]
    assert result.invocations[0].name == "dummy"
    assert result.metadata["visible_names"] is None
    react_stats = result.metadata["react_stats"]
    assert react_stats["iteration_count"] == 2
    assert react_stats["turn_input_sum_tokens"] >= react_stats["turn_input_peak_tokens"]
    assert (
        react_stats["final_call_input_tokens"] == react_stats["turn_input_peak_tokens"]
    )
    assert react_stats["cache_prompt_tokens"] == 220
    assert react_stats["cache_hit_tokens"] == 100
    first_messages = provider.calls[0]["messages"]
    assert not any(
        "未加载工具目录" in str(m.get("content", "")) for m in first_messages
    )


def test_unknown_context_window_leaves_provider_overflow_unmodified() -> None:
    provider = _UnknownWindowOverflowProvider()
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    with pytest.raises(ContextLengthError, match="provider context overflow"):
        asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "overflow"}],
            )
        )

    assert len(provider.calls) == 1


def test_default_reasoner_replays_interrupted_attempt_before_current_input():
    provider = _Provider([LLMResponse(content="final after u2", tool_calls=[])])
    timestamp = datetime.now(UTC)
    inputs = (
        TurnUserInput("u1", 0, "first request", (), {}, timestamp),
        TurnUserInput("u2", 1, "continue with node status", (), {}, timestamp),
    )

    class _Source:
        async def lock(self) -> None:
            return None

        def used_inputs(self) -> tuple[TurnUserInput, ...]:
            return inputs

    replay = [
        {"role": "user", "content": "first request"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "node=ready"},
        {"role": "assistant", "content": "[execution attempt interrupted]"},
    ]
    prior_tool_chain = [
        {
            "text": "",
            "calls": [
                {
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": {},
                    "result": "node=ready",
                }
            ],
        }
    ]
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=ToolRegistry(),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "test context"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                ),
            ),
        ),
    )
    session = SimpleNamespace(
        key="mobile:one",
        created_at=timestamp,
        messages=[{"role": "user", "content": "old canonical"}],
        get_history=lambda max_messages=40, *, start_index=None: [
            {"role": "user", "content": "old canonical"}
        ],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="continue with node status",
        media=[],
        channel="mobile",
        chat_id="one",
        timestamp=timestamp,
        metadata={
            "_control_turn_input_source": _Source(),
            "_control_attempt_replay": replay,
            "_control_prior_tool_chain": prior_tool_chain,
            "_control_prior_input_count": 1,
        },
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert provider.calls[0]["messages"][:-1] == [
        {"role": "system", "content": "test context"},
        {"role": "user", "content": "old canonical"},
        *replay,
        {"role": "user", "content": "continue with node status"},
    ]
    assert provider.calls[0]["messages"][-1] == {
        "role": "assistant",
        "content": "final after u2",
    }
    assert result.reply == "final after u2"
    assert result.tool_chain == prior_tool_chain
    assert result.tools_used == ["lookup"]
    assert "llm_user_content" not in result.context_retry


def test_default_reasoner_blocks_disabled_tool_even_if_model_calls_it():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "message_push", {"message": "天气"})],
            ),
            LLMResponse(content="最终天气", tool_calls=[]),
        ]
    )
    push = _DummyTool("message_push")
    tools = ToolRegistry()
    tools.register(push, always_on=True, risk="external-side-effect")
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(
            reasoner,
            [{"role": "user", "content": "发天气"}],
            disabled_tools={"message_push"},
        )
    )

    first_tool_names = [
        schema["function"]["name"] for schema in provider.calls[0]["tools"]
    ]
    assert "message_push" not in first_tool_names
    assert push.calls == []
    assert result.reply == "最终天气"
    assert result.metadata["tools_used"] == []
    calls = result.metadata["tool_chain"][0]["calls"]
    assert calls[0]["name"] == "message_push"
    assert calls[0]["status"] == "blocked"


def test_default_reasoner_disable_memory_writes_expands_to_memory_write_tools():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "memorize", {"summary": "x"})],
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(
        _DummyTool("memorize"),
        always_on=True,
        risk="write",
        source_type="builtin",
        source_name="memory",
    )
    tools.register(
        _DummyTool("recall_memory"),
        always_on=True,
        risk="read-only",
        source_type="builtin",
        source_name="memory",
    )
    tools.register(_DummyTool("read_file"), always_on=True, risk="read-only")
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "test context"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                ),
            ),
        ),
    )
    session = SimpleNamespace(
        key="telegram:123",
        created_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=40, *, start_index=None: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="telegram",
        chat_id="123",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
        metadata={"disable_memory_writes": True},
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    # 1. memory 来源的写工具被展开禁用，检索与普通工具保留。
    first_tools = cast(list[dict[str, Any]], provider.calls[0]["tools"])
    first_tool_names = [schema["function"]["name"] for schema in first_tools]
    assert "memorize" not in first_tool_names
    assert "recall_memory" in first_tool_names
    assert "read_file" in first_tool_names
    calls = cast(list[dict[str, Any]], result.tool_chain[0]["calls"])
    assert calls[0]["name"] == "memorize"
    assert calls[0]["status"] == "blocked"


def test_default_reasoner_rejects_model_commit_role_override():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "c1",
                        "message_push",
                        {"message": "hi", "_commit_role": "non_passive"},
                    )
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    push = _DummyTool("message_push")
    tools = ToolRegistry()
    tools.register(push, always_on=True, risk="external-side-effect")
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "done"
    assert push.calls == []


def test_default_reasoner_injects_passive_commit_role_internally():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "message_push", {"message": "hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    push = _DummyTool("message_push")
    tools = ToolRegistry()
    tools.register(push, always_on=True, risk="external-side-effect")
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "done"
    assert push.calls == [{"message": "hi", "_commit_role": "passive"}]


def test_default_reasoner_tool_search_cannot_reunlock_disabled_tool():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("s1", "tool_search", {"query": "select:message_push"})
                ],
            ),
            LLMResponse(content="最终天气", tool_calls=[]),
        ]
    )
    push = _DummyTool("message_push")
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(push, always_on=True, risk="external-side-effect")
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
    )

    result = asyncio.run(
        _run_with_compaction_gate(
            reasoner,
            [{"role": "user", "content": "发天气"}],
            disabled_tools={"message_push"},
        )
    )

    first_tool_names = [
        schema["function"]["name"] for schema in provider.calls[0]["tools"]
    ]
    second_tool_names = [
        schema["function"]["name"] for schema in provider.calls[1]["tools"]
    ]
    assert "message_push" not in first_tool_names
    assert "message_push" not in second_tool_names
    assert push.calls == []
    assert result.reply == "最终天气"
    assert "message_push" not in result.metadata["visible_names"]


def test_default_reasoner_zero_max_iterations_is_unlimited():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "dummy", {})]),
            LLMResponse(content="", tool_calls=[ToolCall("c3", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tool = _DummyTool()
    tools = ToolRegistry()
    tools.register(tool, always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=0, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "final"
    assert len(tool.calls) == 3


def test_default_reasoner_stops_on_context_pressure_after_tool_batch(monkeypatch):
    provider = _Provider(
        [
            LLMResponse(
                content="", tool_calls=[ToolCall("c1", "inflate_probe", {"value": 1})]
            ),
            LLMResponse(content="阶段性回复", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_InflateTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(
            model="m",
            max_iterations=0,
            max_tokens=512,
        ),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )
    reasoner.add_after_step_plugin_modules([ContextPressureStopModule()])

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "阶段性回复"
    assert len(provider.calls) == 2
    assert provider.calls[1]["tools"] == []
    summary_messages = json.dumps(provider.calls[1]["messages"], ensure_ascii=False)
    assert "[收尾原因] context_pressure" in summary_messages
    assert "已经使用了哪些工具或操作" in summary_messages
    assert "当前已经做到哪一步" in summary_messages
    assert "还缺什么信息或步骤" in summary_messages
    assert "inflate_probe" in summary_messages
    assert len(result.metadata["tool_chain"]) == 1


def test_default_reasoner_context_pressure_policy_lives_in_after_step_plugin(
    monkeypatch,
):
    provider = _Provider(
        [
            LLMResponse(
                content="", tool_calls=[ToolCall("c1", "inflate_probe", {"value": 1})]
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_InflateTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(
            model="m",
            max_iterations=0,
            max_tokens=512,
        ),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "final"
    assert len(provider.calls) == 2
    assert provider.calls[1]["tools"]


def test_default_reasoner_observes_tool_lifecycle_events():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 7})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tool = _DummyTool()
    tools.register(tool, always_on=True)
    event_bus = EventBus()
    order: list[str] = []
    started_events: list[ToolCallStarted] = []
    completed_events: list[ToolCallCompleted] = []
    event_bus.on(
        ToolCallStarted,
        lambda event: order.append("started") or started_events.append(event),
    )
    event_bus.on(
        ToolCallCompleted,
        lambda event: order.append("completed") or completed_events.append(event),
    )
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "test context"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                ),
            ),
        ),
        event_bus=event_bus,
    )
    session = SimpleNamespace(
        key="telegram:123",
        created_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=40, *, start_index=None: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="telegram",
        chat_id="123",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "final"
    assert order == ["started", "completed"]
    assert started_events[0].session_key == "telegram:123"
    assert started_events[0].channel == "telegram"
    assert started_events[0].chat_id == "123"
    assert started_events[0].iteration == 1
    assert started_events[0].call_id == "c1"
    assert started_events[0].tool_name == "dummy"
    assert started_events[0].arguments == {"x": 7}
    assert completed_events[0].session_key == "telegram:123"
    assert completed_events[0].call_id == "c1"
    assert completed_events[0].tool_name == "dummy"
    assert completed_events[0].arguments == {"x": 7}
    assert completed_events[0].final_arguments == {"x": 7}
    assert completed_events[0].status == "success"
    assert completed_events[0].result_preview == "dummy-ok"


def test_default_reasoner_observes_blocked_tool_lifecycle_events():
    provider = _Provider(
        [
            LLMResponse(
                content="", tool_calls=[ToolCall("c1", "hidden_tool", {"x": 1})]
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    hidden = _DummyTool("hidden_tool")
    tools.register(hidden)
    event_bus = EventBus()
    order: list[str] = []
    started_events: list[ToolCallStarted] = []
    completed_events: list[ToolCallCompleted] = []
    event_bus.on(
        ToolCallStarted,
        lambda event: order.append("started") or started_events.append(event),
    )
    event_bus.on(
        ToolCallCompleted,
        lambda event: order.append("completed") or completed_events.append(event),
    )
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
        event_bus=event_bus,
    )

    result = asyncio.run(
        _run_with_compaction_gate(
            reasoner,
            [{"role": "user", "content": "hi"}],
            tool_event_session_key="telegram:123",
            tool_event_channel="telegram",
            tool_event_chat_id="123",
        )
    )

    assert result.reply == "final"
    assert hidden.calls == []
    assert order == ["started", "completed"]
    assert started_events[0].tool_name == "hidden_tool"
    assert started_events[0].arguments == {"x": 1}
    assert completed_events[0].tool_name == "hidden_tool"
    assert completed_events[0].arguments == {"x": 1}
    assert completed_events[0].final_arguments == {"x": 1}
    assert completed_events[0].status == "blocked"
    assert "select:hidden_tool" in completed_events[0].result_preview


def test_default_reasoner_unlocks_tool_search_visibility():
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("s1", "tool_search", {"query": "hidden"})],
            ),
            LLMResponse(content="", tool_calls=[ToolCall("h1", "hidden_tool", {})]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    hidden = _DummyTool("hidden_tool")
    tools.register(hidden)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "done"
    assert "hidden_tool" in result.metadata["tools_used"]
    assert "hidden_tool" in result.metadata["visible_names"]
    assert len(hidden.calls) == 1


def test_default_reasoner_preflight_includes_deferred_tool_names():
    """调用方（如 _run_agent_loop）负责注入 deferred tools hint；run() 本身不再自动注入。"""
    from agent.core.passive_turn import build_turn_injection_prompt
    from agent.prompting import build_context_frame_content, build_context_frame_message
    from agent.prompting import PromptSectionRender

    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(
        _DummyTool("mcp_github__list_commits"),
        source_type="mcp",
        source_name="github",
    )
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
    )

    # 调用方负责在调用 run() 前注入 hint。
    hint = build_turn_injection_prompt(
        tools=tools,
        tool_search_enabled=True,
        visible_names=tools.get_always_on_names(),
    )
    frame_content = build_context_frame_content(
        [PromptSectionRender(name="tool_hint", content=hint, is_static=False)]
    )
    initial_messages = [
        build_context_frame_message(frame_content),
        {"role": "user", "content": "hi"},
    ]
    asyncio.run(_run_with_compaction_gate(reasoner, initial_messages))

    first_messages = provider.calls[0]["messages"]
    preflight = next(
        str(m.get("content", ""))
        for m in first_messages
        if "未加载工具目录" in str(m.get("content", ""))
    )
    assert "未加载工具目录" in preflight
    assert "mcp_github__list_commits" in preflight
    assert "dummy" not in preflight


def test_default_reasoner_deferred_tool_direct_call_requires_select():
    provider = _Provider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "schedule", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(_DummyTool("schedule"))
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert "schedule" not in result.metadata["tools_used"]
    assert result.reply == "final"
    tool_chain = list(result.metadata["tool_chain"])
    assert len(tool_chain) >= 1
    schedule_call = next(
        (c for c in tool_chain[0]["calls"] if c["name"] == "schedule"), None
    )
    assert schedule_call is not None
    assert "select:" in schedule_call["result"]
    assert "tool_search" in schedule_call["result"]


def test_default_reasoner_preloaded_tool_not_in_deferred_list():
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    tools.register(_DummyTool("schedule"))
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=True,
    )

    asyncio.run(
        _run_with_compaction_gate(
            reasoner,
            [{"role": "user", "content": "hi"}],
            preloaded_tools={"schedule"},
        )
    )

    first_messages = provider.calls[0]["messages"]
    assert not any(
        "未加载工具目录" in str(m.get("content", "")) for m in first_messages
    )


def test_default_reasoner_run_turn_uses_context_render():
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "test context"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                ),
                build_messages=lambda **_: (_ for _ in ()).throw(
                    AssertionError("legacy build_messages should not be used")
                ),
                build_turn_injection_context=lambda **_: (_ for _ in ()).throw(
                    AssertionError("legacy turn_injection should not be used")
                ),
            ),
        ),
    )

    session = SimpleNamespace(
        key="cli:1",
        created_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        messages=[{"role": "assistant", "content": "old"}],
        get_history=lambda max_messages=40, *, start_index=None: [
            {"role": "assistant", "content": "old"}
        ],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "done"


def test_default_reasoner_run_turn_reports_llm_timeout():
    provider = _TimeoutProvider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "test context"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                ),
            ),
        ),
    )
    session = SimpleNamespace(
        key="cli:1",
        created_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=40, *, start_index=None: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="hi",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime(2026, 4, 5, 12, 0, 0),
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "模型流响应中断，请刷新对话重试。"
    assert len(provider.calls) == 1


def test_empty_content_with_thinking_triggers_retry_and_succeeds():
    provider = _Provider(
        [
            LLMResponse(
                content=None,
                tool_calls=[],
                thinking="长思考过程",
                finish_reason="length",
            ),
            LLMResponse(
                content="正式回复",
                tool_calls=[],
                thinking="新思考",
                finish_reason="stop",
            ),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "正式回复"
    assert result.thinking == "新思考"
    assert result.metadata["react_stats"]["finish_reasons"] == [
        "length",
        "stop",
    ]
    retry_call = provider.calls[1]
    assert retry_call["disable_thinking"] is True
    assert [schema["function"]["name"] for schema in retry_call["tools"]] == ["dummy"]
    assert len(provider.calls) == 2


def test_empty_content_with_thinking_retry_can_enter_tool_loop():
    provider = _Provider(
        [
            LLMResponse(content=None, tool_calls=[], thinking="需要写文件"),
            LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "dummy", {})],
                thinking="调用工具",
            ),
            LLMResponse(content="已完成", tool_calls=[]),
        ]
    )
    tool = _DummyTool()
    tools = ToolRegistry()
    tools.register(tool, always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider),
                light_provider=cast(Any, provider),
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "已完成"
    assert tool.calls == [{}]
    assert len(provider.calls) == 3
    assert provider.calls[1]["disable_thinking"] is True
    assert [schema["function"]["name"] for schema in provider.calls[1]["tools"]] == [
        "dummy"
    ]


def test_empty_content_with_thinking_retry_still_empty_falls_back():
    provider = _Provider(
        [
            LLMResponse(content=None, tool_calls=[], thinking="只有思考"),
            LLMResponse(content=None, tool_calls=[], thinking=None),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "模型未返回可用回复，请重试。"
    assert result.thinking == "只有思考"
    assert len(provider.calls) == 2


def test_empty_content_without_thinking_no_retry():
    provider = _Provider(
        [
            LLMResponse(content=None, tool_calls=[], thinking=None),
        ]
    )
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    result = asyncio.run(
        _run_with_compaction_gate(reasoner, [{"role": "user", "content": "hi"}])
    )

    assert result.reply == "模型未返回可用回复，请重试。"
    assert len(provider.calls) == 1


def test_default_reasoner_reuses_snapshot_step_phases(monkeypatch):
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-1",
        before_step_modules=(),
        after_step_modules=(),
    )
    current_snapshot = [snapshot]
    monkeypatch.setattr(
        "agent.core.passive_turn.get_current_runtime_snapshot",
        lambda: current_snapshot[0],
    )

    first = reasoner._runtime_step_phases()
    second = reasoner._runtime_step_phases()

    assert second == first
    assert second[0] is first[0]
    assert second[1] is first[1]

    current_snapshot[0] = SimpleNamespace(
        snapshot_id="snapshot-2",
        before_step_modules=(),
        after_step_modules=(),
    )
    next_snapshot_phases = reasoner._runtime_step_phases()

    assert next_snapshot_phases[0] is not first[0]
    assert next_snapshot_phases[1] is not first[1]
