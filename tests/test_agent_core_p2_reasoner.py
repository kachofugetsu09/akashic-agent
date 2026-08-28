import asyncio
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.config_models import ContextCompactionConfig
from agent.control.turn_scope import (
    ToolGrant,
    TurnExecutionScope,
    bind_turn_scope,
    reset_turn_scope,
)
from agent.core.passive_turn import DefaultReasoner
from agent.control.ports import TurnUserInput
from agent.core.runtime_support import SessionLike, ToolDiscoveryState
from agent.lifecycle.types import AfterStepCtx
from agent.looping.ports import LLMConfig, LLMServices
from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextCompactionError,
    ContextPayloadSegments,
    SUMMARY_HEADINGS,
)
from agent.provider import (
    ContextLengthError,
    LLMProvider,
    LLMResponse,
    ToolCall,
)
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.tool_search import ToolSearchTool
from bus.event_bus import EventBus
from bus.events_lifecycle import ToolCallCompleted, ToolCallStarted, TurnOutputCompleted
from core.error_context import (
    current_provider_attempt,
    current_provider_call_id,
    current_provider_operation,
)
from session.compaction_runtime import CompactionProjection
from session.manager import Session
from session.store import CompactionHead


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
        units = tuple(
            CommittedContextUnit(
                source_from_seq=index,
                consolidated_through_seq=index,
                source_message_ids=(f"test-message-{index}",),
                messages=(dict(message),),
                message_refs=((f"test-message-{index}", index),),
            )
            for index, message in enumerate(history)
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


class _CommittableCompactionRuntime(_MandatoryCompactionRuntime):
    """Commit 直接成功，供真实压缩路径（overflow 强制压缩 / 初始压缩）测试使用。"""

    def __init__(self) -> None:
        self.commit_count = 0

    async def commit_checkpoint(
        self,
        session: SessionLike,
        checkpoint: Any,
        *,
        head: Any,
        scope_channel: str = "",
        scope_chat_id: str = "",
    ) -> SimpleNamespace:
        self.commit_count += 1
        return SimpleNamespace(generation=checkpoint.generation)


def _build_reasoner(**kwargs: Any) -> DefaultReasoner:
    """Construct a reasoner with the mandatory session compaction runtime."""

    return DefaultReasoner(
        compaction_runtime=kwargs.pop("compaction_runtime", None)
        or _MandatoryCompactionRuntime(),
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
    assert result.tools_used == ["dummy"]
    assert result.tool_chain[0]["calls"][0]["name"] == "dummy"
    assert result.visible_names is None
    react_stats = result.react_stats
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
        get_history=lambda max_messages=40: [
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
    assert result.tools_used == []
    calls = result.tool_chain[0]["calls"]
    assert calls[0]["name"] == "message_push"
    assert calls[0]["status"] == "blocked"


def test_default_reasoner_does_not_interpret_legacy_memory_write_metadata():
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
        get_history=lambda max_messages=40: [],
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

    first_tools = cast(list[dict[str, Any]], provider.calls[0]["tools"])
    first_tool_names = [schema["function"]["name"] for schema in first_tools]
    assert "memorize" in first_tool_names
    assert "recall_memory" in first_tool_names
    assert "read_file" in first_tool_names
    calls = cast(list[dict[str, Any]], result.tool_chain[0]["calls"])
    assert calls[0]["name"] == "memorize"
    assert calls[0]["status"] != "blocked"


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
    assert result.visible_names is not None
    assert "message_push" not in result.visible_names


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
        get_history=lambda max_messages=40: [],
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


def test_default_reasoner_observes_output_completed_before_after_step():
    provider = _Provider([LLMResponse(content="final", tool_calls=[])])
    tools = ToolRegistry()
    event_bus = EventBus()
    order: list[str] = []
    completed_events: list[TurnOutputCompleted] = []
    event_bus.on(
        TurnOutputCompleted,
        lambda event: order.append("output_completed")
        or completed_events.append(event),
    )

    async def slow_after_step(_event: AfterStepCtx) -> None:
        order.append("after_step_start")
        await asyncio.sleep(0.05)
        order.append("after_step_end")

    event_bus.on(AfterStepCtx, slow_after_step)
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
    assert completed_events
    assert completed_events[0].session_key == "telegram:123"
    assert completed_events[0].channel == "telegram"
    assert completed_events[0].chat_id == "123"
    # 输出完成信号必须在 AfterStep 收尾完成之前发出，慢插件不得推迟解锁
    assert order.index("output_completed") < order.index("after_step_end")


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
    assert "hidden_tool" in result.tools_used
    assert result.visible_names is not None
    assert "hidden_tool" in result.visible_names
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

    assert "schedule" not in result.tools_used
    assert result.reply == "final"
    tool_chain = list(result.tool_chain)
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
        get_history=lambda max_messages=40: [{"role": "assistant", "content": "old"}],
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


def test_default_reasoner_session_history_read_false_reaches_provider_without_history():
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=1, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "system"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                )
            ),
        ),
    )
    session = SimpleNamespace(
        key="programmatic:stateless",
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        get_history=lambda max_messages=500: [
            {"role": "assistant", "content": "must-not-reach-provider"}
        ],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="current",
        media=[],
        channel="programmatic",
        chat_id="stateless",
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        metadata={"skip_session_history": True},
    )

    result = asyncio.run(reasoner.run_turn(msg=msg, session=cast(Any, session)))

    assert result.reply == "done"
    messages = provider.calls[0]["messages"]
    assert [message["content"] for message in messages[:2]] == ["system", "current"]
    assert all(message["content"] != "must-not-reach-provider" for message in messages)


@pytest.mark.asyncio
async def test_turn_scope_preloads_only_authorized_deferred_tool() -> None:
    provider = _Provider([LLMResponse(content="done", tool_calls=[])])
    tools = ToolRegistry()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(_DummyTool("scoped_decision"))
    tools.register(_DummyTool("other_deferred"))
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
        context=cast(
            Any,
            SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "test context"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ],
                )
            ),
        ),
    )
    session = SimpleNamespace(
        key="programmatic:scoped",
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=500: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="decide",
        media=[],
        metadata={},
        channel="programmatic",
        chat_id="scoped",
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
    )
    token = bind_turn_scope(
        TurnExecutionScope(
            preloaded_tools=("scoped_decision",),
            tool_grant=ToolGrant.only(("scoped_decision",)),
        )
    )
    try:
        result = await reasoner.run_turn(msg=msg, session=cast(Any, session))
    finally:
        reset_turn_scope(token)

    assert result.reply == "done"
    first_tool_names = {
        schema["function"]["name"] for schema in provider.calls[0]["tools"]
    }
    assert first_tool_names == {"scoped_decision"}


@pytest.mark.asyncio
async def test_scoped_budget_adds_one_terminal_only_decision_round() -> None:
    provider = _Provider(
        [
            LLMResponse(
                content="",
                thinking="still investigating",
                tool_calls=[],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("hallucinated-research", "research", {}),
                    ToolCall("decision", "share_content", {}),
                    ToolCall("late-research", "research", {}),
                ],
            ),
        ]
    )
    tools = ToolRegistry()
    research = _DummyTool("research")
    decision = _DummyTool("share_content")
    tools.register(research)
    tools.register(decision)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=10, max_tokens=512),
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
                )
            ),
        ),
    )
    session = SimpleNamespace(
        key="programmatic:wake",
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=500: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="investigate",
        media=[],
        metadata={},
        channel="wake",
        chat_id="wake",
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
    )
    token = bind_turn_scope(
        TurnExecutionScope(
            preloaded_tools=("research", "share_content"),
            terminal_tools=("share_content",),
            tool_grant=ToolGrant.only(("research", "share_content")),
            max_iterations=1,
        )
    )
    try:
        result = await reasoner.run_turn(msg=msg, session=cast(Any, session))
    finally:
        reset_turn_scope(token)

    assert result.tools_used == ["share_content"]
    assert research.calls == []
    assert len(provider.calls) == 2
    assert {schema["function"]["name"] for schema in provider.calls[1]["tools"]} == {
        "share_content"
    }
    assert any(
        message.get("role") == "user"
        and "调查预算已经用完" in message.get("content", "")
        for message in provider.calls[1]["messages"]
    )


@pytest.mark.asyncio
async def test_scoped_terminal_correction_cannot_execute_non_terminal_tool() -> None:
    provider = _Provider(
        [
            LLMResponse(content="I will explain instead.", tool_calls=[]),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("malicious-research", "research", {}),
                    ToolCall("decision", "share_content", {}),
                ],
            ),
        ]
    )
    tools = ToolRegistry()
    research = _DummyTool("research")
    decision = _DummyTool("share_content")
    tools.register(research)
    tools.register(decision)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, provider), light_provider=cast(Any, provider)
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=3, max_tokens=512),
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
                )
            ),
        ),
    )
    session = SimpleNamespace(
        key="programmatic:wake",
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=500: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="decide",
        media=[],
        metadata={},
        channel="wake",
        chat_id="wake",
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
    )
    token = bind_turn_scope(
        TurnExecutionScope(
            preloaded_tools=("research", "share_content"),
            terminal_tools=("share_content",),
            tool_grant=ToolGrant.only(("research", "share_content")),
            max_iterations=3,
        )
    )
    try:
        result = await reasoner.run_turn(msg=msg, session=cast(Any, session))
    finally:
        reset_turn_scope(token)

    assert result.tools_used == ["share_content"]
    assert research.calls == []
    assert len(provider.calls) == 2
    assert {schema["function"]["name"] for schema in provider.calls[1]["tools"]} == {
        "share_content"
    }


@pytest.mark.asyncio
async def test_turn_scope_missing_preload_fails_before_provider_call() -> None:
    provider = _Provider([LLMResponse(content="must not run", tool_calls=[])])
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
        tool_search_enabled=True,
        context=cast(Any, SimpleNamespace(render=lambda *_args, **_kwargs: None)),
    )
    session = SimpleNamespace(
        key="programmatic:missing",
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=500: [],
        last_consolidated=0,
    )
    msg = SimpleNamespace(
        content="decide",
        media=[],
        metadata={},
        channel="programmatic",
        chat_id="missing",
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
    )
    token = bind_turn_scope(
        TurnExecutionScope(
            preloaded_tools=("missing_decision",),
            tool_grant=ToolGrant.only(("missing_decision",)),
        )
    )
    try:
        with pytest.raises(RuntimeError, match="preload Tool 未注册: missing_decision"):
            await reasoner.run_turn(msg=msg, session=cast(Any, session))
    finally:
        reset_turn_scope(token)

    assert provider.calls == []


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
        get_history=lambda max_messages=40: [],
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


def test_default_reasoner_observes_output_completed_on_timeout_error():
    provider = _TimeoutProvider()
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    event_bus = EventBus()
    completed_events: list[TurnOutputCompleted] = []
    event_bus.on(TurnOutputCompleted, completed_events.append)
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
        key="cli:1",
        created_at=datetime(2026, 4, 5, 12, 0, 0, tzinfo=UTC),
        messages=[],
        get_history=lambda max_messages=40: [],
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
    assert completed_events
    assert completed_events[0].session_key == "cli:1"
    assert completed_events[0].channel == "cli"
    assert completed_events[0].chat_id == "1"


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
    assert result.react_stats["finish_reasons"] == [
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


def test_default_reasoner_uses_one_default_step_phase_pair():
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
    first = reasoner._runtime_step_phases()
    second = reasoner._runtime_step_phases()

    assert second == first
    assert second[0] is first[0]
    assert second[1] is first[1]


# ── 首 token 观测：turn-first 里程碑 ─────────────────────────────────


def _milestone_events(
    caplog: pytest.LogCaptureFixture,
    event: str,
) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], record.akashic_fields)
        for record in caplog.records
        if getattr(record, "akashic_fields", None) is not None
        and record.akashic_fields.get("event") == event
    ]


def _counts_map(counts: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in counts.split() if "=" in part)


def _provider_call_id(fields: dict[str, object]) -> str:
    return _counts_map(cast(str, fields["counts"]))["provider_call_id"]


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        chunk = self._chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk

    async def close(self) -> None:
        return None


def _thinking_chunk(text: str = "ponder") -> SimpleNamespace:
    return SimpleNamespace(
        id="chunk-think",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None, tool_calls=[], reasoning_content=text
                ),
                finish_reason=None,
            )
        ],
    )


def _tool_chunk(name: str = "dummy") -> SimpleNamespace:
    return SimpleNamespace(
        id="chunk-tool",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=0,
                            id="c1",
                            function=SimpleNamespace(name=name, arguments='{"x":1}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )


def _answer_chunk(content: str = "final") -> SimpleNamespace:
    return SimpleNamespace(
        id="chunk-answer",
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=[]),
                finish_reason="stop",
            )
        ],
    )


class _FakeClock:
    """Controllable monotonic clock；只由测试显式推进，不 sleep。"""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


class _BlockedRequestStartProvider(_ProviderContextBudget):
    """chat 返回前人为阻塞 100ms（模拟请求建立/上游等待），再回传首 delta。"""

    def __init__(self, response: LLMResponse, clock: _FakeClock) -> None:
        self._response = response
        self._clock = clock
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        self._clock.advance_ms(100.0)
        delta_sink = kwargs.get("on_content_delta")
        if delta_sink is not None:
            await delta_sink({"thinking_delta": "deliberate"})
        return self._response


class _DeltaEmitterProvider(_ProviderContextBudget):
    """每次 chat 都流式回传 thinking+content delta，模拟真实流式消费。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        delta_sink = kwargs.get("on_content_delta")
        if delta_sink is not None:
            await delta_sink({"thinking_delta": "ponder"})
            await delta_sink({"content_delta": "draft"})
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


class _ToolFirstProvider(_ProviderContextBudget):
    """纯 tool-call 响应：不流式任何 delta，first_any 只能来自 tool kind。"""

    def __init__(
        self, responses: list[LLMResponse], clock: _FakeClock | None = None
    ) -> None:
        self._responses = list(responses)
        self._clock = clock
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if self._clock is not None:
            self._clock.advance_ms(100.0)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


class _FailingProvider(_ProviderContextBudget):
    """普通 provider 异常：attempt 必须以 error 终态闭合。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        raise RuntimeError("provider exploded")


class _CancelledProvider(_ProviderContextBudget):
    """provider 抛 CancelledError：attempt 必须以 cancelled 终态闭合。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        raise asyncio.CancelledError


class _OverflowThenSuccessProvider(_ProviderContextBudget):
    """attempt1 抛 ContextLengthError；强制压缩 summary 返回合法摘要；attempt2 成功。"""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs.get("on_content_delta") is None:
            return LLMResponse(content="\n".join(SUMMARY_HEADINGS))
        business_calls = [
            call for call in self.calls if call.get("on_content_delta") is not None
        ]
        if len(business_calls) == 1:
            raise ContextLengthError("provider context overflow")
        return self._response


class _SlowCompactionProvider(_ProviderContextBudget):
    """初始压缩 summary 慢（500ms），业务 chat TTFT 快（100ms），验证两者分离。"""

    context_window = 1_000_000

    def __init__(self, response: LLMResponse, clock: _FakeClock) -> None:
        self._response = response
        self._clock = clock
        self.calls: list[dict[str, Any]] = []

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        if any(
            "<session-context-compaction>" in str(message.get("content", ""))
            for message in messages
        ):
            return 10
        return 900_000

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        return 3

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs.get("on_content_delta") is None:
            self._clock.advance_ms(500.0)
            return LLMResponse(content="\n".join(SUMMARY_HEADINGS))
        self._clock.advance_ms(100.0)
        delta_sink = kwargs.get("on_content_delta")
        if delta_sink is not None:
            await delta_sink({"thinking_delta": "deliberate"})
        return self._response


class _BoundaryHitProvider(_ProviderContextBudget):
    """估算越过软边界：无压缩候选时 gate 报 error；summary 阶段抛 CancelledError 时 gate 报 cancelled。"""

    context_window = 1_000_000

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        if any(
            "<session-context-compaction>" in str(message.get("content", ""))
            for message in messages
        ):
            return 10
        return 900_000

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        return 3

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        raise asyncio.CancelledError


class _SlowSinkProvider(_ProviderContextBudget):
    """下游回调消费慢（200ms）：first-delta 采样必须发生在回调之前，不被污染。"""

    def __init__(self, response: LLMResponse, clock: _FakeClock) -> None:
        self._response = response
        self._clock = clock
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        delta_sink = kwargs.get("on_content_delta")
        if delta_sink is not None:
            await delta_sink({"thinking_delta": "fast"})
            self._clock.advance_ms(200.0)
        return self._response


def _compaction_reasoner(
    provider: object,
    runtime: _CommittableCompactionRuntime,
) -> DefaultReasoner:
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    return _build_reasoner(
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
        compaction_runtime=runtime,
        context_compaction=ContextCompactionConfig(keep_recent_tokens=1),
    )


def _single_tool_round_reasoner(provider: object) -> DefaultReasoner:
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    return _build_reasoner(
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


async def _stream_delta_sink(delta: dict[str, str]) -> None:
    return None


def test_turn_first_ttft_includes_request_establishment_delay(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    provider = _BlockedRequestStartProvider(LLMResponse(content="final"), clock)
    reasoner = _single_tool_round_reasoner(provider)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "hi"}],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert len(starts) == 1
    call_id = _provider_call_id(starts[0])
    assert len(call_id) == 32
    assert str(starts[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    first_thinking = _milestone_events(caplog, "tl:turn.first_thinking")
    assert len(first_thinking) == 1
    assert str(first_thinking[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    duration_ms = first_thinking[0].get("duration_ms")
    assert isinstance(duration_ms, (int, float))
    assert duration_ms >= 100.0
    done = _milestone_events(caplog, "tl:provider.call.done")
    assert len(done) == 1
    assert done[0].get("outcome") == "done"
    assert str(done[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )


def test_two_tool_rounds_emit_single_turn_first(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _DeltaEmitterProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    reasoner = _single_tool_round_reasoner(provider)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "hi"}],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    assert result.tools_used == ["dummy"]
    assert len(provider.calls) == 2
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert len(starts) == 2
    first_call_id = _provider_call_id(starts[0])
    second_call_id = _provider_call_id(starts[1])
    assert first_call_id != second_call_id
    assert [str(item.get("counts")) for item in starts] == [
        f"call_ordinal=1 provider_attempt=1 provider_call_id={first_call_id}",
        f"call_ordinal=2 provider_attempt=1 provider_call_id={second_call_id}",
    ]
    done = _milestone_events(caplog, "tl:provider.call.done")
    assert len(done) == 2
    assert [str(item.get("outcome")) for item in done] == ["done", "done"]
    assert len(_milestone_events(caplog, "tl:turn.first_any")) == 1
    assert len(_milestone_events(caplog, "tl:turn.first_thinking")) == 1
    assert len(_milestone_events(caplog, "tl:turn.first_answer")) == 1
    # turn.first 携带发出该事件时所属逻辑 call 的 provider_call_id。
    assert (
        _provider_call_id(_milestone_events(caplog, "tl:turn.first_any")[0])
        == first_call_id
    )
    assert (
        _provider_call_id(_milestone_events(caplog, "tl:turn.first_thinking")[0])
        == first_call_id
    )
    # _DeltaEmitterProvider 每轮同时发 thinking+content：first_answer 也在 round1 发出。
    assert (
        _provider_call_id(_milestone_events(caplog, "tl:turn.first_answer")[0])
        == first_call_id
    )


def test_tool_call_first_records_turn_first_any(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    provider = _ToolFirstProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {})]),
            LLMResponse(content="final", tool_calls=[]),
        ],
        clock=clock,
    )
    reasoner = _single_tool_round_reasoner(provider)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "hi"}],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    first_any = _milestone_events(caplog, "tl:turn.first_any")
    assert len(first_any) == 1
    call_id = _provider_call_id(first_any[0])
    assert str(first_any[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id} kind=tool"
    )
    first_any_duration = first_any[0].get("duration_ms")
    assert isinstance(first_any_duration, (int, float))
    assert first_any_duration >= 100.0
    assert not _milestone_events(caplog, "tl:turn.first_thinking")
    assert not _milestone_events(caplog, "tl:turn.first_answer")


# ── provider call / compaction 里程碑：结构化 outcome 与 attempt 闭合 ──────────


def test_initial_compaction_slow_keeps_provider_ttft_separate(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    runtime = _CommittableCompactionRuntime()
    provider = _SlowCompactionProvider(LLMResponse(content="final"), clock)
    reasoner = _compaction_reasoner(provider, runtime)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [
                    {"role": "user", "content": "old one"},
                    {"role": "user", "content": "current"},
                ],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    assert runtime.commit_count == 1
    prepares = _milestone_events(caplog, "tl:compaction.prepare.done")
    assert len(prepares) == 1
    assert prepares[0].get("outcome") == "done"
    prepare_duration = prepares[0].get("duration_ms")
    assert isinstance(prepare_duration, (int, float))
    assert prepare_duration >= 500.0
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert len(starts) == 1
    call_id = _provider_call_id(starts[0])
    assert str(prepares[0].get("counts")) == (
        f"call_ordinal=1 provider_call_id={call_id} "
        "trigger=soft_limit force=false compacted=true"
    )
    assert str(starts[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    # 初始 compaction gate 与业务 call 属于同一逻辑调用：call_id 一致。
    assert (
        _provider_call_id(_milestone_events(caplog, "tl:compaction.prepare.start")[0])
        == call_id
    )
    first_thinking = _milestone_events(caplog, "tl:turn.first_thinking")
    assert len(first_thinking) == 1
    assert str(first_thinking[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    first_duration = first_thinking[0].get("duration_ms")
    assert isinstance(first_duration, (int, float))
    assert first_duration < 200.0
    assert first_duration < prepare_duration


def test_context_overflow_sequence_closes_retry_then_attempt_two(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _CommittableCompactionRuntime()
    provider = _OverflowThenSuccessProvider(LLMResponse(content="recovered"))
    reasoner = _compaction_reasoner(provider, runtime)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [
                    {"role": "user", "content": "old one"},
                    {"role": "user", "content": "current"},
                ],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "recovered"
    assert runtime.commit_count == 1
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert len(starts) == 2
    attempt_one_id = _provider_call_id(starts[0])
    attempt_two_id = _provider_call_id(starts[1])
    # 两个 overflow attempts 属于同一个逻辑 call：共享 provider_call_id。
    assert attempt_one_id == attempt_two_id
    assert [str(item.get("counts")) for item in starts] == [
        f"call_ordinal=1 provider_attempt=1 provider_call_id={attempt_one_id}",
        f"call_ordinal=1 provider_attempt=2 provider_call_id={attempt_one_id}",
    ]
    retry = _milestone_events(caplog, "tl:provider.call.retry")
    assert len(retry) == 1
    assert retry[0].get("outcome") == "context_overflow"
    assert retry[0].get("duration_ms") is not None
    assert str(retry[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={attempt_one_id}"
    )
    prepares = _milestone_events(caplog, "tl:compaction.prepare.done")
    assert [str(item.get("counts")) for item in prepares] == [
        f"call_ordinal=1 provider_call_id={attempt_one_id} "
        "trigger=soft_limit force=false compacted=false",
        f"call_ordinal=1 provider_call_id={attempt_one_id} "
        "trigger=context_overflow force=true compacted=true",
    ]
    assert all(item.get("outcome") == "done" for item in prepares)
    done = _milestone_events(caplog, "tl:provider.call.done")
    assert len(done) == 1
    assert done[0].get("outcome") == "done"
    assert str(done[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=2 provider_call_id={attempt_one_id}"
    )
    assert done[0].get("duration_ms") is not None
    assert not _milestone_events(caplog, "tl:provider.call.error")
    assert not _milestone_events(caplog, "tl:provider.call.cancelled")


def test_provider_error_closes_attempt_with_error_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _FailingProvider()
    reasoner = _single_tool_round_reasoner(provider)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        with pytest.raises(RuntimeError, match="provider exploded"):
            asyncio.run(
                _run_with_compaction_gate(
                    reasoner,
                    [{"role": "user", "content": "boom"}],
                )
            )

    errors = _milestone_events(caplog, "tl:provider.call.error")
    assert len(errors) == 1
    assert errors[0].get("outcome") == "error"
    assert errors[0].get("duration_ms") is not None
    call_id = _provider_call_id(errors[0])
    assert str(errors[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    assert not _milestone_events(caplog, "tl:provider.call.done")
    assert not _milestone_events(caplog, "tl:provider.call.retry")
    assert not _milestone_events(caplog, "tl:provider.call.cancelled")


def test_provider_cancelled_closes_attempt_with_cancelled_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _CancelledProvider()
    reasoner = _single_tool_round_reasoner(provider)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                _run_with_compaction_gate(
                    reasoner,
                    [{"role": "user", "content": "boom"}],
                )
            )

    cancelled = _milestone_events(caplog, "tl:provider.call.cancelled")
    assert len(cancelled) == 1
    assert cancelled[0].get("outcome") == "cancelled"
    assert cancelled[0].get("duration_ms") is not None
    call_id = _provider_call_id(cancelled[0])
    assert str(cancelled[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    assert not _milestone_events(caplog, "tl:provider.call.done")
    assert not _milestone_events(caplog, "tl:provider.call.error")
    assert not _milestone_events(caplog, "tl:provider.call.retry")


def test_unknown_window_overflow_closes_attempt_with_error_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        with pytest.raises(ContextLengthError, match="provider context overflow"):
            asyncio.run(
                _run_with_compaction_gate(
                    reasoner,
                    [{"role": "user", "content": "overflow"}],
                )
            )

    assert len(provider.calls) == 1
    errors = _milestone_events(caplog, "tl:provider.call.error")
    assert len(errors) == 1
    assert errors[0].get("outcome") == "error"
    assert errors[0].get("duration_ms") is not None
    call_id = _provider_call_id(errors[0])
    assert str(errors[0].get("counts")) == (
        f"call_ordinal=1 provider_attempt=1 provider_call_id={call_id}"
    )
    assert not _milestone_events(caplog, "tl:provider.call.retry")
    assert not _milestone_events(caplog, "tl:provider.call.done")
    assert not _milestone_events(caplog, "tl:provider.call.cancelled")


def test_compaction_prepare_error_records_error_then_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _CommittableCompactionRuntime()
    provider = _BoundaryHitProvider()
    reasoner = _compaction_reasoner(provider, runtime)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        with pytest.raises(
            ContextCompactionError,
            match="context_compaction_no_closed_prefix",
        ):
            asyncio.run(
                _run_with_compaction_gate(
                    reasoner,
                    [{"role": "user", "content": "only one unit"}],
                )
            )

    prepare_errors = _milestone_events(caplog, "tl:compaction.prepare.error")
    assert len(prepare_errors) == 1
    assert prepare_errors[0].get("outcome") == "error"
    assert prepare_errors[0].get("duration_ms") is not None
    call_id = _provider_call_id(prepare_errors[0])
    assert str(prepare_errors[0].get("counts")) == (
        f"call_ordinal=1 provider_call_id={call_id} " "trigger=soft_limit force=false"
    )
    assert not _milestone_events(caplog, "tl:compaction.prepare.done")
    assert not _milestone_events(caplog, "tl:compaction.prepare.cancelled")
    assert not _milestone_events(caplog, "tl:provider.call.start")


def test_compaction_prepare_cancelled_records_cancelled_then_propagates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _CommittableCompactionRuntime()
    provider = _BoundaryHitProvider()
    reasoner = _compaction_reasoner(provider, runtime)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                _run_with_compaction_gate(
                    reasoner,
                    [
                        {"role": "user", "content": "old one"},
                        {"role": "user", "content": "current"},
                    ],
                )
            )

    prepare_cancelled = _milestone_events(caplog, "tl:compaction.prepare.cancelled")
    assert len(prepare_cancelled) == 1
    assert prepare_cancelled[0].get("outcome") == "cancelled"
    assert prepare_cancelled[0].get("duration_ms") is not None
    call_id = _provider_call_id(prepare_cancelled[0])
    assert str(prepare_cancelled[0].get("counts")) == (
        f"call_ordinal=1 provider_call_id={call_id} " "trigger=soft_limit force=false"
    )
    assert not _milestone_events(caplog, "tl:compaction.prepare.done")
    assert not _milestone_events(caplog, "tl:compaction.prepare.error")
    assert not _milestone_events(caplog, "tl:provider.call.start")


def test_slow_downstream_callback_does_not_pollute_first_delta(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("time.monotonic", clock)
    provider = _SlowSinkProvider(LLMResponse(content="final"), clock)
    reasoner = _single_tool_round_reasoner(provider)

    with caplog.at_level(logging.INFO, logger="agent.core.passive_turn"):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "hi"}],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    first_thinking = _milestone_events(caplog, "tl:turn.first_thinking")
    assert len(first_thinking) == 1
    first_duration = first_thinking[0].get("duration_ms")
    assert isinstance(first_duration, (int, float))
    assert first_duration < 150.0
    done = _milestone_events(caplog, "tl:provider.call.done")
    assert len(done) == 1
    done_duration = done[0].get("duration_ms")
    assert isinstance(done_duration, (int, float))
    assert done_duration >= 200.0


# ── 中性 provider_call_id：高层 call/first 与底层 transport/http join ──────────


def test_provider_call_id_joins_high_and_low_level_milestones(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """同一次真实 _call_provider 驱动：两个连续 tool round 的 call ID 不同，
    且各 round 的高层 call/first 与底层 transport/http/raw 按 provider_call_id join。"""
    fake = _FakeClient(
        [
            _FakeStream([_thinking_chunk(), _tool_chunk()]),
            _FakeStream([_answer_chunk("final")]),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    reasoner = _single_tool_round_reasoner(LLMProvider(api_key="k"))

    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "hi"}],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    assert result.tools_used == ["dummy"]
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert [str(item.get("counts")) for item in starts] == [
        f"call_ordinal=1 provider_attempt=1 provider_call_id={_provider_call_id(starts[0])}",
        f"call_ordinal=2 provider_attempt=1 provider_call_id={_provider_call_id(starts[1])}",
    ]
    call_1 = _provider_call_id(starts[0])
    call_2 = _provider_call_id(starts[1])
    assert call_1 != call_2

    # 底层 transport/http 从 neutral context 读同一身份，不靠 active turn 猜。
    transport_starts = _milestone_events(caplog, "tl:provider.transport.start")
    assert len(transport_starts) == 2
    assert _provider_call_id(transport_starts[0]) == call_1
    assert _provider_call_id(transport_starts[1]) == call_2
    assert len(_milestone_events(caplog, "tl:provider.transport.done")) == 2
    http_starts = _milestone_events(caplog, "tl:provider.http.start")
    assert len(http_starts) == 2
    assert _provider_call_id(http_starts[0]) == call_1
    assert _provider_call_id(http_starts[1]) == call_2
    assert len(_milestone_events(caplog, "tl:provider.http.done")) == 2
    # 各 transport span_id 仍各自唯一。
    span_1 = _counts_map(cast(str, transport_starts[0]["counts"]))["span_id"]
    span_2 = _counts_map(cast(str, transport_starts[1]["counts"]))["span_id"]
    assert span_1 != span_2
    # 低层事件都标 business 且 attempt 准确。
    for entry in (*transport_starts, *http_starts):
        counts = _counts_map(cast(str, entry["counts"]))
        assert counts["provider_operation"] == "business"
        assert counts["provider_attempt"] == "1"
    # 每个 stream 各自采样一次 raw.first_*，携带所属 call。
    raw_first = _milestone_events(caplog, "tl:provider.raw.first_any")
    assert len(raw_first) == 2
    assert _provider_call_id(raw_first[0]) == call_1
    assert _provider_call_id(raw_first[1]) == call_2
    # 高层 turn.first 携带发出事件时所属 call；首 delta 回调仍在下游消费前采样。
    assert (
        _provider_call_id(_milestone_events(caplog, "tl:turn.first_any")[0]) == call_1
    )
    assert (
        _provider_call_id(_milestone_events(caplog, "tl:turn.first_thinking")[0])
        == call_1
    )


def test_context_overflow_attempts_share_call_id_across_layers(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """context overflow 的两个 provider attempt 共用 provider_call_id（attempt 1/2），
    强制压缩摘要 nonstream 标记 compaction_summary + attempt=0，
    摘要后业务 stream 分别恢复 attempt 1/2 的 business 标签。"""
    fake = _FakeClient(
        [
            RuntimeError("maximum context length exceeded for model"),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="\n".join(SUMMARY_HEADINGS), tool_calls=[]
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
            _FakeStream([_answer_chunk("recovered")]),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    runtime = _CommittableCompactionRuntime()
    reasoner = _compaction_reasoner(
        LLMProvider(api_key="k", context_window=2000), runtime
    )

    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [
                    {"role": "user", "content": "old one"},
                    {"role": "user", "content": "current"},
                ],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "recovered"
    assert runtime.commit_count == 1
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert [str(item.get("counts")) for item in starts] == [
        f"call_ordinal=1 provider_attempt=1 provider_call_id={_provider_call_id(starts[0])}",
        f"call_ordinal=1 provider_attempt=2 provider_call_id={_provider_call_id(starts[1])}",
    ]
    call_id = _provider_call_id(starts[0])
    assert _provider_call_id(starts[1]) == call_id
    # 低层 transport：attempt 1 error 与 attempt 2 done 共享同一 call ID。
    transport_starts = _milestone_events(caplog, "tl:provider.transport.start")
    assert len(transport_starts) == 2
    assert _provider_call_id(transport_starts[0]) == call_id
    assert _provider_call_id(transport_starts[1]) == call_id
    assert (
        _counts_map(cast(str, transport_starts[0]["counts"]))["provider_attempt"] == "1"
    )
    assert (
        _counts_map(cast(str, transport_starts[1]["counts"]))["provider_attempt"] == "2"
    )
    transport_errors = _milestone_events(caplog, "tl:provider.transport.error")
    assert len(transport_errors) == 1
    assert _provider_call_id(transport_errors[0]) == call_id
    transport_done = _milestone_events(caplog, "tl:provider.transport.done")
    assert len(transport_done) == 1
    assert _provider_call_id(transport_done[0]) == call_id
    http_errors = _milestone_events(caplog, "tl:provider.http.error")
    assert len(http_errors) == 1
    assert _provider_call_id(http_errors[0]) == call_id
    # 强制压缩摘要 nonstream：与所属 compaction/call 的 provider_call_id 一致；
    # 摘要不是业务 retry，attempt 明确为 0。
    nonstream_starts = _milestone_events(caplog, "tl:provider.nonstream.start")
    assert len(nonstream_starts) == 1
    nonstream_counts = _counts_map(cast(str, nonstream_starts[0]["counts"]))
    assert nonstream_counts["provider_call_id"] == call_id
    assert nonstream_counts["provider_operation"] == "compaction_summary"
    assert nonstream_counts["provider_attempt"] == "0"
    nonstream_done = _milestone_events(caplog, "tl:provider.nonstream.done")
    assert len(nonstream_done) == 1
    done_counts = _counts_map(cast(str, nonstream_done[0]["counts"]))
    assert done_counts["provider_operation"] == "compaction_summary"
    assert done_counts["provider_attempt"] == "0"
    assert _provider_call_id(nonstream_done[0]) == call_id
    # 摘要结束后业务 stream operation 恢复 business，attempt=2。
    for entry in transport_starts[1:]:
        counts = _counts_map(cast(str, entry["counts"]))
        assert counts["provider_operation"] == "business"
    assert not _milestone_events(caplog, "tl:provider.call.error")
    assert not _milestone_events(caplog, "tl:provider.call.cancelled")


def test_initial_compaction_summary_nonstream_carries_compaction_operation(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """compaction.prepare 期间的摘要 nonstream：operation=compaction_summary +
    attempt=0，provider_call_id 与所属 compaction/call 一致；done 后业务
    stream 恢复 business + attempt=1。"""
    fake = _FakeClient(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="\n".join(SUMMARY_HEADINGS), tool_calls=[]
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
            _FakeStream([_answer_chunk("final")]),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    runtime = _CommittableCompactionRuntime()
    reasoner = _compaction_reasoner(
        LLMProvider(api_key="k", context_window=2000), runtime
    )

    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [
                    {"role": "user", "content": "old one"},
                    {"role": "user", "content": "x" * 5000},
                    {"role": "user", "content": "current"},
                ],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "final"
    assert runtime.commit_count == 1
    prepares = _milestone_events(caplog, "tl:compaction.prepare.done")
    assert len(prepares) == 1
    call_id = _provider_call_id(prepares[0])
    starts = _milestone_events(caplog, "tl:provider.call.start")
    assert len(starts) == 1
    assert _provider_call_id(starts[0]) == call_id
    # compaction.prepare 期间 nonstream start/done：operation=compaction_summary，
    # attempt=0（摘要不是业务 attempt，业务 attempt 1 在摘要 done 后才开始）。
    nonstream_starts = _milestone_events(caplog, "tl:provider.nonstream.start")
    assert len(nonstream_starts) == 1
    nonstream_start_counts = _counts_map(cast(str, nonstream_starts[0]["counts"]))
    assert nonstream_start_counts["provider_call_id"] == call_id
    assert nonstream_start_counts["provider_operation"] == "compaction_summary"
    assert nonstream_start_counts["provider_attempt"] == "0"
    nonstream_done = _milestone_events(caplog, "tl:provider.nonstream.done")
    assert len(nonstream_done) == 1
    nonstream_done_counts = _counts_map(cast(str, nonstream_done[0]["counts"]))
    assert _provider_call_id(nonstream_done[0]) == call_id
    assert nonstream_done_counts["provider_operation"] == "compaction_summary"
    assert nonstream_done_counts["provider_attempt"] == "0"
    # nonstream 总 span 自己生成唯一 span_id，与业务 transport span 不同。
    assert "span_id=" in cast(str, nonstream_starts[0]["counts"])
    transport_starts = _milestone_events(caplog, "tl:provider.transport.start")
    assert len(transport_starts) == 1
    transport_counts = _counts_map(cast(str, transport_starts[0]["counts"]))
    assert transport_counts["provider_call_id"] == call_id
    assert transport_counts["provider_operation"] == "business"
    assert transport_counts["provider_attempt"] == "1"
    assert nonstream_start_counts["span_id"] != transport_counts["span_id"]
    assert not _milestone_events(caplog, "tl:provider.nonstream.error")
    assert not _milestone_events(caplog, "tl:provider.nonstream.cancelled")


def test_business_nonstream_marked_business(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """预算收尾总结走 _call_provider 非流式业务请求：nonstream 标 business，
    不因 disable_thinking/model 被误标 compaction_summary。"""
    fake = _FakeClient(
        [
            _FakeStream([_tool_chunk()]),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="先停在这里，保留当前进度", tool_calls=[]
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    tools = ToolRegistry()
    tools.register(_DummyTool(), always_on=True)
    reasoner = _build_reasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=cast(Any, LLMProvider(api_key="k")),
                light_provider=cast(Any, LLMProvider(api_key="k")),
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=1, max_tokens=512),
        tools=tools,
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
    )

    with caplog.at_level(logging.INFO):
        result = asyncio.run(
            _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "hi"}],
                on_content_delta=_stream_delta_sink,
            )
        )

    assert result.reply == "先停在这里，保留当前进度"
    nonstream_starts = _milestone_events(caplog, "tl:provider.nonstream.start")
    assert len(nonstream_starts) == 1
    counts = _counts_map(cast(str, nonstream_starts[0]["counts"]))
    assert counts["provider_operation"] == "business"
    nonstream_done = _milestone_events(caplog, "tl:provider.nonstream.done")
    assert len(nonstream_done) == 1
    assert (
        _counts_map(cast(str, nonstream_done[0]["counts"]))["provider_operation"]
        == "business"
    )
    assert not _milestone_events(caplog, "tl:provider.nonstream.error")
    assert not _milestone_events(caplog, "tl:provider.nonstream.cancelled")


def test_provider_neutral_identity_not_leaked_after_error(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_call_provider 业务异常终态与 compaction summary 异常终态后，中性
    ContextVar 均精确 reset，不跨 turn/task 泄漏。"""

    async def _drive_summary_error() -> tuple[str, int, str]:
        fake = _FakeClient([RuntimeError("summary exploded")])
        monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
        runtime = _CommittableCompactionRuntime()
        reasoner = _compaction_reasoner(
            LLMProvider(api_key="k", context_window=2000), runtime
        )
        try:
            await _run_with_compaction_gate(
                reasoner,
                [
                    {"role": "user", "content": "old one"},
                    {"role": "user", "content": "x" * 5000},
                    {"role": "user", "content": "current"},
                ],
            )
            raise AssertionError("expected ContextCompactionError")
        except ContextCompactionError:
            pass
        return (
            current_provider_call_id.get(),
            current_provider_attempt.get(),
            current_provider_operation.get(),
        )

    async def _drive_business_error() -> tuple[str, int, str]:
        reasoner = _single_tool_round_reasoner(_FailingProvider())
        try:
            await _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "boom"}],
            )
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
        return (
            current_provider_call_id.get(),
            current_provider_attempt.get(),
            current_provider_operation.get(),
        )

    with caplog.at_level(logging.INFO):
        summary_reset = asyncio.run(_drive_summary_error())
        business_reset = asyncio.run(_drive_business_error())

    assert summary_reset == ("", 0, "")
    assert business_reset == ("", 0, "")
    # 摘要异常本身按 attempt=0 + compaction_summary 标记，验证标签在失败瞬间成立。
    nonstream_errors = _milestone_events(caplog, "tl:provider.nonstream.error")
    assert len(nonstream_errors) == 1
    error_counts = _counts_map(cast(str, nonstream_errors[0]["counts"]))
    assert error_counts["provider_operation"] == "compaction_summary"
    assert error_counts["provider_attempt"] == "0"
    assert current_provider_call_id.get() == ""
    assert current_provider_attempt.get() == 0
    assert current_provider_operation.get() == ""


def test_provider_neutral_identity_not_leaked_after_cancel(
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_call_provider 业务取消终态与 compaction summary 取消终态后，中性
    ContextVar 均精确 reset。"""

    async def _drive_summary_cancel() -> tuple[str, int, str]:
        fake = _FakeClient([asyncio.CancelledError("summary cancelled")])
        monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
        runtime = _CommittableCompactionRuntime()
        reasoner = _compaction_reasoner(
            LLMProvider(api_key="k", context_window=2000), runtime
        )
        try:
            await _run_with_compaction_gate(
                reasoner,
                [
                    {"role": "user", "content": "old one"},
                    {"role": "user", "content": "x" * 5000},
                    {"role": "user", "content": "current"},
                ],
            )
            raise AssertionError("expected CancelledError")
        except asyncio.CancelledError:
            pass
        return (
            current_provider_call_id.get(),
            current_provider_attempt.get(),
            current_provider_operation.get(),
        )

    async def _drive_business_cancel() -> tuple[str, int, str]:
        reasoner = _single_tool_round_reasoner(_CancelledProvider())
        try:
            await _run_with_compaction_gate(
                reasoner,
                [{"role": "user", "content": "boom"}],
            )
            raise AssertionError("expected CancelledError")
        except asyncio.CancelledError:
            pass
        return (
            current_provider_call_id.get(),
            current_provider_attempt.get(),
            current_provider_operation.get(),
        )

    with caplog.at_level(logging.INFO):
        summary_reset = asyncio.run(_drive_summary_cancel())
        business_reset = asyncio.run(_drive_business_cancel())

    assert summary_reset == ("", 0, "")
    assert business_reset == ("", 0, "")
    # 摘要取消本身按 attempt=0 + compaction_summary 标记，验证标签在取消瞬间成立。
    nonstream_cancelled = _milestone_events(caplog, "tl:provider.nonstream.cancelled")
    assert len(nonstream_cancelled) == 1
    cancelled_counts = _counts_map(cast(str, nonstream_cancelled[0]["counts"]))
    assert cancelled_counts["provider_operation"] == "compaction_summary"
    assert cancelled_counts["provider_attempt"] == "0"
    assert current_provider_call_id.get() == ""
    assert current_provider_attempt.get() == 0
    assert current_provider_operation.get() == ""
