import asyncio
import json
from collections import OrderedDict
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.core.types import ContextRenderResult, ContextRequest, ReasonerResult
from agent.looping.ports import LLMConfig
from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextPayloadSegments,
)
from agent.provider import ContentSafetyError, ContextLengthError
from session.compaction_runtime import CompactionProjection
from session.store import CompactionHead


class _ProviderContextBudget:
    """Expose the provider budget contract required by the compaction gate."""

    context_window = 100_000
    runtime_id = "safety-retry-test"

    def estimate_context_tokens(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> int:
        return max(1, len(json.dumps([messages, tools], ensure_ascii=False)) // 3)

    def estimate_appended_message_tokens(self, messages: list[dict]) -> int:
        if not messages:
            return 0
        return max(1, len(json.dumps(messages, ensure_ascii=False)) // 3)

    async def chat(self, **_: object) -> None:
        raise AssertionError("run_turn test must not bypass the mocked reasoner.run")


class _MandatoryCompactionRuntime:
    """Project canonical test history into the mandatory compaction port."""

    async def projection(
        self,
        session: object,
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection:
        raw_history = getattr(session, "messages", None)
        if not isinstance(raw_history, list):
            raise AssertionError("test session history must be a list")
        history = [dict(message) for message in raw_history]
        message_ids = tuple(
            f"safety-retry-message-{index}" for index in range(len(history))
        )
        units: tuple[CommittedContextUnit, ...] = ()
        if history:
            units = (
                CommittedContextUnit(
                    source_from_seq=0,
                    consolidated_through_seq=len(history) - 1,
                    source_message_ids=message_ids,
                    messages=tuple(history),
                    message_refs=tuple(
                        (message_id, index)
                        for index, message_id in enumerate(message_ids)
                    ),
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
                session_key=str(getattr(session, "key", "safety-retry")),
                parent_generation=0,
                next_generation=1,
            ),
        )

    async def recover_pending(self, session: object) -> None:
        return None

    async def commit_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("test compaction gate unexpectedly attempted a commit")


def _stub_turn_injection_context(
    *, turn_injection_prompt: str | None = None
) -> dict[str, str]:
    if not turn_injection_prompt:
        return {}
    return {"turn_injection": turn_injection_prompt}


def _msg():
    return SimpleNamespace(
        content="hello",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime.now(timezone.utc),
    )


def _session():
    history = [{"role": "user", "content": str(i)} for i in range(6)]
    return SimpleNamespace(
        key="s:1",
        created_at=datetime(2026, 8, 8, tzinfo=timezone.utc),
        messages=history,
        get_history=lambda max_messages=500: [
            dict(message) for message in history
        ],
        last_consolidated=3,
    )


def _make_reasoner(
    *,
    discovery: ToolDiscoveryState,
    tool_search_enabled: bool,
    render: object | None = None,
):
    def _render(request: ContextRequest, **kwargs: object) -> ContextRenderResult:
        return ContextRenderResult(
            system_prompt="test context",
            turn_injection_context=_stub_turn_injection_context(
                turn_injection_prompt=request.turn_injection_prompt
            ),
            messages=[
                {"role": "system", "content": "test context"},
                *list(request.history),
                {"role": "user", "content": request.current_message},
            ],
            debug_breakdown=[],
        )

    provider = _ProviderContextBudget()
    return DefaultReasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=provider,
                light_provider=provider,
            ),
        ),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=256),
        tools=cast(
            Any,
            SimpleNamespace(
                get_always_on_names=lambda: {"always"},
                get_deferred_names=lambda visible=None: {
                    "builtin": [],
                    "mcp": {},
                },
                get_schemas=lambda names=None: [],
                get_tool=lambda name: None,
            ),
        ),
        discovery=discovery,
        tool_search_enabled=tool_search_enabled,
        context=cast(Any, SimpleNamespace(render=render or _render)),
        compaction_runtime=_MandatoryCompactionRuntime(),
    )


def test_reasoner_run_turn_content_safety_returns_user_error_without_retry():
    discovery = ToolDiscoveryState()
    discovery._unlocked = {"s:1": OrderedDict({"old": None})}
    reasoner = _make_reasoner(discovery=discovery, tool_search_enabled=True)
    reasoner.run = AsyncMock(side_effect=ContentSafetyError("blocked"))

    session = _session()
    original_messages = list(session.messages)
    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, session)))

    assert result.reply == "你的消息触发了安全审查，无法处理。"
    assert reasoner.run.await_count == 1
    assert result.context_retry["selected_plan"] is None
    assert result.context_retry["attempts"] == [
        {
            "name": "full_context",
            "history_window": 6,
            "disabled_sections": [],
        }
    ]
    assert "x" not in discovery._unlocked["s:1"]
    assert session.messages == original_messages
    assert session.last_consolidated == 3


def test_reasoner_run_turn_success_updates_discovery_with_full_context_plan():
    discovery = ToolDiscoveryState()
    discovery._unlocked = {"s:1": OrderedDict({"old": None})}
    reasoner = _make_reasoner(discovery=discovery, tool_search_enabled=True)
    reasoner.run = AsyncMock(
        return_value=ReasonerResult(
            reply="ok",
            metadata={"tools_used": ["tool_search", "x"], "tool_chain": []},
        )
    )

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, _session())))

    assert result.reply == "ok"
    assert result.tools_used == ["tool_search", "x"]
    assert result.tool_chain == []
    assert result.thinking is None
    assert result.context_retry["selected_plan"] == "full_context"
    assert result.context_retry["trimmed_sections"] == []
    assert result.context_retry["attempts"] == [
        {
            "name": "full_context",
            "history_window": 6,
            "disabled_sections": [],
        }
    ]
    assert "x" in discovery._unlocked["s:1"]
    assert reasoner.run.await_count == 1


def test_reasoner_run_turn_context_length_returns_final_user_error():
    reasoner = _make_reasoner(discovery=ToolDiscoveryState(), tool_search_enabled=False)
    reasoner.run = AsyncMock(side_effect=ContextLengthError("long"))

    session = _session()
    original_messages = list(session.messages)
    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, session)))

    assert "上下文过长" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []
    assert result.context_retry["selected_plan"] is None
    assert result.context_retry["attempts"] == [
        {
            "name": "full_context",
            "history_window": 6,
            "disabled_sections": [],
        }
    ]
    assert reasoner.run.await_count == 1
    assert session.messages == original_messages


def test_reasoner_run_turn_keeps_full_context_without_dynamic_or_history_trimming():
    calls: list[dict[str, object]] = []

    def _render(request: ContextRequest, **kwargs: object) -> ContextRenderResult:
        calls.append(
            {
                "history": list(request.history),
                "disabled_sections": set(request.disabled_sections or set()),
            }
        )
        return ContextRenderResult(
            system_prompt="test context",
            messages=[
                {"role": "system", "content": "test context"},
                *list(request.history),
                {"role": "user", "content": request.current_message},
            ],
        )

    reasoner = _make_reasoner(
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        render=_render,
    )
    reasoner.run = AsyncMock(side_effect=ContextLengthError("long"))
    session = _session()
    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=cast(Any, session)))

    assert "上下文过长" in str(result.reply)
    assert len(calls) == 1
    assert calls[0]["history"] == session.messages
    assert calls[0]["disabled_sections"] == set()
    assert result.context_retry["attempts"] == [
        {
            "name": "full_context",
            "history_window": len(session.messages),
            "disabled_sections": [],
        }
    ]
