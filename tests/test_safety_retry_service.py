import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agent.core.reasoner import DefaultReasoner
from agent.core.runtime_support import ToolDiscoveryState
from agent.core.runtime_support import LLMServices
from agent.core.types import ReasonerResult
from agent.looping.ports import LLMConfig
from agent.provider import ContentSafetyError, ContextLengthError


def _stub_runtime_guard_context(*, preflight_prompt: str | None = None) -> dict[str, str]:
    if not preflight_prompt:
        return {}
    return {"preflight": preflight_prompt}


def _msg():
    return SimpleNamespace(
        content="hello",
        media=[],
        channel="cli",
        chat_id="1",
        timestamp=datetime.now(timezone.utc),
    )


def _session():
    return SimpleNamespace(
        key="s:1",
        messages=[{"role": "user", "content": str(i)} for i in range(6)],
        get_history=lambda max_messages: [{"role": "user", "content": str(i)} for i in range(6)],
        last_consolidated=3,
    )


def _make_reasoner(*, discovery: ToolDiscoveryState, tool_search_enabled: bool):
    return DefaultReasoner(
        llm=LLMServices(provider=SimpleNamespace(chat=AsyncMock()), light_provider=SimpleNamespace()),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=256),
        tools=SimpleNamespace(get_always_on_names=lambda: {"always"}, get_schemas=lambda names=None: []),
        discovery=discovery,
        memory_port=SimpleNamespace(),
        tool_search_enabled=tool_search_enabled,
        memory_window=10,
        context=SimpleNamespace(
            build_messages=lambda **kwargs: kwargs["history"] + [{"role": "user"}],
            build_runtime_guard_context=_stub_runtime_guard_context,
        ),
        session_manager=SimpleNamespace(save_async=AsyncMock()),
    )


def test_reasoner_run_turn_retries_and_updates_discovery():
    discovery = ToolDiscoveryState()
    discovery._unlocked = {"s:1": OrderedDict({"old": None})}
    reasoner = _make_reasoner(discovery=discovery, tool_search_enabled=True)
    reasoner.run = AsyncMock(
        side_effect=[
            ContentSafetyError("blocked"),
            ReasonerResult(
                reply="ok",
                metadata={"tools_used": ["tool_search", "x"], "tool_chain": []},
            ),
        ]
    )

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=_session()))

    assert result.reply == "ok"
    assert result.tools_used == ["tool_search", "x"]
    assert result.tool_chain == []
    assert result.thinking is None
    assert result.context_retry["selected_plan"] == "trim_skills_catalog"
    assert "x" in discovery._unlocked["s:1"]


def test_reasoner_run_turn_context_length_all_fail_returns_fallback():
    reasoner = _make_reasoner(discovery=ToolDiscoveryState(), tool_search_enabled=False)
    reasoner.run = AsyncMock(side_effect=[ContextLengthError("long")] * 7)

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=_session()))
    assert "上下文过长" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []


def test_reasoner_run_turn_context_length_trims_dynamic_sections_before_history():
    calls: list[dict] = []

    def _build_messages(**kwargs):
        calls.append(
            {
                "history_len": len(kwargs["history"]),
                "disabled_sections": set(kwargs.get("disabled_sections") or set()),
            }
        )
        return kwargs["history"] + [{"role": "user"}]

    discovery = ToolDiscoveryState()
    reasoner = DefaultReasoner(
        llm=LLMServices(provider=SimpleNamespace(chat=AsyncMock()), light_provider=SimpleNamespace()),
        llm_config=LLMConfig(model="m", max_iterations=4, max_tokens=256),
        tools=SimpleNamespace(get_always_on_names=lambda: {"always"}, get_schemas=lambda names=None: []),
        discovery=discovery,
        memory_port=SimpleNamespace(),
        tool_search_enabled=False,
        memory_window=10,
        context=SimpleNamespace(
            build_messages=_build_messages,
            build_runtime_guard_context=_stub_runtime_guard_context,
        ),
        session_manager=SimpleNamespace(save_async=AsyncMock()),
    )
    reasoner.run = AsyncMock(
        side_effect=[
            ContextLengthError("long"),
            ReasonerResult(
                reply="ok",
                metadata={"tools_used": [], "tool_chain": []},
            ),
        ]
    )

    result = asyncio.run(reasoner.run_turn(msg=_msg(), session=_session()))
    assert result.reply == "ok"
    assert result.tools_used == []
    assert result.tool_chain == []
    assert calls[0]["history_len"] == 6
    assert calls[0]["disabled_sections"] == set()
    assert calls[1]["history_len"] == 6
    assert calls[1]["disabled_sections"] == {"skills_catalog"}
