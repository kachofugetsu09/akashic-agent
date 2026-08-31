"""Shared fixtures for private reasoner calls that must install the compaction gate."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, cast

from agent.core.passive_turn import DefaultReasoner, _PassThroughTurn
from agent.core.runtime_support import SessionLike
from plugins.compaction.engine import ContextPayloadSegments
from agent.plugin_composition import ModelRole, ProviderTurnInput
from agent.core.passive_turn import build_turn_injection_prompt
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)
from plugins.compaction.runtime import CompactionProjection
from session.manager import Session
from session.store import CompactionHead
from tests.model_plugin_fakes import BoundChatModelFake


class TestCompactionRuntime:
    """Expose an empty ledger projection for isolated direct-call tests."""

    async def projection(
        self,
        session: SessionLike,
        *,
        prefix: list[dict[str, Any]],
        current_anchor: list[dict[str, Any]],
        pending: list[dict[str, Any]],
    ) -> CompactionProjection:
        _ = session, prefix, current_anchor, pending
        return CompactionProjection(
            segments=ContextPayloadSegments(
                prefix=(),
                committed_units=(),
                current_anchor=(),
                pending=(),
            ),
            active=None,
            head=CompactionHead(
                session_key=session.key,
                parent_generation=0,
                next_generation=1,
            ),
        )

    async def recover_pending(self, session: SessionLike) -> None:
        _ = session

    async def commit_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError("direct-call fixture unexpectedly attempted compaction commit")

    def checkpoint_suppresses_post_commit(self, _checkpoint: object) -> bool:
        return False


def install_test_projection(
    reasoner: DefaultReasoner,
    runtime: object | None = None,
) -> object:
    """Bind the source-neutral pass-through service to a direct-call fixture."""

    _ = runtime

    class _Projection:
        async def open_turn(self, input: ProviderTurnInput) -> _PassThroughTurn:
            return _PassThroughTurn(
                [
                    dict(message)
                    for unit in input.history_units
                    for message in unit.messages()
                ]
            )

    projection = _Projection()
    reasoner._provider_request_projection = lambda: projection  # type: ignore[method-assign]
    return projection


def _session(key: str) -> Session:
    return Session(
        key=key,
        created_at=datetime(2026, 8, 8, tzinfo=UTC),
        messages=[],
        last_consolidated=0,
    )


async def run_reasoner_with_compaction_gate(
    reasoner: DefaultReasoner,
    initial_messages: list[dict[str, Any]],
    *,
    agent_model: BoundChatModelFake,
    fallback_model: BoundChatModelFake | None = None,
    session_key: str = "test:direct",
    runner: Callable[..., Awaitable[Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Run a direct reasoner fixture with one complete payload gate."""

    runtime = getattr(reasoner, "_test_request_projection", None)
    if runtime is None:
        runtime = install_test_projection(reasoner)
        reasoner._test_request_projection = runtime

    payload = [dict(message) for message in initial_messages]
    if not payload or payload[0].get("role") != "system":
        payload.insert(0, {"role": "system", "content": "test context"})
    session = _session(session_key)
    projection = await runtime.open_turn(
        ProviderTurnInput(
            session_key=session.key,
            session_created_at=session.created_at.isoformat(),
            history_units=(),
        )
    )
    state = reasoner._build_request_state(
        agent_model=agent_model,
        fallback_model=fallback_model or agent_model,
        projection=projection,
        initial_messages=payload,
        history_count=0,
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="direct",
    )
    invoke = runner or reasoner.run
    kwargs.setdefault("agent_model", agent_model)
    return await invoke(payload, request_state=state, **kwargs)


async def run_test_agent_loop(
    loop: Any,
    provider: Any,
    initial_messages: list[dict[str, Any]],
    *,
    request_time: datetime | None = None,
    preloaded_tools: set[str] | None = None,
) -> tuple[str, list[str], list[dict[str, Any]], set[str] | None, str | None]:
    """Run the narrow Reasoner seam with explicit public model bindings."""

    reasoner = loop._reasoner
    model = "test-model"
    agent_model = BoundChatModelFake(provider, model=model)
    fallback_model = BoundChatModelFake(
        provider,
        model=model,
        role=ModelRole.DEFAULT,
    )
    reasoner._test_request_projection = install_test_projection(reasoner)
    visible = preloaded_tools if loop._tool_search_enabled else None
    hint = build_turn_injection_prompt(
        tools=loop.tools,
        tool_search_enabled=loop._tool_search_enabled,
        visible_names=visible,
    )
    payload = list(initial_messages)
    if hint:
        hint_message = build_context_frame_message(
            build_context_frame_content(
                [PromptSectionRender("turn_injection", hint, False)]
            )
        )
        if payload and payload[-1].get("role") == "user":
            payload = [*payload[:-1], hint_message, payload[-1]]
        else:
            payload.append(hint_message)
    result = await run_reasoner_with_compaction_gate(
        reasoner,
        payload,
        agent_model=agent_model,
        fallback_model=fallback_model,
        session_key="test:loop",
        request_time=request_time,
        preloaded_tools=preloaded_tools,
        preflight_injected=True,
    )
    return (
        result.reply,
        result.tools_used,
        result.tool_chain,
        result.visible_names,
        result.thinking,
    )
