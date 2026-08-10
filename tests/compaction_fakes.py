"""Shared fixtures for private reasoner calls that must install the compaction gate."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, cast

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import SessionLike
from agent.model_runtime.context_compaction import ContextPayloadSegments
from session.compaction_runtime import CompactionProjection
from session.manager import Session
from session.store import CompactionHead


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
    session_key: str = "test:direct",
    runner: Callable[..., Awaitable[Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Run a direct reasoner fixture with one complete payload gate."""

    runtime = reasoner._compaction_runtime
    if runtime is None:
        runtime = TestCompactionRuntime()
        reasoner._compaction_runtime = runtime

    payload = [dict(message) for message in initial_messages]
    if not payload or payload[0].get("role") != "system":
        payload.insert(0, {"role": "system", "content": "test context"})
    session = _session(session_key)
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
        history_count=0,
        attempt_replay=[],
        prior_tool_groups=0,
        channel="test",
        chat_id="direct",
    )
    invoke = runner or reasoner.run
    return await invoke(payload, compaction_state=state, **kwargs)


def install_compaction_gate(loop: Any) -> Any:
    """Adapt AgentLoop's private direct-call helper to the mandatory gate."""

    reasoner = loop._reasoner
    runtime = TestCompactionRuntime()
    reasoner._compaction_runtime = runtime
    original_run = reasoner.run

    async def run_with_gate(
        initial_messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        return await run_reasoner_with_compaction_gate(
            reasoner,
            initial_messages,
            session_key="test:loop",
            runner=original_run,
            **kwargs,
        )

    reasoner.run = cast(Any, run_with_gate)
    return loop
