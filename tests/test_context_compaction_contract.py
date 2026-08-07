from __future__ import annotations

import asyncio

from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextCompactor,
    ContextPayloadSegments,
)
from agent.model_runtime.types import LLMResponse


_SUMMARY = """## Goal
goal
## Constraints & Preferences
constraints
## Progress
### Done
done
### In Progress
in progress
### Blocked
blocked
## Key Decisions
decisions
## Next Steps
next
## Critical Context
critical
"""


class _Provider:
    def __init__(self, *, context_window: int = 100_000, fail: bool = False) -> None:
        self.context_window = context_window
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def estimate_context_tokens(self, messages, tools):
        return sum(int(message.get("tokens", 1)) for message in messages) + len(tools)

    def estimate_appended_message_tokens(self, messages):
        return sum(int(message.get("tokens", 1)) for message in messages)

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("summary provider unavailable")
        return LLMResponse(content=_SUMMARY)


def _unit(seq: int, token_count: int, *, prefix: str = "m") -> CommittedContextUnit:
    return CommittedContextUnit(
        source_from_seq=seq,
        consolidated_through_seq=seq,
        source_message_ids=(f"{prefix}{seq}",),
        messages=({"role": "user", "content": f"u{seq}", "tokens": token_count},),
        message_refs=((f"{prefix}{seq}", seq),),
    )


def _run(coro):
    return asyncio.run(coro)


def test_tail_crosses_twenty_thousand_tokens_and_keeps_refs() -> None:
    units = (_unit(1, 10_000), _unit(2, 15_000), _unit(3, 5_000))
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=units,
        current_anchor=({"role": "user", "content": "current", "tokens": 1},),
    )
    provider = _Provider()
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=20_000,
    )
    messages = segments.flatten()
    result = _run(compactor.prepare(messages, pending_start=4, tools=[], force=True))

    assert result.compacted
    assert [item["id"] for item in result.checkpoint.retained_tail] == ["m2", "m3"]


def test_single_interaction_splits_only_after_closed_tool_batches() -> None:
    messages = (
        {"role": "user", "content": "u", "id": "m1", "seq": 1},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}], "id": "m2", "seq": 2},
        {"role": "tool", "tool_call_id": "c1", "content": "r", "id": "m3", "seq": 3},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}], "id": "m4", "seq": 4},
        {"role": "tool", "tool_call_id": "c2", "content": "r", "id": "m5", "seq": 5},
        {"role": "assistant", "content": "done", "id": "m6", "seq": 6},
    )
    unit = CommittedContextUnit(
        source_from_seq=1,
        consolidated_through_seq=6,
        source_message_ids=tuple(f"m{i}" for i in range(1, 7)),
        messages=messages,
        message_refs=tuple((f"m{i}", i) for i in range(1, 7)),
    )
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(unit,),
        current_anchor=(),
    )
    compactor = ContextCompactor(
        provider=_Provider(),
        model="m",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=5,
    )

    candidates = compactor._candidate_units()
    assert [tuple(item.source_message_ids) for item in candidates] == [
        ("m1", "m2", "m3"),
        ("m4", "m5"),
        ("m6",),
    ]


def test_generation_comes_from_store_head_and_temporary_projection_does_not_consume_it() -> None:
    committed = _unit(1, 100)
    committed_tail = _unit(2, 100)
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(committed, committed_tail),
        current_anchor=({"role": "user", "content": "q", "tokens": 1},),
    )
    provider = _Provider()
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        ledger_parent_generation=7,
        next_generation=8,
        keep_recent_tokens=1,
    )
    result = _run(
        compactor.prepare(segments.flatten(), pending_start=3, tools=[], force=True)
    )
    assert result.checkpoint.generation == 8
    assert result.checkpoint.parent_generation == 7

    active = ContextPayloadSegments(
        prefix=(),
        committed_units=(),
        current_anchor=({"role": "user", "content": "q", "tokens": 1},),
        active_batches=(
            (
                {"role": "assistant", "tool_calls": [{"id": "c"}], "tokens": 1},
                {"role": "tool", "tool_call_id": "c", "content": "r", "tokens": 1},
            ),
            (
                {"role": "assistant", "tool_calls": [{"id": "d"}], "tokens": 1},
                {"role": "tool", "tool_call_id": "d", "content": "r", "tokens": 1},
            ),
        ),
    )
    temporary = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="s",
        payload_segments=active,
        max_output_tokens=100,
        keep_recent_tokens=1,
    )
    result = _run(
        temporary.prepare(active.flatten(), pending_start=5, tools=[], force=True)
    )
    assert not result.checkpoint.committable
    assert result.checkpoint.generation == 0


def test_mixed_segments_preserve_anchor_before_active_batches() -> None:
    active_batch = (
        {"role": "assistant", "tool_calls": [{"id": "c"}], "tokens": 200},
        {
            "role": "tool",
            "tool_call_id": "c",
            "content": "ACTIVE_SHOULD_NOT_PERSIST",
            "tokens": 200,
        },
    )
    segments = ContextPayloadSegments(
        prefix=({"role": "system", "content": "prefix", "tokens": 1},),
        committed_units=(_unit(1, 100), _unit(2, 100)),
        current_anchor=({"role": "user", "content": "anchor", "tokens": 1},),
        active_batches=(active_batch, active_batch),
        pending=({"role": "assistant", "content": "pending", "tokens": 1},),
    )
    compactor = ContextCompactor(
        provider=_Provider(context_window=1_000),
        model="m",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
    )
    messages = segments.flatten()
    result = _run(compactor.prepare(messages, pending_start=8, tools=[], force=True))

    roles = [message["role"] for message in messages]
    assert roles[:4] == ["system", "system", "user", "user"]
    assert roles[3:5] == ["user", "system"]
    assert roles[5:7] == ["assistant", "tool"]
    assert roles[-1] == "assistant"
    assert result.pending_start == 7
    assert result.checkpoint.committable
    assert "ACTIVE_SHOULD_NOT_PERSIST" not in result.checkpoint.summary
    assert "ACTIVE_SHOULD_NOT_PERSIST" not in str(result.checkpoint.retained_tail)


def test_summary_uses_current_once_then_distinct_fallback_once_with_own_budget() -> None:
    current = _Provider(context_window=500, fail=True)
    fallback = _Provider(context_window=2_000)
    unit = _unit(1, 100)
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(unit, _unit(2, 100)),
        current_anchor=({"role": "user", "content": "q", "tokens": 1},),
    )
    compactor = ContextCompactor(
        provider=current,
        model="current",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        fallback_provider=fallback,
        fallback_model="default",
        keep_recent_tokens=1,
    )
    _run(compactor.prepare(segments.flatten(), pending_start=3, tools=[], force=True))

    assert len(current.calls) == 1
    assert len(fallback.calls) == 1
    assert fallback.calls[0]["max_tokens"] <= fallback.context_window
