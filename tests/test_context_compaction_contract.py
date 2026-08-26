from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent.model_runtime.context_compaction import (
    CommittedContextUnit,
    ContextCompactionError,
    ContextCompactor,
    ContextPayloadSegments,
    _summary_output_limit,
)
from agent.model_runtime.types import LLMResponse, ModelUsage
from agent.provider import LLMProvider
from agent.tool_runtime import append_tool_result


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


class _Provider(LLMProvider):
    context_window: int = 0
    runtime_id: str = ""

    def __init__(
        self,
        *,
        context_window: int = 100_000,
        fail: bool = False,
        runtime_id: str = "main",
    ) -> None:
        self.context_window = context_window
        self.fail = fail
        self.runtime_id = runtime_id
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


def _execution_batch(
    call_id: str,
    *,
    name: str,
    arguments: dict[str, object],
    result: dict[str, object],
) -> tuple[dict[str, Any], ...]:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tokens": 10,
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
        }
    ]
    append_tool_result(
        messages,
        tool_call_id=call_id,
        content=json.dumps(result),
        tool_name=name,
        execution_status="success",
    )
    messages[-1]["tokens"] = 10
    return tuple(messages)


def _run(coro):
    return asyncio.run(coro)


def _call_message_content(call: dict[str, object]) -> str:
    """Read a provider fixture call after validating its JSON-like shape."""

    raw_messages = call.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise AssertionError("provider call must contain a non-empty messages list")
    first = raw_messages[0]
    if not isinstance(first, dict) or "content" not in first:
        raise AssertionError("provider call first message must contain content")
    return str(first["content"])


def _call_int(call: dict[str, object], field: str) -> int:
    """Read one integer request field from a provider fixture call."""

    value = call.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"provider call {field} must be an integer")
    return value


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


def test_tail_below_twenty_thousand_tokens_has_no_legal_cut() -> None:
    units = (_unit(1, 5_000), _unit(2, 5_000))
    compactor = ContextCompactor(
        provider=_Provider(),
        model="m",
        scope_id="s",
        payload_segments=ContextPayloadSegments(
            prefix=(),
            committed_units=units,
            current_anchor=(),
        ),
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=20_000,
    )

    with pytest.raises(
        ContextCompactionError,
        match="no_valid_cut_before_keep_recent_target",
    ):
        compactor._select_units(list(units))


class _UsageProvider(_Provider):
    def __init__(self) -> None:
        super().__init__(context_window=100)
        self._summary_index = 0

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        self._summary_index += 1
        return LLMResponse(
            content=_SUMMARY,
            usage=ModelUsage(
                input_tokens=10 * self._summary_index,
                output_tokens=self._summary_index,
                request_count=1,
                covered_request_count=1,
            ),
        )


def test_committed_and_temporary_summary_usage_are_aggregated() -> None:
    active_batch = (
        {"role": "assistant", "tool_calls": [{"id": "c1"}], "tokens": 15},
        {"role": "tool", "tool_call_id": "c1", "content": "r", "tokens": 15},
    )
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(_unit(1, 30), _unit(2, 30)),
        current_anchor=({"role": "user", "content": "q", "tokens": 1},),
        active_batches=(active_batch, active_batch),
    )
    provider = _UsageProvider()
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=10,
        next_generation=1,
        keep_recent_tokens=20,
    )

    result = _run(
        compactor.prepare(
            segments.flatten(),
            pending_start=7,
            tools=[],
            force=True,
        )
    )

    assert len(provider.calls) == 2
    assert result.summary_usage is not None
    assert result.summary_usage.input_tokens == 30
    assert result.summary_usage.output_tokens == 3
    assert result.summary_usage.request_count == 2
    assert result.checkpoint is not None
    assert result.checkpoint.generation == 1
    assert result.checkpoint.summary_usage is not None
    assert result.checkpoint.summary_usage.input_tokens == 10
    assert result.checkpoint.summary_usage.output_tokens == 1


def test_single_interaction_remains_atomic_after_closed_tool_batches() -> None:
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
        ("m1", "m2", "m3", "m4", "m5", "m6"),
    ]


def test_live_shell_execution_blocks_cut_until_terminal_evidence_arrives() -> None:
    live = _execution_batch(
        "shell-call",
        name="shell",
        arguments={"command": "python train.py"},
        result={"process_status": "running", "execution_id": 4201},
    )
    closed = (
        {"role": "assistant", "content": "", "tokens": 10, "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "done", "tokens": 10},
    )
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(),
        current_anchor=({"role": "user", "content": "finish training", "tokens": 1},),
        active_batches=(live, closed, closed),
    )
    provider = _Provider(context_window=100)
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="shell-session",
        payload_segments=segments,
        max_output_tokens=10,
        next_generation=1,
        keep_recent_tokens=20,
    )
    messages = segments.flatten()

    with pytest.raises(ContextCompactionError, match="no_closed_prefix"):
        _run(compactor.prepare(messages, pending_start=7, tools=[], force=True))

    terminal = _execution_batch(
        "stdin-call",
        name="write_stdin",
        arguments={"execution_id": 4201},
        result={"process_status": "succeeded", "exit_code": 0},
    )
    batch_start = len(messages)
    messages.extend(terminal)
    compactor.record_completed_batch(messages, batch_start=batch_start)

    completed = _run(
        compactor.prepare(
            messages,
            pending_start=compactor.pending_start,
            tools=[],
            force=True,
        )
    )

    assert completed.compacted is True
    assert provider.calls
    summary_input = _call_message_content(provider.calls[0])
    assert "python train.py" in summary_input
    assert "4201" in summary_input
    assert "succeeded" in str(messages[-1]["content"])


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
    provider = _Provider(context_window=1_000)
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
    )
    messages = segments.flatten()
    result = _run(compactor.prepare(messages, pending_start=8, tools=[], force=True))

    compaction_blocks = [
        message
        for message in messages
        if message.get("role") == "system"
        and "<session-context-compaction>" in str(message.get("content"))
    ]
    assert len(compaction_blocks) == 1
    assert "ACTIVE_SHOULD_NOT_PERSIST" not in str(compaction_blocks[0])
    assert "ACTIVE_SHOULD_NOT_PERSIST" in _call_message_content(provider.calls[-1])
    assert result.pending_start == 5
    assert result.checkpoint.committable
    assert "ACTIVE_SHOULD_NOT_PERSIST" not in result.checkpoint.summary
    assert "ACTIVE_SHOULD_NOT_PERSIST" not in str(result.checkpoint.retained_tail)


def test_summary_uses_current_once_then_distinct_fallback_once_with_own_budget() -> None:
    current = _Provider(context_window=500, fail=True, runtime_id="agent")
    fallback = _Provider(context_window=2_000, runtime_id="main")
    current.model = "selected-model"
    fallback.model = "default-model"
    unit = _unit(1, 100)
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(unit, _unit(2, 100)),
        current_anchor=({"role": "user", "content": "q", "tokens": 1},),
    )
    compactor = ContextCompactor(
        provider=current,
        model="startup-current",
        scope_id="s",
        payload_segments=segments,
        max_output_tokens=100,
        next_generation=1,
        fallback_provider=fallback,
        fallback_model="startup-default",
        keep_recent_tokens=1,
    )
    result = _run(
        compactor.prepare(segments.flatten(), pending_start=3, tools=[], force=True)
    )

    assert len(current.calls) == 1
    assert len(fallback.calls) == 1
    assert current.calls[0]["model"] == "selected-model"
    assert fallback.calls[0]["model"] == "default-model"
    assert _call_int(fallback.calls[0], "max_tokens") <= fallback.context_window
    assert result.checkpoint is not None
    assert result.checkpoint.model_runtime_id == "main"
    assert result.checkpoint.model == "default-model"


def test_summary_checkpoint_records_selected_provider_model_and_runtime() -> None:
    provider = _Provider(runtime_id="selected")
    provider.model = "selected-model"
    compactor = ContextCompactor(
        provider=provider,
        model="startup-model",
        scope_id="selected-runtime",
        payload_segments=ContextPayloadSegments(
            prefix=(),
            committed_units=(_unit(1, 100), _unit(2, 100)),
            current_anchor=(),
        ),
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
    )

    result = _run(
        compactor.prepare(
            compactor._segments.flatten(),
            pending_start=2,
            tools=[],
            force=True,
        )
    )

    assert provider.calls[0]["model"] == "selected-model"
    assert result.checkpoint is not None
    assert result.checkpoint.model_runtime_id == "selected"
    assert result.checkpoint.model == "selected-model"


def test_summary_does_not_duplicate_same_selected_main_provider() -> None:
    provider = _Provider(runtime_id="main")
    compactor = ContextCompactor(
        provider=provider,
        model="main-model",
        scope_id="same-provider",
        payload_segments=ContextPayloadSegments(
            prefix=(),
            committed_units=(_unit(1, 100), _unit(2, 100)),
            current_anchor=(),
        ),
        max_output_tokens=100,
        next_generation=1,
        fallback_provider=provider,
        fallback_model="main-model",
        keep_recent_tokens=1,
    )

    _run(compactor.prepare(compactor._segments.flatten(), pending_start=2, tools=[], force=True))

    assert len(provider.calls) == 1


def test_logical_interaction_inputs_only_enter_temporary_summary() -> None:
    active = (
        {
            "role": "user",
            "content": "U1",
            "tokens": 2,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1"}],
            "tokens": 2,
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "result-1",
            "tokens": 2,
        },
    )
    active_tail = (
        {
            "role": "user",
            "content": "U2",
            "tokens": 2,
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c2"}],
            "tokens": 2,
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "content": "result-2",
            "tokens": 2,
        },
    )
    current_query = {
        "logical_interaction_inputs": ["U1", "U2", "U3"],
    }
    temporary_provider = _Provider()
    temporary_segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(),
        current_anchor=(),
        active_batches=(active, active_tail),
        pending=({"role": "user", "content": "U3", "tokens": 1},),
    )
    temporary = ContextCompactor(
        provider=temporary_provider,
        model="m",
        scope_id="temporary-interaction",
        current_query=current_query,
        payload_segments=temporary_segments,
        max_output_tokens=100,
        keep_recent_tokens=1,
    )
    temporary_messages = temporary_segments.flatten()
    _run(
        temporary.prepare(
            temporary_messages,
            pending_start=6,
            tools=[],
            force=True,
        )
    )
    temporary_prompt = _call_message_content(temporary_provider.calls[0])
    assert all(value in temporary_prompt for value in ("U1", "U2", "U3"))

    committed_provider = _Provider()
    committed_segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(
            _unit(1, 2, prefix="history-"),
            _unit(2, 2, prefix="history-"),
        ),
        current_anchor=(),
    )
    committed = ContextCompactor(
        provider=committed_provider,
        model="m",
        scope_id="committed-interaction",
        current_query=current_query,
        payload_segments=committed_segments,
        max_output_tokens=100,
        next_generation=1,
        keep_recent_tokens=1,
    )
    committed_messages = committed_segments.flatten()
    _run(
        committed.prepare(
            committed_messages,
            pending_start=2,
            tools=[],
            force=True,
        )
    )
    committed_prompt = _call_message_content(committed_provider.calls[0])
    assert all(value not in committed_prompt for value in ("U1", "U2", "U3"))


def test_summary_output_limit_keeps_strict_input_boundary() -> None:
    summary_input = [{"role": "user", "content": "summary", "tokens": 1}]
    assert _summary_output_limit(_Provider(context_window=8_193), summary_input) == 8_191
    assert _summary_output_limit(_Provider(context_window=8_192), summary_input) == 8_190
    with pytest.raises(ContextCompactionError, match="summary_input_exceeds_window"):
        _summary_output_limit(_Provider(context_window=2), summary_input)

    capped = _Provider(context_window=8_193)
    capped.max_output_tokens = 123
    assert _summary_output_limit(capped, summary_input) == 123


def test_summary_reduces_oversized_history_in_bounded_unit_chunks() -> None:
    class _ChunkProvider(_Provider):
        def __init__(self) -> None:
            super().__init__(context_window=10)

        def estimate_context_tokens(self, messages, tools):
            content = str(messages[0].get("content", "")) if messages else ""
            units = sum(content.count(f'"content":"u{seq}"') for seq in range(1, 5))
            previous = 2 if "[Previous compaction summary]" in content else 0
            return 1 + previous + units * 3

        def estimate_appended_message_tokens(self, messages):
            return sum(int(message.get("tokens", 1)) for message in messages)

    provider = _ChunkProvider()
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=tuple(_unit(seq, 3) for seq in range(1, 5)),
        current_anchor=(),
    )
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="chunked-summary",
        payload_segments=segments,
        max_output_tokens=1,
        next_generation=1,
        keep_recent_tokens=1,
    )

    result = _run(
        compactor.prepare(segments.flatten(), pending_start=4, tools=[], force=True)
    )

    assert result.compacted
    assert len(provider.calls) == 2
    assert "[Previous compaction summary]" in _call_message_content(provider.calls[1])
    assert all(_call_int(call, "max_tokens") > 0 for call in provider.calls)


def test_request_output_limit_moves_hard_edge_for_each_payload() -> None:
    class _BoundaryProvider(_Provider):
        def __init__(self) -> None:
            super().__init__(context_window=100)

        def estimate_context_tokens(self, messages, tools):
            if any(
                "<session-context-compaction>" in str(message.get("content", ""))
                for message in messages
            ):
                return 1
            return 60

        def estimate_appended_message_tokens(self, messages):
            return 1

    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(_unit(1, 1), _unit(2, 1)),
        current_anchor=(),
    )

    below_edge = ContextCompactor(
        provider=_BoundaryProvider(),
        model="m",
        scope_id="hard-edge-below",
        payload_segments=segments,
        max_output_tokens=20,
        next_generation=1,
        keep_recent_tokens=1,
    )
    below = _run(
        below_edge.prepare(
            below_edge._segments.flatten(),
            pending_start=2,
            tools=[],
            max_output_tokens=20,
        )
    )
    assert not below.compacted

    above_edge = ContextCompactor(
        provider=_BoundaryProvider(),
        model="m",
        scope_id="hard-edge-above",
        payload_segments=segments,
        max_output_tokens=20,
        next_generation=1,
        keep_recent_tokens=1,
    )
    above = _run(
        above_edge.prepare(
            above_edge._segments.flatten(),
            pending_start=2,
            tools=[],
            max_output_tokens=50,
        )
    )
    assert above.compacted


def test_soft_limit_uses_fixed_context_window_ratio() -> None:
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(_unit(1, 36), _unit(2, 36)),
        current_anchor=({"role": "user", "content": "current", "tokens": 2},),
    )
    compactor = ContextCompactor(
        provider=_Provider(context_window=100),
        model="m",
        scope_id="fixed-soft-limit",
        payload_segments=segments,
        max_output_tokens=0,
        next_generation=1,
        keep_recent_tokens=1,
    )

    result = _run(
        compactor.prepare(
            segments.flatten(),
            pending_start=3,
            tools=[],
        )
    )

    assert result.compacted


def test_unknown_context_window_estimates_but_never_compacts() -> None:
    provider = _Provider(context_window=0)
    segments = ContextPayloadSegments(
        prefix=(),
        committed_units=(_unit(1, 100), _unit(2, 100)),
        current_anchor=({"role": "user", "content": "current", "tokens": 2},),
    )
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="unknown-window",
        payload_segments=segments,
        max_output_tokens=512,
        next_generation=1,
        keep_recent_tokens=1,
    )

    result = _run(
        compactor.prepare(
            segments.flatten(),
            pending_start=3,
            tools=[{"type": "function", "function": {"name": "tool"}}],
            force=True,
        )
    )

    assert result.compacted is False
    assert result.checkpoint is None
    assert result.estimated_tokens > 0
    assert provider.calls == []


def test_same_turn_temporary_summary_replaces_previous_projection() -> None:
    class _SentinelProvider(_Provider):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            marker = f"C{len(self.calls)}"
            return LLMResponse(content=_SUMMARY.replace("goal", marker))

    active = (
        {"role": "assistant", "tool_calls": [{"id": "a"}], "tokens": 2},
        {"role": "tool", "tool_call_id": "a", "content": "active", "tokens": 2},
    )
    initial_segments = ContextPayloadSegments(
        prefix=(),
        committed_units=tuple(_unit(index, 2, prefix="old-") for index in range(1, 5)),
        current_anchor=(),
        active_batches=(active, active),
    )
    provider = _SentinelProvider()
    compactor = ContextCompactor(
        provider=provider,
        model="m",
        scope_id="same-turn",
        payload_segments=initial_segments,
        max_output_tokens=100,
        ledger_parent_generation=0,
        next_generation=1,
        keep_recent_tokens=1,
    )
    first_messages = initial_segments.flatten()
    first = _run(
        compactor.prepare(
            first_messages,
            pending_start=len(first_messages),
            tools=[],
            force=True,
        )
    )
    assert first.checkpoint is not None
    assert first.checkpoint.generation == 1
    assert len(provider.calls) == 2

    compactor.acknowledge_committed_checkpoint(1)
    next_units = tuple(
        _unit(index, 2, prefix="next-") for index in range(10, 12)
    )
    compactor._committed_units = list(next_units)
    compactor._completed_batches = []
    compactor._segments = ContextPayloadSegments(
        prefix=compactor._segments.prefix,
        committed_units=next_units,
        current_anchor=(),
        temporary_summary=compactor._segments.temporary_summary,
    )
    second_messages = compactor._segments.flatten()
    second = _run(
        compactor.prepare(
            second_messages,
            pending_start=len(second_messages),
            tools=[],
            force=True,
        )
    )
    assert second.checkpoint is not None
    assert second.checkpoint.generation == 2
    assert len(provider.calls) == 4
    temporary_prompt = _call_message_content(provider.calls[3])
    assert "C3" in temporary_prompt
    assert "C2" in temporary_prompt
    blocks = [
        message
        for message in second_messages
        if message.get("role") == "system"
        and "<session-context-compaction>" in str(message.get("content"))
    ]
    assert len(blocks) == 1
    assert "C4" in str(blocks[0]["content"])
    assert "C2" not in str(blocks[0]["content"])
