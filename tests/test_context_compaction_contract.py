from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor,
    CapabilitySources,
    ContextLengthError,
    LLMResponse,
    ModelCapabilities,
    ModelRequest,
    ModelRole,
    ModelContinuation,
    RateLimitError,
)
from plugins.compaction.message_summary import (
    HEADINGS,
    SummaryError,
    _request,
    closed_groups,
    summarize,
    summary_groups,
    window_starts,
)
from plugins.context.api import ContextOverflow, Materials, Summary
from plugins.context.plugin import ContextBuilder
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from plugins.turn_projection.plugin import TurnProjection
from session.message import CallRef, ContentPart, Control, Input, Message, Output, ToolCall, ToolResult


def message(seq: int, body, source: str = "conversation") -> Message:
    return Message(str(seq), "s", seq, datetime(2026, 9, 5, tzinfo=UTC), "test", source, body)


class Projection:
    context_window = 1000
    max_tool_schemas = 10

    def __init__(self, estimate: int = 500) -> None:
        self.estimate_value = estimate
        self.seen: tuple[Message, ...] = ()
        self.continuation = ModelContinuation("model-binding", {"opaque": "kept"})

    def render(self, messages, *, after_seq, summary_reference=None, fresh=False):
        self.seen = tuple(messages)
        return ModelRequest(
            messages=tuple(
                {
                    "role": "user" if isinstance(item.body, Input) else "assistant",
                    "content": str(item.body.parts),
                }
                for item in messages
                if item.seq > after_seq and not isinstance(item.body, Control)
            ),
            continuation=self.continuation,
        )

    def estimate(self, request):
        return self.estimate_value


def summary_text() -> str:
    return "\n".join(heading + "\nPreserved facts." for heading in HEADINGS)


def model(
    store: ModelsStore,
    complete,
    *,
    identity: str = "main",
    window: int = 10_000,
    estimate_context=None,
):
    descriptor = BoundModelDescriptor(
        binding_id=identity,
        plugin_snapshot_id="snapshot",
        model_revision=0,
        model_id=identity,
        connection_id="fixture",
        driver_id="fixture",
        driver_contract_version="1",
        auth_identity="fixture",
        model=identity,
        role=ModelRole.AGENT,
        reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=window, max_output_tokens=800),
        capability_sources=CapabilitySources(),
        capability_digest="fixture",
    )

    class Driver:
        max_tool_schemas = None

        def estimate_context_tokens(self, messages, tools=()):
            if estimate_context is not None:
                return estimate_context(messages, tools)
            return len(str(messages)) // 4

        def estimate_appended_message_tokens(self, messages):
            return len(str(messages)) // 4

        async def complete(self, request):
            return await complete(request)

    return _BoundChat(descriptor, Driver(), store)


def test_context_overflow_keeps_real_messages_and_model_continuation() -> None:
    snapshot = (message(0, Input((ContentPart("text", "large"),))),)
    projection = Projection(estimate=900)
    with pytest.raises(ContextOverflow) as caught:
        ContextBuilder().build(
            snapshot,
            materials=Materials("trusted"),
            model=projection,
            max_output_tokens=200,
        )
    assert caught.value.request.continuation is projection.continuation
    assert projection.seen == snapshot
    assert snapshot[0].body.parts[0].value == "large"


def test_context_summary_requires_exact_settled_message_prefix() -> None:
    snapshot = (
        message(0, Output((ToolCall("tool", {}),), "continue")),
        message(1, ToolResult(CallRef("0", 0), "success", ())),
        message(2, Output((ContentPart("text", "answer"),), "complete")),
        message(3, Input((ContentPart("text", "current"),))),
    )
    projection = Projection()
    request = ContextBuilder().build(
        snapshot,
        materials=Materials("", summary=Summary("summary@2", ("0", "1", "2"), "saved")),
        model=projection,
        max_output_tokens=100,
    )
    assert projection.seen == snapshot
    assert projection.continuation is request.continuation
    assert projection.seen[0].message_id == "0"
    assert '"summary":"saved"' in request.messages[0]["content"]
    with pytest.raises(ValueError, match="尚未结算"):
        ContextBuilder().build(
            snapshot,
            materials=Materials("", summary=Summary("bad", ("0",), "incomplete")),
            model=Projection(),
            max_output_tokens=100,
        )


def test_message_groups_preserve_complete_turn_and_tool_batch() -> None:
    rows = (
        message(0, Input((ContentPart("text", "input"),))),
        message(1, Output((ToolCall("call", {}),), "continue")),
        message(2, ToolResult(CallRef("1", 0), "success", ())),
        message(3, Output((ContentPart("text", "answer"),), "complete")),
        message(4, Input((ContentPart("text", "open"),))),
        message(5, Output((ToolCall("next", {}),), "continue")),
    )
    projection = TurnProjection()
    groups = closed_groups(rows, projection)
    assert groups == (rows[:4],)
    assert window_starts(rows, projection) == (0, 4)
    assert summary_groups(groups, rows) == groups


@pytest.mark.asyncio
async def test_summary_provider_overflow_bisects_complete_groups_without_truncating_source(tmp_path) -> None:
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    requests: list[ModelRequest] = []

    async def complete(request):
        requests.append(request)
        if str(request.messages).count('"message_id"') > 1:
            raise ContextLengthError("provider rejected payload")
        return LLMResponse(summary_text())

    provider = model(store, complete)
    groups = tuple(
        (message(index, Output((ContentPart("text", f"body {index}"),), "complete")),)
        for index in range(3)
    )
    original = groups
    summary, calls = await summarize(groups, previous="", model=provider, fallback=provider)
    assert summary == summary_text()
    assert len(requests) == 5
    assert len(calls) == 3
    assert groups == original


def test_summary_request_stops_at_current_soft_watermark(tmp_path) -> None:
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()

    async def complete(_request):
        pytest.fail("the soft-watermark rejection happens before provider I/O")

    provider = model(
        store,
        complete,
        window=100,
        estimate_context=lambda _messages, _tools: 74,
    )
    with pytest.raises(SummaryError, match="软水位"):
        _request(
            provider,
            "",
            ((message(0, Output((ContentPart("text", "facts"),), "complete")),),),
        )


@pytest.mark.asyncio
async def test_summary_fallback_only_handles_provider_failure(tmp_path) -> None:
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    used: list[ModelRequest] = []

    async def failed(_request):
        raise RateLimitError("limited")

    async def fallback(request):
        used.append(request)
        return LLMResponse(summary_text())

    primary = model(store, failed, identity="primary")
    default = model(store, fallback, identity="default")
    groups = ((message(0, Output((ContentPart("text", "facts"),), "complete")),),)
    summary, calls = await summarize(groups, previous="", model=primary, fallback=default)
    assert summary == summary_text()
    assert len(used) == 1
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_single_oversized_message_is_rejected_without_source_truncation(tmp_path) -> None:
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()

    async def complete(_request):
        pytest.fail("oversized message must not reach provider")

    provider = model(store, complete, window=1000)
    original = message(0, Output((ContentPart("text", "long original" * 1000),), "complete"))
    with pytest.raises(SummaryError, match="完整消息组"):
        await summarize(((original,),), previous="", model=provider, fallback=provider)
    assert original.body.parts[0].value == "long original" * 1000
