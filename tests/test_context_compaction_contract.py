from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent.plugin_composition.models import ModelContinuation, ModelRequest
from plugins.compaction.message_summary import SummaryError, _request
from plugins.context.api import ContextOverflow, Materials
from plugins.context.plugin import ContextBuilder
from session.message import ContentPart, Input, Message, Output


def message(seq: int, body) -> Message:
    return Message(str(seq), "s", seq, datetime(2026, 9, 5, tzinfo=UTC), "test", "conversation", body)


class Projection:
    context_window = 1000
    max_tool_schemas = 10

    def __init__(self) -> None:
        self.seen: tuple[Message, ...] = ()
        self.continuation = ModelContinuation("model-binding", {"opaque": "kept"})

    def render(self, messages, *, after_seq, summary_reference=None, fresh=False):
        self.seen = tuple(messages)
        return ModelRequest(
            messages=tuple(
                {"role": "user", "content": str(item.body.parts)}
                for item in messages
                if item.seq > after_seq
            ),
            continuation=self.continuation,
        )

    def estimate(self, _request):
        return 900


def test_context_overflow_keeps_real_messages_and_model_continuation() -> None:
    snapshot = (message(0, Input((ContentPart("text", "large"),))),)
    projection = Projection()
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


def test_summary_request_stops_at_current_soft_watermark() -> None:
    class SoftWatermarkModel:
        descriptor = SimpleNamespace(
            capabilities=SimpleNamespace(context_window=100),
        )

        def estimate_context_tokens(self, _messages, _tools=()):
            return 74

    with pytest.raises(SummaryError, match="软水位"):
        _request(
            SoftWatermarkModel(),
            "",
            ((message(0, Output((ContentPart("text", "facts"),), "complete")),),),
        )
