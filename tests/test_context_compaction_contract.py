from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.plugin_composition.models import ModelContinuation, ModelRequest
from plugins.compaction.message_summary import SummaryError, _request
from plugins.context.api import ContextOverflow, Materials
from plugins.context.plugin import ContextBuilder
from session.message import ContentPart, Input, Message, Output
from tests.model_plugin_fakes import BoundChatModelFake


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

    def estimate(self, request):
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
    request = ContextBuilder().build(
        snapshot,
        materials=Materials("trusted"),
        model=projection,
        max_output_tokens=0,
    )
    assert request.max_output_tokens == 0
    assert request.continuation is projection.continuation
    assert projection.seen == snapshot


def test_summary_request_accepts_soft_watermark_and_rejects_above_it() -> None:
    class SoftWatermarkProvider:
        context_window = 100
        max_tool_schemas = None

        def __init__(self, tokens):
            self.tokens = tokens

        def estimate_context_tokens(self, messages, tools=()):
            return self.tokens

        def estimate_appended_message_tokens(self, messages):
            return 0

        async def chat(self, **kwargs):
            raise AssertionError("soft-watermark request must not call provider")

    groups = ((message(0, Output((ContentPart("text", "facts"),), "complete")),),)
    assert _request(BoundChatModelFake(SoftWatermarkProvider(74)), "", groups)
    with pytest.raises(SummaryError, match="软水位"):
        _request(
            BoundChatModelFake(SoftWatermarkProvider(75)),
            "",
            groups,
        )
