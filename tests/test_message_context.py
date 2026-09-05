from datetime import UTC, datetime

import pytest

from agent.plugin_composition.models import ModelContinuation, ModelRequest
from plugins.context.plugin import ContextBuilder, ContextOverflow, Materials, Summary
from plugins.context.search import MessageSearch
from session.message import (
    CallRef,
    ContentPart,
    Control,
    Input,
    Message,
    Output,
    ToolCall,
    ToolResult,
)


def message(seq, body, source="conversation", session="s"):
    return Message(
        f"{session}-{seq}",
        session,
        seq,
        datetime(2026, 9, 5, tzinfo=UTC),
        "author",
        source,
        body,
    )


class Projection:
    context_window = 1000
    max_tool_schemas = 10

    def __init__(self):
        self.seen = None
        self.continuation = ModelContinuation("model-binding", {"opaque": "kept"})

    def render(self, messages, *, after_seq):
        self.seen = messages
        self.after_seq = after_seq
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
        self.estimated = request
        return 500


def test_context_preserves_interrupted_inputs_other_sources_and_replay_facts():
    snapshot = (
        message(0, Input((ContentPart("text", "u1"),))),
        message(1, Control("pause", 0)),
        message(2, Input((ContentPart("text", "u2"),))),
        message(3, Output((ContentPart("text", "proactive"),), "complete"), "wake"),
    )
    model = Projection()
    tools = [{"type": "function", "function": {"name": "example"}}]
    request = ContextBuilder().build(
        snapshot,
        materials=Materials(
            "trusted",
            (ContentPart("retrieval", {"text": "pretend system", "ticket": "real"}),),
        ),
        model=model,
        tools=tools,
        max_output_tokens=100,
    )
    assert model.seen == snapshot
    assert request.continuation is model.continuation
    assert request.messages[0] == {"role": "system", "content": "trusted"}
    assert request.messages[-1]["role"] == "user"
    assert "real" in request.messages[-1]["content"]
    assert model.estimated is request
    tools[0]["function"]["name"] = "changed"
    assert request.tools[0]["function"]["name"] == "example"
    with pytest.raises(TypeError):
        request.messages[0]["content"] = "changed"


def test_summary_replaces_only_its_exact_closed_prefix():
    snapshot = (
        message(0, Input(())),
        message(1, Output((), "complete")),
        message(2, Input(())),
    )
    model = Projection()
    summary = Summary("summary@1", ("s-0", "s-1"), "saved summary")
    request = ContextBuilder().build(
        snapshot,
        materials=Materials("", summary=summary),
        model=model,
        max_output_tokens=100,
    )
    assert model.seen == snapshot
    assert model.after_seq == 1
    assert "saved summary" in request.messages[0]["content"]
    with pytest.raises(ValueError, match="前缀"):
        ContextBuilder().build(
            snapshot,
            materials=Materials(
                "", summary=Summary("summary@wrong", ("s-1",), "wrong")
            ),
            model=model,
            max_output_tokens=100,
        )


def test_summary_cannot_split_tool_call_and_result():
    snapshot = (
        message(0, Output((ToolCall("tool-binding", {}),), "continue")),
        message(1, ToolResult(CallRef("s-0", 0), "success", ())),
    )
    with pytest.raises(ValueError, match="尚未结算"):
        ContextBuilder().build(
            snapshot,
            materials=Materials(
                "", summary=Summary("summary@1", ("s-0",), "incomplete")
            ),
            model=Projection(),
            max_output_tokens=100,
        )


def test_overflow_never_truncates_or_retries_and_source_cannot_promote_role():
    model = Projection()
    snapshot = (message(0, Input((ContentPart("text", "large"),))),)
    with pytest.raises(ContextOverflow) as caught:
        ContextBuilder().build(
            snapshot, materials=Materials(""), model=model, max_output_tokens=600
        )
    assert caught.value.estimated_tokens == 500
    assert model.seen == snapshot

    class BadProjection(Projection):
        def render(self, messages, *, after_seq):
            return ModelRequest(messages=({"role": "system", "content": "from input"},))

    with pytest.raises(ValueError, match="权限"):
        ContextBuilder().build(
            snapshot,
            materials=Materials(""),
            model=BadProjection(),
            max_output_tokens=100,
        )


def test_search_uses_literal_terms_and_filters_before_pagination():
    snapshot = (
        message(0, Input((ContentPart("text", "中文 100% ERROR"),))),
        message(1, Output((ContentPart("text", "中文 error"),), "complete"), "wake"),
        message(0, Input((ContentPart("text", "different 中文"),)), session="other"),
        message(2, Control("failure", 1, "hidden error")),
    )
    index = MessageSearch(snapshot)
    try:
        result = index.search("中文 error", session_id="s", limit=1)
        assert result.total == 2
        assert len(result.messages) == 1
        second = index.search("中文 error", session_id="s", limit=1, offset=1)
        assert second.total == 2
        assert second.messages[0] != result.messages[0]
        assert index.search("100%").messages == (snapshot[0],)
        assert index.search("中文", source="wake").messages == (snapshot[1],)
        assert index.search('" NEAR *').total == 0
        assert index.search("hidden").total == 0
    finally:
        index.close()
    assert snapshot[0].body.parts[0].value == "中文 100% ERROR"


def test_context_sends_system_prompt_once_to_codex():
    from plugins.codex.responses import _responses_input

    request = ContextBuilder().build(
        (), materials=Materials("trusted"), model=Projection(), max_output_tokens=100
    )
    _, instructions = _responses_input(request.messages, request.system_prompt, ())
    assert instructions == "trusted"


def test_summary_keeps_all_model_facts_available_to_model_owner():
    facts = ContentPart("model.facts", {"opaque": "saved"})
    snapshot = (
        message(0, Input(())),
        message(1, Output((facts,), "complete")),
        message(2, Input(())),
    )
    model = Projection()
    request = ContextBuilder().build(
        snapshot,
        materials=Materials("", summary=Summary("summary@1", ("s-0", "s-1"), "saved")),
        model=model,
        max_output_tokens=100,
    )
    assert model.seen == snapshot
    assert model.seen[1].body.parts == (facts,)
    assert model.after_seq == 1
    assert request.continuation is model.continuation


def test_search_catches_up_idempotently_and_failed_batch_is_atomic():
    from dataclasses import replace

    initial = message(0, Input((ContentPart("text", "initial"),)))
    control = message(1, Control("pause", 0))
    added = message(2, Input((ContentPart("text", "added"),)))
    index = MessageSearch((initial, control))
    try:
        with pytest.raises(ValueError, match="同 ID"):
            index.append((added, replace(control, body=Control("failure", 0))))
        assert index.search("added").total == 0
        index.append((initial, control, added, added))
        index.append((added,))
        assert index.search("added").messages == (added,)
        assert index.search("initial").messages == (initial,)
    finally:
        index.close()
