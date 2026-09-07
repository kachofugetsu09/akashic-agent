from datetime import UTC, datetime

import pytest

from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, ContextLengthError, InvalidRequestError,
    LLMResponse, ModelCapabilities, ModelRequest, ModelRole, RateLimitError,
)
from plugins.compaction.message_summary import HEADINGS, SummaryError, closed_groups, summarize, summary_groups, window_starts
from plugins.turn_projection.plugin import TurnProjection
from plugins.models.state import _BoundChat
from plugins.models.store import ModelsStore
from session.message import CallRef, ContentPart, Control, Input, Message, Output, ToolCall, ToolResult


def message(seq, body, source="conversation"):
    return Message(str(seq), "s", seq, datetime.now(UTC), "test", source, body)


def model(store, complete, *, identity="main", window=10000):
    descriptor = BoundModelDescriptor(
        binding_id=identity, plugin_snapshot_id="snapshot", model_revision=0,
        model_id=identity, connection_id="fixture", driver_id="fixture", driver_contract_version="1",
        auth_identity="fixture", model=identity, role=ModelRole.AGENT, reasoning_effort=None,
        capabilities=ModelCapabilities(context_window=window, max_output_tokens=800),
        capability_sources=CapabilitySources(), capability_digest="fixture",
    )
    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools=()):
            return len(str(messages)) // 4
        async def complete(self, request):
            return await complete(request)
    return _BoundChat(descriptor, Driver(), store)


def summary_text():
    return "\n".join(heading + "\nPreserved facts." for heading in HEADINGS)


def test_closed_prefix_never_splits_interleaved_sources_or_tool_batches():
    rows = (
        message(0, Input((ContentPart("text", "current input"),))),
        message(1, Output((ToolCall("a", {}), ToolCall("b", {})), "continue")),
        message(2, Output((ToolCall("c", {}),), "continue"), "wake"),
        message(3, ToolResult(CallRef("1", 0), "success", ())),
        message(4, ToolResult(CallRef("2", 0), "success", ()), "wake"),
        message(5, Input((ContentPart("text", "second input"),))),
        message(6, ToolResult(CallRef("1", 1), "success", ())),
        message(7, Input((ContentPart("text", "unanswered tail"),))),
    )
    assert closed_groups(rows[:6], TurnProjection()) == ()
    assert closed_groups(rows, TurnProjection()) == (rows[:7],)


def test_completed_turns_merge_overlapping_sources_but_open_batches_can_compact():
    rows = (
        message(0, Input((ContentPart("text", "input"),))),
        message(1, Output((ToolCall("tool", {}),), "continue")),
        message(2, ToolResult(CallRef("1", 0), "success", ())),
        message(3, Input((ContentPart("text", "other input"),)), "wake"),
        message(4, Output((ContentPart("text", "answer"),), "complete")),
        message(5, Output((ContentPart("text", "other answer"),), "complete"), "wake"),
        message(6, Input((ContentPart("text", "open input"),))),
        message(7, Output((ToolCall("next", {}),), "continue")),
        message(8, ToolResult(CallRef("7", 0), "success", ())),
    )
    projection = TurnProjection()
    assert closed_groups(rows, projection) == (rows[:6], rows[6:])
    assert window_starts(rows, projection) == (0, 6)
    # 先前摘要可能在当时的 open batch 后结束；本代只继续其实际剩余区间。
    assert closed_groups(rows, projection, after=3) == (rows[3:6], rows[6:])


@pytest.mark.parametrize("late_result", [False, True])
def test_abandon_closes_summary_prefix_without_inventing_a_tool_result(tmp_path, late_result):
    from plugins.context.api import Materials, Summary
    from plugins.context.plugin import ContextBuilder
    from plugins.models.content import render_content
    from plugins.models.projection import MessageProjection

    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    async def complete(request):
        pytest.fail("projection must not call model")
    provider = model(store, complete)
    rows = (
        message(0, Input((ContentPart("text", "old work"),))),
        message(1, Output((ToolCall("old-tool", {}),), "continue")),
        message(2, Control("abandon", 1)),
        message(3, Input((ContentPart("text", "new work"),))),
        *((message(4, ToolResult(CallRef("1", 0), "success", (ContentPart("text", "late fact"),))),)
          if late_result else ()),
    )
    assert closed_groups(rows, TurnProjection())[0] == rows[:3]
    projection = MessageProjection(provider, source="conversation", read_call=store.read_call,
        render_content=lambda part: render_content(part, artifacts={}), tool_name=lambda binding: "old-tool",
        keep_input_ids=("3",))
    request = ContextBuilder().build(rows, materials=Materials("", summary=Summary("saved", ("0", "1", "2"), "abandoned work")),
                                     model=projection, max_output_tokens=100)
    assert [row["role"] for row in request.messages] == ["user", "user"]
    assert request.messages[-1]["content"][0]["text"] == "new work"
    assert "late fact" not in str(request.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_summary", [False, True])
async def test_summary_provider_excludes_late_abandoned_result_across_generations(tmp_path, prior_summary):
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    requests = []
    async def complete(request):
        requests.append(request.messages)
        return LLMResponse(summary_text())
    provider = model(store, complete)
    rows = (
        message(0, Input((ContentPart("text", "old input"),))),
        message(1, Output((ToolCall("old-call", {}),), "continue")),
        message(2, Control("abandon", 1)),
        message(3, Output((ToolCall("independent-call", {}),), "continue"), "wake"),
        message(4, Input((ContentPart("text", "new input"),))),
        message(5, ToolResult(CallRef("1", 0), "success", (ContentPart("text", "late excluded fact"),))),
        message(6, ToolResult(CallRef("3", 0), "success", (ContentPart("text", "independent observation"),)), "wake"),
        message(7, Output((ContentPart("text", "new answer"),), "complete")),
    )
    original = tuple(rows)
    groups = closed_groups(rows, TurnProjection(), after=3 if prior_summary else 0)
    _, calls = await summarize(summary_groups(groups, rows), previous=summary_text() if prior_summary else "",
                               model=provider, fallback=provider)
    assert "late excluded fact" not in str(requests)
    assert "independent observation" in str(requests)
    assert "new answer" in str(requests)
    assert calls and all(store.read_call(identity)["state"] == "success" for identity in calls)
    assert rows == original and rows[5].body.parts[0].value == "late excluded fact"


@pytest.mark.asyncio
async def test_provider_overflow_halves_complete_groups_and_keeps_each_successful_call(tmp_path):
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    requests, successes = [], []
    async def complete(request):
        requests.append(request)
        assert request.max_output_tokens == 0
        text = request.messages[0]["content"]
        if text.count('"message_id"') > 1:
            raise ContextLengthError("actual tokenizer exceeds estimate")
        successes.append(text)
        return LLMResponse(summary_text())
    provider = model(store, complete)
    groups = tuple((message(index, Output((ContentPart("text", "body " + str(index)),), "complete")),)
                   for index in range(3))
    summary, calls = await summarize(groups, previous="", model=provider, fallback=provider)
    assert summary == summary_text()
    assert len(requests) == 5 and len(calls) == 3
    assert all(store.read_call(identity)["state"] == "success" for identity in calls)
    assert [next(str(index) for index in range(3) if f'"message_id":"{index}"' in text)
            for text in successes] == ["0", "1", "2"]
    assert "Preserved facts." in successes[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RateLimitError("limited"), InvalidRequestError("bad payload")])
async def test_default_fallback_is_fixed_and_does_not_hide_request_contract_errors(tmp_path, failure):
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    used = []
    async def failed(request):
        raise failure
    async def fallback(request):
        used.append(request)
        return LLMResponse(summary_text())
    primary = model(store, failed)
    default = model(store, fallback, identity="fixed-default")
    groups = ((message(0, Output((ContentPart("text", "facts"),), "complete")),),)
    if isinstance(failure, InvalidRequestError):
        with pytest.raises(InvalidRequestError):
            await summarize(groups, previous="", model=primary, fallback=default)
        assert used == []
    else:
        _, calls = await summarize(groups, previous="", model=primary, fallback=default)
        assert len(used) == 1
        assert store.read_call(calls[0])["binding"]["binding_id"] == "fixed-default"


@pytest.mark.asyncio
async def test_oversized_single_group_is_rejected_without_truncating_original(tmp_path):
    store = ModelsStore(tmp_path / "models.db", tmp_path / "backups")
    store.initialize()
    async def complete(request):
        pytest.fail("oversized group must not reach provider")
    provider = model(store, complete, window=1000)
    original = message(0, Output((ContentPart("text", "long original" * 1000),), "complete"))
    with pytest.raises(SummaryError, match="完整消息组"):
        await summarize(((original,),), previous="", model=provider, fallback=provider)
    assert original.body.parts[0].value == "long original" * 1000
