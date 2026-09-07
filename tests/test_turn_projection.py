from datetime import UTC, datetime

import pytest

from plugins.turn_projection.plugin import TurnProjection
from agent.plugin_composition import CompositionRoot, ServiceKey
from plugins.turn_projection.plugin import apply
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


def message(seq, body, *, source="conversation", session="test"):
    return Message(
        f"m{seq}",
        session,
        seq,
        datetime(2026, 9, 5, tzinfo=UTC),
        "author",
        source,
        body,
    )


def test_source_interleaving_and_interrupted_inputs_share_one_turn():
    rows = [
        message(0, Input((ContentPart("text", "first"),))),
        message(1, Control("pause", 0)),
        message(2, Input((ContentPart("text", "more"),))),
        message(3, Output((ContentPart("text", "wake"),), "complete"), source="wake"),
        message(4, Output((ContentPart("text", "answer"),), "complete")),
    ]
    projection = TurnProjection()
    (turn,) = projection.project(rows, "conversation")
    assert turn.message_ids == ("m0", "m2", "m4")
    assert turn.status == "complete"
    assert turn.ending_message_id == "m4"
    (wake,) = projection.project(rows, "wake")
    assert wake.message_ids == ("m3",)


def test_abandoned_calls_cannot_join_the_next_turn():
    rows = [
        message(0, Input(())),
        message(1, Output((ToolCall("binding", {}),), "continue")),
        message(2, Control("abandon", 1)),
        message(3, Input(())),
        message(
            4, ToolResult(CallRef("m1", 0), "success", (ContentPart("text", "late"),))
        ),
        message(5, Output((), "complete")),
    ]
    abandoned, answer = TurnProjection().project(rows, "conversation")
    assert abandoned.status == "abandoned"
    assert abandoned.message_ids == ("m0", "m1")
    assert abandoned.ending_message_id == "m2"
    assert answer.message_ids == ("m3", "m5")
    assert answer.after_seq == 1
    assert rows[4].body.parts[0].value == "late"


def test_tool_results_keep_real_call_positions_and_arrival_order():
    rows = [
        message(
            0,
            Output(
                (
                    ContentPart("text", "thinking"),
                    ToolCall("b1", {}),
                    ToolCall("b2", {}),
                ),
                "continue",
            ),
        ),
        message(1, ToolResult(CallRef("m0", 2), "denied", ())),
        message(2, ToolResult(CallRef("m0", 1), "success", ())),
        message(3, Output((), "complete")),
    ]
    (turn,) = TurnProjection().project(rows, "conversation")
    assert turn.message_ids == ("m0", "m3")
    assert turn.observations == ((CallRef("m0", 2), "m1"), (CallRef("m0", 1), "m2"))


def test_repeated_reads_and_new_plugin_instances_do_not_own_progress():
    prefix = [message(0, Input(())), message(1, Control("failure", 0))]
    projection = TurnProjection()
    (open_turn,) = projection.project(prefix, "conversation")
    assert open_turn.status == "open"
    full = [*prefix, message(2, Input(())), message(3, Output((), "quiet"))]
    assert projection.project(full, "conversation") == TurnProjection().project(
        full, "conversation"
    )
    (complete,) = projection.project(full, "conversation")
    assert complete.message_ids == ("m0", "m2", "m3")
    assert complete.status == "quiet"
    assert projection.project(prefix, "conversation") == (open_turn,)


def test_abandon_cutoff_preserves_later_input_and_drops_old_observation():
    rows = [
        message(0, Input(())),
        message(1, Output((ToolCall("b", {}),), "continue")),
        message(2, ToolResult(CallRef("m1", 0), "success", ())),
        message(3, Input(())),
        message(4, Control("abandon", 1)),
        message(5, Output((), "complete")),
    ]
    abandoned, answer = TurnProjection().project(rows, "conversation")
    assert abandoned.message_ids == ("m0", "m1")
    assert answer.message_ids == ("m3", "m5")


def test_projection_rejects_mixed_sessions_or_reordered_log():
    a = message(0, Input(()))
    for rows in (
        [a, message(1, Input(()), session="other")],
        [message(2, Input(())), a],
        [a, a],
    ):
        with pytest.raises(ValueError):
            TurnProjection().project(rows, "conversation")


def test_message_content_cannot_change_through_caller_owned_objects():
    source = {"nested": [{"value": "original"}]}
    part = ContentPart("example.data", source)
    call = ToolCall("b", source)
    source["nested"][0]["value"] = "changed"
    assert part.value["nested"][0]["value"] == "original"
    assert call.arguments["nested"][0]["value"] == "original"
    with pytest.raises(TypeError):
        part.value["nested"][0]["value"] = "mutated"


def test_body_parts_detach_mutable_input_sequences():
    parts = [ContentPart("text", "original")]
    bodies = (
        Input(parts),
        Output(parts, "complete"),
        ToolResult(CallRef("m0", 0), "success", parts),
    )
    parts.clear()
    assert all(body.parts[0].value == "original" for body in bodies)


def test_other_sources_do_not_change_an_open_turn_boundary():
    rows = [message(0, Input(())), message(1, Control("pause", 0))]
    projection = TurnProjection()
    before = projection.project(rows, "conversation")
    rows.append(message(2, Output((), "complete"), source="wake"))
    assert projection.project(rows, "conversation") == before
    assert before[0].through_seq == 1


def test_finished_outputs_cannot_contain_unstarted_calls():
    with pytest.raises(ValueError, match="continue"):
        Output((ToolCall("b", {}),), "complete")


def test_invalid_body_values_fail_at_message_construction():
    with pytest.raises(TypeError, match="reason"):
        Control("pause", 0, ["mutable"])
    with pytest.raises(TypeError, match="JSON 对象"):
        ToolCall("b", ["not an object"])
    with pytest.raises(TypeError, match="Input"):
        Input((ToolCall("b", {}),))
    with pytest.raises(TypeError, match="body"):
        message(0, {"kind": "input"})


@pytest.mark.asyncio
async def test_public_service_loads_under_arbitrary_plugin_identity():
    root = CompositionRoot("turn-consumer")
    key = ServiceKey[TurnProjection]("turn.projection.v1")
    results = []

    async def provider(ctx):
        await apply(ctx, None)

    async def consumer(ctx):
        projection = ctx.require(key)
        results.append(
            projection.project([message(0, Output((), "quiet"))], "conversation")
        )

    try:
        await root.mount(consumer, name="independent-consumer", inject=(key,))
        assert results == []
        await root.mount(provider, name="external-provider")
        assert results[0][0].status == "quiet"
    finally:
        await root.dispose()
