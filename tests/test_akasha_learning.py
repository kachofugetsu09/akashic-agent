"""Akasha feedback restores only the causal prefix required by a receipt."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

import pytest

from agent.plugin_composition.bindings import Bindings
from session.message import CallRef, ContentPart, Input, Message, Output, ToolCall, ToolResult
from plugins.akasha._boundaries import TOOLS
from plugins.akasha import learning as learning_module
from plugins.akasha.domain.model import Turn, TurnFeedback
from plugins.akasha.infrastructure.consumption import Applied, Consumption
from plugins.akasha.learning import Learning
from plugins.akasha.projection import Sample
from plugins.content.api import legacy_post_commit_effect
from plugins.turn_projection.plugin import TurnProjection


def applied(name: str, seq: int) -> Applied:
    """Build one fixed historical learning source without a graph fixture."""
    return Applied(
        learning_binding="recorded-learning",
        session_id="conversation",
        ending=(seq + 1, f"{name}-output"),
        members=((seq, f"{name}-input"), (seq + 1, f"{name}-output")),
        observations=(),
        source_digest="0" * 64,
    )


def message(message_id: str, seq: int, body: Input | Output | ToolResult) -> Message:
    """Keep fixture messages accepted facts with a stable causal order."""
    return Message(
        message_id=message_id,
        session_id="conversation",
        seq=seq,
        recorded_at=datetime(2026, 9, 29, tzinfo=UTC),
        author="assistant" if isinstance(body, (Output, ToolResult)) else "user",
        source="conversation",
        body=body,
    )


class RecordedBindings:
    """Expose the recorded Akasha tool owner used by real receipt validation."""

    def describe(self, identity: str, service: object) -> Mapping[str, object]:
        assert identity == "recorded-tool"
        assert service is TOOLS
        return {"tool": {"owner": "akasha"}}


def test_empty_feedback_does_not_scan_the_historical_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A receipt-free restored turn does not need message-node attribution."""
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    current = message("current-input", 4, Input((ContentPart("text", "question"),)))
    state = Consumption(cutover_heads=(), applied=(applied("past", 0), applied("future", 2)))

    def unexpected_scan(*args: object, **kwargs: object) -> dict[str, int]:
        raise AssertionError("empty feedback must not scan historical members")

    monkeypatch.setattr(learning_module, "message_nodes", unexpected_scan)
    previous = cast(tuple[Turn, ...], (object(),))
    assert learning.feedback(Sample(current, (current,), ()), previous, state,
                             cast(Bindings, object())) == TurnFeedback()


def test_recorded_feedback_keeps_prefix_mapping_and_rejects_future_targets() -> None:
    """Receipts still use their exact prior prefix and cannot name a future node."""
    learning = Learning(TurnProjection(), owner="akasha", post_commit_effect=legacy_post_commit_effect)
    current = message("current-input", 2, Input((ContentPart("text", "question"),)))
    request = message("request", 3, Output((ToolCall("recorded-tool", {}),), "continue"))
    state = Consumption(cutover_heads=(), applied=(applied("past", 0), applied("future", 2)))
    previous = cast(tuple[Turn, ...], (object(),))

    def receipt(message_id: str, targets: tuple[str, ...]) -> Message:
        return message(message_id, 4, ToolResult(
            CallRef("request", 0), "success", (
                ContentPart("akasha.feedback", {
                    "action": "remember", "target_message_ids": targets, "reason": "recorded",
                }),
            ),
        ))

    sample = Sample(request, (current, request), (receipt("past-and-current", ("past-input", "current-input")),))
    bindings = cast(Bindings, RecordedBindings())
    assert learning.feedback(sample, previous, state, bindings) == TurnFeedback((0, 1), (), 3.0)

    future = Sample(request, (current, request), (receipt("future", ("future-input",)),))
    with pytest.raises(ValueError, match="反馈目标没有对应的已学习消息"):
        _ = learning.feedback(future, previous, state, bindings)
