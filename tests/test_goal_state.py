"""目标状态机的纯值测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from plugins.goal.state import (
    BLOCKED_AFTER_EMPTY_TURNS,
    GoalStatus,
    apply_model_update,
    mark_continuation,
    new_goal,
    pause,
    resume,
    set_token_budget,
    settle_round,
    status_text,
)


def _goal(**kwargs):
    return new_goal(
        session_id="s", goal_id="g", objective="ship it",
        settled_through_seq=3, **kwargs,
    )


def test_new_goal_normalizes_and_rejects_bad_objective():
    goal = _goal()
    assert goal.objective == "ship it"
    assert goal.status is GoalStatus.ACTIVE
    assert goal.settled_through_seq == 3
    with pytest.raises(ValueError):
        new_goal(session_id="s", goal_id="g", objective="  ")
    with pytest.raises(ValueError):
        new_goal(session_id="s", goal_id="g", objective="x" * 4001)


def test_continuation_counter_is_per_goal():
    goal = _goal()
    first = mark_continuation(goal, "goal-continuation:g:1")
    second = mark_continuation(first, "goal-continuation:g:2")
    assert (first.continuations, second.continuations) == (1, 2)
    assert second.continuation_id == "goal-continuation:g:2"
    with pytest.raises(ValueError):
        mark_continuation(second, "")


def test_activity_resets_empty_and_empty_blocks_after_three():
    goal = _goal()
    for _ in range(BLOCKED_AFTER_EMPTY_TURNS - 1):
        goal = settle_round(
            goal, activity=False, tokens=0, coverage="exact",
            settled_through_seq=goal.settled_through_seq + 1,
            last_settled_message_id="m",
        )
        assert goal.status is GoalStatus.ACTIVE
    goal = settle_round(
        goal, activity=True, tokens=0, coverage="exact",
        settled_through_seq=goal.settled_through_seq + 1,
        last_settled_message_id="m",
    )
    assert goal.consecutive_empty == 0
    for _ in range(BLOCKED_AFTER_EMPTY_TURNS):
        goal = settle_round(
            goal, activity=False, tokens=0, coverage="exact",
            settled_through_seq=goal.settled_through_seq + 1,
            last_settled_message_id="m",
        )
    assert goal.status is GoalStatus.BLOCKED
    assert goal.consecutive_empty == BLOCKED_AFTER_EMPTY_TURNS


def test_budget_limit_is_terminal_and_blocks_more_work():
    goal = _goal(token_budget=100)
    goal = settle_round(
        goal, activity=True, tokens=60, coverage="exact",
        settled_through_seq=4, last_settled_message_id="a",
    )
    assert goal.status is GoalStatus.ACTIVE
    goal = settle_round(
        goal, activity=True, tokens=50, coverage="exact",
        settled_through_seq=5, last_settled_message_id="b",
    )
    assert goal.status is GoalStatus.BUDGET_LIMITED
    assert goal.tokens_used == 110
    with pytest.raises(ValueError):
        settle_round(
            goal, activity=True, tokens=1, coverage="exact",
            settled_through_seq=6, last_settled_message_id="c",
        )


def test_model_updates_only_from_active_and_only_three_statuses():
    goal = _goal()
    assert apply_model_update(goal, "complete").status is GoalStatus.COMPLETE
    assert apply_model_update(goal, "blocked").status is GoalStatus.BLOCKED
    assert apply_model_update(goal, "paused").status is GoalStatus.PAUSED
    with pytest.raises(ValueError):
        apply_model_update(goal, "active")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        apply_model_update(apply_model_update(goal, "complete"), "blocked")


def test_resume_only_from_paused_or_blocked_and_resets_empty():
    goal = _goal()
    paused = pause(goal, "user")
    resumed = resume(paused)
    assert resumed.status is GoalStatus.ACTIVE
    assert resumed.consecutive_empty == 0
    with pytest.raises(ValueError):
        resume(resumed)
    with pytest.raises(ValueError):
        resume(_goal(token_budget=1))


def test_budget_can_be_set_and_lowered_below_usage():
    goal = _goal(token_budget=1000)
    goal = set_token_budget(goal, 10)
    assert goal.token_budget == 10
    goal = settle_round(
        goal, activity=True, tokens=20, coverage="exact",
        settled_through_seq=4, last_settled_message_id="m",
    )
    assert goal.status is GoalStatus.BUDGET_LIMITED
    raised = set_token_budget(goal, 100)
    assert raised.status is GoalStatus.ACTIVE
    assert raised.token_budget == 100
    lowered = set_token_budget(raised, 5)
    assert lowered.status is GoalStatus.BUDGET_LIMITED


def test_continuation_cap_pauses_active_goal():
    from dataclasses import replace
    goal = _goal()
    goal = replace(goal, continuations=100)
    settled = settle_round(
        goal, activity=True, tokens=0, coverage="exact",
        settled_through_seq=4, last_settled_message_id="m",
    )
    assert settled.status is GoalStatus.PAUSED
    assert "上限" in (settled.last_reason or "")


def test_resume_and_mark_continuation_reject_exhausted_cap():
    from dataclasses import replace
    exhausted = replace(_goal(), continuations=100, status=GoalStatus.PAUSED)
    with pytest.raises(ValueError):
        resume(exhausted)
    with pytest.raises(ValueError):
        mark_continuation(exhausted, "goal-continuation:g:101")


def test_budget_recovery_at_continuation_cap_pauses_instead_of_active():
    from dataclasses import replace
    exhausted = replace(
        _goal(), status=GoalStatus.BUDGET_LIMITED, continuations=100,
        token_budget=10, tokens_used=10,
    )
    recovered = set_token_budget(exhausted, 100)
    assert recovered.status is GoalStatus.PAUSED
    assert "上限" in (recovered.last_reason or "")


def test_inactive_goal_can_record_final_round_usage():
    goal = _goal()
    goal = apply_model_update(goal, "complete")
    settled = settle_round(
        goal, activity=True, tokens=13, coverage="exact",
        settled_through_seq=4, last_settled_message_id="m",
        allow_inactive=True,
    )
    assert settled.status is GoalStatus.COMPLETE
    assert settled.tokens_used == 13
    assert settled.settled_through_seq == 4
    assert settled.last_reason == "目标已完成"
    with pytest.raises(ValueError):
        settle_round(
            goal, activity=True, tokens=1, coverage="exact",
            settled_through_seq=4, last_settled_message_id="m",
        )


def test_status_text_contains_objective_and_remaining():
    goal = _goal(token_budget=100)
    goal = settle_round(
        goal, activity=True, tokens=40, coverage="partial",
        settled_through_seq=4, last_settled_message_id="m",
    )
    text = status_text(goal)
    assert "ship it" in text
    assert "60" in text
    assert "partial" in text
