"""目标状态机的纯值测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from plugins.goal.state import (
    GoalStatus,
    apply_model_update,
    mark_continuation,
    new_goal,
    pause,
    resume,
    set_token_budget,
    settle_round,
)


def _goal(**kwargs):
    return new_goal(
        session_id="s", goal_id="g", objective="ship it",
        settled_through_seq=3, **kwargs,
    )


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


def test_resume_and_mark_continuation_reject_exhausted_cap():
    exhausted = replace(_goal(), continuations=100, status=GoalStatus.PAUSED)
    with pytest.raises(ValueError):
        resume(exhausted)
    with pytest.raises(ValueError):
        mark_continuation(exhausted, "goal-continuation:g:101")


def test_budget_recovery_at_continuation_cap_pauses_instead_of_active():
    exhausted = replace(
        _goal(), status=GoalStatus.BUDGET_LIMITED, continuations=100,
        token_budget=10, tokens_used=10,
    )
    recovered = set_token_budget(exhausted, 100)
    assert recovered.status is GoalStatus.PAUSED
    assert "上限" in (recovered.last_reason or "")
