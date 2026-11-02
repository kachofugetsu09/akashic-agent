"""`/goal` 命令的解析、状态变更与幂等。"""

from __future__ import annotations

import pytest

from plugins.commands.contract import CommandInvocation
from plugins.goal.command import GoalCommand
from plugins.goal.state import GoalStatus
from tests.support.goal_harness import goal_harness


def _call(raw: str, message_id: str = "m1", session: str = "s") -> CommandInvocation:
    return CommandInvocation(
        name="goal", raw_input=raw, session_key=session,
        channel="web", chat_id="room", sender="user", message_id=message_id,
    )


@pytest.mark.asyncio
async def test_status_set_pause_resume_clear(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        command = GoalCommand(harness.store)
        assert "没有目标" in (await command(_call(""))).text
        set_result = await command(_call(" ship the feature ", "m-set"))
        assert set_result.kind == "success"
        assert "目标已设置" in set_result.text
        assert (await harness.store.get("s")).status is GoalStatus.ACTIVE

        duplicate = await command(_call(" ship the feature ", "m-set"))
        assert duplicate.kind == "success"
        assert "目标已设置" in duplicate.text
        assert (await harness.store.get("s")).objective == "ship the feature"

        reused = await command(_call("another", "m-set"))
        assert reused.kind == "error"

        paused = await command(_call("pause", "m-pause"))
        assert paused.kind == "success"
        assert (await harness.store.get("s")).status is GoalStatus.PAUSED

        resumed = await command(_call("resume", "m-resume"))
        assert resumed.kind == "success"
        assert (await harness.store.get("s")).status is GoalStatus.ACTIVE

        budget = await command(_call("budget 1000", "m-budget"))
        assert budget.kind == "success"
        assert (await harness.store.get("s")).token_budget == 1000

        cleared_budget = await command(_call("budget clear", "m-budget-clear"))
        assert cleared_budget.kind == "success"
        assert (await harness.store.get("s")).token_budget is None

        cleared = await command(_call("clear", "m-clear"))
        assert cleared.kind == "success"
        assert await harness.store.get("s") is None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_free_text_starting_with_subcommand_word_is_objective(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        command = GoalCommand(harness.store)
        reset = await command(_call("reset the deploy config", "m-reset"))
        assert reset.kind == "success"
        assert (await harness.store.get("s")).objective == "reset the deploy config"
        assert (await command(_call("clear", "m-clear"))).kind == "success"

        paused = await command(_call("pause the rollout tonight", "m-pause"))
        assert paused.kind == "success"
        assert (await harness.store.get("s")).objective == "pause the rollout tonight"
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_budget_can_resume_a_budget_limited_goal(tmp_path):
    from plugins.goal.state import settle_round
    harness = goal_harness(tmp_path)
    try:
        command = GoalCommand(harness.store)
        await command(_call("ship", "m-set"))
        await command(_call("budget 10", "m-budget"))
        goal = await harness.store.get("s")
        assert goal is not None
        goal = settle_round(
            goal, activity=True, tokens=10, coverage="exact",
            settled_through_seq=goal.settled_through_seq + 1,
            last_settled_message_id="m",
        )
        await harness.store.apply(
            "s", "m-settle", "update", "a" * 64,
            lambda current: (goal, "settled"),
        )
        assert (await harness.store.get("s")).status is GoalStatus.BUDGET_LIMITED

        raised = await command(_call("budget 100", "m-budget-raise"))
        assert raised.kind == "success"
        assert (await harness.store.get("s")).status is GoalStatus.ACTIVE

        lowered = await command(_call("budget 5", "m-budget-lower"))
        assert lowered.kind == "success"
        assert (await harness.store.get("s")).status is GoalStatus.BUDGET_LIMITED
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_invalid_inputs_do_not_create_goals(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        command = GoalCommand(harness.store)
        assert (await command(_call("", "m-empty"))).kind == "success"
        assert (await command(_call("x" * 4001, "m-long"))).kind == "error"
        assert (await command(_call("budget 0", "m-budget"))).kind == "error"
        assert (await command(_call("budget", "m-budget-missing"))).kind == "error"
        assert await harness.store.get("s") is None
    finally:
        harness.log.close()
