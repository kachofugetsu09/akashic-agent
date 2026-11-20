"""目标自动续跑：折叠、结算、空转熔断、抢占与预算。"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, cast

import pytest

from plugins.goal.runtime import GoalRuntime
from plugins.goal.state import GoalStatus, new_goal
from plugins.ledger.contract import (
    ContentPart,
    ContentReferences,
    Control,
    Output,
)
from plugins.models.contract import ModelCallStats, ModelUsage
from tests.support.goal_harness import goal_harness, user_input


class _RuntimeCatalog:
    def __init__(self, log):
        self._log = log

    def reader(self, session_id):
        return self._log.reader(session_id)

    async def follow(self, **kwargs):
        yield {}
        await asyncio.Event().wait()


def _stats(call_id: str) -> ModelCallStats:
    return ModelCallStats(
        call_record_id=call_id, model="fixture", state="success",
        first_token_ms=None, duration_ms=None,
        usage=ModelUsage(input_tokens=10, cached_input_tokens=2, output_tokens=5, coverage="exact"),
    )


def _writer(log, session_id="s"):
    return log.writer(
        session_id, author="assistant", source="conversation",
        body_types=(Output,),
        content={
            "text": lambda part: ContentReferences(),
            "model.facts": lambda part: ContentReferences(),
        },
    )


def _append_output(log, message_id, text="done", call_id=None, finish: Literal["continue", "complete", "quiet"] = "complete"):
    parts = [ContentPart("text", text)]
    if call_id is not None:
        parts.append(ContentPart("model.facts", {"call_record_id": call_id}))
    writer = _writer(log)
    message = writer.append(message_id, Output(tuple(parts), finish))
    writer.expire()
    return message


def _append_control(log, message_id, action, reason=None, through_seq=None):
    writer = log.writer(
        "s", author="app", source="conversation", body_types=(Control,),
        content={},
    )
    message = writer.append(
        message_id, Control(action, through_seq if through_seq is not None else 0, reason),
    )
    writer.expire()
    return message


async def _create_goal(harness, settled_through_seq=-1, token_budget=None):
    def action(current):
        assert current is None
        return (
            new_goal(
                session_id="s", goal_id="g", objective="ship it",
                token_budget=token_budget, settled_through_seq=settled_through_seq,
            ),
            "created",
        )

    goal, _ = await harness.store.apply(
        "s", "op-create", "create", "0" * 64, action,
    )
    assert goal is not None
    return goal


@pytest.mark.asyncio
async def test_restart_does_not_duplicate_pending_continuation(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        first = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await first.advance("s")
        # 模拟重启：新 runtime 看到同一个 pending goal Input，不应再写一条。
        restarted = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await restarted.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None and goal.continuations == 1
        assert harness.log.reader("s").get("goal-continuation:g:1") is not None
        assert harness.log.reader("s").get("goal-continuation:g:2") is None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_terminal_output_before_user_input_does_not_append_continuation(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        terminal = _append_output(harness.log, "goal-final", text="done", call_id="c1")
        user_input(harness.log, "s", "user-1", "wait, change direction")
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.ACTIVE
        assert goal.continuations == 1
        assert goal.tokens_used == 13
        assert goal.settled_through_seq == terminal.seq
        assert harness.log.reader("s").get("goal-continuation:g:2") is None

        # 用户轮次结束后才恢复目标。
        _append_output(harness.log, "user-final", "answered")
        await runtime.advance("s")
        assert harness.log.reader("s").get("goal-continuation:g:2") is not None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_exhausted_budget_stops_goal(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0, token_budget=10)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        _append_output(harness.log, "goal-final", text="done", call_id="c1")
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.BUDGET_LIMITED
        assert harness.log.reader("s").get("goal-continuation:g:2") is None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_three_failed_rounds_block_goal(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        for index in range(3):
            await runtime.advance("s")
            head = await harness.store.conversation_head("s")
            _append_control(
                harness.log, f"failure-{index}", "failure",
                reason="model failed", through_seq=head,
            )
            await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.BLOCKED
        assert goal.consecutive_empty == 3
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_user_input_preempts_round_without_pausing_goal(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        user_input(harness.log, "s", "user-1", "let me add something")
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.ACTIVE
        assert goal.settled_through_seq == harness.log.reader("s").get("user-1").seq
        assert goal.consecutive_empty == 0
        # 用户轮次结束后必须能继续，而不是反复折叠同一次抢占。
        _append_output(harness.log, "user-final", "answered")
        await runtime.advance("s")
        assert harness.log.reader("s").get("goal-continuation:g:2") is not None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_pause_after_preemption_does_not_continue(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        user_input(harness.log, "s", "user-1", "let me add something")
        await runtime.advance("s")
        head = await harness.store.conversation_head("s")
        _append_control(harness.log, "pause-1", "pause", reason="用户停止", through_seq=head)
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.PAUSED
        assert harness.log.reader("s").get("goal-continuation:g:2") is None
    finally:
        harness.log.close()
