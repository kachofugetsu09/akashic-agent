"""目标自动续跑：折叠、结算、空转熔断、抢占与预算。"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, Literal, cast

import pytest

from plugins.goal.runtime import GoalRuntime, _usage, fold_round
from plugins.goal.state import GoalStatus, new_goal
from plugins.ledger.contract import (
    ContentPart,
    ContentReferences,
    Control,
    Input,
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
        import asyncio
        await asyncio.Event().wait()


def _stats(call_id: str) -> ModelCallStats:
    return ModelCallStats(
        call_record_id=call_id, model="fixture", state="success",
        first_token_ms=None, duration_ms=None,
        usage=ModelUsage(input_tokens=10, cached_input_tokens=2, output_tokens=5, coverage="exact"),
    )


def _partial_stats(call_id: str) -> ModelCallStats:
    return ModelCallStats(
        call_record_id=call_id, model="fixture", state="success",
        first_token_ms=None, duration_ms=None,
        usage=ModelUsage(input_tokens=10, output_tokens=5, coverage="partial"),
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
async def test_advance_appends_first_continuation_after_setup_turn(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.continuations == 1
        message = harness.log.reader("s").get("goal-continuation:g:1")
        assert message is not None and isinstance(message.body, Input)
    finally:
        harness.log.close()


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
async def test_advance_settles_activity_and_appends_next(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        _append_output(harness.log, "goal-final", text="done", call_id="c1")
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.continuations == 2
        assert goal.tokens_used == 13  # (10-2) + 5
        assert goal.consecutive_empty == 0
        assert goal.status is GoalStatus.ACTIVE
        # 下一条续跑提示必须来自结算后的目标，包含刚计入的用量。
        next_input = harness.log.reader("s").get("goal-continuation:g:2")
        assert next_input is not None
        text = next(str(part.value) for part in next_input.body.parts if part.kind == "text")
        assert "13" in text
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
async def test_terminal_output_before_pause_control_pauses_goal(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        _append_output(harness.log, "goal-final", text="done", call_id="c1")
        head = await harness.store.conversation_head("s")
        _append_control(harness.log, "pause-after-final", "pause", reason="用户停止", through_seq=head)
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.PAUSED
        assert goal.continuations == 1
        assert harness.log.reader("s").get("goal-continuation:g:2") is None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_preemption_with_user_final_output_resumes_without_new_message(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        user_input(harness.log, "s", "user-1", "let me add something")
        _append_output(harness.log, "user-final", "answered")
        await runtime.advance("s")
        # 抢占结算后仍有用户终态输出：runtime 自己再排一次，不等新消息。
        await runtime.advance("s")
        assert harness.log.reader("s").get("goal-continuation:g:2") is not None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_completion_during_round_records_final_usage(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        terminal = _append_output(harness.log, "goal-final", text="done", call_id="c1")
        from dataclasses import replace
        await harness.store.apply(
            "s", "op-complete", "update", "1" * 64,
            lambda current: (
                replace(current, status=GoalStatus.COMPLETE)
                if current is not None else current,
                "complete",
            ),
        )
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.COMPLETE
        assert goal.tokens_used == 13
        assert goal.settled_through_seq == terminal.seq
        assert harness.log.reader("s").get("goal-continuation:g:2") is None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_follow_queues_inactive_goal_with_pending_continuation(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        from dataclasses import replace
        await harness.store.apply(
            "s", "op-complete", "update", "1" * 64,
            lambda current: (
                replace(current, status=GoalStatus.COMPLETE)
                if current is not None else current,
                "complete",
            ),
        )
        task = asyncio.create_task(runtime.follow())
        try:
            session = await asyncio.wait_for(runtime._queue.get(), 1)
            assert session == "s"
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_failure_with_prior_activity_counts_as_empty(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        _append_output(harness.log, "progress", text="working", finish="continue")
        head = await harness.store.conversation_head("s")
        _append_control(harness.log, "failure-1", "failure", reason="model failed", through_seq=head)
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.consecutive_empty == 1
        assert goal.status is GoalStatus.ACTIVE
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_partial_coverage_with_budget_pauses_instead_of_spending(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        await _create_goal(harness, settled_through_seq=0, token_budget=1000)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _partial_stats)
        await runtime.advance("s")
        _append_output(harness.log, "goal-final", text="done", call_id="c1")
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None
        assert goal.status is GoalStatus.PAUSED
        assert "无法确认" in (goal.last_reason or "")
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


@pytest.mark.asyncio
async def test_complete_status_stops_continuation(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        goal = await _create_goal(harness, settled_through_seq=0)
        runtime = GoalRuntime(harness.store, cast(Any, _RuntimeCatalog(harness.log)), _stats)
        await runtime.advance("s")
        from dataclasses import replace
        await harness.store.apply(
            "s", "op-complete", "update", "1" * 64,
            lambda current: (
                replace(current, status=GoalStatus.COMPLETE)
                if current is not None else current,
                "complete",
            ),
        )
        await runtime.advance("s")
        goal = await harness.store.get("s")
        assert goal is not None and goal.status is GoalStatus.COMPLETE
        assert goal.continuations == 1
    finally:
        harness.log.close()


def test_usage_without_model_facts_is_unavailable():
    assert _usage((), _stats) == (0, "unavailable")


def test_fold_round_marks_successful_tool_result_as_activity(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        _append_output(harness.log, "setup")
        goal = new_goal(
            session_id="s", goal_id="g", objective="ship", settled_through_seq=0,
        )
        # 续跑输入
        writer = harness.log.writer(
            "s", author="goal", source="conversation", body_types=(Input,),
            content={"text": lambda part: ContentReferences()},
        )
        writer.append("goal-continuation:g:1", Input((ContentPart("text", "continue"),)))
        writer.expire()
        # 成功工具结果
        from plugins.ledger.contract import CallRef, ToolCall, ToolResult
        harness.log.save_binding("tool", {"target": "test"})
        call_writer = harness.log.writer(
            "s", author="assistant", source="conversation", body_types=(Output,),
            content={}, check_call=lambda call: None,
        )
        call_writer.append("tool-call", Output((ToolCall("tool", {}),), "continue"))
        call_writer.expire()
        result_writer = harness.log.writer(
            "s", author="assistant", source="conversation", body_types=(ToolResult,),
            content={"text": lambda part: ContentReferences()},
            call_ref=CallRef("tool-call", 0),
        )
        result_writer.append(
            "tool-result", ToolResult(CallRef("tool-call", 0), "success",
                                      (ContentPart("text", "ok"),)),
        )
        result_writer.expire()
        _append_output(harness.log, "final")
        messages = harness.log.reader("s").snapshot()
        folded = fold_round(goal, messages)
        assert folded is not None
        assert folded.activity is True
        assert folded.terminal is not None
    finally:
        harness.log.close()
