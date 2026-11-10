"""目标状态与续跑事务的确定性 monkey 测试。"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Literal, cast

import pytest

from plugins.goal.runtime import GoalRuntime
from plugins.goal.state import (
    GoalStatus, apply_model_update, new_goal, pause, resume, set_token_budget,
)
from plugins.goal.store import decode_goal, digest, encode_goal
from plugins.ledger.contract import (
    CallRef, ContentPart, ContentReferences, Control, Output, ToolCall, ToolResult,
)
from plugins.models.contract import ModelCallStats, ModelUsage
from tests.support.goal_harness import goal_harness, user_input


class _Catalog:
    def __init__(self, log):
        self._log = log

    def reader(self, session_id):
        return self._log.reader(session_id)


def _stats(call_id: str) -> ModelCallStats:
    if call_id.startswith("missing"):
        raise KeyError(call_id)
    coverage = "exact" if call_id.startswith("exact") else "partial"
    return ModelCallStats(
        call_record_id=call_id, model="fixture", state="success",
        first_token_ms=None, duration_ms=None,
        usage=ModelUsage(
            input_tokens=10, cached_input_tokens=2, output_tokens=5,
            coverage=coverage,
        ),
    )


def _append_output(log, message_id, text="done", call_id=None, finish: Literal["continue", "complete", "quiet"] = "complete", session_id="s"):
    parts = [ContentPart("text", text)]
    if call_id is not None:
        parts.append(ContentPart("model.facts", {"call_record_id": call_id}))
    writer = log.writer(
        session_id, author="assistant", source="conversation", body_types=(Output,),
        content={
            "text": lambda part: ContentReferences(),
            "model.facts": lambda part: ContentReferences(),
        },
    )
    writer.append(message_id, Output(tuple(parts), finish))
    writer.expire()


def _append_failure(log, message_id, session_id="s"):
    head = log.reader(session_id).head(source="conversation")
    writer = log.writer(
        session_id, author="app", source="conversation", body_types=(Control,), content={},
    )
    writer.append(message_id, Control("failure", max(0, head), "model failed"))
    writer.expire()


def _append_tool_round(log, message_id, session_id="s"):
    log.save_binding("tool", {"target": "monkey"})
    call_writer = log.writer(
        session_id, author="assistant", source="conversation", body_types=(Output,),
        content={}, check_call=lambda call: None,
    )
    call_writer.append(
        message_id, Output((ToolCall("tool", {}),), "continue"),
    )
    call_writer.expire()
    result_writer = log.writer(
        session_id, author="assistant", source="conversation", body_types=(ToolResult,),
        content={"text": lambda part: ContentReferences()},
        call_ref=CallRef(message_id, 0),
    )
    result_writer.append(
        message_id + "-result",
        ToolResult(CallRef(message_id, 0), "success", (ContentPart("text", "ok"),)),
    )
    result_writer.expire()


async def _create(harness, rng, *, token_budget=None, session_id="s"):
    head = await harness.store.conversation_head(session_id)
    goal_id = f"g-{rng.randrange(10**9)}"

    def action(current):
        if current is not None and current.status is not GoalStatus.COMPLETE:
            raise ValueError("unfinished")
        return (
            new_goal(
                session_id=session_id, goal_id=goal_id, objective="ship it",
                token_budget=token_budget, settled_through_seq=head,
            ),
            "created",
        )

    return await harness.store.apply(
        session_id, f"create-{goal_id}", "create", digest({"objective": "ship it"}),
        action,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(200))
async def test_monkey_goal_invariants(tmp_path, seed):
    rng = random.Random(seed)
    harness = goal_harness(tmp_path)
    log = harness.log
    try:
        _append_output(log, "setup", "setup")
        budget = 10 if seed % 3 == 0 else rng.choice([None, 100, 1000])
        goal, _ = await _create(harness, rng, token_budget=budget)
        assert goal is not None
        runtime = GoalRuntime(harness.store, cast(Any, _Catalog(log)), _stats)

        # 每个种子至少完成一次“续跑 + 结算”，让 _usage 和结算路径真的执行。
        await runtime.advance("s")
        _append_output(log, f"forced-{seed}", "done", call_id=f"exact-{seed}")
        await runtime.advance("s")

        settled = (await harness.store.get("s"))
        assert settled is not None
        assert settled.settled_through_seq >= goal.settled_through_seq
        assert settled.tokens_used >= 13
        if budget == 10:
            assert settled.status is GoalStatus.BUDGET_LIMITED

        previous_goal_id = settled.goal_id
        previous_continuations = settled.continuations
        previous_tokens = settled.tokens_used
        previous_settled = settled.settled_through_seq

        # 加权动作：advance/输出为主，clear 只占很小一部分。
        actions = (
            ["advance"] * 4 + ["output"] * 4 + ["tool"] * 2
            + ["failure", "user", "pause", "resume", "complete", "blocked", "budget", "clear", "create"]
        )
        for step in range(40):
            action = rng.choice(actions)
            if action == "advance":
                await runtime.advance("s")
            elif action == "output":
                call_id = rng.choice([
                    None, f"exact-{seed}-{step}", f"partial-{seed}-{step}",
                    f"missing-{seed}-{step}",
                ])
                _append_output(
                    log, f"out-{seed}-{step}",
                    rng.choice(["done", "", "progress"]), call_id=call_id,
                )
            elif action == "tool":
                _append_tool_round(log, f"tool-{seed}-{step}")
            elif action == "failure":
                _append_failure(log, f"fail-{seed}-{step}")
            elif action == "user":
                user_input(log, "s", f"user-{seed}-{step}", "interrupt")
            elif action == "pause":
                try:
                    await harness.store.apply(
                        "s", f"pause-{seed}-{step}", "pause", digest({"pause": step}),
                        lambda current: (None, "noop") if current is None
                        else (pause(current, "monkey"), "paused"),
                    )
                except ValueError:
                    pass
            elif action == "resume":
                try:
                    await harness.store.apply(
                        "s", f"resume-{seed}-{step}", "resume", digest({"resume": step}),
                        lambda current: (None, "noop") if current is None
                        else (resume(current), "resumed"),
                    )
                except ValueError:
                    pass
            elif action == "complete":
                try:
                    await harness.store.apply(
                        "s", f"complete-{seed}-{step}", "update", digest({"complete": step}),
                        lambda current: (None, "noop") if current is None
                        else (apply_model_update(current, "complete"), "complete"),
                    )
                except ValueError:
                    pass
            elif action == "blocked":
                try:
                    await harness.store.apply(
                        "s", f"blocked-{seed}-{step}", "update", digest({"blocked": step}),
                        lambda current: (None, "noop") if current is None
                        else (apply_model_update(current, "blocked"), "blocked"),
                    )
                except ValueError:
                    pass
            elif action == "budget":
                try:
                    new_budget = rng.choice([None, 10, 100, 1000])
                    await harness.store.apply(
                        "s", f"budget-{seed}-{step}", "budget",
                        digest({"budget": new_budget}),
                        lambda current, new_budget=new_budget: (
                            None, "noop"
                        ) if current is None else (
                            set_token_budget(current, new_budget), "budget",
                        ),
                    )
                except ValueError:
                    pass
            elif action == "clear":
                try:
                    await harness.store.apply(
                        "s", f"clear-{seed}-{step}", "clear", digest({"clear": step}),
                        lambda current: (None, "cleared"),
                    )
                except ValueError:
                    pass
            else:
                try:
                    await _create(harness, rng)
                except ValueError:
                    pass
            if rng.random() < 0.1:
                runtime = GoalRuntime(harness.store, cast(Any, _Catalog(log)), _stats)

            current = await harness.store.get("s")
            if current is None:
                previous_goal_id = None
                continue
            assert decode_goal("s", encode_goal(current)) == current
            if current.goal_id == previous_goal_id:
                assert current.continuations >= previous_continuations
                assert current.tokens_used >= previous_tokens
                assert current.settled_through_seq >= previous_settled
            previous_goal_id = current.goal_id
            previous_continuations = current.continuations
            previous_tokens = current.tokens_used
            previous_settled = current.settled_through_seq

        # 结构事实：一个 Session 最多一条 goal owner 记录。
        goals = [
            key for key, _record in log.owner("goal").list()
            if key.startswith("goal:")
        ]
        assert len(goals) <= 1
    finally:
        log.close()


@pytest.mark.asyncio
async def test_monkey_terminal_rules_are_reachable(tmp_path):
    harness = goal_harness(tmp_path)
    log = harness.log
    try:
        _append_output(log, "setup", "setup")
        rng = random.Random(7)
        goal, _ = await _create(harness, rng, token_budget=10)
        assert goal is not None
        runtime = GoalRuntime(harness.store, cast(Any, _Catalog(log)), _stats)
        await runtime.advance("s")
        _append_output(log, "budget-out", "done", call_id="exact-budget")
        await runtime.advance("s")
        current = await harness.store.get("s")
        assert current is not None and current.status is GoalStatus.BUDGET_LIMITED

        # 另一个 Session 走空转熔断。
        _append_output(log, "setup-2", "setup", session_id="s2")
        await _create(harness, rng, session_id="s2")
        runtime2 = GoalRuntime(harness.store, cast(Any, _Catalog(log)), _stats)
        await runtime2.advance("s2")
        for index in range(3):
            _append_output(log, f"empty-{index}", "", session_id="s2")
            await runtime2.advance("s2")
        blocked = await harness.store.get("s2")
        assert blocked is not None and blocked.status is GoalStatus.BLOCKED
    finally:
        log.close()
