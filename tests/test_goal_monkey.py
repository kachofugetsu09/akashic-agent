"""目标状态与续跑事务的确定性 monkey 测试。"""

from __future__ import annotations

import random

import pytest

from plugins.goal.runtime import GoalRuntime
from plugins.goal.state import (
    GoalStatus, apply_model_update, new_goal, pause, resume, set_token_budget,
)
from plugins.goal.store import decode_goal, digest, encode_goal
from plugins.ledger.contract import ContentPart, ContentReferences, Control, Output
from tests.support.goal_harness import goal_harness, user_input


class _Catalog:
    def __init__(self, log):
        self._log = log

    def reader(self, session_id):
        return self._log.reader(session_id)


def _missing_usage(call_id: str):
    raise KeyError(call_id)


def _append_output(log, message_id, text="done"):
    writer = log.writer(
        "s", author="assistant", source="conversation", body_types=(Output,),
        content={"text": lambda part: ContentReferences()},
    )
    writer.append(message_id, Output((ContentPart("text", text),), "complete"))
    writer.expire()


def _append_failure(log, message_id):
    head = log.reader("s").head(source="conversation")
    writer = log.writer(
        "s", author="app", source="conversation", body_types=(Control,), content={},
    )
    writer.append(message_id, Control("failure", max(0, head), "model failed"))
    writer.expire()


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(200))
async def test_monkey_goal_invariants(tmp_path, seed):
    rng = random.Random(seed)
    harness = goal_harness(tmp_path)
    log = harness.log
    try:
        _append_output(log, "setup", "setup")

        async def create():
            head = harness.store.conversation_head("s")

            def action(current):
                if current is not None and current.status is not GoalStatus.COMPLETE:
                    raise ValueError("unfinished")
                return (
                    new_goal(
                        session_id="s", goal_id=f"g-{rng.randrange(10**9)}",
                        objective="ship it", settled_through_seq=head,
                    ),
                    "created",
                )

            try:
                return await harness.store.apply(
                    "s", f"create-{rng.randrange(10**9)}", "create",
                    digest({"objective": "ship it"}), action,
                )
            except Exception:
                return None, ""

        goal, _ = await create()
        runtime = GoalRuntime(harness.store, _Catalog(log), _missing_usage)
        previous_goal_id = goal.goal_id if goal is not None else None
        previous_continuations = goal.continuations if goal is not None else 0
        previous_tokens = goal.tokens_used if goal is not None else 0
        previous_settled = goal.settled_through_seq if goal is not None else -1

        for step in range(50):
            action = rng.randrange(11)
            if action == 0:
                await runtime.advance("s")
            elif action == 1:
                _append_output(log, f"out-{step}", rng.choice(["done", "", "progress"]))
            elif action == 2:
                _append_failure(log, f"fail-{step}")
            elif action == 3:
                user_input(log, "s", f"user-{step}", "interrupt")
            elif action == 4:
                try:
                    await harness.store.apply(
                        "s", f"pause-{step}", "pause", digest({"pause": step}),
                        lambda current: (None, "noop") if current is None
                        else (pause(current, "monkey"), "paused"),
                    )
                except Exception:
                    pass
            elif action == 5:
                await create()
            elif action == 6:
                try:
                    await harness.store.apply(
                        "s", f"clear-{step}", "clear", digest({"clear": step}),
                        lambda current: (None, "cleared"),
                    )
                except Exception:
                    pass
            elif action == 7:
                try:
                    await harness.store.apply(
                        "s", f"resume-{step}", "resume", digest({"resume": step}),
                        lambda current: (None, "noop") if current is None
                        else (resume(current), "resumed"),
                    )
                except Exception:
                    pass
            elif action == 8:
                try:
                    await harness.store.apply(
                        "s", f"complete-{step}", "update", digest({"complete": step}),
                        lambda current: (None, "noop") if current is None
                        else (apply_model_update(current, "complete"), "complete"),
                    )
                except Exception:
                    pass
            elif action == 9:
                try:
                    await harness.store.apply(
                        "s", f"blocked-{step}", "update", digest({"blocked": step}),
                        lambda current: (None, "noop") if current is None
                        else (apply_model_update(current, "blocked"), "blocked"),
                    )
                except Exception:
                    pass
            else:
                try:
                    budget = rng.choice([None, 100, 1000])
                    await harness.store.apply(
                        "s", f"budget-{step}", "budget", digest({"budget": budget}),
                        lambda current, budget=budget: (None, "noop") if current is None
                        else (set_token_budget(current, budget), "budget"),
                    )
                except Exception:
                    pass
            if rng.random() < 0.1:
                runtime = GoalRuntime(harness.store, _Catalog(log), _missing_usage)

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
    finally:
        log.close()
