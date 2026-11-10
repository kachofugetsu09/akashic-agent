"""目标 owner 记录的 CAS、幂等与续跑追加测试。"""

from __future__ import annotations

from typing import cast

import pytest

from plugins.goal.state import Goal, GoalStatus, new_goal
from plugins.goal.store import RoundSettlement, digest, goal_key
from plugins.ledger.contract import (
    ContentReferences,
    ContentPart,
    MessageConflict,
    Output,
)
from tests.support.goal_harness import goal_harness, user_input


async def _create(store, session_id="s", operation="op-create", objective="ship it"):
    def action(current):
        assert current is None
        return (
            new_goal(
                session_id=session_id, goal_id="g", objective=objective,
                settled_through_seq=0,
            ),
            "created",
        )

    return await store.apply(
        session_id, operation, "create", digest({"objective": objective}), action,
        expected_head=await store.conversation_head(session_id),
    )


@pytest.mark.asyncio
async def test_create_is_idempotent_and_digest_guarded(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        goal, text = await _create(harness.store)
        assert text == "created"
        assert goal is not None and goal.status is GoalStatus.ACTIVE
        again, again_text = await _create(harness.store)
        assert again == goal and again_text == "created"
        with pytest.raises(MessageConflict):
            await harness.store.apply(
                "s", "op-create", "create", digest({"objective": "other"}),
                lambda current: (current, "noop"),
                expected_head=await harness.store.conversation_head("s"),
            )
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_start_appends_continuation_and_settle_appends_next(tmp_path):
    harness = goal_harness(tmp_path)
    log = harness.log
    try:
        goal, _ = await _create(harness.store)
        assert goal is not None
        version = harness.store._open().read(goal_key("s")).version
        head = await harness.store.conversation_head("s")
        started = await harness.store.start(
            "s", expected_goal_id="g", expected_version=version, expected_head=head,
            message_id="goal-continuation:g:1", text="continue",
        )
        assert started.continuations == 1
        inputs = [
            message for message in log.reader("s").snapshot()
            if message.message_id == "goal-continuation:g:1"
        ]
        assert len(inputs) == 1
        writer = log.writer(
            "s", author="assistant", source="conversation", body_types=(Output,),
            content={"text": lambda part: ContentReferences()},
        )
        terminal = writer.append(
            "final", Output((ContentPart("text", "done"),), "complete"),
        )
        writer.expire()
        version = harness.store._open().read(goal_key("s")).version
        settled = await harness.store.settle(
            "s", expected_goal_id="g", expected_version=version,
            expected_head=terminal.seq,
            settlement=RoundSettlement(
                activity=True, tokens=10, coverage="exact",
                settled_through_seq=terminal.seq,
                last_settled_message_id="final", reason=None, append=True,
            ),
            next_message_id="goal-continuation:g:2", next_text="continue again",
        )
        assert settled.continuations == 2
        assert settled.tokens_used == 10
        assert log.reader("s").get("goal-continuation:g:2") is not None
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_settle_rejects_head_change(tmp_path):
    harness = goal_harness(tmp_path)
    log = harness.log
    try:
        goal, _ = await _create(harness.store)
        assert goal is not None
        version = harness.store._open().read(goal_key("s")).version
        head = await harness.store.conversation_head("s")
        await harness.store.start(
            "s", expected_goal_id="g", expected_version=version, expected_head=head,
            message_id="goal-continuation:g:1", text="continue",
        )
        writer = log.writer(
            "s", author="assistant", source="conversation", body_types=(Output,),
            content={"text": lambda part: ContentReferences()},
        )
        terminal = writer.append(
            "final", Output((ContentPart("text", "done"),), "complete"),
        )
        writer.expire()
        stale_head = terminal.seq
        user_input(log, "s", "user-1", "let me add something")
        version = harness.store._open().read(goal_key("s")).version
        with pytest.raises(MessageConflict):
            await harness.store.settle(
                "s", expected_goal_id="g", expected_version=version,
                expected_head=stale_head,
                settlement=RoundSettlement(
                    activity=True, tokens=0, coverage="exact",
                    settled_through_seq=terminal.seq,
                    last_settled_message_id="final", reason=None, append=True,
                ),
                next_message_id="goal-continuation:g:2", next_text="continue",
            )
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_operation_receipts_are_bounded(tmp_path, monkeypatch):
    from plugins.goal import store as store_module
    monkeypatch.setattr(store_module, "MAX_OPERATIONS", 3)
    harness = goal_harness(tmp_path)
    try:
        for index in range(8):
            await harness.store.apply(
                "s", f"op-{index}", "clear", digest({"clear": index}),
                lambda current: (None, "cleared"),
            )
        keys = [
            key for key, _record in harness.store._open().list()
            if key.startswith("goalop:")
        ]
        assert len(keys) <= 3
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_runtime_sessions_includes_inactive_with_pending_continuation(tmp_path):
    from dataclasses import replace
    harness = goal_harness(tmp_path)
    try:
        await _create(harness.store)
        assert await harness.store.runtime_sessions() == frozenset({"s"})

        await harness.store.apply(
            "s", "op-complete", "update", digest({"status": "complete"}),
            lambda current: (
                replace(
                    cast(Goal, current), status=GoalStatus.COMPLETE,
                    continuation_id="goal-continuation:g:1",
                ),
                "complete",
            ),
        )
        assert await harness.store.active_sessions() == frozenset()
        assert await harness.store.runtime_sessions() == frozenset({"s"})

        await harness.store.apply(
            "s", "op-settled", "update", digest({"settled": True}),
            lambda current: (replace(cast(Goal, current), continuation_id=None), "settled"),
        )
        assert await harness.store.runtime_sessions() == frozenset()
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_active_sessions_only_returns_active(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        await _create(harness.store)
        assert await harness.store.active_sessions() == frozenset({"s"})
        await harness.store.apply(
            "s", "op-pause", "pause", digest({"pause": True}),
            lambda current: (None, "cleared"),
        )
        assert await harness.store.active_sessions() == frozenset()
    finally:
        harness.log.close()
