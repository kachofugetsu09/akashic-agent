"""目标 owner 记录的 CAS、幂等与续跑追加测试。"""

from __future__ import annotations

import pytest

from plugins.goal.state import new_goal
from plugins.goal.store import RoundSettlement, digest
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
async def test_start_appends_continuation_and_settle_appends_next(tmp_path):
    harness = goal_harness(tmp_path)
    log = harness.log
    try:
        goal, _ = await _create(harness.store)
        assert goal is not None
        version = await harness.store.version("s")
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
        version = await harness.store.version("s")
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
        version = await harness.store.version("s")
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
        version = await harness.store.version("s")
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
