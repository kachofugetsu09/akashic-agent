"""目标工具的 prepare/invoke/query 语义。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from plugins.goal.state import GoalStatus
from plugins.goal.tools import CreateGoal, GetGoal, UpdateGoal
from plugins.ledger.contract import CallRef, ContentPart, Message, Output
from plugins.tools.contract import CallSource
from tests.support.goal_harness import goal_harness


def _source(session_id: str = "s") -> CallSource:
    message = Message(
        message_id="call-message", session_id=session_id, seq=0,
        recorded_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
        author="assistant", source="conversation",
        body=Output((ContentPart("text", "work"),), "complete"),
    )
    return CallSource(CallRef("call-message", 0), (message,))


@pytest.mark.asyncio
async def test_create_get_update_and_replace_after_complete(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        create = CreateGoal(harness.store)
        get = GetGoal(harness.store)
        update = UpdateGoal(harness.store)
        source = _source()

        prepared = await create.prepare({"objective": " ship it "}, source)
        assert prepared == {"session_id": "s", "objective": "ship it", "token_budget": None}
        result = await create.invoke("call-1", prepared)  # type: ignore[arg-type]
        assert result.outcome == "success"
        goal = await harness.store.get("s")
        assert goal is not None and goal.status is GoalStatus.ACTIVE

        second = await create.prepare({"objective": "other"}, source)
        blocked = await create.invoke("call-2", second)  # type: ignore[arg-type]
        assert blocked.outcome == "error"
        assert "尚未完成" in str(blocked.parts[0].value)

        updated = await update.prepare({"status": "complete"}, source)
        assert isinstance(updated, dict)
        result = await update.invoke("call-3", updated)
        assert result.outcome == "success"
        assert (await harness.store.get("s")).status is GoalStatus.COMPLETE

        replacement = await create.prepare({"objective": "new goal"}, source)
        created = await create.invoke("call-4", replacement)  # type: ignore[arg-type]
        assert created.outcome == "success"
        assert (await harness.store.get("s")).objective == "new goal"

        fetched = await get.invoke("call-5", {"session_id": "s"})
        assert fetched.outcome == "success"
        assert "new goal" in str(fetched.parts[0].value)
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_update_rejects_missing_goal_and_invalid_transition(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        update = UpdateGoal(harness.store)
        missing = await update.prepare({"status": "complete"}, _source())
        assert isinstance(missing, str)
        assert "没有目标" in missing

        create = CreateGoal(harness.store)
        prepared = await create.prepare({"objective": "ship"}, _source())
        await create.invoke("call-2", prepared)  # type: ignore[arg-type]
        completed = await update.prepare({"status": "complete"}, _source())
        assert isinstance(completed, dict)
        await update.invoke("call-3", completed)
        again = await update.prepare({"status": "blocked"}, _source())
        assert isinstance(again, dict)
        rejected = await update.invoke("call-4", again)
        assert rejected.outcome == "error"
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_update_prepared_snapshot_rejects_replacement_goal(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        create = CreateGoal(harness.store)
        update = UpdateGoal(harness.store)
        source = _source()
        prepared_create = await create.prepare({"objective": "first"}, source)
        await create.invoke("call-1", prepared_create)  # type: ignore[arg-type]
        prepared_update = await update.prepare({"status": "complete"}, source)
        assert isinstance(prepared_update, dict)

        # 模拟目标在 prepare 与 invoke 之间被清除并替换。
        await harness.store.apply(
            "s", "op-clear", "clear", "2" * 64,
            lambda current: (None, "cleared"),
        )
        replacement = await create.prepare({"objective": "second"}, source)
        await create.invoke("call-2", replacement)  # type: ignore[arg-type]

        stale = await update.invoke("call-3", prepared_update)
        assert stale.outcome == "error"
        assert "变化" in str(stale.parts[0].value)
        assert (await harness.store.get("s")).objective == "second"
    finally:
        harness.log.close()


@pytest.mark.asyncio
async def test_prepare_rejects_bad_arguments(tmp_path):
    harness = goal_harness(tmp_path)
    try:
        create = CreateGoal(harness.store)
        assert isinstance(await create.prepare({"objective": ""}, _source()), str)
        assert isinstance(await create.prepare({"objective": "x", "token_budget": 0}, _source()), str)
        assert isinstance(await create.prepare({"objective": "x"}, None), str)
        update = UpdateGoal(harness.store)
        assert isinstance(await update.prepare({"status": "active"}, _source()), str)
        get = GetGoal(harness.store)
        assert isinstance(await get.prepare({"unexpected": 1}, _source()), str)
    finally:
        harness.log.close()
