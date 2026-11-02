"""目标工具：get_goal / create_goal / update_goal。

工具只改变目标状态，不运行模型；自动续跑由 runtime 负责。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from plugins.ledger.contract import ContentPart, MessageConflict
from plugins.tools.contract import CallSource, Result

from .state import (
    Goal,
    GoalStatus,
    apply_model_update,
    new_goal,
    normalize_objective,
    normalize_token_budget,
    now_utc,
)
from .store import GoalStale, GoalStore, digest

GET_GOAL_TOOL = "get_goal"
CREATE_GOAL_TOOL = "create_goal"
UPDATE_GOAL_TOOL = "update_goal"
_RETRIES = 3


class GoalConflict(ValueError):
    """目标状态不允许这次变更；作为工具错误返回，不写操作回执。"""


class CreateGoalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    objective: str = Field(min_length=1, max_length=4000)
    token_budget: int | None = Field(default=None, gt=0)


class UpdateGoalInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["complete", "blocked", "paused"]


def _session(source: CallSource | None) -> str:
    if source is None or not source.messages:
        raise ValueError("目标工具需要当前调用的会话前缀")
    return source.messages[-1].session_id


def _snapshot(goal: Goal) -> dict[str, object]:
    return {
        "goal_id": goal.goal_id,
        "objective": goal.objective,
        "status": goal.status.value,
        "token_budget": goal.token_budget,
        "tokens_used": goal.tokens_used,
        "remaining_tokens": goal.remaining_tokens,
        "continuations": goal.continuations,
        "consecutive_empty": goal.consecutive_empty,
        "usage_coverage": goal.usage_coverage,
        "last_reason": goal.last_reason,
    }


def _text(payload: Mapping[str, object]) -> Result:
    return Result("success", (ContentPart("text", json.dumps(payload, ensure_ascii=False)),))


class GetGoal:
    idempotent = True
    parallel = True

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None,
    ) -> Mapping[str, object] | str:
        if arguments:
            return "get_goal 不接收参数"
        try:
            return {"session_id": _session(source)}
        except ValueError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        goal = await self._store.get(cast(str, arguments["session_id"]))
        return _text({"goal": None if goal is None else _snapshot(goal)})

    async def query(self, key: str) -> Result | None:
        return None


class CreateGoal:
    idempotent = True

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None,
    ) -> Mapping[str, object] | str:
        try:
            request = CreateGoalInput.model_validate(dict(arguments))
            session_id = _session(source)
            objective = normalize_objective(request.objective)
            token_budget = normalize_token_budget(request.token_budget)
        except (ValidationError, ValueError) as error:
            return str(error)
        return {
            "session_id": session_id,
            "objective": objective,
            "token_budget": token_budget,
        }

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        session_id = cast(str, arguments["session_id"])
        objective = cast(str, arguments["objective"])
        token_budget = cast(int | None, arguments["token_budget"])
        request: dict[str, object] = {"objective": objective, "token_budget": token_budget}
        last_error: Exception | None = None
        for _attempt in range(_RETRIES):
            head = await self._store.conversation_head(session_id)

            def action(current: Goal | None, *, head: int = head) -> tuple[Goal | None, str]:
                if current is not None and current.status is not GoalStatus.COMPLETE:
                    raise GoalConflict(
                        "当前目标尚未完成；先完成或清除它，create_goal 只替换 complete 的目标"
                    )
                goal = new_goal(
                    session_id=session_id,
                    goal_id=f"goal-{digest([session_id, objective, key])[:24]}",
                    objective=objective,
                    token_budget=token_budget,
                    settled_through_seq=head,
                )
                return goal, f"目标已创建：{objective}"

            try:
                goal, response = await self._store.apply(
                    session_id, key, "create", digest(request), action,
                    expected_head=head,
                )
            except MessageConflict as error:
                last_error = error
                continue
            except GoalConflict as error:
                return Result("error", (ContentPart("text", str(error)),))
            return _text({
                "response": response,
                "goal": None if goal is None else _snapshot(goal),
            })
        return Result("error", (ContentPart("text", f"目标创建冲突，请重试：{last_error}"),))

    async def query(self, key: str) -> Result | None:
        # 操作 key 不含 Session，跨会话扫描可能返回别的目标；重放由 invoke 的
        # OWNER_STATE 幂等事务完成，这里不提供跨会话查询。
        return None

class UpdateGoal:
    idempotent = True

    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None,
    ) -> Mapping[str, object] | str:
        try:
            request = UpdateGoalInput.model_validate(dict(arguments))
            session_id = _session(source)
        except (ValidationError, ValueError) as error:
            return str(error)
        snapshot = await self._store.get_with_version(session_id)
        if snapshot is None:
            return "当前会话没有目标"
        goal, version = snapshot
        return {
            "session_id": session_id,
            "status": request.status,
            "goal_id": goal.goal_id,
            "version": version,
        }

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        session_id = cast(str, arguments["session_id"])
        status = cast(Literal["complete", "blocked", "paused"], arguments["status"])
        goal_id = arguments.get("goal_id")
        version = arguments.get("version")
        if not isinstance(goal_id, str) or not isinstance(version, int):
            return Result("error", (ContentPart("text", "目标操作缺少 prepare 快照"),))

        def action(current: Goal | None) -> tuple[Goal | None, str]:
            if current is None:
                raise GoalConflict("当前会话没有目标")
            goal = apply_model_update(current, status, now=now_utc())
            return goal, f"目标状态已更新为 {goal.status.value}"

        try:
            goal, response = await self._store.apply(
                session_id, key, "update", digest({"status": status}), action,
                expected_goal=(goal_id, version),
            )
        except GoalStale as error:
            return Result("error", (ContentPart("text", str(error)),))
        except GoalConflict as error:
            return Result("error", (ContentPart("text", str(error)),))
        except ValueError as error:
            return Result("error", (ContentPart("text", str(error)),))
        return _text({
            "response": response,
            "goal": None if goal is None else _snapshot(goal),
        })

    async def query(self, key: str) -> Result | None:
        return None
