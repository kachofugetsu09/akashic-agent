"""目标状态机：只处理值，不碰磁盘、消息或插件服务。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

OBJECTIVE_MAX_LENGTH = 4000
# 与 Codex 的 blocked audit 对齐：同一阻塞至少连续三个目标轮次才允许标记 blocked。
BLOCKED_AFTER_EMPTY_TURNS = 3
# 单目标自动续跑上限；达到后暂停，清除后重新设置目标才能继续。
DEFAULT_MAX_CONTINUATIONS = 100
DEFAULT_MAX_TOKEN_BUDGET = 100_000_000

GoalMutationStatus = Literal["complete", "blocked", "paused"]
UsageCoverage = Literal["exact", "partial", "unavailable"]
_COVERAGE_RANK = {"exact": 0, "partial": 1, "unavailable": 2}


class GoalStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    BUDGET_LIMITED = "budget_limited"
    COMPLETE = "complete"


_TERMINAL = frozenset({GoalStatus.COMPLETE, GoalStatus.BUDGET_LIMITED})
# 不会自动续跑、但用户可以显式恢复的状态。
_STOPPED = frozenset({GoalStatus.PAUSED, GoalStatus.BLOCKED})


@dataclass(frozen=True, slots=True)
class Goal:
    session_id: str
    goal_id: str
    objective: str
    status: GoalStatus
    token_budget: int | None
    tokens_used: int
    continuations: int
    consecutive_empty: int
    settled_through_seq: int
    last_settled_message_id: str | None
    continuation_id: str | None
    usage_coverage: UsageCoverage
    created_at: datetime
    updated_at: datetime
    last_reason: str | None = None

    @property
    def remaining_tokens(self) -> int | None:
        if self.token_budget is None:
            return None
        return max(0, self.token_budget - self.tokens_used)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL

    @property
    def stopped(self) -> bool:
        return self.status in _TERMINAL or self.status in _STOPPED

    @property
    def active(self) -> bool:
        return self.status is GoalStatus.ACTIVE


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def worst_coverage(*values: UsageCoverage) -> UsageCoverage:
    return max(values, key=_COVERAGE_RANK.__getitem__)


def normalize_objective(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("目标必须是非空文本")
    objective = value.strip()
    if not objective:
        raise ValueError("目标必须是非空文本")
    if len(objective) > OBJECTIVE_MAX_LENGTH:
        raise ValueError(f"目标最多 {OBJECTIVE_MAX_LENGTH} 个字符（收到 {len(objective)}）")
    return objective


def normalize_token_budget(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError("token 预算必须是正整数")
    if value > DEFAULT_MAX_TOKEN_BUDGET:
        raise ValueError(f"token 预算最多 {DEFAULT_MAX_TOKEN_BUDGET}")
    return value


def new_goal(
    *, session_id: str, goal_id: str, objective: str,
    token_budget: int | None = None, settled_through_seq: int = -1,
    now: datetime | None = None,
) -> Goal:
    if not session_id or not goal_id:
        raise ValueError("目标需要 Session 与 goal_id")
    if type(settled_through_seq) is not int or settled_through_seq < -1:
        raise ValueError("结算游标必须 >= -1")
    stamp = now or now_utc()
    return Goal(
        session_id=session_id,
        goal_id=goal_id,
        objective=normalize_objective(objective),
        status=GoalStatus.ACTIVE,
        token_budget=normalize_token_budget(token_budget),
        tokens_used=0,
        continuations=0,
        consecutive_empty=0,
        settled_through_seq=settled_through_seq,
        last_settled_message_id=None,
        continuation_id=None,
        usage_coverage="exact",
        created_at=stamp,
        updated_at=stamp,
    )


def _touch(goal: Goal, now: datetime, **changes: object) -> Goal:
    return replace(goal, updated_at=now, **changes)


def mark_continuation(
    goal: Goal, message_id: str, *, now: datetime | None = None,
) -> Goal:
    """写入下一条续跑输入时调用；续跑次数是单个 goal_id 内的单调计数。"""
    if not goal.active:
        raise ValueError(f"目标状态 {goal.status} 不能继续")
    if not message_id:
        raise ValueError("续跑消息身份不能为空")
    if goal.continuations >= DEFAULT_MAX_CONTINUATIONS:
        raise ValueError("已达到自动续跑上限；请清除后重新设置目标")
    return _touch(
        goal, now or now_utc(),
        continuations=goal.continuations + 1,
        continuation_id=message_id,
    )


def settle_round(
    goal: Goal, *,
    activity: bool,
    tokens: int,
    coverage: UsageCoverage,
    settled_through_seq: int,
    last_settled_message_id: str | None,
    reason: str | None = None,
    preempted: bool = False,
    allow_inactive: bool = False,
    now: datetime | None = None,
) -> Goal:
    """结算一个已结束的目标轮次；状态推进与计数只在这里发生。

    ``allow_inactive`` 只用于目标在轮次进行中被模型标记为 complete/blocked/paused
    后的收尾：仍记录该轮 token 与 coverage，但不改终态，也不触发续跑。
    """
    if not goal.active and not allow_inactive:
        raise ValueError(f"目标状态 {goal.status} 不接受轮次结算")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("token 增量必须是非负整数")
    if type(settled_through_seq) is not int or settled_through_seq < goal.settled_through_seq:
        raise ValueError("结算游标不能回退")
    stamp = now or now_utc()
    # 用户抢占不是模型空转：不增加也不清零连续空转计数。
    empty = goal.consecutive_empty if preempted else (
        0 if activity else goal.consecutive_empty + 1
    )
    if not goal.active:
        # 终态目标只补记最后一轮用量；completed/blocked/paused 的原因保持不变。
        return _touch(
            goal, stamp,
            tokens_used=goal.tokens_used + tokens,
            settled_through_seq=settled_through_seq,
            last_settled_message_id=last_settled_message_id,
            continuation_id=None,
            usage_coverage=worst_coverage(goal.usage_coverage, coverage),
        )
    updated = _touch(
        goal, stamp,
        tokens_used=goal.tokens_used + tokens,
        consecutive_empty=empty,
        settled_through_seq=settled_through_seq,
        last_settled_message_id=last_settled_message_id,
        continuation_id=None,
        usage_coverage=worst_coverage(goal.usage_coverage, coverage),
        last_reason=reason if reason is not None else goal.last_reason,
    )
    if updated.token_budget is not None and updated.tokens_used >= updated.token_budget:
        return _touch(
            updated, stamp,
            status=GoalStatus.BUDGET_LIMITED,
            last_reason="目标 token 预算已耗尽",
        )
    if empty >= BLOCKED_AFTER_EMPTY_TURNS:
        return _touch(
            updated, stamp,
            status=GoalStatus.BLOCKED,
            last_reason=f"连续 {BLOCKED_AFTER_EMPTY_TURNS} 个目标轮次没有产出",
        )
    if updated.continuations >= DEFAULT_MAX_CONTINUATIONS:
        return _touch(
            updated, stamp,
            status=GoalStatus.PAUSED,
            last_reason="达到自动续跑上限",
        )
    return updated


def apply_model_update(
    goal: Goal, status: GoalMutationStatus, *, now: datetime | None = None,
) -> Goal:
    """模型只能完成、阻塞或暂停一个 active 目标。"""
    if not goal.active:
        raise ValueError(f"目标状态 {goal.status} 不接受模型状态更新")
    if status not in ("complete", "blocked", "paused"):
        raise ValueError(f"未知的目标状态更新: {status}")
    stamp = now or now_utc()
    reason = {
        "complete": "目标已完成",
        "blocked": "目标被阻塞",
        "paused": "目标已暂停",
    }[status]
    return _touch(goal, stamp, status=GoalStatus(status), last_reason=reason)


def pause(goal: Goal, reason: str, *, now: datetime | None = None) -> Goal:
    if not goal.active:
        raise ValueError(f"目标状态 {goal.status} 不能暂停")
    return _touch(goal, now or now_utc(), status=GoalStatus.PAUSED, last_reason=reason)


def resume(goal: Goal, *, now: datetime | None = None) -> Goal:
    if goal.status not in _STOPPED:
        raise ValueError(f"目标状态 {goal.status} 不能恢复")
    if goal.continuations >= DEFAULT_MAX_CONTINUATIONS:
        raise ValueError("已达到自动续跑上限；请清除后重新设置目标")
    return _touch(
        goal, now or now_utc(),
        status=GoalStatus.ACTIVE,
        consecutive_empty=0,
        last_reason=None,
    )


def set_token_budget(
    goal: Goal, budget: int | None, *, now: datetime | None = None,
) -> Goal:
    if goal.status not in {
        GoalStatus.ACTIVE, GoalStatus.PAUSED, GoalStatus.BUDGET_LIMITED,
    }:
        raise ValueError(f"目标状态 {goal.status} 不能修改预算")
    stamp = now or now_utc()
    updated = _touch(goal, stamp, token_budget=normalize_token_budget(budget))
    if updated.token_budget is not None and updated.tokens_used >= updated.token_budget:
        return _touch(
            updated, stamp,
            status=GoalStatus.BUDGET_LIMITED,
            last_reason="目标 token 预算已耗尽",
        )
    if goal.status is GoalStatus.BUDGET_LIMITED:
        if goal.continuations >= DEFAULT_MAX_CONTINUATIONS:
            return _touch(
                updated, stamp,
                status=GoalStatus.PAUSED,
                last_reason="达到自动续跑上限",
            )
        return _touch(
            updated, stamp,
            status=GoalStatus.ACTIVE,
            last_reason=None,
        )
    return updated


def status_text(goal: Goal) -> str:
    lines = [
        f"目标：{goal.objective}",
        f"状态：{goal.status.value}",
        f"续跑：{goal.continuations} 轮",
        f"连续空转：{goal.consecutive_empty} 轮",
        f"用量：{goal.tokens_used} tokens"
        + ("（未设预算）" if goal.token_budget is None
           else f" / {goal.token_budget}（剩余 {goal.remaining_tokens}）"),
        f"用量置信度：{goal.usage_coverage}",
    ]
    if goal.last_reason:
        lines.append(f"最近原因：{goal.last_reason}")
    return "\n".join(lines)
