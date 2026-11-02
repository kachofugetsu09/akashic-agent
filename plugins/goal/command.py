"""`/goal` 命令：设置、查看、暂停、恢复、清除与预算。"""

from __future__ import annotations

from plugins.commands.contract import CommandInvocation, CommandResult
from plugins.ledger.contract import MessageConflict

from .state import (
    Goal,
    GoalStatus,
    new_goal,
    normalize_objective,
    normalize_token_budget,
    pause,
    resume,
    set_token_budget,
    status_text,
)
from .store import GoalStore, digest

_CLEAR_ALIASES = frozenset({"clear", "stop", "off", "reset", "none", "cancel"})
_RETRIES = 3


def _error(text: str) -> CommandResult:
    return CommandResult("error", text)


def _success(text: str) -> CommandResult:
    return CommandResult("success", text)


class GoalCommand:
    def __init__(self, store: GoalStore) -> None:
        self._store = store

    async def __call__(self, invocation: CommandInvocation) -> CommandResult:
        raw = invocation.raw_input.strip()
        session_id = invocation.session_key
        if not raw:
            goal = await self._store.get(session_id)
            return _success("当前没有目标" if goal is None else status_text(goal))
        lowered = raw.lower()
        if lowered in _CLEAR_ALIASES:
            return await self._clear(session_id, invocation)
        if lowered == "pause":
            return await self._pause(session_id, invocation)
        if lowered == "resume":
            return await self._resume(session_id, invocation)
        if lowered == "budget" or lowered.startswith("budget "):
            return await self._budget(session_id, invocation, raw)
        # 自由文本 objective 里可能以 clear/pause 等词开头；只有整条输入
        # 等于子命令时才走控制分支，其余一律当目标。
        return await self._set(session_id, invocation, raw)

    async def _set(
        self, session_id: str, invocation: CommandInvocation, raw: str,
    ) -> CommandResult:
        try:
            objective = normalize_objective(raw)
        except ValueError as error:
            return _error(str(error))
        last_error: Exception | None = None
        for _attempt in range(_RETRIES):
            head = await self._store.conversation_head(session_id)

            def action(current: Goal | None, *, head: int = head) -> tuple[Goal | None, str]:
                if current is not None and current.status is not GoalStatus.COMPLETE:
                    raise ValueError("当前目标尚未完成；先完成或清除它")
                goal = new_goal(
                    session_id=session_id,
                    goal_id=f"goal-{digest([session_id, objective, invocation.message_id])[:24]}",
                    objective=objective,
                    settled_through_seq=head,
                )
                return goal, f"目标已设置：{objective}"

            try:
                goal, text = await self._store.apply(
                    session_id, invocation.message_id, "command", digest({"set": objective}),
                    action, expected_head=head,
                )
            except MessageConflict as error:
                last_error = error
                continue
            except ValueError as error:
                return _error(str(error))
            return _success(text + "\n" + status_text(goal) if goal is not None else text)
        return _error(f"目标设置冲突，请重试：{last_error}")

    async def _clear(self, session_id: str, invocation: CommandInvocation) -> CommandResult:
        def action(current: Goal | None) -> tuple[Goal | None, str]:
            if current is None:
                return None, "当前没有目标"
            return None, f"目标已清除：{current.objective}"

        _goal, text = await self._store.apply(
            session_id, invocation.message_id, "command", digest({"clear": True}), action,
        )
        return _success(text)

    async def _pause(self, session_id: str, invocation: CommandInvocation) -> CommandResult:
        def action(current: Goal | None) -> tuple[Goal | None, str]:
            if current is None:
                raise ValueError("当前没有目标")
            goal = pause(current, "用户暂停目标")
            return goal, f"目标已暂停：{goal.objective}"

        try:
            _goal, text = await self._store.apply(
                session_id, invocation.message_id, "command", digest({"pause": True}), action,
            )
        except ValueError as error:
            return _error(str(error))
        return _success(text)

    async def _resume(self, session_id: str, invocation: CommandInvocation) -> CommandResult:
        def action(current: Goal | None) -> tuple[Goal | None, str]:
            if current is None:
                raise ValueError("当前没有目标")
            goal = resume(current)
            return goal, f"目标已恢复：{goal.objective}"

        try:
            _goal, text = await self._store.apply(
                session_id, invocation.message_id, "command", digest({"resume": True}), action,
            )
        except ValueError as error:
            return _error(str(error))
        return _success(text)

    async def _budget(
        self, session_id: str, invocation: CommandInvocation, raw: str,
    ) -> CommandResult:
        parts = raw.split(maxsplit=2)
        if len(parts) != 2:
            return _error("用法：/goal budget <正整数> 或 /goal budget clear")
        value = parts[1].strip().lower()
        try:
            budget = None if value == "clear" else normalize_token_budget(int(value))
        except (ValueError, TypeError) as error:
            return _error(f"预算无效：{error}")

        def action(current: Goal | None) -> tuple[Goal | None, str]:
            if current is None:
                raise ValueError("当前没有目标")
            goal = set_token_budget(current, budget)
            return goal, (
                "目标预算已清除" if budget is None
                else f"目标预算已设为 {budget} tokens"
            )

        try:
            goal, text = await self._store.apply(
                session_id, invocation.message_id, "command",
                digest({"budget": budget}), action,
            )
        except ValueError as error:
            return _error(str(error))
        return _success(text + ("" if goal is None else "\n" + status_text(goal)))
