"""目标状态的只读公共合同。"""

from __future__ import annotations

from typing import Protocol

from agent.plugin_composition import ServiceKey

from .state import Goal


class GoalStatusReader(Protocol):
    async def get(self, session_id: str) -> Goal | None: ...
    async def active_sessions(self) -> frozenset[str]: ...
    async def runtime_sessions(self) -> frozenset[str]: ...


GOAL_STATUS = ServiceKey[GoalStatusReader]("goal.status.v1")
