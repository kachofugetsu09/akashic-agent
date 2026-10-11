"""Narrow directory capability provided by the ordinary file tools plugin."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, Effect, ServiceKey


@dataclass(frozen=True, slots=True)
class DirectorySnapshot:
    path: str | None
    revision: int | None


class WorkingDirectories(Protocol):
    async def register_default(
        self, ctx: Context, *, dimension: str, read: Callable[[str], str | None],
    ) -> Effect: ...
    def snapshot(self, session_id: str) -> DirectorySnapshot: ...
    async def check_directory(self, path: str, *, base_dir: str | None = None) -> str: ...
    async def inspect(self, path: str | None) -> Mapping[str, object]: ...
    async def browse(self, path: str, *, after: str | None = None) -> Mapping[str, object]: ...


WORKING_DIRECTORY = ServiceKey[WorkingDirectories]("standard_tools.working_directory.v1")


class SkillReader(Protocol):
    """技能目录只读投影的窄输入。"""

    async def list_skills(self) -> tuple[Mapping[str, object], ...]: ...

    async def list_sources(self) -> tuple[Mapping[str, object], ...]: ...


# RuntimeInspection._bind_optional 将此 key 传给 ctx.inject 的技能目录子 Fiber。
SKILL_INSPECTION = ServiceKey[SkillReader]("standard_tools.skill_inspection.v1")
