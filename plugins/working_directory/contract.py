"""会话工作目录的窄能力；文件、命令等工具在调用时借用，缺席时按各自配置目录工作。"""

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
    # 按 Session 当前目录固定目标路径；返回实际路径与当时依赖的目录。
    async def resolve_target(
        self, session_id: str | None, path: str, *, legacy_base: str | None = None,
    ) -> tuple[str, str | None]: ...


WORKING_DIRECTORY = ServiceKey[WorkingDirectories]("working_directory.v1")

