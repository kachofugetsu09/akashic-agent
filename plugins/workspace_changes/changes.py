"""会话工作目录的改动统计：未提交的改动按需对比；每轮的窗口见 turns.py。"""
from __future__ import annotations

from dataclasses import asdict

from agent.plugin_composition import Context, ServiceKey
from core.common.file_io import run_file_io
from plugins.standard_tools.contract import WORKING_DIRECTORY

from .git import Changes, Git, GitUnavailable, Repo
from .turns import TurnWindows


class WorkspaceChanges:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self.git = Git(ctx)
        self.windows = TurnWindows(ctx, self.git, self.repo)

    # 工作目录在调用时读取；仓库按这一刻的目录解析。
    async def repo(self, session: str) -> Repo:
        directories = self._ctx.require(WORKING_DIRECTORY)
        snapshot = await run_file_io(lambda: directories.snapshot(session))
        return await self.git.resolve(snapshot.path)

    # 未提交：拍一次工作树快照与 HEAD 比较，含未跟踪的新文件；unborn 时与空树比较。
    async def uncommitted(self, session: str) -> dict[str, object]:
        try:
            async with self._ctx.runtime_scope():
                repo = await self.repo(session)
                snapshot = await self.git.snapshot(repo)
                return view(repo, await self.git.diff(repo, snapshot.base, snapshot.tree))
        except GitUnavailable as error:
            return {"status": "unavailable", **error.view()}


def view(repo: Repo, changes: Changes) -> dict[str, object]:
    return {
        "status": "ok", "repo": repo.toplevel, "submodules": repo.submodules,
        "files": [asdict(item) for item in changes.files], "patch": changes.patch, "truncated": changes.truncated,
    }


WORKSPACE_CHANGES = ServiceKey[WorkspaceChanges]("workspace_changes.service.v1")
