"""会话工作目录的改动：未提交的差异与每轮前后的差异，只读统计，不改用户的 index、HEAD 与 refs。"""
from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESSES
from plugins.standard_tools.contract import WORKING_DIRECTORY
from plugins.ui.contract import UI

from . import dashboard
from .changes import WORKSPACE_CHANGES, WorkspaceChanges

api_version = 3
name = "workspace_changes"
version = "1.0.0"
desc = "统计会话工作目录（git 仓库）每轮与未提交的改动"
author = "Akashic Core"
inject = (PROCESSES, WORKING_DIRECTORY)
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context) -> None:
    _ = await ctx.provide(WORKSPACE_CHANGES, WorkspaceChanges(ctx))

    # UI 缺席时只是没有浏览器入口，统计照常进行。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(child, dashboard=lambda: dashboard)

    _ = await ctx.inject((UI, WORKSPACE_CHANGES), register_ui, name="workspace-changes-ui")
