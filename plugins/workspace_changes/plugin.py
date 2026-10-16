"""会话工作目录的改动：未提交的差异与每轮前后的差异，只读统计，不改用户的 index、HEAD 与 refs。"""
from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESSES
from plugins.ledger.contract import MESSAGE_CATALOG, OWNER_STATE
from plugins.sources.contract import SOURCE_CHANGED_V3
from plugins.standard_tools.contract import WORKING_DIRECTORY
from plugins.turn_projection.contract import TURN_PROJECTION
from plugins.ui.contract import UI

from . import dashboard
from .changes import WORKSPACE_CHANGES, WorkspaceChanges

api_version = 3
name = "workspace_changes"
version = "1.0.0"
desc = "统计会话工作目录（git 仓库）每轮与未提交的改动"
author = "Akashic Core"
inject = (PROCESSES, WORKING_DIRECTORY, MESSAGE_CATALOG, OWNER_STATE, TURN_PROJECTION)
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context) -> None:
    changes = WorkspaceChanges(ctx)
    _ = await ctx.provide(WORKSPACE_CHANGES, changes)
    # 插件停用时不采集：窗口任务与监听随本插件的生命周期。
    _ = await ctx.spawn(changes.windows.run(), name="workspace-changes-turns")
    _ = await ctx.on(SOURCE_CHANGED_V3, changes.windows.changed)

    # UI 缺席时只是没有浏览器入口，统计照常进行。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(child, dashboard=lambda: dashboard)

    _ = await ctx.inject((UI, WORKSPACE_CHANGES), register_ui, name="workspace-changes-ui")
