"""设计工作台：基于 open-pencil 的设计画布，文档按会话存放。"""
from agent.plugin_composition import Context
from plugins.ui.contract import UI

from . import dashboard
from .store import DESIGN_STORE, DesignStore

api_version = 3
name = "design"
version = "1.0.0"
desc = "工作台里的设计画布（open-pencil）：用户和 Agent 在同一份设计文档上编辑"


async def apply(ctx: Context) -> None:
    _ = await ctx.provide(DESIGN_STORE, DesignStore(ctx.runtime.data_dir / "documents"))

    # UI 缺席时只是没有浏览器入口，文档照常保存在本插件的数据目录里。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(child, dashboard=lambda: dashboard)

    _ = await ctx.inject((UI, DESIGN_STORE), register_ui, name="design-ui")
