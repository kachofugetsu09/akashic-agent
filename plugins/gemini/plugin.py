from agent.plugin_composition import Context
from plugins.models.contract import MODEL_DRIVERS, register_connection_type_ui

from .driver import definition

api_version = 3
name = "gemini"
version = "1.0.0"
desc = "Gemini 原生 GenerateContent 驱动"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context) -> None:
    """注册原生模型驱动和可选设置界面。"""
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
    await register_connection_type_ui(ctx)
