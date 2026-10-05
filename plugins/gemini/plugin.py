from agent.plugin_composition import MODEL_DRIVERS, Context

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
    """注册原生 Gemini 对话驱动。"""
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
