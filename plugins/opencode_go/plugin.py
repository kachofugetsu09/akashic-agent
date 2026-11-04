from __future__ import annotations

from agent.plugin_composition import Context
from plugins.models.contract import MODEL_DRIVERS, register_connection_type_ui

from .driver import definition

api_version = 3
name = "opencode-go"
version = "1.0.0"
desc = "OpenCode Go Chat Completions models and local login import"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context) -> None:
    """注册模型驱动；配置界面只影响可选子分支。"""
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
    await register_connection_type_ui(ctx)
