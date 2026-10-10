from __future__ import annotations

from agent.plugin_composition import MODEL_DRIVERS, Context
from plugins.ui.contract import UI

from .driver import definition

api_version = 3
name = "opencode-go"
version = "1.0.0"
desc = "OpenCode Go Chat Completions models and local login import"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
workspace_roots = ()
workspace_files = ()


async def _register_ui(ctx: Context) -> None:
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("models.connection-types.v1",),
        provides=(),
        contract_digests={
            "models.connection-types.v1": "8c304d85090a65a4a66cd777a5c2a88e6908de147964dc91e7f84336979ea561",
        },
    )


async def apply(ctx: Context) -> None:
    """注册模型驱动；配置界面只影响可选子分支。"""
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
    _ = await ctx.inject((UI,), _register_ui, name="ui")
