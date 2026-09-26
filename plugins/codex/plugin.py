from __future__ import annotations

from agent.plugin_composition import MODEL_DRIVERS, Context
from agent.plugin_composition.ui import UI

from .driver import definition

api_version = 3
name = "codex"
version = "1.0.0"
desc = "ChatGPT login, Codex catalog, and Responses transport"
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
            "models.connection-types.v1": "005155186b59c61f0d67311ce2e0f06dba016d516ba32f3142f0eef754208a4f",
        },
    )


async def apply(ctx: Context) -> None:
    """注册模型驱动；配置界面只影响可选子分支。"""
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
    _ = await ctx.inject((UI,), _register_ui, name="ui")
