from __future__ import annotations

from agent.plugin_composition import MODEL_DRIVERS, Context

from .driver import definition

api_version = 3
name = "codex"
version = "1.0.0"
desc = "ChatGPT login, Codex catalog, and Responses transport"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context, config: object) -> None:
    """Register the Codex driver in this candidate Root."""

    _ = config
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
