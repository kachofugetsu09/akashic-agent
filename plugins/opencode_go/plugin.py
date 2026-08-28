from __future__ import annotations

from agent.plugin_composition import MODEL_DRIVERS, Context

from .driver import definition

api_version = 3
name = "opencode-go"
version = "1.0.0"
desc = "OpenCode Go Chat Completions models and local login import"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context, config: object) -> None:
    """Register this artifact's OpenCode Go model driver."""

    _ = config
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
