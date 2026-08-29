from __future__ import annotations

from agent.plugin_composition import MODEL_DRIVERS, Context

from .driver import definition

api_version = 3
name = "openai-compatible"
version = "1.0.0"
desc = "OpenAI-compatible Chat Completions and embeddings"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context, config: object) -> None:
    """Register this artifact's one provider-neutral model driver contribution."""

    _ = config
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
