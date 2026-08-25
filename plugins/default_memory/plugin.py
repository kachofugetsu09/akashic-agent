from __future__ import annotations

from agent.plugin_composition import Context, EMBEDDING_MEMORY_PLUGIN

api_version = 3
name = "default_memory"
version = "3.0.0"
desc = "Default memory baseline claim"
author = "Akashic Core"
inject = ()
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context, config: object) -> None:
    """Declare the baseline memory role without a privileged runtime."""

    _ = config
    _ = await ctx.provide(EMBEDDING_MEMORY_PLUGIN, object())
