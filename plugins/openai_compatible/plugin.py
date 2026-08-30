from __future__ import annotations

from agent.plugin_composition import MODEL_DRIVERS, Context

from .driver import definition

api_version = 3
name = "openai-compatible"
version = "1.0.0"
desc = "OpenAI-compatible Chat Completions and embeddings"
author = "Akashic Core"
inject = (MODEL_DRIVERS,)
web_module = "web_module.js"
web_requires = ("models.connection-types.v1",)
web_provides = ()
web_contract_digests = {
    "models.connection-types.v1": "005155186b59c61f0d67311ce2e0f06dba016d516ba32f3142f0eef754208a4f",
}
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context, config: object) -> None:
    """Register this artifact's one provider-neutral model driver contribution."""

    _ = config
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition())
